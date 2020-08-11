########################################################################################################################
# Module: inference/model.py
# Description: Objects and functions relating to the map-matching state-space model.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

from typing import Union
from collections import OrderedDict

import numpy as np
from numba import njit
from scipy.special import gammainc, digamma, gamma
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm as lognorm_dist
from tweedie import tweedie


@njit
def _intersection_prior_evaluate(between_obs_route: np.ndarray,
                                 intersection_penalisation: float) -> float:
    """
    Evaluate intersection prior - njitted
    :param between_obs_route: edges traversed between observation times
    :return: intersection prior density evaluation
    """
    intersection_col = between_obs_route[:-1, -2]
    return (1 / intersection_col[intersection_col > 0] * intersection_penalisation).prod()


@njit
def _likelihood_evaluate(route_cart_coords: np.ndarray,
                         observation: np.ndarray,
                         gps_sd: float,
                         likelihood_d_truncate: float) -> Union[float, np.ndarray]:
    """
    Evaluate probability of generating observation from cartesian coords - njitted
    Vectorised to evaluate over many cart_coords for a single observation
    Isotropic Gaussian with standard dev self.gps_sd
    :param route_cart_coords: shape = (_, 2), cartesian coordinates - positions along road network
    :param observation: shape = (2,) observed GPS cartesian coordinate
    :return: shape = (_,) likelihood evaluations
    """
    squared_deviations = np.sum((observation - route_cart_coords) ** 2, axis=1)
    evals = np.exp(-0.5 / gps_sd ** 2 * squared_deviations)

    if likelihood_d_truncate < np.inf:
        evals *= squared_deviations < likelihood_d_truncate ** 2

    return evals


class MapMatchingModel:

    def __init__(self):
        self.gps_sd = 5.67
        self.gps_sd_bounds = (0, np.inf)
        self.likelihood_d_truncate = np.inf

        self.intersection_penalisation = 1

        self.deviation_beta = 0.0263
        self.deviation_beta_bounds = (0, np.inf)

        self.zero_dist_prob_neg_exponent = 0.123
        self.min_zero_dist_prob = 0.01
        self.max_zero_dist_prob = 0.9

        self.max_speed = 50
        self.distance_params = OrderedDict()
        self.distance_params_bounds = OrderedDict()

    initiate_speed_mean = 10
    initiate_speed_sd = 5
    initiate_zero_dist_prob_neg_exponent = 0.123

    def zero_dist_prob(self,
                       time_interval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Probability of travelling a distance of exactly zero
        :param time_interval: time between last observation and newly received observation
        :return: probability of travelling zero metres in time_interval
        """
        prob = np.exp(- self.zero_dist_prob_neg_exponent * time_interval)
        prob = np.where(prob < self.min_zero_dist_prob, self.min_zero_dist_prob, prob)
        prob = np.where(prob > self.max_zero_dist_prob, self.max_zero_dist_prob, prob)
        return prob

    def distance_prior_evaluate(self,
                                distance: Union[float, np.ndarray],
                                time_interval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate distance prior/transition density
        Vectorised to handle multiple evaluations at once
        :param distance: metres
            array if multiple evaluations at once
        :param time_interval: seconds, time between observations
        :return: distance prior density evaluation(s)
        """
        raise NotImplementedError

    def distance_prior_gradient(self,
                                distance: Union[float, np.ndarray],
                                time_interval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate gradient of distance prior/transition density in distance_params
        Vectorised to handle multiple evaluations at once
        :param distance: metres
            array if multiple evaluations at once
        :param time_interval: seconds, time between observations
        :return: distance prior density evaluation(s)
        """
        raise AttributeError("Distance prior gradient not implemented")

    def prior_bound(self, time_interval):
        """
        Extracts bound on the prior/transition density
        :param time_interval: seconds, time between observations
        :return: bound on distance prior density
        """
        raise AttributeError("Prior bound not implemented")

    def d_max(self, time_interval):
        """
        Initiates default value of the maximum distance possibly travelled in the time interval.
        Assumes a maximum possible speed.
        :param time_interval: float
            seconds
            time between observations
        :return: float
            defaulted d_max
        """
        return self.max_speed * time_interval

    def deviation_prior_evaluate(self,
                                 previous_cart_coord: np.ndarray,
                                 route_cart_coords: np.ndarray,
                                 distances: np.ndarray) -> np.ndarray:
        """
        Evaluate deviation prior/transition density
        Vectorised to handle multiple evaluations at once
        :param previous_cart_coord: shape = (2,) or (_, 2) cartesian coordinate(s) at previous observation time
        :param route_cart_coords: shape = (_, 2), cartesian coordinates - positions along road network
        :param distances: shape = (_,) route distances between previous_cart_coord(s) and route_cart_coords
        :return: deviation prior density evaluation(s)
        """
        if self.deviation_beta == 0:
            return np.ones(len(route_cart_coords))

        deviations = np.sqrt(np.sum((previous_cart_coord - route_cart_coords) ** 2, axis=1))
        diffs = np.abs(deviations - distances)
        return np.exp(-diffs * self.deviation_beta)

    def intersection_prior_evaluate(self,
                                    between_obs_route: np.ndarray) -> float:
        """
        Evaluate intersection prior.
        :param between_obs_route: edges traversed between observation times
        :return: intersection prior density evaluation
        """
        return _intersection_prior_evaluate(between_obs_route, self.intersection_penalisation)

    def likelihood_evaluate(self,
                            route_cart_coords: np.ndarray,
                            observation: np.ndarray) -> Union[float, np.ndarray]:
        """
        Evaluate probability of generating observation from cartesian coords
        Vectorised to evaluate over many cart_coords for a single observation
        Isotropic Gaussian with standard dev self.gps_sd
        :param route_cart_coords: shape = (_, 2), cartesian coordinates - positions along road network
        :param observation: shape = (2,) observed GPS cartesian coordinate
        :return: shape = (_,) likelihood evaluations
        """
        return _likelihood_evaluate(route_cart_coords, observation, self.gps_sd, self.likelihood_d_truncate)


class GammaMapMatchingModel(MapMatchingModel):

    def __init__(self):
        super().__init__()
        self.distance_params = OrderedDict({'a_speed': 1.,
                                            'b_speed': 0.036})
        self.distance_params_bounds = OrderedDict({'a_speed': (1e-20, np.inf),
                                                   'b_speed': (1e-20, np.inf)})

    def distance_prior_sample(self,
                              time_interval: float) -> float:
        """
        Sample from distance prior
        :param time_interval: time between last observation and newly received observation
        :return: a distance sample in R+ from the prior
        """
        zero_dist_prob = self.zero_dist_prob(time_interval)
        if np.random.uniform() < zero_dist_prob:
            return 0.
        return np.random.gamma(self.distance_params['a_speed'], 1 / self.distance_params['b_speed']) * time_interval

    def distance_prior_evaluate(self,
                                distance: Union[float, np.ndarray],
                                time_interval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate distance prior/transition density
        Vectorised to handle multiple evaluations at once
        :param distance: metres
            array if multiple evaluations at once
        :param time_interval: seconds, time between observations
        :return: distance prior density evaluation(s)
        """
        zero_dist_prob = self.zero_dist_prob(time_interval)

        distance = np.atleast_1d(distance)

        out_arr = np.ones_like(distance) * zero_dist_prob

        non_zero_inds = distance > 1e-5

        if np.sum(non_zero_inds) > 0:
            if np.any(np.atleast_1d(distance[non_zero_inds]) < 0):
                raise ValueError("Gamma pdf takes only positive values")

            time_int_check = time_interval[non_zero_inds] if isinstance(time_interval, np.ndarray) else time_interval
            zero_dist_prob_check = zero_dist_prob[non_zero_inds] if isinstance(time_interval, np.ndarray) \
                else zero_dist_prob

            # out_arr[non_zero_inds] = gamma_dist.pdf(distance[non_zero_inds] / time_int_check,
            #                                         a=self.distance_params['a_speed'],
            #                                         scale=1 / self.distance_params['b_speed']) \
            #                          * (1 - zero_dist_prob_check) / time_int_check

            speeds = distance[non_zero_inds] / time_int_check
            out_arr[non_zero_inds] = self.distance_params['b_speed'] ** self.distance_params['a_speed'] \
                                     / gamma(self.distance_params['a_speed']) \
                                     * speeds ** (self.distance_params['a_speed'] - 1) \
                                     * np.exp(-self.distance_params['b_speed'] * speeds) \
                                     * (1 - zero_dist_prob_check) / time_int_check

        return np.squeeze(out_arr)

    def distance_prior_gradient(self,
                                distance: Union[float, np.ndarray],
                                time_interval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate gradient of distance prior/transition density in distance_params
        Gradient at distances >0 only
        Vectorised to handle multiple evaluations at once
        :param distance: metres
            array if multiple evaluations at once
        :param time_interval: seconds, time between observations
        :return: distance prior gradient evaluation(s)
        """

        distance = np.atleast_1d(distance)

        out_arr = np.zeros((2, len(distance)))

        non_zero_inds = distance > 1e-5

        if np.sum(non_zero_inds) > 0:
            if np.any(np.atleast_1d(distance[non_zero_inds]) < 0):
                raise ValueError("Gamma pdf takes only positive values")

            time_int_check = time_interval[non_zero_inds] if isinstance(time_interval, np.ndarray) else time_interval

            positive_pdf_evals = gamma_dist.pdf(distance[non_zero_inds] / time_int_check,
                                                a=self.distance_params['a_speed'],
                                                scale=1 / self.distance_params['b_speed']) / time_int_check

            out_arr[0, non_zero_inds] = (np.log(self.distance_params['b_speed'])
                                         - digamma(self.distance_params['a_speed'])
                                         + np.log(distance[non_zero_inds] / time_int_check))

            out_arr[1, non_zero_inds] = (self.distance_params['a_speed'] / self.distance_params['b_speed']
                                         - distance[non_zero_inds] / time_int_check)

            out_arr[:, non_zero_inds] *= positive_pdf_evals

        return np.squeeze(out_arr)

    def prior_bound(self,
                    time_interval: float) -> float:
        """
        Extracts bound on the prior/transition density
        :param time_interval: seconds, time between observations
        :return: bound on distance prior density
        """
        zero_dist_prob = self.zero_dist_prob(time_interval)

        if self.distance_params['a_speed'] < 1:
            raise ValueError("Distance prior not bounded")

        gamma_mode = (self.distance_params['a_speed'] - 1) / self.distance_params['b_speed']

        distance_bound = max(self.distance_params['b_speed'] ** self.distance_params['a_speed'] \
                             / gamma(self.distance_params['a_speed']) \
                             * gamma_mode ** (self.distance_params['a_speed'] - 1) \
                             * np.exp(-self.distance_params['b_speed'] * gamma_mode) * (1 - zero_dist_prob)
                             / time_interval,
                             zero_dist_prob)
        return distance_bound


def pdf_gamma_mv(vals: Union[float, np.ndarray],
                 mean: float,
                 var: float) -> Union[float, np.ndarray]:
    """
    Evaluates Gamma pdf (uses moment matching based on received mean and variance), up to normalisation constant
    :param vals: values to be evaluated
    :param mean: inputted distribution mean
    :param var: inputted distribution variance
    :return: Gamma pdf evaulations, same length as vals
    """
    gamma_beta = mean / var
    gamma_alpha = mean * gamma_beta

    # gamma_func_alpha = gamma(gamma_alpha)
    # return gamma_beta ** gamma_alpha / gamma_func_alpha * vals ** (gamma_alpha - 1) * np.exp(-gamma_beta * vals)
    return gamma_dist.pdf(vals, a=gamma_alpha, scale=1 / gamma_beta)


def cdf_gamma_mv(vals: Union[float, np.ndarray],
                 mean: float,
                 var: float) -> Union[float, np.ndarray]:
    """
    Evaluates Gamma cdf (uses moment matching based on received mean and variance).
    :param vals: values to be evaluated
    :param mean: inputted distribution mean
    :param var: inputted distribution variance
    :return: Gamma cdf evaulations, same length as vals
    """
    gamma_beta = mean / var
    gamma_alpha = mean * gamma_beta

    if any(np.atleast_1d(vals) <= 0):
        print(vals)
        raise ValueError("Gamma pdf takes only positive values")

    return gammainc(gamma_alpha, gamma_beta * vals)
#
#
# class LogNormalMapMatchingModel(MapMatchingModel):
#     # distance_params_bounds = [(1e-20, np.inf)] * 3
#     distance_params_bounds = [(1e-20, np.inf), (1e-20, np.inf), (0.01, 0.25)]
#
#     def __init__(self):
#         super().__init__()
#         self.distance_params = {'scale_speed': self.initiate_speed_mean ** 2
#                                                / np.sqrt(self.initiate_speed_mean ** 2 + self.initiate_speed_sd ** 2)}
#         self.distance_params['s_speed'] = np.log(1 + self.initiate_speed_sd ** 2 / self.initiate_speed_mean ** 2)
#
#         self.distance_params['zero_dist_prob_neg_exponent'] = self.initiate_zero_dist_prob_neg_exponent
#
#     def zero_dist_prob(self,
#                        time_interval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """
#         Probability of travelling a distance of exactly zero
#         :param time_interval: time between last observation and newly received observation
#         :return: probability of travelling zero metres in time_interval
#         """
#         return np.exp(- self.distance_params['zero_dist_prob_neg_exponent'] * time_interval)
#
#     def distance_prior_evaluate(self,
#                                 distance: Union[float, np.ndarray],
#                                 time_interval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """
#         Evaluate distance prior/transition density
#         Vectorised to handle multiple evaluations at once
#         :param time_interval: seconds, time between observations
#         :param distance: metres
#             array if multiple evaluations at once
#         :return: distance prior density evaluation(s)
#         """
#         zero_dist_prob = self.zero_dist_prob(time_interval)
#
#         distance = np.atleast_1d(distance)
#
#         out_arr = np.ones_like(distance) * zero_dist_prob
#
#         non_zero_inds = distance > 0
#
#         if np.sum(non_zero_inds) > 0:
#             if np.any(np.atleast_1d(distance[non_zero_inds]) < 0):
#                 raise ValueError("Gamma pdf takes only positive values")
#
#             time_int_check = time_interval[non_zero_inds] if isinstance(time_interval, np.ndarray) else time_interval
#             zero_dist_prob_check = zero_dist_prob[non_zero_inds] if isinstance(time_interval, np.ndarray) \
#                 else zero_dist_prob
#
#             out_arr[non_zero_inds] = lognorm_dist.pdf(distance[non_zero_inds] / time_int_check,
#                                                       scale=self.distance_params['scale_speed'],
#                                                       s=self.distance_params['s_speed']) \
#                                      * (1 - zero_dist_prob_check)
#
#         return np.squeeze(out_arr)
#
#     def prior_bound(self,
#                     time_interval: float) -> float:
#         """
#         Extracts bound on the prior/transition density
#         :param time_interval: seconds, time between observations
#         :return: bound on distance prior density
#         """
#         zero_dist_prob = self.zero_dist_prob(time_interval)
#
#         distance_bound = max(np.exp(-self.distance_params['s_speed'] ** 2)
#                              / (self.distance_params['s_speed'] * self.distance_params['scale_speed']
#                                 * np.exp(-self.distance_params['s_speed']) * np.sqrt(2 * np.pi)),
#                              zero_dist_prob)
#
#         return distance_bound if self.deviation_beta == np.inf else distance_bound / self.deviation_beta
#
#
# class TweedieMapMatchingModel(MapMatchingModel):
#     # deviation_beta = 3.464
#     deviation_beta = 8
#     gps_sd = 7.512
#     distance_params = {'p_speed': 1.906,
#                        'mu_speed': 8.650,
#                        'phi_speed': 2.528}
#
#     distance_params_bounds = [(1 + 1e-20, 2 - 1e-20), (1e-20, np.inf), (1e-20, np.inf)]
#
#     def distance_prior_evaluate(self,
#                                 distance: Union[float, np.ndarray],
#                                 time_interval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """
#         Evaluate distance prior/transition density
#         Vectorised to handle multiple evaluations at once
#         :param time_interval: seconds, time between observations
#         :param distance: metres
#             array if multiple evaluations at once
#         :return: distance prior density evaluation(s)
#         """
#         return tweedie.pdf(distance / time_interval,
#                            p=self.distance_params['p_speed'],
#                            mu=self.distance_params['mu_speed'],
#                            phi=self.distance_params['phi_speed'])
