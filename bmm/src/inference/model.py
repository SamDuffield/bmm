########################################################################################################################
# Module: inference/model.py
# Description: Objects and functions relating to the state-space model.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

from typing import Union

import numpy as np
from numba import njit
from scipy.special import gammainc
from scipy.stats import gamma as gamma_dist


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
    gps_sd = 25
    intersection_penalisation = 1
    deviation_beta = 10
    max_speed = 35
    likelihood_d_truncate = np.inf
    distance_params = {}

    def distance_prior_evaluate(self,
                                distance: Union[float, np.ndarray],
                                time_interval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate distance prior/transition density
        Vectorised to handle multiple evaluations at once
        :param time_interval: seconds, time between observations
        :param distance: metres
            array if multiple evaluations at once
        :return: distance prior density evaluation(s)
        """
        raise NotImplementedError

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
        if self.deviation_beta == np.inf:
            return np.ones(len(route_cart_coords))

        deviations = np.sqrt(np.sum((previous_cart_coord - route_cart_coords) ** 2, axis=1))
        diffs = np.abs(deviations - distances)
        return np.exp(-diffs / self.deviation_beta) / self.deviation_beta

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
    speed_mean = 7.44
    speed_sd = 6.88

    distance_params = {'b_speed': speed_mean / speed_sd ** 2}
    distance_params['a_speed'] = speed_mean * distance_params['b_speed']
    distance_params['zero_dist_prob_neg_exponent'] = 0.25

    # asymp_min_zero_prob = 0.01
    # def zero_dist_prob(self,
    #                    time_interval: float) -> float:
    #     """
    #     Probability of travelling a distance of exactly zero
    #     :param time_interval: time between last observation and newly received observation
    #     :return: probability of travelling zero metres in time_interval
    #     """
    #     exp_param = (np.log(1 - self.asymp_min_zero_prob) - np.log(0.044 - self.asymp_min_zero_prob)) / 15
    #     return self.asymp_min_zero_prob + (1 - self.asymp_min_zero_prob) * np.exp(- exp_param * time_interval)

    def zero_dist_prob(self,
                       time_interval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Probability of travelling a distance of exactly zero
        :param time_interval: time between last observation and newly received observation
        :return: probability of travelling zero metres in time_interval
        """
        return np.exp(- self.distance_params['zero_dist_prob_neg_exponent'] * time_interval)

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
        :param time_interval: seconds, time between observations
        :param distance: metres
            array if multiple evaluations at once
        :return: distance prior density evaluation(s)
        """
        zero_dist_prob = self.zero_dist_prob(time_interval)

        distance = np.atleast_1d(distance)

        out_arr = np.ones_like(distance) * zero_dist_prob

        non_zero_inds = distance > 0

        if np.sum(non_zero_inds) > 0:
            if np.any(np.atleast_1d(distance[non_zero_inds]) < 0):
                raise ValueError("Gamma pdf takes only positive values")

            time_int_check = time_interval[non_zero_inds] if isinstance(time_interval, np.ndarray) else time_interval
            zero_dist_prob_check = zero_dist_prob[non_zero_inds] if isinstance(time_interval, np.ndarray)\
                else zero_dist_prob

            out_arr[non_zero_inds] = gamma_dist.pdf(distance[non_zero_inds] / time_int_check,
                                                    a=self.distance_params['a_speed'],
                                                    scale=1 / self.distance_params['b_speed'])\
                                     * (1 - zero_dist_prob_check)

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

        distance_bound = max(gamma_dist.pdf(gamma_mode,
                                            a=self.distance_params['a_speed'],
                                            scale=1 / self.distance_params['b_speed']) * (1 - zero_dist_prob),
                             zero_dist_prob)

        return distance_bound if self.deviation_beta == np.inf else distance_bound / self.deviation_beta


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
