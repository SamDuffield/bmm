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
        self.gps_sd = None
        self.gps_sd_bounds = (0, np.inf)
        self.likelihood_d_truncate = np.inf

        self.deviation_beta = None
        self.deviation_beta_bounds = (0, np.inf)

        self.max_speed = 35
        self.distance_params = OrderedDict()
        self.distance_params_bounds = OrderedDict()

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

    def pos_distance_prior_bound(self, time_interval: float) -> float:
        """
        Extracts bound on the distance component of the prior/transition density given the distance is > 0
        :param time_interval: seconds, time between observations
        :return: bound on distance prior density
        """
        raise AttributeError("Prior bound not implemented")

    def distance_prior_bound(self, time_interval: float) -> float:
        """
        Extracts bound on the distance component of the prior/transition density
        :param time_interval: seconds, time between observations
        :return: bound on distance prior density
        """
        raise AttributeError("Prior bound not implemented")

    def d_max(self, time_interval: float) -> float:
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


class ExponentialMapMatchingModel(MapMatchingModel):

    def __init__(self,
                 zero_dist_prob_neg_exponent: float = 0.133,
                 lambda_speed: float = 0.068,
                 deviation_beta: float = 0.052,
                 gps_sd: float = 5.23):
        super().__init__()
        self.min_zero_dist_prob = 0.01
        self.max_zero_dist_prob = 0.5
        self.distance_params = OrderedDict({'zero_dist_prob_neg_exponent': zero_dist_prob_neg_exponent,
                                            'lambda_speed': lambda_speed})
        self.distance_params_bounds = OrderedDict({'zero_dist_prob_neg_exponent': (-np.log(self.max_zero_dist_prob)/15,
                                                                                   -np.log(self.min_zero_dist_prob)/15),
                                                   'lambda_speed': (1e-20, np.inf)})
        self.deviation_beta = deviation_beta
        self.gps_sd = gps_sd

    def zero_dist_prob(self,
                       time_interval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Probability of travelling a distance of exactly zero
        :param time_interval: time between last observation and newly received observation
        :return: probability of travelling zero metres in time_interval
        """
        prob = np.exp(- self.distance_params['zero_dist_prob_neg_exponent'] * time_interval)
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
        zero_dist_prob = self.zero_dist_prob(time_interval)

        distance = np.atleast_1d(distance)

        out_arr = np.ones_like(distance) * zero_dist_prob

        non_zero_inds = distance > 1e-5

        if np.sum(non_zero_inds) > 0:
            if np.any(np.atleast_1d(distance[non_zero_inds]) < 0):
                raise ValueError("Exponential pdf takes only positive values")

            time_int_check = time_interval[non_zero_inds] if isinstance(time_interval, np.ndarray) else time_interval
            zero_dist_prob_check = zero_dist_prob[non_zero_inds] if isinstance(time_interval, np.ndarray) \
                else zero_dist_prob

            speeds = distance[non_zero_inds] / time_int_check
            out_arr[non_zero_inds] = self.distance_params['lambda_speed'] \
                                     * np.exp(-self.distance_params['lambda_speed'] * speeds) \
                                     * (1 - zero_dist_prob_check) / time_int_check

        return np.squeeze(out_arr)

    def distance_prior_gradient(self,
                                distance: Union[float, np.ndarray],
                                time_interval: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate gradient of distance prior/transition density in distance_params
        Vectorised to handle multiple evaluations at once
        :param distance: metres
            array if multiple evaluations at once
        :param time_interval: seconds, time between observations
        :return: distance prior gradient evaluation(s)
        """

        distance = np.atleast_1d(distance)
        speeds = distance / time_interval

        out_arr = np.zeros((2, len(distance)))

        non_zero_inds = distance > 1e-5

        if np.any(np.atleast_1d(distance[non_zero_inds]) < 0):
            raise ValueError("Exponential pdf takes only positive values")

        time_int_check = time_interval[non_zero_inds] if isinstance(time_interval, np.ndarray) else time_interval

        out_arr[0] = (- time_interval * ~non_zero_inds
                      + non_zero_inds
                      * self.distance_params['lambda_speed'] * np.exp(-self.distance_params['lambda_speed'] * speeds)) \
                     * self.zero_dist_prob(time_interval)

        out_arr[1, non_zero_inds] = (1 - self.zero_dist_prob(time_int_check)) \
                                    * np.exp(
            -self.distance_params['lambda_speed'] * speeds[non_zero_inds]) / time_int_check \
                                    * (1 - self.distance_params['lambda_speed'] * speeds[non_zero_inds])

        return np.squeeze(out_arr)

    def distance_prior_bound(self,
                             time_interval: float) -> float:
        """
        Extracts bound on the prior/transition density
        :param time_interval: seconds, time between observations
        :return: bound on distance prior density
        """
        zero_dist_prob = self.zero_dist_prob(time_interval)

        distance_bound = max(zero_dist_prob,
                             (1 - zero_dist_prob) * self.distance_params['lambda_speed'] / time_interval)
        return distance_bound

    def pos_distance_prior_bound(self, time_interval: float) -> float:
        """
        Extracts bound on the distance component of the prior/transition density given the distance is > 0
        :param time_interval: seconds, time between observations
        :return: bound on distance prior density
        """
        zero_dist_prob = self.zero_dist_prob(time_interval)
        return (1 - zero_dist_prob) * self.distance_params['lambda_speed'] / time_interval
