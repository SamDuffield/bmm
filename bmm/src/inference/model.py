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
    r"""
    Class defining the state-space model used for map-matching.

    **Transition density** (assuming constant time interval)
        .. math:: p(x_t, e_t | x_{t-1}) \propto \gamma(d_t) \exp(-\beta|d^\text{gc}_t - d_t|)\mathbb{1}[d_t < d_\text{max}],

    where :math:`d_t` is the distance between positions :math:`x_{t-1}` and :math:`x_{t}` along the series of edges
    :math:`e_{t-1}`, restricted to the graph/road network.
    :math:`d^\text{gc}_t` is the *great circle distance* between :math:`x_{t-1}` and :math:`x_{t}`,
    not restricted to the graph/road network.

    The :math:`\exp(-\beta|d^\text{gc}_t - d_t|)` term penalises non-direct or windy routes where :math:`\beta` is a
    parameter stored in ``self.deviation_beta``, yet to be defined.

    :math:`d_\text{max}` is defined by ``self.d_max`` function (metres)
    and ``self.max_speed`` parameter (metres per second), defaults to 35.

    The :math:`\gamma(d_t)` term penalises overly lengthy routes and is yet to be defined.

    **Observation density**
        .. math:: p(y_t| x_{t}) = \mathcal{N}(y_t \mid x_t, \sigma_\text{GPS}^2 \mathbb{I}_2),

    where :math:`\sigma_\text{GPS}` is the standard deviation (metres) of the GPS noise stored in ``self.gps_sd``,
    yet to be defined. Additional optional ``self.likelihood_d_truncate`` for truncated Gaussian noise, defaults to inf.

    The parameters ``self.deviation_beta``, ``self.gps_sd`` and the distance prior parameters defined in
    ``self.distance_params`` and ``self.distance_params_bounds`` can be tuned using expectation-maximisation with
    ``bmm.offline_em``.

    For more details see https://arxiv.org/abs/2012.04602.

    """

    __module__ = 'bmm'

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
    r"""
    Class defining the state-space model used for map-matching with exponential prior on distance travelled.

    **Transition density** (assuming constant time interval)
        .. math:: p(x_t, e_t | x_{t-1}) \propto \gamma(d_t) \exp(-\beta|d^\text{gc}_t - d_t|)\mathbb{1}[d_t < d_\text{max}],

    where :math:`d_t` is the distance between positions :math:`x_{t-1}` and :math:`x_{t}` along the series of edges
    :math:`e_{t-1}`, restricted to the graph/road network.
    :math:`d^\text{gc}_t` is the *great circle distance* between :math:`x_{t-1}` and :math:`x_{t}`,
    not restricted to the graph/road network.

    The :math:`\exp(-\beta|d^\text{gc}_t - d_t|)` term penalises non-direct or windy routes where :math:`\beta` is a
    parameter stored in ``self.deviation_beta`` defaults to 0.052.

    :math:`d_\text{max}` is defined by ``self.d_max`` function (metres) and ``self.max_speed`` parameter
    (metres per second), defaults to 35.

    The :math:`\gamma(d_t)` term
        .. math:: \gamma(d_t) = p^0\mathbb{1}[d_t = 0] + (1-p^0) \mathbb{1}[d_t > 0] \lambda \exp(-\lambda d_t/\Delta t),

    penalises overly lengthy routes, defined as an exponential distribution with
    probability mass at :math:`d_t=0` to account for traffic, junctions etc.

    where :math:`p^0 = \exp(-r^0 \Delta t)` with :math:`\Delta t` being the time interval between observations.
    The :math:`r^0` parameter is stored in ``self.zero_dist_prob_neg_exponent`` and defaults to 0.133.
    Exponential distribution parameter :math:`\lambda` is stored in ``self.lambda_speed`` and defaults to 0.068.

    **Observation density**
        .. math:: p(y_t| x_{t}) = \mathcal{N}(y_t \mid x_t, \sigma_\text{GPS}^2 \mathbb{I}_2),

    where :math:`\sigma_\text{GPS}` is the standard deviation (metres) of the GPS noise stored in ``self.gps_sd``,
    defaults to 5.23. Additional optional ``self.likelihood_d_truncate`` for truncated Gaussian noise, defaults to inf.

    The parameters ``self.deviation_beta``, ``self.gps_sd`` as well as the distance prior parameters
    ``self.zero_dist_prob_neg_exponent`` and ``self.lambda_speed`` can be tuned using expectation-maximisation
    with ``bmm.offline_em``.

    For more details see https://arxiv.org/abs/2012.04602.

    :param zero_dist_prob_neg_exponent: Positive parameter such that stationary probability
        is :math:`p^0 = \exp(-r^0 \Delta t)`, defaults to 0.133.
    :param lambda_speed: Positive parameter of exponential distribution over average speed between observations.
    :param deviation_beta: Positive parameter of exponential distribution over route deviation.
    :param gps_sd: Positive parameter defining standard deviation of GPS noise in metres.

    """
    __module__ = 'bmm'

    def __init__(self,
                 zero_dist_prob_neg_exponent: float = 0.133,
                 lambda_speed: float = 0.068,
                 deviation_beta: float = 0.052,
                 gps_sd: float = 5.23):
        """
        Initiate parameters of map-matching model with exponential prior on distance travelled between observations.
        :param zero_dist_prob_neg_exponent: Positive parameter such that stationary probability is
        :math:`p^0 = \exp(-r^0 \Delta t)`, defaults to 0.133.
        :param lambda_speed: Positive parameter of exponential distribution over average speed between observations.
        :param deviation_beta: Positive parameter of exponential distribution over route deviation.
        :param gps_sd: Positive parameter defining standard deviation of GPS noise in metres.
        """
        super().__init__()
        self.min_zero_dist_prob = 0.01
        self.max_zero_dist_prob = 0.5
        self.distance_params = OrderedDict({'zero_dist_prob_neg_exponent': zero_dist_prob_neg_exponent,
                                            'lambda_speed': lambda_speed})
        self.distance_params_bounds = OrderedDict(
            {'zero_dist_prob_neg_exponent': (-np.log(self.max_zero_dist_prob) / 15,
                                             -np.log(self.min_zero_dist_prob) / 15),
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
