########################################################################################################################
# Module: inference/model.py
# Description: Functions related to the state-space model, i.e. prior density on the distance travelled.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import numpy as np
from scipy.special import gamma, gammainc


intersection_penalisation = 1


class MapMatchingModel:

    gps_sd = None
    max_speed = None
    intersection_penalisation = 1

    def zero_dist_prob(self, time_interval):
        raise NotImplementedError

    def distance_prior_evaluate(self, distance, time_interval):
        raise NotImplementedError

    def distance_prior_bound(self, time_interval):
        raise NotImplementedError("Distance bound not implemented")

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


class SimpleMapMatchingModel(MapMatchingModel):

    gps_sd = 7
    max_speed = 35
    intersection_penalisation = 1

    speed_mean = 7.44
    speed_sd = 6.88

    def zero_dist_prob(self, time_interval):
        return 0.044

    def distance_prior_sample(self, time_interval):
        zero_dist_prob = self.zero_dist_prob(time_interval)
        if np.random.uniform() < zero_dist_prob:
            return 0.

        gamma_beta = self.speed_mean / self.speed_sd ** 2
        gamma_alpha = self.speed_mean * gamma_beta
        return np.random.gamma(gamma_alpha, 1 / gamma_beta)

    def distance_prior_evaluate(self, distance, time_interval):
        zero_dist_prob = self.zero_dist_prob(time_interval)

        distance = np.atleast_1d(distance)

        out_arr = np.ones_like(distance) * zero_dist_prob

        non_zero_inds = distance > 0

        if np.sum(non_zero_inds) > 0:
            out_arr[non_zero_inds] = pdf_gamma_mv(distance[non_zero_inds] / time_interval,
                                                  self.speed_mean,
                                                  self.speed_sd ** 2) \
                                     * (1 - zero_dist_prob)

        return np.squeeze(out_arr)

    def distance_prior_bound(self, time_interval):
        zero_dist_prob = self.zero_dist_prob(time_interval)
        gamma_beta = self.speed_mean / self.speed_sd ** 2
        gamma_alpha = self.speed_mean * gamma_beta

        if gamma_alpha < 1:
            raise ValueError("Distance prior not bounded")

        gamma_mode = (gamma_alpha - 1) / gamma_beta

        return max(pdf_gamma_mv(gamma_mode, self.speed_mean, self.speed_sd ** 2) * (1 - zero_dist_prob), zero_dist_prob)


def pdf_gamma_mv(vals, mean, var):
    """
    Evaluates Gamma pdf (uses moment matching based on received mean and variance).
    :param vals: float or np.array
        values to be evaluated
    :param mean: float
        inputted distribution mean
    :param var: float
        inputted distribution variance
    :return: np.array, same length as vals
        Gamma pdf evaulations
    """
    gamma_beta = mean / var
    gamma_alpha = mean * gamma_beta

    gamma_func_alpha = gamma(gamma_alpha)

    if any(np.atleast_1d(vals) <= 0):
        print(vals)
        raise ValueError("Gamma pdf takes only positive values")

    return gamma_beta ** gamma_alpha / gamma_func_alpha * vals ** (gamma_alpha - 1) * np.exp(-gamma_beta * vals)


def cdf_gamma_mv(vals, mean, var):
    """
    Evaluates Gamma cdf (uses moment matching based on received mean and variance).
    :param vals: float or np.array
        values to be evaluated
    :param mean: float
        inputted distribution mean
    :param var: float
        inputted distribution variance
    :return: np.array, same length as vals
        Gamma pdf evaulations
    """
    gamma_beta = mean / var
    gamma_alpha = mean * gamma_beta

    if any(np.atleast_1d(vals) <= 0):
        print(vals)
        raise ValueError("Gamma pdf takes only positive values")

    return gammainc(gamma_alpha, gamma_beta*vals)











































#
#
#
#
# def default_d_max(d_max, time_interval, max_speed=35):
#     """
#     Initiates default value of the maximum distance possibly travelled in the time interval.
#     Assumes a maximum possible speed.
#     :param d_max: float or None
#         metres
#         value to be checked
#     :param time_interval: float
#         seconds
#         time between observations
#     :param max_speed: float
#         metres per second
#         assumed maximum possible speed
#     :return: float
#         defaulted d_max
#     """
#     return max_speed * time_interval if d_max is None else d_max
#
#
#
# def distance_prior(distance, time_interval, speed_mean=7.44, speed_var=47.38, zero_prob=0.044):
#     """
#     Evaluates prior probability of distance travelled in time interval.
#     :param distance: float or np.array
#         metres
#         distance values to be evaluated
#     :param time_interval: float
#         seconds
#         time between observations
#     :param speed_mean: float
#         metres
#         mean of gamma prior
#     :param speed_var: float
#         metres^2
#         variance of gamma prior
#     :param zero_prob: float in [0,1]
#         probability of distance = 0 metres
#     :return: float or np.array with values in [0,1], same length as distance
#         prior pdf evaluations
#     """
#     distance = np.atleast_1d(distance)
#
#     out_arr = np.ones_like(distance) * zero_prob
#
#     non_zero_inds = distance > 0
#
#     if np.sum(non_zero_inds) > 0:
#         out_arr[non_zero_inds] = pdf_gamma_mv(distance[non_zero_inds]/time_interval, speed_mean, speed_var)\
#             * (1 - zero_prob)
#
#     return np.squeeze(out_arr)
#
#
# def get_distance_prior_bound(speed_mean=7.44, speed_var=47.38, zero_prob=0.044):
#     """
#     Extracts upper bound for distance prior
#     :param speed_mean: float
#         metres
#         mean of gamma prior
#     :param speed_var: float
#         metres^2
#         variance of gamma prior
#     :param zero_prob: float in [0,1]
#         probability of distance = 0 metres
#     :return: float
#         upper bound
#     """
#
#     gamma_beta = speed_mean / speed_var
#     gamma_alpha = speed_mean * gamma_beta
#
#     if gamma_alpha < 1:
#         raise ValueError("Distance prior not bounded")
#
#     gamma_mode = (gamma_alpha - 1) / gamma_beta
#
#     return max(pdf_gamma_mv(gamma_mode, speed_mean, speed_var) * (1-zero_prob), zero_prob)

