########################################################################################################################
# Module: inference/model.py
# Description: Functions related to the state-space model, i.e. prior density on the distance travelled.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import numpy as np
from scipy import stats


def default_d_max(d_max, time_interval, max_speed=35):
    """
    Initiates default value of the maximum distance possibly travelled in the time interval.
    Assumes a maximum possible speed.
    :param d_max: float or None
        metres
        value to be checked
    :param time_interval: float
        seconds
        time between observations
    :param max_speed: float
        metres per second
        assumed maximum possible speed
    :return: float
        defaulted d_max
    """
    return max_speed * time_interval if d_max is None else d_max


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

    if any(np.atleast_1d(vals) <= 0):
        print(vals)
        raise ValueError("Gamma pdf takes only positive values")

    return stats.gamma.pdf(vals, a=gamma_alpha, scale=1/gamma_beta)


def distance_prior(distance, time_interval, speed_mean=7.21, speed_var=47.55):
    """
    Evaluates prior probability of distance travelled in time interval.
    :param distance: float or np.array
        metres
        distance values to be evaluated
    :param time_interval: float
        seconds
        time between observations
    :param speed_mean: float
        metres
        mean of gamma prior
    :param speed_var: float
        metres^2
        variance of gamma prior
    :return: float or np.array with values in [0,1], same length as distance
        prior pdf evaluations
    """
    return pdf_gamma_mv(distance/time_interval, speed_mean, speed_var)

