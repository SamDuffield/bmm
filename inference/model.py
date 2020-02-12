########################################################################################################################
# Module: inference/model.py
# Description: Functions related to the state-space model, i.e. prior density on the distance travelled.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import numpy as np
from scipy.special import gamma as gamma_func


def pdf_gamma_mv(vals, mean, var):
    """
    Evaluates Gamma pdf with parameters adjusted to give an inputted mean and variance.
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
        raise ValueError("Gamma pdf takes only positive values")

    return gamma_beta ** gamma_beta / gamma_func(gamma_alpha) * vals ** (gamma_alpha - 1) * np.exp(-gamma_beta * vals)


def distance_prior(distance, mean=108, var=10700):
    """
    Evaluates prior probability of distance, assumes time interval of 15 seconds.
    :param distance: float or np.array
        distance values to be evaluated
    :return: float or np.array with values in [0,1], same length as distance
        prior pdf evaluations
    """
    return pdf_gamma_mv(distance, mean, var)










