########################################################################################################################
# Module: inference/parameters.py
# Description: Expectation maximisation to infer maximal likelihood hyperparameters.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

from typing import Union, Tuple

import numpy as np
from numba import njit
from networkx.classes import MultiDiGraph
from scipy.optimize import minimize

from bmm.src.inference.model import MapMatchingModel
from bmm.src.inference.smc import offline_map_match, get_time_interval_array
from bmm.src.tools.edges import observation_time_rows


def offline_em(graph: MultiDiGraph,
               mm_model: MapMatchingModel,
               timestamps: Union[list, float],
               polylines: list,
               n_ffbsi: int = 100,
               n_iter: int = 10,
               **kwargs):
    """
    Run expectation maximisation to optimise prior hyperparameters.
    Updates the hyperparameters of mm_model in place.
    :param graph: encodes road network, simplified and projected to UTM
    :param mm_model: MapMatchingModel - of which parameters will be updated
    :param timestamps: seconds
        either float if all times between observations are the same, or a series of timestamps in seconds/UNIX timestamp
        if timestamps given, must be in a list matching dimensions of polylines
    :param polylines: UTM polylines
    :param n_iter: number of EM iterations
    :return: dict of optimised parameters
    """

    if isinstance(polylines, np.ndarray):
        polylines = [polylines]

    if isinstance(timestamps, float):
        timestamps = [timestamps] * len(polylines)

    time_interval_arrs = [get_time_interval_array(timestamps_single, len(polyline))
                          for timestamps_single, polyline in zip(timestamps, polylines)]

    for k in range(n_iter):
        # Run FFBSi over all given polylines with latest hyperparameters
        map_matchings = [offline_map_match(graph,
                                           polyline,
                                           n_ffbsi,
                                           time_ints_single,
                                           mm_model,
                                           n_samps=n_ffbsi,
                                           **kwargs)
                         for time_ints_single, polyline in zip(time_interval_arrs, polylines)]

        # Optimise hyperparameters
        optimise_hyperparameters(mm_model, map_matchings, timestamps, polylines)


def extract_mm_quantities(map_matching: list,
                          polyline: np.ndarray) -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Extract required statistics for parameter optimisation from map-matching results.
    :param map_matching: MMParticles.particles list
    :param polyline: for single route
    :return: distances, deviations and squared observation-position distances
    """
    distances = []
    devs = np.array([])
    sq_obs_dists = np.array([])
    for particle in map_matching:
        particle_obs_time_rows = observation_time_rows(particle)
        distances_particle = particle_obs_time_rows[1:, -1]
        distances.append(distances_particle)

        devs_particle = np.abs(distances_particle - np.sqrt(np.sum(np.square(particle_obs_time_rows[1:, 5:7]
                                                                             - particle_obs_time_rows[:-1, 5:7]),
                                                                   axis=1)))
        devs = np.append(devs, devs_particle)

        sq_obs_dists = np.append(sq_obs_dists, np.sum(np.square(particle_obs_time_rows[:, 5:7] - polyline), axis=1))

    return distances, devs, sq_obs_dists


def optimise_hyperparameters(mm_model: MapMatchingModel,
                             map_matchings: list,
                             time_interval_arrs: list,
                             polylines: list):
    """
    For given map-matching results, optimise model hyperparameters.
    Updates mm_model hyperparameters in place
    :param mm_model: MapMatchingModel
    :param map_matchings: list of MMParticles objects
    :param time_interval_arrs: time interval arrays for each route
    :param polylines: observations for each route
    """

    # Get key quantities
    distances = np.array([])
    time_interval_arrs_concat = np.array([])
    devs = np.array([])
    sq_obs_dists = np.array([])
    for map_matching, time_interval_arr, polyline in zip(map_matchings, time_interval_arrs, polylines):
        distances_single, devs_single, sq_obs_dists_single = extract_mm_quantities(map_matching.particles,
                                                                                   polyline)
        distances = np.append(distances, np.concatenate(distances_single))
        time_interval_arrs_concat = np.append(time_interval_arrs_concat,
                                              np.concatenate([time_interval_arr] * len(map_matching)))

        devs = np.append(devs, devs_single)
        sq_obs_dists = np.append(sq_obs_dists, sq_obs_dists_single)

    # Optimise distance params
    def distance_optim_func(distance_params_vals: np.ndarray) -> float:
        for i, k in enumerate(mm_model.distance_params.keys):
            mm_model.distance_params[k] = distance_params_vals[i]

        return -np.sum(mm_model.distance_prior_evaluate(distances, time_interval_arrs_concat))\
               / len(map_matchings[0])

    # Optimise distance params
    optim_dist_params = minimize(distance_optim_func,
                                 np.array([a for a in mm_model.distance_params.values()]))
    for i, k in enumerate(mm_model.distance_params.keys):
        mm_model.distance_params[k] = optim_dist_params.x[i]

    # Optimise deviation beta
    mm_model.deviation_beta = np.mean(devs)

    # Optimise GPS noise
    mm_model.deviation_beta = np.mean(sq_obs_dists) / 2
