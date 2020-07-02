########################################################################################################################
# Module: inference/parameters.py
# Description: Expectation maximisation to infer maximal likelihood hyperparameters.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

from typing import Union, Tuple
import pickle

import numpy as np
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
    :param n_ffbsi: number of samples for FFBSi algorithm
    :param n_iter: number of EM iterations
    :return: dict of optimised parameters
    """

    params_track = {'distance_params': {key: np.asarray(value) for key, value in mm_model.distance_params.items()},
                    'deviation_beta': np.asarray(mm_model.deviation_beta),
                    'gps_sd': np.asarray(mm_model.gps_sd)}

    if isinstance(polylines, np.ndarray):
        polylines = [polylines]

    if isinstance(timestamps, (float, int)):
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
                                           **kwargs)
                         for time_ints_single, polyline in zip(time_interval_arrs, polylines)]

        # Optimise hyperparameters
        optimise_hyperparameters(mm_model, map_matchings, time_interval_arrs, polylines)

        # Update tracking of hyperparameters
        params_track = update_params_track(params_track, mm_model)

        print(f'EM iter: {k}')
        print(params_track)
        pickle.dump(params_track, open('param_track.pickle', 'wb'))

    return params_track


def update_params_track(params_track: dict,
                        mm_model: MapMatchingModel) -> dict:
    """
    Appends latest value to tracking of hyperparameter tuning
    :param params_track: dict of hyperparameters
    :param mm_model: MapMatchingModel with hyperparameters updated
    :return: params_track with new hyperparameters updated
    """
    params_track['distance_params'] = {key: np.append(params_track['distance_params'][key], value)
                                       for key, value in mm_model.distance_params.items()}
    params_track['deviation_beta'] = np.append(params_track['deviation_beta'], mm_model.deviation_beta)
    params_track['gps_sd'] = np.append(params_track['gps_sd'], mm_model.gps_sd)
    return params_track


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
        for i, k in enumerate(mm_model.distance_params.keys()):
            mm_model.distance_params[k] = distance_params_vals[i]
        return -np.sum(np.log(mm_model.distance_prior_evaluate(distances, time_interval_arrs_concat)))

    # Optimise distance params
    optim_dist_params = minimize(distance_optim_func,
                                 np.array([a for a in mm_model.distance_params.values()]),
                                 method='powell',
                                 bounds=[(1e-20, np.inf)] * len(mm_model.distance_params))
                                 # bounds=[(1 + 1e-20, 2 - 1e-20), (1e-20, np.inf), (1e-20, np.inf)])

    for i, k in enumerate(mm_model.distance_params.keys()):
        mm_model.distance_params[k] = optim_dist_params.x[i]

    # Optimise deviation beta
    mm_model.deviation_beta = max(devs.mean(), 10)

    # Optimise GPS noise
    mm_model.gps_sd = min(np.sqrt(np.mean(sq_obs_dists) / 2), 7.5)
    # gps_roots = np.roots([mm_model.gps_sd_lambda/len(sq_obs_dists) * len(map_matchings[0]),
    #                       1,
    #                       0,
    #                       -np.mean(sq_obs_dists) / 2])
    # gps_roots = gps_roots[np.isreal(gps_roots)]
    # gps_roots = gps_roots[gps_roots > 0]
    # mm_model.gps_sd = float(gps_roots[0])

