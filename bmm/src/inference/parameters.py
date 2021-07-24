########################################################################################################################
# Module: inference/parameters.py
# Description: Expectation maximisation to infer maximum likelihood hyperparameters.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

from typing import Union, Tuple
import pickle

import numpy as np
from networkx.classes import MultiDiGraph
from scipy.optimize import minimize

from bmm.src.inference.particles import MMParticles
from bmm.src.inference.model import MapMatchingModel
from bmm.src.inference.smc import get_time_interval_array, offline_map_match
from bmm.src.tools.edges import observation_time_rows


def offline_em(graph: MultiDiGraph,
               mm_model: MapMatchingModel,
               timestamps: Union[list, float],
               polylines: list,
               save_path: str,
               n_ffbsi: int = 100,
               n_iter: int = 10,
               gradient_stepsize_scale: float = 1e-3,
               gradient_stepsize_neg_exp: float = 0.5,
               **kwargs):
    """
    Run expectation maximisation to optimise parameters of bmm.MapMatchingModel object.
    Updates the hyperparameters of mm_model in place.

    :param graph: encodes road network, simplified and projected to UTM
    :param mm_model: MapMatchingModel - of which parameters will be updated
    :param timestamps: seconds, either float if all times between observations are the same, or a series of timestamps
            in seconds/UNIX timestamp, if timestamps given, must be of matching dimensions to polylines
    :param polylines: UTM polylines
    :param save_path: path to save learned parameters
    :param n_ffbsi: number of samples for FFBSi algorithm
    :param n_iter: number of EM iterations
    :param gradient_stepsize_scale: starting stepsize
    :param gradient_stepsize_neg_exp: rate of decay of stepsize, in [0.5, 1]
    :param kwargs: additional arguments for FFBSi
    :return: dict of optimised parameters

    """

    params_track = {'distance_params': {key: np.asarray(value) for key, value in mm_model.distance_params.items()},
                    'deviation_beta': np.asarray(mm_model.deviation_beta),
                    'gps_sd': np.asarray(mm_model.gps_sd)}

    if isinstance(polylines, np.ndarray):
        polylines = [polylines]

    if isinstance(timestamps, (float, int)):
        timestamps = [timestamps] * len(polylines)

    # If no deviation prior - can optimise prior directly, otherwise can only take gradient step
    no_deviation_prior = mm_model.deviation_beta_bounds[1] == 0
    if no_deviation_prior:
        mm_model.deviation_beta = 0

    time_interval_arrs_full = [get_time_interval_array(timestamps_single, len(polyline))
                               for timestamps_single, polyline in zip(timestamps, polylines)]

    for k in range(n_iter):
        # Run FFBSi over all given polylines with latest hyperparameters
        mm_ind = 0
        map_matchings = []
        time_interval_arrs_int = []
        polylines_int = []
        for time_ints_single, polyline in zip(time_interval_arrs_full, polylines):
            print(f'Polyline {mm_ind}')
            success = True
            try:
                mm = offline_map_match(graph,
                                       polyline,
                                       n_ffbsi,
                                       time_ints_single,
                                       mm_model,
                                       store_norm_quants=not no_deviation_prior,
                                       **kwargs)
            except ValueError:
                print(f'Map-matching {mm_ind} failed')
                success = False
            if success:
                map_matchings.append(mm)
                time_interval_arrs_int.append(time_ints_single)
                polylines_int.append(polyline)
            mm_ind += 1

        if no_deviation_prior:
            # Optimise hyperparameters
            optimise_hyperparameters(mm_model, map_matchings, time_interval_arrs_int, polylines_int)
        else:
            # Take gradient step
            gradient_em_step(mm_model, map_matchings, time_interval_arrs_int, polylines_int,
                             gradient_stepsize_scale / (k + 1) ** gradient_stepsize_neg_exp)

        # Update tracking of hyperparameters
        params_track = update_params_track(params_track, mm_model)

        print(f'EM iter: {k}')
        print(params_track)
        pickle.dump(params_track, open(save_path, 'wb'))

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


def extract_mm_quantities(map_matching: MMParticles,
                          polyline: np.ndarray,
                          extract_devs: bool = True) -> tuple:
    """
    Extract required statistics for parameter optimisation from map-matching results.
    :param map_matching: MMParticles.particles list
    :param polyline: for single route
    :param extract_devs: whether to extract deviations (and gradient quantities)
    :return: distances, deviations and squared observation-position distances
    """
    distances = np.array([])
    devs = np.array([])
    sq_obs_dists = np.array([])
    for particle in map_matching:
        particle_obs_time_rows = observation_time_rows(particle)
        distances_particle = particle_obs_time_rows[1:, -1]
        distances = np.append(distances, distances_particle)

        if extract_devs:
            devs_particle = np.abs(distances_particle - np.sqrt(np.sum(np.square(particle_obs_time_rows[1:, 5:7]
                                                                                 - particle_obs_time_rows[:-1, 5:7]),
                                                                       axis=1)))
            devs = np.append(devs, devs_particle)

        sq_obs_dists = np.append(sq_obs_dists, np.sum(np.square(particle_obs_time_rows[:, 5:7] - polyline), axis=1))

    if extract_devs:
        # Z, *dZ/dalpha, dZ/dbeta where alpha = distance_params and beta = deviation_beta
        dev_norm_quants = np.concatenate(map_matching.dev_norm_quants)
        return distances, (devs, dev_norm_quants), sq_obs_dists
    else:
        return distances, sq_obs_dists


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
    sq_obs_dists = np.array([])
    for map_matching, time_interval_arr, polyline in zip(map_matchings, time_interval_arrs, polylines):
        distances_single, sq_obs_dists_single = extract_mm_quantities(map_matching,
                                                                      polyline,
                                                                      extract_devs=False)
        distances = np.append(distances, np.concatenate(distances_single))
        time_interval_arrs_concat = np.append(time_interval_arrs_concat,
                                              np.concatenate([time_interval_arr] * len(map_matching)))
        sq_obs_dists = np.append(sq_obs_dists, sq_obs_dists_single)

    # # Optimise zero dist prob
    # def zero_dist_prob_root_func(neg_exp: float) -> float:
    #     return - np.sum(- time_interval_arrs_concat * (distances < 1e-5)
    #                     + time_interval_arrs_concat * np.exp(-neg_exp * time_interval_arrs_concat)
    #                     / (1 - np.exp(-neg_exp * time_interval_arrs_concat)) * (distances >= 1e-5))
    #
    # mm_model.zero_dist_prob_neg_exponent = root_scalar(zero_dist_prob_root_func, bracket=(1e-3, 1e20)).root
    #
    # pos_distances = distances[distances > 1e-5]
    # pos_time_interval_arrs_concat = time_interval_arrs_concat[distances > 1e-5]

    pos_distances = distances
    pos_time_interval_arrs_concat = time_interval_arrs_concat

    bounds = list(mm_model.distance_params_bounds.values())
    bounds = [(a - 1e-5, a + 1e-5) if a == b else (a, b) for a, b in bounds]

    # Optimise distance params
    def distance_minim_func(distance_params_vals: np.ndarray) -> float:
        for i, k in enumerate(mm_model.distance_params.keys()):
            mm_model.distance_params[k] = distance_params_vals[i]
        return -np.sum(np.log(mm_model.distance_prior_evaluate(pos_distances, pos_time_interval_arrs_concat)))

    # Optimise distance params
    optim_dist_params = minimize(distance_minim_func,
                                 np.array([a for a in mm_model.distance_params.values()]),
                                 # method='powell',
                                 bounds=bounds)

    for i, k in enumerate(mm_model.distance_params.keys()):
        mm_model.distance_params[k] = optim_dist_params.x[i]

    # Optimise GPS noise
    mm_model.gps_sd = min(max(np.sqrt(sq_obs_dists.mean() / 2),
                              mm_model.gps_sd_bounds[0]),
                          mm_model.gps_sd_bounds[1])


def gradient_em_step(mm_model: MapMatchingModel,
                     map_matchings: list,
                     time_interval_arrs: list,
                     polylines: list,
                     stepsize: float):
    """
    For given map-matching results, take gradient step on prior hyperparameters (but fully optimise gps_sd)
    Updates mm_model hyperparameters in place
    :param mm_model: MapMatchingModel
    :param map_matchings: list of MMParticles objects
    :param time_interval_arrs: time interval arrays for each route
    :param polylines: observations for each route
    :param stepsize: stepsize for gradient step (applied to each coord)
    """
    n_particles = map_matchings[0].n

    # Get key quantities
    distances = np.array([])
    time_interval_arrs_concat = np.array([])
    devs = np.array([])
    sq_obs_dists = np.array([])
    dev_norm_quants = []
    for map_matching, time_interval_arr, polyline in zip(map_matchings, time_interval_arrs, polylines):
        distances_single, devs_and_norms_single, sq_obs_dists_single = extract_mm_quantities(map_matching,
                                                                                             polyline)
        distances = np.append(distances, distances_single)
        time_interval_arrs_concat = np.append(time_interval_arrs_concat,
                                              np.concatenate([time_interval_arr] * len(map_matching)))

        devs_single, dev_norm_quants_single = devs_and_norms_single
        devs = np.append(devs, devs_single)
        dev_norm_quants.append(dev_norm_quants_single)

        sq_obs_dists = np.append(sq_obs_dists, sq_obs_dists_single)

    # Z, *dZ/dalpha, dZ/dbeta where alpha = distance_params and beta = deviation_beta
    dev_norm_quants = np.concatenate(dev_norm_quants)

    pos_distances = distances
    pos_time_interval_arrs_concat = time_interval_arrs_concat
    pos_dev_norm_quants = dev_norm_quants
    pos_devs = devs

    distance_gradient_evals = (mm_model.distance_prior_gradient(pos_distances, pos_time_interval_arrs_concat)
                               / mm_model.distance_prior_evaluate(pos_distances, pos_time_interval_arrs_concat)
                               - pos_dev_norm_quants[:, 1:-1].T / pos_dev_norm_quants[:, 0]).sum(axis=1) \
                              / n_particles

    deviation_beta_gradient_evals = (-pos_devs - pos_dev_norm_quants[:, -1] /
                                     pos_dev_norm_quants[:, 0]).sum() \
                                    / n_particles

    # Take gradient step in distance params
    for i, k in enumerate(mm_model.distance_params.keys()):
        bounds = mm_model.distance_params_bounds[k]
        mm_model.distance_params[k] = min(max(
            mm_model.distance_params[k] + stepsize * distance_gradient_evals[i],
            bounds[0]), bounds[1])

    # Take gradient step in deviation beta
    mm_model.deviation_beta = min(max(
        mm_model.deviation_beta + stepsize * deviation_beta_gradient_evals,
        mm_model.deviation_beta_bounds[0]), mm_model.deviation_beta_bounds[1])

    # Optimise GPS noise
    mm_model.gps_sd = min(max(np.sqrt(sq_obs_dists.mean() / 2),
                              mm_model.gps_sd_bounds[0]),
                          mm_model.gps_sd_bounds[1])

