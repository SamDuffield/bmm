########################################################################################################################
# Module: inference/parameters.py
# Description: Expectation maximisation to infer maximum likelihood hyperparameters.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

from typing import Union, Tuple
import pickle
from time import time as tm
import inspect

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
    Run expectation maximisation to optimise prior hyperparameters.
    Updates the hyperparameters of mm_model in place.
    :param graph: encodes road network, simplified and projected to UTM
    :param mm_model: MapMatchingModel - of which parameters will be updated
    :param timestamps: seconds
        either float if all times between observations are the same, or a series of timestamps in seconds/UNIX timestamp
        if timestamps given, must be in a list matching dimensions of polylines
    :param polylines: UTM polylines
    :param save_path: path to save learned parameters
    :param n_ffbsi: number of samples for FFBSi algorithm
    :param n_iter: number of EM iterations
    :param gradient_stepsize_scale: starting stepsize
    :param gradient_stepsize_neg_exp: rate of decay of stepsize, in [0.5, 1]
    :param **kwargs additional arguments for FFBSi
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

    # # Optimise zero dist prob
    # def zero_dist_prob_root_func(neg_exp: float) -> float:
    #     return - np.sum(- time_interval_arrs_concat * (distances < 1e-5)
    #                     + time_interval_arrs_concat * np.exp(-neg_exp * time_interval_arrs_concat)
    #                     / (1 - np.exp(-neg_exp * time_interval_arrs_concat)) * (distances >= 1e-5))
    #
    # mm_model.zero_dist_prob_neg_exponent = root_scalar(zero_dist_prob_root_func, bracket=(1e-5, 1e20)).root
    # pos_distances = distances[distances > 1e-5]
    # pos_time_interval_arrs_concat = time_interval_arrs_concat[distances > 1e-5]
    # pos_dev_norm_quants = dev_norm_quants[distances > 1e-5]
    # pos_devs = devs[distances > 1e-5]

    pos_distances = distances
    pos_time_interval_arrs_concat = time_interval_arrs_concat
    pos_dev_norm_quants = dev_norm_quants
    pos_devs = devs

    # non_zero_inds = pos_dev_norm_quants[:, 0] > 0
    #
    # distance_gradient_evals = (mm_model.distance_prior_gradient(pos_distances, pos_time_interval_arrs_concat)[:,
    #                            non_zero_inds]
    #                            / mm_model.distance_prior_evaluate(pos_distances, pos_time_interval_arrs_concat)[
    #                                non_zero_inds]
    #                            - pos_dev_norm_quants[non_zero_inds, 1:-1].T / pos_dev_norm_quants[
    #                                non_zero_inds, 0]).sum(axis=1) \
    #                           / n_particles
    #
    # deviation_beta_gradient_evals = (-pos_devs[non_zero_inds] - pos_dev_norm_quants[non_zero_inds, -1] /
    #                                  pos_dev_norm_quants[non_zero_inds, 0]).sum() \
    #                                 / n_particles

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

#
#
#
#
# def optimise_hyperparameters(mm_model: MapMatchingModel,
#                              map_matchings: list,
#                              time_interval_arrs: list,
#                              polylines: list):
#     """
#     For given map-matching results, optimise model hyperparameters.
#     Updates mm_model hyperparameters in place
#     :param mm_model: MapMatchingModel
#     :param map_matchings: list of MMParticles objects
#     :param time_interval_arrs: time interval arrays for each route
#     :param polylines: observations for each route
#     """
#     # Get key quantities
#     distances = np.array([])
#     time_interval_arrs_concat = np.array([])
#     devs = np.array([])
#     sq_obs_dists = np.array([])
#     for map_matching, time_interval_arr, polyline in zip(map_matchings, time_interval_arrs, polylines):
#         distances_single, devs_single, sq_obs_dists_single = extract_mm_quantities(map_matching.particles,
#                                                                                    polyline)
#         distances = np.append(distances, np.concatenate(distances_single))
#         time_interval_arrs_concat = np.append(time_interval_arrs_concat,
#                                               np.concatenate([time_interval_arr] * len(map_matching)))
#
#         devs = np.append(devs, devs_single)
#         sq_obs_dists = np.append(sq_obs_dists, sq_obs_dists_single)
#
#     # Optimise distance params
#     def distance_minim_func(distance_params_vals: np.ndarray) -> float:
#         for i, k in enumerate(mm_model.distance_params.keys()):
#             mm_model.distance_params[k] = distance_params_vals[i]
#         return -np.sum(np.log(mm_model.distance_prior_evaluate(distances, time_interval_arrs_concat)))
#
#     # Optimise distance params
#     optim_dist_params = minimize(distance_minim_func,
#                                  np.array([a for a in mm_model.distance_params.values()]),
#                                  method='powell',
#                                  bounds=mm_model.distance_params_bounds)
#
#     for i, k in enumerate(mm_model.distance_params.keys()):
#         mm_model.distance_params[k] = optim_dist_params.x[i]
#
#     # Optimise GPS noise
#     mm_model.gps_sd = min(max(np.sqrt(sq_obs_dists.mean() / 2),
#                               mm_model.gps_sd_bounds[0]),
#                           mm_model.gps_sd_bounds[1])
#
#     # Optimise deviation beta
#     # mm_model.deviation_beta = min(max(devs[distances > 1e-5].mean(),
#     #                                   mm_model.deviation_beta_bounds[0]),
#     #                               mm_model.deviation_beta_bounds[1])
#
#     # # gps_sd regularisation
#     # gps_roots = np.roots([mm_model.gps_sd_lambda/len(sq_obs_dists) * len(map_matchings[0]),
#     #                       1,
#     #                       0,
#     #                       -np.mean(sq_obs_dists) / 2])
#     # gps_roots = gps_roots[np.isreal(gps_roots)]
#     # gps_roots = gps_roots[gps_roots > 0]
#     # mm_model.gps_sd = float(gps_roots[0])
#
# # # Plot dist loss function
# lambda_linsp = np.linspace(0.01, 0.5, 100)
# nexpp0_linsp = np.linspace(0.01, 0.6, 100)
#
# dist_eval_mat = np.array([[-distance_minim_func(np.array([1., lam, p0]))
#                            for lam in lambda_linsp]
#                           for p0 in nexpp0_linsp])
#
# brute_optim_inds = np.unravel_index(dist_eval_mat.argmax(), dist_eval_mat.shape)
#
# # plt.contourf(lambda_linsp, nexpp0_linsp, dist_eval_mat)
# # plt.scatter(lambda_linsp[brute_optim_inds[1]], nexpp0_linsp[brute_optim_inds[0]])
# # plt.xlabel('lambda')
# # plt.ylabel('-15 * log(p0)')
#
# plt.figure()
# plt.contourf(lambda_linsp, np.exp(-15*nexpp0_linsp), dist_eval_mat)
# plt.scatter(lambda_linsp[brute_optim_inds[1]], np.exp(-15*nexpp0_linsp[brute_optim_inds[0]]))
# plt.xlabel('lambda')
# plt.ylabel('p0')
#
#

#
# def offline_map_match_dev_quants(cam_graph: MultiDiGraph,
#                                  polyline: np.ndarray,
#                                  n_samps: int,
#                                  timestamps: Union[float, np.ndarray],
#                                  mm_model: MapMatchingModel = GammaMapMatchingModel(),
#                                  proposal: str = 'optimal',
#                                  d_refine: int = 1,
#                                  initial_d_truncate: float = None,
#                                  max_rejections: int = 20,
#                                  ess_threshold: float = 1,
#                                  **kwargs) -> MMParticles:
#     """
#     Runs offline map-matching. I.e. receives a full polyline and returns an equal probability collection
#     of trajectories.
#     Additionally stores prior/transition density normalising quantities for each filter particle
#     Forward-filtering backward-simulation implementation - no fixed-lag approximation needed for offline inference.
#     :param cam_graph: encodes road network, simplified and projected to UTM
#     :param polyline: series of cartesian coordinates in UTM
#     :param n_samps: int
#         number of particles
#     :param timestamps: seconds
#         either float if all times between observations are the same, or a series of timestamps in seconds/UNIX timestamp
#     :param mm_model: MapMatchingModel
#     :param proposal: either 'optimal' or 'aux_dist'
#         defaults to optimal (discretised) proposal
#     :param d_refine: metres, resolution of distance discretisation
#     :param initial_d_truncate: distance beyond which to assume zero likelihood probability at time zero
#         defaults to 5 * mm_model.gps_sd
#     :param max_rejections: number of rejections to attempt before doing full fixed-lag stitching
#         0 will do full fixed-lag stitching and track ess_stitch
#     :param ess_threshold: in [0,1], particle filter resamples if ess < ess_threshold * n_samps
#     :param kwargs: optional parameters to pass to proposal
#         i.e. d_max, d_refine or var
#         as well as ess_threshold for backward simulation update
#     :return: MMParticles object
#     """
#     if proposal == 'optimal':
#         proposal_func = optimal_proposal
#     else:
#         raise ValueError("Proposal " + str(proposal) + "not recognised, see bmm.proposals for valid options")
#
#     num_obs = len(polyline)
#
#     ess_all = max_rejections == 0
#
#     start = tm()
#
#     filter_particles = [None] * num_obs
#     adjusted_weights = np.zeros((num_obs, n_samps))
#
#     # Initiate filter_particles
#     filter_particles[0] = initiate_particles(cam_graph, polyline[0], n_samps, mm_model=mm_model,
#                                              d_refine=d_refine, d_truncate=initial_d_truncate,
#                                              ess_all=ess_all)
#     adjusted_weights[0] = 1 / n_samps
#     live_weights = adjusted_weights[0].copy()
#
#     ess_pf = np.zeros(num_obs)
#     ess_pf[0] = n_samps
#
#     print("0 PF ESS: " + str(ess_pf[0]))
#
#     if 'd_refine' in inspect.getfullargspec(proposal_func)[0]:
#         kwargs['d_refine'] = d_refine
#
#     time_interval_arr = get_time_interval_array(timestamps, num_obs)
#
#     dev_norm_quants = np.zeros((num_obs - 1, n_samps, len(mm_model.distance_params) + 2))
#
#     # Forward filtering, storing x_t-1, x_t ~ p(x_t-1:t|y_t)
#     for i in range(num_obs - 1):
#         if ess_pf[i] < ess_threshold * n_samps:
#             # live_particles = multinomial(filter_particles[i], live_weights)
#             resample_inds = np.random.choice(n_samps, n_samps, replace=True, p=live_weights)
#             live_particles = filter_particles[i].copy()
#             live_particles.particles = [live_particles.particles[ind].copy() for ind in resample_inds]
#             live_weights = np.ones(n_samps) / n_samps
#             not_prop_inds = np.arange(n_samps)[~np.isin(np.arange(n_samps), resample_inds)]
#
#         else:
#             resample_inds = np.arange(n_samps)
#             live_particles = filter_particles[i]
#             not_prop_inds = []
#
#         temp_weights = np.zeros(n_samps)
#         filter_particles[i + 1] = live_particles.copy()
#         for j in range(n_samps):
#             prop_output = proposal_func(cam_graph, live_particles[j], polyline[i + 1],
#                                         time_interval_arr[i],
#                                         mm_model,
#                                         full_smoothing=False,
#                                         store_norm_quants=True,
#                                         **kwargs)
#
#             filter_particles[i + 1][j], temp_weights[j], dev_norm_quants[i, resample_inds[j]] = prop_output
#
#         for k in not_prop_inds:
#             if filter_particles[i][k] is not None:
#                 prop_output = proposal_func(cam_graph, filter_particles[i][k].copy(), polyline[i + 1],
#                                             time_interval_arr[i],
#                                             mm_model,
#                                             full_smoothing=False,
#                                             store_norm_quants=True,
#                                             **kwargs)
#
#                 _, _, dev_norm_quants[i, k] = prop_output
#
#         temp_weights *= live_weights
#         temp_weights /= np.sum(temp_weights)
#         adjusted_weights[i + 1] = temp_weights.copy()
#         live_weights = temp_weights.copy()
#         ess_pf[i + 1] = 1 / np.sum(temp_weights ** 2)
#
#         print(str(filter_particles[i + 1].latest_observation_time) + " PF ESS: " + str(ess_pf[i + 1]))
#
#     # Backward simulation
#     out_particles = backward_simulate(cam_graph,
#                                       filter_particles, adjusted_weights,
#                                       time_interval_arr,
#                                       mm_model,
#                                       max_rejections,
#                                       verbose=True,
#                                       dev_norm_quants=dev_norm_quants)
#
#     end = tm()
#     out_particles.time = end - start
#     return out_particles



#def optim_plus_devs(cam_graph: MultiDiGraph,
#                     mm_model: MapMatchingModel,
#                     map_matchings: list,
#                     time_interval_arrs: list,
#                     polylines: list,
#                     max_iter: int = 10,
#                     **kwargs):
#     """
#     For given map-matching results, take gradient step on prior hyperparameters (but fully optimise gps_sd)
#     Updates mm_model hyperparameters in place
#     :param cam_graph: encodes road network, simplified and projected to UTM
#     :param mm_model: MapMatchingModel
#     :param map_matchings: list of MMParticles objects
#     :param time_interval_arrs: time interval arrays for each route
#     :param polylines: observations for each route
#     :param max_iter: maximum number of iterations for minimiser
#     :param **kwargs additional arguments (d_refine, num_inter_cut_off)
#     """
#     # Get key quantities
#     distances = np.array([])
#     time_interval_arrs_concat = np.array([])
#     devs = np.array([])
#     sq_obs_dists = np.array([])
#     for map_matching, time_interval_arr, polyline in zip(map_matchings, time_interval_arrs, polylines):
#         distances_single, devs_single, sq_obs_dists_single = extract_dists_devs(map_matching, polyline)
#         distances = np.append(distances, distances_single)
#         time_interval_arrs_concat = np.append(time_interval_arrs_concat,
#                                               np.concatenate([time_interval_arr] * len(map_matching)))
#         devs = np.append(devs, devs_single)
#         sq_obs_dists = np.append(sq_obs_dists, sq_obs_dists_single)
#
#     dist_bounds = list(mm_model.distance_params_bounds.values())
#     bounds = [(a - 1e-5, a + 1e-5) if a == b else (a, b) for a, b in dist_bounds] + [(1e-5, np.inf)]
#
#     prop_kwargs = {}
#     if 'd_refine' in kwargs:
#         prop_kwargs['d_refine'] = kwargs['d_refine']
#     if 'num_inter_cut_off' in kwargs:
#         prop_kwargs['num_inter_cut_off'] = kwargs['num_inter_cut_off']
#
#     # Optimise distance params
#     def param_minim_func(params_vals: np.ndarray) -> float:
#         for i, k in enumerate(mm_model.distance_params.keys()):
#             mm_model.distance_params[k] = params_vals[i]
#
#         mm_model.deviation_beta = params_vals[-1]
#
#         norm_consts = np.array([])
#         for mm_ind in range(len(map_matchings)):
#             print(f'Optimisation mm_ind: {mm_ind}')
#
#             obs_row_particles = [observation_time_rows(p)[:-1] for p in map_matchings[mm_ind]]
#             m = len(obs_row_particles[0])
#             n = len(obs_row_particles)
#             norm_consts_mm = np.zeros((m, n))
#
#             for time_ind in range(m):
#                 for part_ind in range(n):
#                     norm_consts_mm[time_ind, part_ind] = optimal_proposal(cam_graph,
#                                                                           obs_row_particles[part_ind][time_ind][None],
#                                                                           None,
#                                                                           time_interval_arrs[mm_ind][time_ind],
#                                                                           mm_model,
#                                                                           full_smoothing=False,
#                                                                           only_norm_const=True,
#                                                                           **prop_kwargs)
#             norm_consts = np.append(norm_consts, norm_consts_mm.T)
#
#         return -np.sum(np.log(mm_model.distance_prior_evaluate(distances, time_interval_arrs_concat))
#                        - mm_model.deviation_beta * devs - np.log(norm_consts))
#
#     # Optimise distance params
#     optim_dist_params = minimize(param_minim_func,
#                                  np.array([a for a in mm_model.distance_params.values()]
#                                           + [mm_model.deviation_beta]),
#                                  method='trust-constr',
#                                  bounds=bounds,
#                                  options={'maxiter': max_iter})
#
#     for i, k in enumerate(mm_model.distance_params.keys()):
#         mm_model.distance_params[k] = optim_dist_params.x[i]
#
#     mm_model.deviation_beta = optim_dist_params.x[-1]
#
#     # Optimise GPS noise
#     mm_model.gps_sd = min(max(np.sqrt(sq_obs_dists.mean() / 2),
#                               mm_model.gps_sd_bounds[0]),
#                           mm_model.gps_sd_bounds[1])



# def extract_dists_devs(map_matching: MMParticles,
#                        polyline: np.ndarray) -> tuple:
#     """
#     Extract distances and deviations from a map_matching.
#     :param map_matching: MMParticles.particles list
#     :param polyline: for single route
#     :param extract_devs: whether to extract deviations (and gradient quantities)
#     :return: distances, deviations and squared observation-position distances
#     """
#     distances = np.array([])
#     devs = np.array([])
#     sq_obs_dists = np.array([])
#     for particle in map_matching:
#         particle_obs_time_rows = observation_time_rows(particle)
#         distances_particle = particle_obs_time_rows[1:, -1]
#         distances = np.append(distances, distances_particle)
#
#         devs_particle = np.abs(distances_particle - np.sqrt(np.sum(np.square(particle_obs_time_rows[1:, 5:7]
#                                                                              - particle_obs_time_rows[:-1, 5:7]),
#                                                                    axis=1)))
#         devs = np.append(devs, devs_particle)
#
#         sq_obs_dists = np.append(sq_obs_dists, np.sum(np.square(particle_obs_time_rows[:, 5:7] - polyline), axis=1))
#
#     return distances, devs, sq_obs_dists



# def gradient_em_step(cam_graph: MultiDiGraph,
#                      mm_model: MapMatchingModel,
#                      map_matchings: list,
#                      time_interval_arrs: list,
#                      polylines: list,
#                      stepsize: float,
#                      **kwargs):
#     """
#     For given map-matching results, take gradient step on prior hyperparameters (but fully optimise gps_sd)
#     Updates mm_model hyperparameters in place
#     :param cam_graph: encodes road network, simplified and projected to UTM
#     :param mm_model: MapMatchingModel
#     :param map_matchings: list of MMParticles objects
#     :param time_interval_arrs: time interval arrays for each route
#     :param polylines: observations for each route
#     :param stepsize: stepsize for gradient step (applied to each coord)
#     :param **kwargs additional arguments (d_refine, num_inter_cut_off)
#     """
#     # Get key quantities
#     n = map_matchings[0].n
#     distances = np.array([])
#     time_interval_arrs_concat = np.array([])
#     devs = np.array([])
#     sq_obs_dists = np.array([])
#     for map_matching, time_interval_arr, polyline in zip(map_matchings, time_interval_arrs, polylines):
#         distances_single, devs_single, sq_obs_dists_single = extract_dists_devs(map_matching, polyline)
#         distances = np.append(distances, distances_single)
#         time_interval_arrs_concat = np.append(time_interval_arrs_concat,
#                                               np.concatenate([time_interval_arr] * len(map_matching)))
#         devs = np.append(devs, devs_single)
#         sq_obs_dists = np.append(sq_obs_dists, sq_obs_dists_single)
#
#     prop_kwargs = {}
#     if 'd_refine' in kwargs:
#         prop_kwargs['d_refine'] = kwargs['d_refine']
#     if 'num_inter_cut_off' in kwargs:
#         prop_kwargs['num_inter_cut_off'] = kwargs['num_inter_cut_off']
#
#     # Z, *dZ/dalpha, dZ/dbeta where alpha = distance_params and beta = deviation_beta
#     norm_quants = np.empty((len(distances), len(mm_model.distance_params) + 2))
#     nq_ind = 0
#     for mm_ind in range(len(map_matchings)):
#         print(f'Gradient mm_ind: {mm_ind}')
#
#         obs_row_particles = [observation_time_rows(p)[:-1] for p in map_matchings[mm_ind]]
#         m = len(obs_row_particles[0])
#         norm_quants_mm = np.zeros((m, n, len(mm_model.distance_params) + 2))
#
#         for time_ind in range(m):
#             for part_ind in range(n):
#                 norm_quants_mm[time_ind, part_ind] = optimal_proposal(cam_graph,
#                                                                       obs_row_particles[part_ind][time_ind][None],
#                                                                       None,
#                                                                       time_interval_arrs[mm_ind][time_ind],
#                                                                       mm_model,
#                                                                       full_smoothing=False,
#                                                                       only_norm_const=True,
#                                                                       store_norm_quants=True,
#                                                                       **prop_kwargs)
#         for part_ind in range(n):
#             norm_quants[nq_ind:(nq_ind + m)] = norm_quants_mm[:, part_ind]
#             nq_ind += m
#
#     distance_gradient_evals = (mm_model.distance_prior_gradient(distances, time_interval_arrs_concat)
#                                / mm_model.distance_prior_evaluate(distances, time_interval_arrs_concat)
#                                - norm_quants[:, 1:-1].T / norm_quants[:, 0]).sum(axis=1) \
#                               / n
#
#     deviation_beta_gradient_evals = (-devs - norm_quants[:, -1] /
#                                      norm_quants[:, 0]).sum() \
#                                     / n
#
#     # Take gradient step in distance params
#     for i, k in enumerate(mm_model.distance_params.keys()):
#         bounds = mm_model.distance_params_bounds[k]
#         mm_model.distance_params[k] = min(max(
#             mm_model.distance_params[k] + stepsize * distance_gradient_evals[i],
#             bounds[0]), bounds[1])
#
#     # Take gradient step in deviation beta
#     mm_model.deviation_beta = min(max(
#         mm_model.deviation_beta + stepsize * deviation_beta_gradient_evals,
#         mm_model.deviation_beta_bounds[0]), mm_model.deviation_beta_bounds[1])
#
#     # Optimise GPS noise
#     mm_model.gps_sd = min(max(np.sqrt(sq_obs_dists.mean() / 2),
#                               mm_model.gps_sd_bounds[0]),
#                           mm_model.gps_sd_bounds[1])
