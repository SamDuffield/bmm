########################################################################################################################
# Module: inference/backward.py
# Description: Implementation of backward simulation for particle smoothing.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

from typing import Union, Optional

import numpy as np
from networkx.classes import MultiDiGraph

from bmm.src.inference.resampling import multinomial
from bmm.src.tools.edges import get_geometry
from bmm.src.inference.particles import MMParticles
from bmm.src.inference.model import MapMatchingModel


def full_backward_sample(fixed_particle: np.ndarray,
                         first_edge_fixed: np.ndarray,
                         first_edge_fixed_length: float,
                         filter_particles: MMParticles,
                         adjusted_weights: Union[list, np.ndarray],
                         time_interval: float,
                         next_time_index: int,
                         mm_model: MapMatchingModel,
                         return_ess_back: bool = False,
                         return_sampled_index: bool = False) \
        -> Union[Optional[np.ndarray], tuple]:
    """
    Evaluate full interacting weights, normalise and backwards sample a past coordinate
    for a single fixed particle of future coordinates
    :param fixed_particle: trajectory post backwards sampling time
    :param first_edge_fixed: first row of fixed particle
    :param first_edge_fixed_length: metres
    :param filter_particles: proposal particles to be sampled
    :param adjusted_weights: non-interacting weights for filter_particles
    :param time_interval: time between observations at backwards sampling time
    :param next_time_index: index of second observation time in fixed_particle
    :param mm_model: MapMatchingModel
    :param return_ess_back: whether to calculate and return the ESS of the full interacting weights
    :param return_sampled_index: whether to return index of selected back sample
    :return: appended particle (and ess_back if return_ess_back)
    """
    n = filter_particles.n

    smoothing_distances = np.empty(n)
    smoothing_distances[:] = np.nan

    distances_j_to_k = np.empty(n)
    new_prev_cart_coords = np.empty((n, 2))

    for k in range(n):
        if adjusted_weights[k] == 0:
            continue

        filter_particle = filter_particles[k]

        # Check first fixed edge and last filter edge coincide
        if np.array_equal(first_edge_fixed[1:4], filter_particle[-1, 1:4]):
            # Check that fixed edge overtakes filter edge. i.e. distance isn't negative
            if np.array_equal(filter_particle[-1, 1:4], fixed_particle[next_time_index, 1:4]) and \
                    filter_particle[-1, 4] > fixed_particle[next_time_index, 4]:
                continue

            distances_j_to_k[k] = np.round((first_edge_fixed[4] - filter_particle[-1, 4]) * first_edge_fixed_length, 5)
            smoothing_distances[k] = fixed_particle[next_time_index, -1] + distances_j_to_k[k]

            if smoothing_distances[k] < 0:
                raise ValueError('Negative smoothing distance')

            new_prev_cart_coords[k] = filter_particle[-1, 5:7]

    possible_inds = ~np.isnan(smoothing_distances)
    if not np.any(possible_inds):
        if return_ess_back:
            if return_sampled_index:
                return None, 0, 0
            else:
                return None, 0
        else:
            if return_sampled_index:
                return None, 0
            else:
                return None

    smoothing_weights = adjusted_weights[possible_inds] \
                        * mm_model.distance_prior_evaluate(smoothing_distances[possible_inds],
                                                           time_interval) \
                        * mm_model.deviation_prior_evaluate(new_prev_cart_coords[possible_inds],
                                                            fixed_particle[None, next_time_index, 5:7],
                                                            smoothing_distances[possible_inds])

    smoothing_weights /= smoothing_weights.sum()

    sampled_index = np.where(possible_inds)[0][np.random.choice(len(smoothing_weights), 1, p=smoothing_weights)[0]]

    fixed_particle[1:(next_time_index + 1), -1] += distances_j_to_k[sampled_index]

    out_particle = np.append(filter_particles[sampled_index], fixed_particle[1:], axis=0)

    ess_back = 1 / (smoothing_weights ** 2).sum()

    if return_ess_back:
        if return_sampled_index:
            return out_particle, ess_back, sampled_index
        else:
            return out_particle, ess_back
    else:
        if return_sampled_index:
            return out_particle, sampled_index
        else:
            return out_particle


def rejection_backward_sample(fixed_particle: np.ndarray,
                              first_edge_fixed: np.ndarray,
                              first_edge_fixed_length: float,
                              filter_particles: MMParticles,
                              filter_weights: np.ndarray,
                              time_interval: float,
                              next_time_index: int,
                              prior_bound: float,
                              mm_model: MapMatchingModel,
                              max_rejections: int,
                              return_sampled_index: bool = False,
                              break_on_zero: bool = False) -> Union[Optional[np.ndarray], tuple, int]:
    """
    Attempt up to max_rejections of rejection sampling to backwards sample a single particle
    :param fixed_particle: trajectory prior to stitching time
    :param first_edge_fixed: first row of fixed particle
    :param first_edge_fixed_length: metres
    :param filter_particles: proposal particles to be sampled
    :param filter_weights: weights for filter_particles
    :param time_interval: time between observations at backwards sampling time
    :param next_time_index: index of second observation time in fixed_particle
    :param prior_bound: bound on distance transition density (given positive if break_on_zero)
    :param mm_model: MapMatchingModel
    :param max_rejections: number of rejections to attempt, if none succeed return None
    :param return_sampled_index: whether to return index of selected back sample
    :param break_on_zero: whether to return 0 if smoothing_distance=0
    :return: appended particle
    """
    n = filter_particles.n

    for k in range(max_rejections):
        filter_index = np.random.choice(n, 1, p=filter_weights)[0]
        filter_particle = filter_particles[filter_index]

        if not np.array_equal(first_edge_fixed[1:4], filter_particle[-1, 1:4]):
            continue
        elif np.array_equal(fixed_particle[next_time_index, 1:4], filter_particle[-1, 1:4]) and \
                filter_particle[-1, 4] > fixed_particle[next_time_index, 4]:
            continue

        distance_j_to_k = np.round((first_edge_fixed[4] - filter_particle[-1, 4]) * first_edge_fixed_length, 5)

        smoothing_distance = fixed_particle[next_time_index, -1] + distance_j_to_k

        if break_on_zero and smoothing_distance < 1e-5:
            return (0, filter_index) if return_sampled_index else 0

        smoothing_distance_prior = mm_model.distance_prior_evaluate(smoothing_distance, time_interval)
        smoothing_deviation_prior = mm_model.deviation_prior_evaluate(filter_particle[-1, 5:7],
                                                                      fixed_particle[None, next_time_index, 5:7],
                                                                      smoothing_distance)
        accept_prob = smoothing_distance_prior * smoothing_deviation_prior / prior_bound
        if accept_prob > (1 - 1e-5) or np.random.uniform() < accept_prob:
            fixed_particle[1:(next_time_index + 1), -1] += distance_j_to_k
            out_part = np.append(filter_particle, fixed_particle[1:], axis=0)
            if return_sampled_index:
                return out_part, filter_index
            else:
                return out_part

    return (None, 0) if return_sampled_index else None


def backward_simulate(graph: MultiDiGraph,
                      filter_particles: MMParticles,
                      filter_weights: np.ndarray,
                      time_interval_arr: np.ndarray,
                      mm_model: MapMatchingModel,
                      max_rejections: int,
                      verbose: bool = False,
                      store_ess_back: bool = None,
                      store_norm_quants: bool = False) -> MMParticles:
    """
    Given particle filter output, run backwards simulation to output smoothed trajectories
    :param graph: encodes road network, simplified and projected to UTM
    :param filter_particles: marginal outputs from particle filter
    :param filter_weights: weights
    :param time_interval_arr: times between observations, must be length one less than filter_particles
    :param mm_model: MapMatchingModel
    :param max_rejections: number of rejections to attempt before doing full fixed-lag stitching
        0 will do full backward simulation and track ess_back
    :param verbose: print ess_pf or ess_back
    :param store_ess_back: whether to store ess_back (if possible) in MMParticles object
    :param store_norm_quants: if True normalisation quantities returned in out_particles
    :return: MMParticles object
    """
    n_samps = filter_particles[-1].n
    num_obs = len(filter_particles)

    if len(time_interval_arr) + 1 != num_obs:
        raise ValueError("time_interval_arr must be length one less than that of filter_particles")

    full_sampling = max_rejections == 0
    if store_ess_back is None:
        store_ess_back = full_sampling

    # Multinomial resample end particles if weighted
    if np.all(filter_weights[-1] == filter_weights[-1][0]):
        out_particles = filter_particles[-1].copy()
    else:
        out_particles = multinomial(filter_particles[-1], filter_weights[-1])
    if full_sampling:
        ess_back = np.zeros((num_obs, n_samps))
        ess_back[0] = 1 / (filter_weights[-1] ** 2).sum()
    else:
        ess_back = None

    if num_obs < 2:
        return out_particles

    if store_norm_quants:
        norm_quants = np.zeros((num_obs - 1, *filter_particles[0].prior_norm.shape))

    for i in range(num_obs - 2, -1, -1):
        next_time = filter_particles[i + 1].latest_observation_time

        if not full_sampling:
            pos_prior_bound = mm_model.pos_distance_prior_bound(time_interval_arr[i])
            prior_bound = mm_model.distance_prior_bound(time_interval_arr[i])
            store_out_parts = out_particles.copy()

        if filter_particles[i].prior_norm.ndim == 2:
            prior_norm = filter_particles[i].prior_norm[:, 0]
        else:
            prior_norm = filter_particles[i].prior_norm
        adjusted_weights = filter_weights[i].copy()
        good_inds = np.logical_and(adjusted_weights != 0, prior_norm != 0)
        adjusted_weights[good_inds] /= prior_norm[good_inds]
        adjusted_weights[~good_inds] = 0
        adjusted_weights /= adjusted_weights.sum()

        if store_norm_quants:
            sampled_inds = np.zeros(n_samps, dtype=int)

        resort_to_full = False
        for j in range(n_samps):
            fixed_particle = out_particles[j].copy()
            first_edge_fixed = fixed_particle[0]
            first_edge_fixed_geom = get_geometry(graph, first_edge_fixed[1:4])
            first_edge_fixed_length = first_edge_fixed_geom.length
            fixed_next_time_index = np.where(fixed_particle[:, 0] == next_time)[0][0]

            if full_sampling:
                back_output = full_backward_sample(fixed_particle,
                                                   first_edge_fixed,
                                                   first_edge_fixed_length,
                                                   filter_particles[i],
                                                   adjusted_weights,
                                                   time_interval_arr[i],
                                                   fixed_next_time_index,
                                                   mm_model,
                                                   return_ess_back=True,
                                                   return_sampled_index=store_norm_quants)

                if store_norm_quants:
                    out_particles[j], ess_back[i, j], sampled_inds[j] = back_output
                else:
                    out_particles[j], ess_back[i, j] = back_output

            else:
                back_output = rejection_backward_sample(fixed_particle,
                                                        first_edge_fixed,
                                                        first_edge_fixed_length,
                                                        filter_particles[i],
                                                        adjusted_weights,
                                                        time_interval_arr[i],
                                                        fixed_next_time_index,
                                                        pos_prior_bound,
                                                        mm_model,
                                                        max_rejections,
                                                        return_sampled_index=store_norm_quants,
                                                        break_on_zero=True)

                first_back_output = back_output[0] if store_norm_quants else back_output

                if first_back_output is None:
                    back_output = full_backward_sample(fixed_particle,
                                                       first_edge_fixed,
                                                       first_edge_fixed_length,
                                                       filter_particles[i],
                                                       adjusted_weights,
                                                       time_interval_arr[i],
                                                       fixed_next_time_index,
                                                       mm_model,
                                                       return_ess_back=False,
                                                       return_sampled_index=store_norm_quants)

                if isinstance(first_back_output, int) and first_back_output == 0:
                    resort_to_full = True
                    break

                if store_norm_quants:
                    out_particles[j], sampled_inds[j] = back_output
                else:
                    out_particles[j] = back_output

        if resort_to_full:
            if store_norm_quants:
                sampled_inds = np.zeros(n_samps, dtype=int)
            for j in range(n_samps):
                fixed_particle = store_out_parts[j]
                first_edge_fixed = fixed_particle[0]
                first_edge_fixed_geom = get_geometry(graph, first_edge_fixed[1:4])
                first_edge_fixed_length = first_edge_fixed_geom.length
                fixed_next_time_index = np.where(fixed_particle[:, 0] == next_time)[0][0]

                back_output = rejection_backward_sample(fixed_particle,
                                                        first_edge_fixed,
                                                        first_edge_fixed_length,
                                                        filter_particles[i],
                                                        adjusted_weights,
                                                        time_interval_arr[i],
                                                        fixed_next_time_index,
                                                        prior_bound,
                                                        mm_model,
                                                        max_rejections,
                                                        return_sampled_index=store_norm_quants,
                                                        break_on_zero=False)

                first_back_output = back_output[0] if store_norm_quants else back_output

                if first_back_output is None:
                    back_output = full_backward_sample(fixed_particle,
                                                       first_edge_fixed,
                                                       first_edge_fixed_length,
                                                       filter_particles[i],
                                                       adjusted_weights,
                                                       time_interval_arr[i],
                                                       fixed_next_time_index,
                                                       mm_model,
                                                       return_ess_back=False,
                                                       return_sampled_index=store_norm_quants)

                if store_norm_quants:
                    out_particles[j], sampled_inds[j] = back_output
                else:
                    out_particles[j] = back_output

        if store_norm_quants:
            norm_quants[i] = filter_particles[i].prior_norm[sampled_inds]

        none_inds = np.array([p is None or None in p for p in out_particles])
        good_inds = ~none_inds
        n_good = good_inds.sum()
        if n_good < n_samps:
            none_inds_res_indices = np.random.choice(n_samps, n_samps - n_good, p=good_inds / n_good)
            for i_none, j_none in enumerate(np.where(none_inds)[0]):
                out_particles[j_none] = out_particles[none_inds_res_indices[i_none]].copy()
                if store_norm_quants:
                    norm_quants[:, j_none] = norm_quants[:, none_inds_res_indices[i_none]]
            if store_ess_back:
                out_particles.ess_back[i, none_inds] = n_samps

        if verbose:
            if full_sampling:
                print(str(filter_particles[i].latest_observation_time) + " Av Backward ESS: " + str(
                    np.mean(ess_back[i])))
            else:
                print(str(filter_particles[i].latest_observation_time))

        if store_ess_back:
            out_particles.ess_back = ess_back

    if store_norm_quants:
        out_particles.dev_norm_quants = norm_quants

    return out_particles
