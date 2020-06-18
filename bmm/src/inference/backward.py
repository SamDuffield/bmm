########################################################################################################################
# Module: inference/backward.py
# Description: Implementation of backward simulation for particle smoothing.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################
import numpy as np

from bmm.src.inference.resampling import multinomial
from bmm.src.tools.edges import get_geometry


def full_backward_sample(fixed_particle, first_edge_fixed, first_edge_fixed_length,
                         filter_particles,
                         filter_weights,
                         time_interval,
                         next_time_index,
                         distance_prior_evaluate,
                         return_ess_back=False):
    n = filter_particles.n

    # filter_particles_adjusted = [None] * n

    smoothing_distances = np.empty(n)
    smoothing_distances[:] = np.nan

    distances_j_to_k = np.empty(n)

    for k in range(n):
        if filter_weights[k] == 0:
            continue

        filter_particle = filter_particles[k]

        # Check first fixed edge and last filter edge coincide
        if np.array_equal(first_edge_fixed[1:4], filter_particle[-1, 1:4]):
            # Check that fixed edge overtakes filter edge. i.e. distance isn't negative
            if np.array_equal(fixed_particle[next_time_index, 1:4], filter_particle[-1, 1:4]) and \
                    filter_particle[-1, 4] > fixed_particle[next_time_index, 4]:
                continue

            distances_j_to_k[k] = (first_edge_fixed[4] - filter_particle[-1, 4]) * first_edge_fixed_length

            # fixed_particle[1:(next_time_index + 1), -1] += distance_j_to_k

            smoothing_distances[k] = fixed_particle[next_time_index, -1] + distances_j_to_k[k]

            # filter_particles_adjusted[k] = filter_particle

    possible_inds = ~np.isnan(smoothing_distances)

    if not np.any(possible_inds):
        return (None, 0) if return_ess_back else None

    smoothing_weights = np.zeros(n)
    smoothing_weights[possible_inds] = filter_weights[possible_inds] \
                                       * distance_prior_evaluate(smoothing_distances[possible_inds], time_interval)
    smoothing_weights /= smoothing_weights.sum()

    sampled_index = np.random.choice(n, 1, p=smoothing_weights)[0]

    fixed_particle[1:(next_time_index + 1), -1] += distances_j_to_k[sampled_index]

    out_particle = np.append(filter_particles[sampled_index], fixed_particle[1:], axis=0)

    ess_back = 1 / (smoothing_weights ** 2).sum()

    if return_ess_back:
        return out_particle, ess_back
    else:
        return out_particle


def rejection_backward_sample(fixed_particle,
                              first_edge_fixed, first_edge_fixed_length,
                              filter_particles,
                              filter_weights,
                              time_interval,
                              next_time_index,
                              distance_prior_evaluate,
                              distance_prior_bound,
                              max_rejections):
    n = filter_particles.n

    for k in range(max_rejections):
        filter_index = np.random.choice(n, 1, p=filter_weights)[0]
        filter_particle = filter_particles[filter_index]

        if not np.array_equal(first_edge_fixed[1:4], filter_particle[-1, 1:4]):
            continue
        elif np.array_equal(fixed_particle[next_time_index, 1:4], filter_particle[-1, 1:4]) and \
                filter_particle[-1, 4] > fixed_particle[next_time_index, 4]:
            continue

        distance_j_to_k = (first_edge_fixed[4] - filter_particle[-1, 4]) * first_edge_fixed_length

        smoothing_distance = fixed_particle[next_time_index, -1] + distance_j_to_k

        smoothing_distance_prior = distance_prior_evaluate(smoothing_distance, time_interval)

        if np.random.uniform() < smoothing_distance_prior / distance_prior_bound:
            fixed_particle[1:(next_time_index + 1), -1] += distance_j_to_k
            return np.append(filter_particle, fixed_particle[1:], axis=0)

    return None


def backward_simulate(graph,
                      filter_particles, filter_weights,
                      time_interval_arr,
                      mm_model,
                      max_rejections,
                      verbose=False,
                      store_ess_back=None):
    n_samps = filter_particles[-1].n
    num_obs = len(filter_particles)

    if len(time_interval_arr) + 1 != num_obs:
        raise ValueError("time_interval_arr must be length one less than that of filter_particles")

    full_sampling = max_rejections == 0
    if store_ess_back is None:
        store_ess_back = full_sampling
    out_particles = multinomial(filter_particles[-1], filter_weights[-1])
    if full_sampling:
        ess_back = np.zeros((num_obs, n_samps))
        ess_back[0] = 1 / (filter_weights[-1] ** 2).sum()
    else:
        ess_back = None

    if num_obs < 2:
        return out_particles

    for i in range(num_obs - 2, -1, -1):
        next_time = filter_particles[i + 1].latest_observation_time

        for j in range(n_samps):
            fixed_particle = out_particles[j].copy()
            first_edge_fixed = fixed_particle[0]
            first_edge_fixed_geom = get_geometry(graph, first_edge_fixed[1:4])
            first_edge_fixed_length = first_edge_fixed_geom.length
            fixed_next_time_index = np.where(fixed_particle[:, 0] == next_time)[0][0]

            if full_sampling:
                out_particles[j], ess_back[i, j] = full_backward_sample(fixed_particle,
                                                                        first_edge_fixed, first_edge_fixed_length,
                                                                        filter_particles[i],
                                                                        filter_weights[i],
                                                                        time_interval_arr[i],
                                                                        fixed_next_time_index,
                                                                        mm_model.distance_prior_evaluate,
                                                                        True)
            else:
                out_particles[j] = rejection_backward_sample(fixed_particle,
                                                             first_edge_fixed, first_edge_fixed_length,
                                                             filter_particles[i],
                                                             filter_weights[i],
                                                             time_interval_arr[i],
                                                             fixed_next_time_index,
                                                             mm_model.distance_prior_evaluate,
                                                             mm_model.distance_prior_bound(time_interval_arr[i]),
                                                             max_rejections)

                if out_particles[j] is None:
                    out_particles[j] = full_backward_sample(fixed_particle,
                                                            first_edge_fixed, first_edge_fixed_length,
                                                            filter_particles[i],
                                                            filter_weights[i],
                                                            time_interval_arr[i],
                                                            fixed_next_time_index,
                                                            mm_model.distance_prior_evaluate,
                                                            False)

        none_inds = np.array([p is None or None in p for p in out_particles])
        good_inds = ~none_inds
        n_good = good_inds.sum()
        if n_good < n_samps:
            none_inds_res_indices = np.random.choice(n_samps, n_samps - n_good, p=good_inds / n_good)
            for i, j in enumerate(np.where(none_inds)[0]):
                out_particles[j] = out_particles[none_inds_res_indices[i]]
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

    return out_particles

