########################################################################################################################
# Module: inference/offline.py
# Description: Offline smoothing via forward-filtering backward-simulation.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

from time import time as tm
import inspect

import numpy as np

from tools.edges import get_geometry
from inference.particles import MMParticles
from inference.proposal import optimal_proposal
from inference.smc import initiate_particles
from inference.resampling import multinomial
from inference.model import distance_prior, get_distance_prior_bound


def full_backward_sample(fixed_particle, first_edge_fixed, first_edge_fixed_length,
                         filter_particles,
                         filter_weights,
                         time_interval,
                         next_time_index,
                         return_ess_back=False):
    n = filter_particles.n

    filter_particles_adjusted = [None] * n

    smoothing_distances = np.empty(n)
    smoothing_distances[:] = np.nan

    for k in range(n):
        if filter_weights[k] == 0:
            continue

        filter_particle = filter_particles[k].copy()

        # Check first fixed edge and last filter edge coincide
        if np.array_equal(first_edge_fixed[1:4], filter_particle[-1, 1:4]):
            # Check that fixed edge overtakes filter edge. i.e. distance isn't negative
            if np.array_equal(fixed_particle[next_time_index, 1:4], filter_particle[-1, 1:4]) and \
                    filter_particle[-1, 4] > fixed_particle[next_time_index, 4]:
                continue

            distance_j_to_k = (first_edge_fixed[4] - filter_particle[-1, 4]) * first_edge_fixed_length

            fixed_particle[1:next_time_index, -1] += distance_j_to_k

            smoothing_distances[k] = fixed_particle[next_time_index, -1]

            filter_particles_adjusted[k] = filter_particle

    possible_inds = ~np.isnan(smoothing_distances)
    smoothing_weights = np.zeros(n)
    smoothing_weights[possible_inds] = filter_weights[possible_inds] \
                                       * distance_prior(smoothing_distances[possible_inds], time_interval)
    smoothing_weights /= smoothing_weights.sum()

    sampled_index = np.random.choice(n, 1, p=smoothing_weights)[0]

    out_particle = np.append(filter_particles_adjusted[sampled_index], fixed_particle[1:], axis=0)

    ess_back = 1 / (smoothing_weights ** 2).sum()

    if return_ess_back:
        return out_particle, ess_back,
    else:
        return out_particle


def rejection_backward_sample(fixed_particle,
                              first_edge_fixed, first_edge_fixed_length,
                              filter_particles,
                              filter_weights,
                              time_interval,
                              next_time_index,
                              distance_prior_bound,
                              max_rejections):
    n = filter_particles.n

    for k in range(max_rejections):
        filter_index = np.random.choice(n, 1, p=filter_weights)[0]
        filter_particle = filter_particles[filter_index].copy()

        if not np.array_equal(first_edge_fixed[1:4], filter_particle[-1, 1:4]):
            continue
        elif np.array_equal(fixed_particle[next_time_index, 1:4], filter_particle[-1, 1:4]) and \
                filter_particle[-1, 4] > fixed_particle[next_time_index, 4]:
            continue

        distance_j_to_k = (first_edge_fixed[4] - filter_particle[-1, 4]) * first_edge_fixed_length

        fixed_particle[1:next_time_index, -1] += distance_j_to_k

        smoothing_distance = fixed_particle[next_time_index, -1]

        smoothing_distance_prior = distance_prior(smoothing_distance, time_interval)

        if np.random.uniform() < smoothing_distance_prior / distance_prior_bound:
            return np.append(filter_particle, fixed_particle[1:], axis=0)

    return None


def offline_map_match(graph,
                      polyline,
                      n_samps,
                      time_interval,
                      proposal=optimal_proposal,
                      gps_sd=7,
                      d_refine=1,
                      initial_truncation=None,
                      max_rejections=20,
                      ess_threshold=1,
                      **kwargs):
    """
    Runs offline map-matching. I.e. receives a full polyline and returns an equal probability collection
    of trajectories (filter_particles).
    Forward-filtering backward-simulation implementation - therefore no fixed-lag approximation.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param polyline: list-like, length M
        UTM projection
        series of 2D coordinates
    :param n_samps: int
        number of filter_particles
    :param time_interval: float or list-like (length M-1)
        seconds
        time between observations
    :param proposal: function
        function that takes previous trajectory and new observation
        and output new trajectory and (unnormalised weight)
        defaults to the optimal (discrete distance) proposal
    :param gps_sd: float
        standard deviation of GPS noise
        assumes isotropic
    :param d_refine: float
        metres
        discretisation level of distance parameter
        needed for initiate_particle and potentially proposal
    :param initial_truncation: float
        metres
        distance to truncate for sampling initial position
        defaults to 3 * gps_sd
    :param max_rejections: int
        number of rejections before doing full fixed-lag stitching in resampling
        0 will do full fixed-lag stitching and track ess_stitch
    :param ess_threshold: float in [0,1]
        when to resample particle filter
        will resample if ess < ess_threshold * n_samps
    :param kwargs: optional parameters to pass to proposal
        i.e. d_max, d_refine or var
    :return: MMParticles object (from inference.smc)
    """
    num_obs = len(polyline)

    ess_all = max_rejections == 0

    start = tm()

    filter_particles = [None] * num_obs
    filter_weights = np.zeros((num_obs, n_samps))

    # Initiate filter_particles
    filter_particles[0] = initiate_particles(graph, polyline[0], n_samps,
                                             gps_sd=gps_sd, d_refine=d_refine, truncation_distance=initial_truncation,
                                             ess_all=ess_all)
    filter_weights[0] = 1 / n_samps
    live_weights = filter_weights[0].copy()

    ess_pf = np.zeros(num_obs)
    ess_pf[0] = n_samps

    print("0 PF ESS: " + str(ess_pf[0]))

    if 'd_refine' in inspect.getfullargspec(proposal)[0]:
        kwargs['d_refine'] = d_refine

    if isinstance(time_interval, (int, float)):
        time_interval_arr = np.ones(num_obs - 1) * time_interval
    elif len(time_interval) == (num_obs - 1):
        time_interval_arr = time_interval
    else:
        raise ValueError("time_interval must be either float or list-like of length one less than polyline")

    # Forward filtering, storing x_t-1, x_t ~ p(x_t-1:t|y_t)
    for i in range(num_obs - 1):
        if ess_pf[i] < ess_threshold * n_samps:
            live_particles = multinomial(filter_particles[i], live_weights)
            live_weights = np.ones(n_samps) / n_samps
        else:
            live_particles = filter_particles[i]

        temp_weights = np.zeros(n_samps)
        filter_particles[i + 1] = live_particles.copy()
        for j in range(n_samps):
            filter_particles[i + 1][j], temp_weights[j] = proposal(graph, live_particles[j], polyline[i + 1],
                                                                   time_interval_arr[i], gps_sd,
                                                                   full_smoothing=False,
                                                                   **kwargs)
        temp_weights *= live_weights
        temp_weights /= np.sum(temp_weights)
        filter_weights[i + 1] = temp_weights.copy()
        live_weights = temp_weights.copy()
        ess_pf[i + 1] = 1 / np.sum(temp_weights ** 2)

        print(str(filter_particles[i + 1].latest_observation_time) + " PF ESS: " + str(ess_pf[i + 1]))

    # Backward simulation
    full_sampling = max_rejections == 0
    out_particles = multinomial(filter_particles[-1], filter_weights[-1])
    if full_sampling:
        ess_back = np.zeros((num_obs, n_samps))
        ess_back[0] = ess_pf[-1]

        distance_prior_bound = None
    else:
        ess_back = None

        distance_prior_bound = get_distance_prior_bound()

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
                                                                        True)
            else:
                out_particles[j] = rejection_backward_sample(fixed_particle,
                                                             first_edge_fixed, first_edge_fixed_length,
                                                             filter_particles[i],
                                                             filter_weights[i],
                                                             time_interval_arr[i],
                                                             fixed_next_time_index,
                                                             distance_prior_bound,
                                                             max_rejections)

                if out_particles[j] is None:
                    out_particles[j] = full_backward_sample(fixed_particle,
                                                            first_edge_fixed, first_edge_fixed_length,
                                                            filter_particles[i],
                                                            filter_weights[i],
                                                            time_interval_arr[i],
                                                            fixed_next_time_index,
                                                            False)
        if full_sampling:
            print(str(filter_particles[i].latest_observation_time) + " Av Backward ESS: " + str(np.mean(ess_back[i])))
        else:
            print(str(filter_particles[i].latest_observation_time))

    if full_sampling:
        out_particles.ess_back = ess_back

    end = tm()
    out_particles.time = end - start
    return out_particles
