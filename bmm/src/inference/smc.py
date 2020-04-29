########################################################################################################################
# Module: inference/smc.py
# Description: Implementation of sequential Monte Carlo map-matching. Both offline and online.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

from time import time as tm
import inspect

import numpy as np

import bmm.src.tools.edges
from bmm.src.inference.particles import MMParticles
from bmm.src.inference.proposal import optimal_proposal, auxiliary_distance_proposal
from bmm.src.inference.resampling import fixed_lag_stitching, multinomial, fixed_lag_stitch_post_split
from bmm.src.inference.backward import backward_simulate

updates = ('PF', 'BSi')

proposals = ('optimal', 'aux_dist')


def initiate_particles(graph,
                       first_observation,
                       n_samps,
                       gps_sd=7,
                       d_refine=1,
                       truncation_distance=None,
                       ess_all=True,
                       filter_store=False):
    """
    Initiate start of a trajectory by sampling points around the first observation.
    Note that coordinate system of inputs must be the same, typically a UTM projection (not longtitude-latitude!).
    See tools/graph.py and data/preprocess.py to ensure both your grapha and polyline data are projected to UTM.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param first_observation: numpy.ndarray, shape = (2,)
        UTM projection
        coordinate of first observation
    :param n_samps: int
        number of samples to generate
    :param gps_sd: float
        metres
        standard deviation of GPS noise
    :param d_refine: float
        metres
        resolution of distance discretisation
        increase of speed, decrease for accuracy
    :param truncation_distance: float
        metres
        distance beyond which to assume zero likelihood probability
        required only for first observation
    :param ess_all: bool
        if true initiate effective sample size for each particle for each observation
        otherwise initiate effective sample size only for each observation
    :param filter_store: bool
        whether to initiate storage of filter particles and weights
    :return: MMParticles object (from inference.smc)
    """
    if truncation_distance is None:
        truncation_distance = gps_sd * 3

    start = tm()

    # Discretize edges within truncation
    dis_points = bmm.src.tools.edges.get_truncated_discrete_edges(graph, first_observation, d_refine,
                                                                  truncation_distance)

    # Likelihood weights
    weights = np.exp(-0.5 / gps_sd ** 2 * dis_points[:, 4] ** 2)
    weights /= np.sum(weights)

    # Sample indices according to weights
    sampled_indices = np.random.choice(len(weights), n_samps, replace=True, p=weights)

    # Output
    out_particles = MMParticles(dis_points[sampled_indices, :4])

    # Initiate ESS
    if ess_all:
        out_particles.ess_stitch = np.ones((1, out_particles.n)) * out_particles.n
    out_particles.ess_pf = np.array([out_particles.n])

    if filter_store:
        out_particles.filter_particles = [out_particles.copy()]
        out_particles.filter_weights = np.ones((1, n_samps)) / n_samps

    end = tm()
    out_particles.time += end - start

    return out_particles


def update_particles_flpf(graph,
                          particles,
                          new_observation,
                          time_interval,
                          proposal_func,
                          lag=3,
                          gps_sd=7,
                          max_rejections=20,
                          **kwargs):
    """
    Joint fixed-lag update in light of a newly received observation, using raw particle filter output for stitching.
    Propose + reweight then fixed-lag stitching.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param particles: MMParticles object (from inference.particles)
        unweighted particle approximation up to the previous observation time
    :param new_observation: numpy.ndarray, shape = (2,)
        UTM projection
        coordinate of new observation]
    :param time_interval: float
        seconds
        time between last observation and newly received observation
    :param proposal_func: func
        see inference/proposal
    :param lag: int
        fixed lag for resampling/stitching
    :param gps_sd: float
        metres
        standard deviation of GPS noise
    :param max_rejections: int
        number of rejections before doing full fixed-lag stitching in resampling
        0 will do full fixed-lag stitching and track ess_stitch
    :param kwargs:
        any additional arguments to be passed to proposal
        i.e. d_refine or d_max for optimal proposal
    :return: MMParticles object (from inference.smc)
    """
    start = tm()

    # Initiate particle output
    out_particles = particles.copy()

    # Initiate weight output
    weights = np.zeros(particles.n)

    # Propose and weight for each particle
    for j in range(particles.n):
        out_particles[j], weights[j] = proposal_func(graph, out_particles[j], new_observation,
                                                     time_interval, gps_sd, **kwargs)

    # print(sum([p is None for p in out_particles]))

    # Normalise weights
    weights /= sum(weights)

    # Store ESS
    out_particles.ess_pf = np.append(out_particles.ess_pf, 1 / np.sum(weights ** 2))

    ############################## ADD CATCH FOR ALL NONE (i.e. reinitiate particles at new_observation = start new route)

    # Update time intervals
    out_particles.time_intervals = np.append(out_particles.time_intervals, time_interval)

    # Resample
    out_particles = fixed_lag_stitching(graph, out_particles, weights, lag, max_rejections)

    end = tm()
    out_particles.time += end - start

    return out_particles


def update_particles_flbs(graph,
                          particles,
                          new_observation,
                          time_interval,
                          proposal_func,
                          lag=3,
                          gps_sd=7,
                          max_rejections=20,
                          ess_threshold=1,
                          **kwargs):
    """
    Joint fixed-lag update in light of a newly received observation, using raw particle filter output for stitching.
    Propose + reweight then backward simulation + fixed-lag stitching.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param particles: MMParticles object (from inference.particles)
        unweighted particle approximation up to the previous observation time
    :param new_observation: numpy.ndarray, shape = (2,)
        UTM projection
        coordinate of new observation]
    :param time_interval: float
        seconds
        time between last observation and newly received observation
    :param proposal_func: func
        see inference/proposal
    :param lag: int
        fixed lag for resampling/stitching
    :param gps_sd: float
        metres
        standard deviation of GPS noise
    :param max_rejections: int
        number of rejections before doing full fixed-lag stitching in resampling
        0 will do full fixed-lag stitching and track ess_stitch
    :param ess_threshold: float in [0,1]
        when to resample particle filter
        will resample if ess < ess_threshold * n_samps
    :param kwargs:
        any additional arguments to be passed to proposal
        i.e. d_refine or d_max for optimal proposal
    :return: MMParticles object (from inference.smc)
    """
    start = tm()

    # Extract basic quantities
    n = particles.n
    observation_times = particles.observation_times
    m = len(observation_times)
    stitching_required = m > lag

    # Initiate particle output
    out_particles = particles.copy()

    # Initiate weight output
    weights = np.zeros(particles.n)

    # Initiate new filter particles
    latest_filter_particles = out_particles.filter_particles[-1].copy()

    # Which particles to propose from (out_particles have been resampled, filter_particles haven't)
    previous_resample = particles.ess_pf[-1] < ess_threshold * n
    base_particles = out_particles if previous_resample else out_particles.filter_particles[-1]

    # Propose and weight for each particle
    for j in range(n):
        latest_filter_particles[j], weights[j] = proposal_func(graph, base_particles[j], new_observation,
                                                               time_interval, gps_sd,
                                                               full_smoothing=False,
                                                               **kwargs)

    # Update weights if not resampled
    if not previous_resample:
        weights *= particles.filter_weights[-1]

    # Normalise weights
    weights /= sum(weights)

    # Append new filter particles and weights, discard old ones
    start_point = 1 if stitching_required else 0
    out_particles.filter_particles = out_particles.filter_particles[start_point:] + [latest_filter_particles]
    out_particles.filter_weights = np.append(out_particles.filter_weights[start_point:], weights[np.newaxis], axis=0)

    # Store ESS
    out_particles.ess_pf = np.append(out_particles.ess_pf, 1 / np.sum(weights ** 2))

    # Update time intervals
    out_particles.time_intervals = np.append(out_particles.time_intervals, time_interval)

    # Run backward simulation
    backward_particles = backward_simulate(graph,
                                           out_particles.filter_particles, out_particles.filter_weights,
                                           out_particles.time_intervals[max(m - lag, 0):],
                                           max_rejections)

    if stitching_required:
        # Largest time not to be resampled
        max_fixed_time = observation_times[m - lag - 1]

        # Smallest time to be resampled
        min_resample_time = observation_times[m - lag]

        # Initiate
        min_resample_time_indices = np.zeros(n, dtype=int)
        originial_stitching_distances = np.zeros(n)

        # Extract fixed particles
        fixed_particles = out_particles.copy()
        for j in range(n):
            if out_particles[j] is None:
                continue
            max_fixed_time_index = np.where(out_particles[j][:, 0] == max_fixed_time)[0][0]
            fixed_particles[j] = out_particles[j][:(max_fixed_time_index + 1)]
            min_resample_time_indices[j] = np.where(backward_particles[j][:, 0] == min_resample_time)[0][0]
            originial_stitching_distances[j] = backward_particles[j][min_resample_time_indices[j], -1]

        # Stitch
        out_particles = fixed_lag_stitch_post_split(graph,
                                                    fixed_particles,
                                                    backward_particles,
                                                    np.ones(n) / n,
                                                    min_resample_time, min_resample_time_indices,
                                                    originial_stitching_distances,
                                                    max_rejections)

    else:
        out_particles.particles = backward_particles.particles

    end = tm()
    out_particles.time += end - start

    return out_particles


def update_particles(graph,
                     particles,
                     new_observation,
                     time_interval,
                     proposal='optimal',
                     lag=3,
                     gps_sd=7,
                     max_rejections=20,
                     update='PF',
                     **kwargs):
    """
    Runs offline map-matching. I.e. receives a full polyline and returns an equal probability collection
    of trajectories (particles).
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param particles: MMParticles object (from inference.particles)
        unweighted particle approximation up to the previous observation time
    :param new_observation: numpy.ndarray, shape = (2,)
        UTM projection
        coordinate of new observation]
    :param time_interval: float or list-like (length M-1)
        seconds
        time between observations
    :param proposal: str
        either 'optimal' or 'aux_dist'
        see inference/proposal
        defaults to optimal (discretised) proposal
    :param update: str
        PF = new particles from particle filter
        BSi = new particles from backward simulation
    :param lag: int
        fixed lag, the number of observations beyond which to stop resampling
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
    :param kwargs: optional parameters to pass to proposal
        i.e. d_max, d_refine or var
        as well as ess_threshold for backward simulation update
    :return: MMParticles object (from inference.smc)
    """
    if proposal == 'optimal':
        proposal_func = optimal_proposal
    elif proposal == 'aux_dist':
        proposal_func = auxiliary_distance_proposal
    else:
        raise ValueError("Proposal " + str(proposal) + " not recognised, see bmm.proposals for valid options")

    if update == 'PF':
        return update_particles_flpf(graph,
                                     particles,
                                     new_observation,
                                     time_interval,
                                     proposal_func,
                                     lag,
                                     gps_sd,
                                     max_rejections,
                                     **kwargs)
    elif update == 'BSi':
        return update_particles_flbs(graph,
                                     particles,
                                     new_observation,
                                     time_interval,
                                     proposal_func,
                                     lag,
                                     gps_sd,
                                     max_rejections,
                                     **kwargs)
    else:
        raise ValueError("update " + update + " not recognised, see bmm.updates for valid options")


def _offline_map_match_fl(graph,
                          polyline,
                          n_samps,
                          time_interval,
                          proposal='optimal',
                          update='PF',
                          lag=3,
                          gps_sd=7,
                          d_refine=1,
                          initial_truncation=None,
                          max_rejections=20,
                          **kwargs):
    """
    Runs offline map-matching. I.e. receives a full polyline and returns an equal probability collection
    of trajectories (particles).
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param polyline: list-like, length M
        UTM projection
        series of 2D coordinates
    :param n_samps: int
        number of particles
    :param time_interval: float or list-like (length M-1)
        seconds
        time between observations
    :param proposal: str
        either 'optimal' or 'aux_dist'
        see inference/proposal
        defaults to optimal (discretised) proposal
    :param update: str
        PF = new particles from particle filter
        BSi = new particles from backward simulation
    :param lag: int
        fixed lag, the number of observations beyond which to stop resampling
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
    :param kwargs: optional parameters to pass to proposal
        i.e. d_max, d_refine or var
        as well as ess_threshold for backward simulation update
    :return: MMParticles object (from inference.smc)
    """
    if proposal == 'optimal':
        proposal_func = optimal_proposal
    elif proposal == 'aux_dist':
        proposal_func = auxiliary_distance_proposal
    else:
        raise ValueError("Proposal " + str(proposal) + "not recognised, see bmm.proposals for valid options")

    num_obs = len(polyline)

    ess_all = max_rejections == 0

    # Initiate particles
    particles = initiate_particles(graph, polyline[0], n_samps,
                                   gps_sd=gps_sd, d_refine=d_refine, truncation_distance=initial_truncation,
                                   ess_all=ess_all,
                                   filter_store=update == 'BSi')

    print(str(particles.latest_observation_time) + " PF ESS: " + str(np.mean(particles.ess_pf[-1])))

    if 'd_refine' in inspect.getfullargspec(proposal_func)[0]:
        kwargs['d_refine'] = d_refine

    if isinstance(time_interval, (int, float)):
        time_interval_arr = np.ones(num_obs - 1) * time_interval
    elif len(time_interval) == (num_obs - 1):
        time_interval_arr = time_interval
    else:
        raise ValueError("time_interval must be either float or list-like of length one less than polyline")

    if update == 'PF':
        update_func = update_particles_flpf
    elif update == 'BSi':
        update_func = update_particles_flbs
    else:
        raise ValueError('Update of ' + str(update) + ' not understood')

    # Update particles
    for i in range(num_obs - 1):
        particles = update_func(graph, particles, polyline[1 + i], time_interval=time_interval_arr[i],
                                proposal_func=proposal_func, lag=lag, gps_sd=gps_sd, max_rejections=max_rejections,
                                **kwargs)

        print(str(particles.latest_observation_time) + " PF ESS: " + str(np.mean(particles.ess_pf[-1])))

    return particles


def offline_map_match(graph,
                      polyline,
                      n_samps,
                      time_interval,
                      proposal='optimal',
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
    :param proposal: str
        either 'optimal' or 'aux_dist'
        see inference/proposal
        defaults to optimal (discretised) proposal
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
    if proposal == 'optimal':
        proposal_func = optimal_proposal
    elif proposal == 'aux_dist':
        proposal_func = auxiliary_distance_proposal
    else:
        raise ValueError("Proposal " + str(proposal) + "not recognised, see bmm.proposals for valid options")

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

    if 'd_refine' in inspect.getfullargspec(proposal_func)[0]:
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
            filter_particles[i + 1][j], temp_weights[j] = proposal_func(graph, live_particles[j], polyline[i + 1],
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
    out_particles = backward_simulate(graph,
                                      filter_particles, filter_weights,
                                      time_interval_arr,
                                      max_rejections,
                                      verbose=True)

    end = tm()
    out_particles.time = end - start
    return out_particles
