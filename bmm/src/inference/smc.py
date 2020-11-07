########################################################################################################################
# Module: inference/smc.py
# Description: Implementation of sequential Monte Carlo map-matching. Both offline and online.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

from time import time as tm
import inspect
from typing import Callable, Union, Tuple

import numpy as np
from networkx.classes import MultiDiGraph

from bmm.src.tools import edges
from bmm.src.inference.particles import MMParticles
from bmm.src.inference.proposal import optimal_proposal
from bmm.src.inference.resampling import fixed_lag_stitching, multinomial, fixed_lag_stitch_post_split
from bmm.src.inference.backward import backward_simulate
from bmm.src.inference.model import MapMatchingModel, ExponentialMapMatchingModel

updates = ('PF', 'BSi')

proposals = ('optimal', 'aux_dist', 'dist_then_edge')


def get_proposal(proposal_str: str) -> Callable:
    """
    Converts string to proposal function
    :param proposal_str: string indicating which proposal to use, see bmm.proposals for included proposals
    :return: proposal function
    """
    if proposal_str == 'optimal':
        proposal_func = optimal_proposal
    else:
        raise ValueError(f"Proposal {proposal_str} not recognised, see bmm.proposals for valid options")

    return proposal_func


def get_time_interval_array(timestamps: Union[float, np.ndarray],
                            num_obs: int) -> np.ndarray:
    """
    Preprocess timestamp in put
    :param timestamps: either float if all observations equally spaced, list of timestamps (length of polyline)
    or list of time intervals (length of polyline - 1)
    :param num_obs: length of polyline
    :return: array of time intervals (length of polyline - 1)
    """
    if isinstance(timestamps, (int, float)):
        return np.ones(num_obs - 1) * timestamps
    elif len(timestamps) == num_obs:
        return timestamps[1:] - timestamps[:-1]
    elif len(timestamps) == (num_obs - 1):
        return timestamps
    else:
        raise ValueError("timestamps input not understood")


def initiate_particles(graph: MultiDiGraph,
                       first_observation: np.ndarray,
                       n_samps: int,
                       mm_model: MapMatchingModel = ExponentialMapMatchingModel(),
                       d_refine: float = 1,
                       d_truncate: float = None,
                       ess_all: bool = True,
                       filter_store: bool = True) -> MMParticles:
    """
    Initiate start of a trajectory by sampling points around the first observation.
    Note that coordinate system of inputs must be the same, typically a UTM projection (not longtitude-latitude!).
    :param graph: encodes road network, simplified and projected to UTM
    :param mm_model: MapMatchingModel
    :param first_observation: cartesian coordinate in UTM
    :param n_samps: number of samples to generate
    :param d_refine: metres, resolution of distance discretisation
    :param d_truncate: metres, distance beyond which to assume zero likelihood probability
        defaults to 5 * mm_model.gps_sd
    :param ess_all: if true initiate effective sample size for each particle for each observation
        otherwise initiate effective sample size only for each observation
    :param filter_store: whether to initiate storage of filter particles and weights
    :return: MMParticles object
    """
    gps_sd = mm_model.gps_sd

    if d_truncate is None:
        d_truncate = gps_sd * 5

    start = tm()

    # Discretize edges within truncation
    dis_points, dists_to_first_obs = edges.get_truncated_discrete_edges(graph, first_observation,
                                                                        d_refine,
                                                                        d_truncate, True)

    if dis_points.size == 0:
        raise ValueError("No edges found near initial observation: try increasing the initial_truncation")

    # Likelihood weights
    weights = np.exp(-0.5 / gps_sd ** 2 * dists_to_first_obs ** 2)
    weights /= np.sum(weights)

    # Sample indices according to weights
    sampled_indices = np.random.choice(len(weights), n_samps, replace=True, p=weights)

    # Output
    out_particles = MMParticles(dis_points[sampled_indices])

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


def update_particles_flpf(graph: MultiDiGraph,
                          particles: MMParticles,
                          new_observation: np.ndarray,
                          time_interval: float,
                          mm_model: MapMatchingModel,
                          proposal_func: Callable,
                          lag: int = 3,
                          max_rejections: int = 50,
                          **kwargs) -> MMParticles:
    """
    Joint fixed-lag update in light of a newly received observation, uses particle filter trajectories for stitching
    Propose + reweight then fixed-lag stitching.
    :param graph: encodes road network, simplified and projected to UTM
    :param particles: unweighted particle approximation up to the previous observation time
    :param new_observation: cartesian coordinate in UTM
    :param time_interval: time between last observation and newly received observation
    :param mm_model: MapMatchingModel
    :param proposal_func: function to propagate and weight particles
    :param lag: fixed lag for resampling/stitching
    :param max_rejections: number of rejections to attempt before doing full fixed-lag stitching
        0 will do full fixed-lag stitching and track ess_stitch
    :param kwargs:
        any additional arguments to be passed to proposal
        i.e. d_refine or d_max for optimal proposal
    :return: MMParticles object
    """
    start = tm()

    # Propose and weight for each particle
    out_particles, weights, new_norm_constants = propose_particles(proposal_func,
                                                                   None,
                                                                   graph,
                                                                   particles,
                                                                   new_observation,
                                                                   time_interval,
                                                                   mm_model,
                                                                   full_smoothing=True,
                                                                   store_norm_quants=False,
                                                                   **kwargs)

    # Normalise weights
    weights /= sum(weights)

    if np.any(np.isnan(weights)):
        raise ZeroDivisionError('Map-matching failed (all weights zero)')

    # Store norm constants
    if hasattr(out_particles, 'prior_norm'):
        out_particles.prior_norm = np.vstack([out_particles.prior_norm, new_norm_constants])
    else:
        out_particles.prior_norm = new_norm_constants[None]

    # Store ESS
    out_particles.ess_pf = np.append(out_particles.ess_pf, 1 / np.sum(weights ** 2))

    # Update time intervals
    out_particles.time_intervals = np.append(out_particles.time_intervals, time_interval)

    # Resample
    out_particles = fixed_lag_stitching(graph, mm_model, out_particles, weights, lag, max_rejections)

    end = tm()
    out_particles.time += end - start

    return out_particles


def update_particles_flbs(graph: MultiDiGraph,
                          particles: MMParticles,
                          new_observation: np.ndarray,
                          time_interval: float,
                          mm_model: MapMatchingModel,
                          proposal_func: Callable,
                          lag: int = 3,
                          max_rejections: int = 20,
                          ess_threshold: float = 1.,
                          **kwargs) -> MMParticles:
    """
    Joint fixed-lag update in light of a newly received observation, uses partial backward simulation runs for stitching
    Propose + reweight then backward simulation + fixed-lag stitching.
    :param graph: encodes road network, simplified and projected to UTM
    :param particles: unweighted particle approximation up to the previous observation time
    :param new_observation: cartesian coordinate in UTM
    :param time_interval: time between last observation and newly received observation
    :param mm_model: MapMatchingModel
    :param proposal_func: function to propagate and weight particles
    :param lag: fixed lag for resampling/stitching
    :param max_rejections: number of rejections to attempt before doing full fixed-lag stitching
        0 will do full fixed-lag stitching and track ess_stitch
    :param ess_threshold: in [0,1], particle filter resamples if ess < ess_threshold * n_samps
    :param kwargs:
        any additional arguments to be passed to proposal
        i.e. d_refine or d_max for optimal proposal
    :return: MMParticles object
    """
    start = tm()

    filter_particles = particles.filter_particles

    # Extract basic quantities
    n = particles.n
    observation_times = np.append(particles.observation_times, particles.observation_times[-1] + time_interval)
    m = len(observation_times) - 1
    stitching_required = m > lag

    # Initiate particle output
    out_particles = particles.copy()

    # Which particles to propose from (out_particles have been resampled, filter_particles haven't)
    previous_resample = particles.ess_pf[-1] < ess_threshold * n
    base_particles = out_particles if previous_resample else particles.filter_particles[-1].copy()

    latest_filter_particles, weights, temp_prior_norm = propose_particles(proposal_func,
                                                                          None,
                                                                          graph,
                                                                          base_particles,
                                                                          new_observation,
                                                                          time_interval,
                                                                          mm_model,
                                                                          full_smoothing=False,
                                                                          store_norm_quants=False,
                                                                          **kwargs)

    filter_particles[-1].prior_norm = temp_prior_norm

    # Update weights if not resampled
    if not previous_resample:
        weights *= particles.filter_weights[-1]

    # Normalise weights
    weights /= sum(weights)

    # Append new filter particles and weights, discard old ones
    start_point = 1 if stitching_required else 0
    filter_particles = particles.filter_particles[start_point:] + [latest_filter_particles]
    out_particles.filter_weights = np.append(out_particles.filter_weights[start_point:], weights[np.newaxis], axis=0)

    # Store ESS
    out_particles.ess_pf = np.append(out_particles.ess_pf, 1 / np.sum(weights ** 2))

    # Update time intervals
    out_particles.time_intervals = np.append(out_particles.time_intervals, time_interval)

    # Run backward simulation
    backward_particles = backward_simulate(graph,
                                           filter_particles,
                                           out_particles.filter_weights,
                                           out_particles.time_intervals[-lag:] if lag != 0 else [],
                                           mm_model,
                                           max_rejections,
                                           store_ess_back=False,
                                           store_norm_quants=True)
    backward_particles.prior_norm = backward_particles.dev_norm_quants[0]
    del backward_particles.dev_norm_quants

    if stitching_required:
        # Largest time not to be resampled
        max_fixed_time = observation_times[m - lag - 1]

        # Extract fixed particles
        fixed_particles = out_particles.copy()
        for j in range(n):
            if out_particles[j] is None:
                continue
            max_fixed_time_index = np.where(out_particles[j][:, 0] == max_fixed_time)[0][0]
            fixed_particles[j] = out_particles[j][:(max_fixed_time_index + 1)]

        # Stitch
        out_particles = fixed_lag_stitch_post_split(graph,
                                                    fixed_particles,
                                                    backward_particles,
                                                    np.ones(n) / n,
                                                    mm_model,
                                                    max_rejections)

    else:
        out_particles.particles = backward_particles.particles

    out_particles.filter_particles = filter_particles

    end = tm()
    out_particles.time += end - start

    return out_particles


def update_particles(graph: MultiDiGraph,
                     particles: MMParticles,
                     new_observation: np.ndarray,
                     time_interval: float,
                     mm_model: MapMatchingModel = ExponentialMapMatchingModel(),
                     proposal: str = 'optimal',
                     update: str = 'BSi',
                     lag: int = 3,
                     max_rejections: int = 20,
                     **kwargs) -> MMParticles:
    """
    Updates particle approximation in receipt of new observation
    :param graph: encodes road network, simplified and projected to UTM
    :param particles: unweighted particle approximation up to the previous observation time
    :param new_observation: cartesian coordinate in UTM
    :param time_interval: time between last observation and newly received observation
    :param mm_model: MapMatchingModel
    :param proposal: either 'optimal' or 'aux_dist'
        defaults to optimal (discretised) proposal
    :param update:
        'PF' for particle filter fixed-lag update
        'BSi' for backward simulation fixed-lag update
        must be consistent across updates
    :param lag: fixed lag for resampling/stitching
    :param max_rejections: number of rejections to attempt before doing full fixed-lag stitching
        0 will do full fixed-lag stitching and track ess_stitch
    :param kwargs: optional parameters to pass to proposal
        i.e. d_max, d_refine or var
        as well as ess_threshold for backward simulation update
    :return: MMParticles object
    """

    proposal_func = get_proposal(proposal)

    if update == 'PF' or lag == 0:
        return update_particles_flpf(graph,
                                     particles,
                                     new_observation,
                                     time_interval,
                                     mm_model,
                                     proposal_func,
                                     lag,
                                     max_rejections,
                                     **kwargs)
    elif update == 'BSi':
        return update_particles_flbs(graph,
                                     particles,
                                     new_observation,
                                     time_interval,
                                     mm_model,
                                     proposal_func,
                                     lag,
                                     max_rejections,
                                     **kwargs)
    else:
        raise ValueError("update " + update + " not recognised, see bmm.updates for valid options")


def _offline_map_match_fl(graph: MultiDiGraph,
                          polyline: np.ndarray,
                          n_samps: int,
                          timestamps: Union[float, np.ndarray],
                          mm_model: MapMatchingModel = ExponentialMapMatchingModel(),
                          proposal: str = 'optimal',
                          update: str = 'BSi',
                          lag: int = 3,
                          d_refine: int = 1,
                          initial_d_truncate: float = None,
                          max_rejections: int = 20,
                          **kwargs) -> MMParticles:
    """
    Runs offline map-matching but uses online fixed-lag techniques.
    Only recommended for simulation purposes.
    :param graph: encodes road network, simplified and projected to UTM
    :param polyline: series of cartesian coordinates in UTM
    :param n_samps: int
        number of particles
    :param timestamps: seconds
        either float if all times between observations are the same, or a series of timestamps in seconds/UNIX timestamp
    :param mm_model: MapMatchingModel
    :param proposal: either 'optimal' or 'aux_dist'
        defaults to optimal (discretised) proposal
    :param update:
        'PF' for particle filter fixed-lag update
        'BSi' for backward simulation fixed-lag update
        must be consistent across updates
    :param lag: fixed lag for resampling/stitching
    :param d_refine: metres, resolution of distance discretisation
    :param initial_d_truncate: distance beyond which to assume zero likelihood probability at time zero
        defaults to 5 * mm_model.gps_sd
    :param max_rejections: number of rejections to attempt before doing full fixed-lag stitching
        0 will do full fixed-lag stitching and track ess_stitch
    :param kwargs: optional parameters to pass to proposal
        i.e. d_max, d_refine or var
        as well as ess_threshold for backward simulation update
    :return: MMParticles object
    """
    proposal_func = get_proposal(proposal)

    num_obs = len(polyline)

    ess_all = max_rejections == 0

    # Initiate particles
    particles = initiate_particles(graph, polyline[0], n_samps, mm_model=mm_model,
                                   d_refine=d_refine, d_truncate=initial_d_truncate,
                                   ess_all=ess_all,
                                   filter_store=update == 'BSi')

    print(str(particles.latest_observation_time) + " PF ESS: " + str(np.mean(particles.ess_pf[-1])))

    if 'd_refine' in inspect.getfullargspec(proposal_func)[0]:
        kwargs['d_refine'] = d_refine

    time_interval_arr = get_time_interval_array(timestamps, num_obs)

    if update == 'PF' or lag == 0:
        update_func = update_particles_flpf
    elif update == 'BSi':
        update_func = update_particles_flbs
    else:
        raise ValueError('Update of ' + str(update) + ' not understood')

    # Update particles
    for i in range(num_obs - 1):
        particles = update_func(graph, particles, polyline[1 + i], time_interval=time_interval_arr[i],
                                mm_model=mm_model, proposal_func=proposal_func, lag=lag, max_rejections=max_rejections,
                                **kwargs)

        print(str(particles.latest_observation_time) + " PF ESS: " + str(np.mean(particles.ess_pf[-1])))

    return particles


def offline_map_match(graph: MultiDiGraph,
                      polyline: np.ndarray,
                      n_samps: int,
                      timestamps: Union[float, np.ndarray],
                      mm_model: MapMatchingModel = ExponentialMapMatchingModel(),
                      proposal: str = 'optimal',
                      d_refine: int = 1,
                      initial_d_truncate: float = None,
                      max_rejections: int = 20,
                      ess_threshold: float = 1,
                      store_norm_quants: bool = False,
                      **kwargs) -> MMParticles:
    """
    Runs offline map-matching. I.e. receives a full polyline and returns an equal probability collection
    of trajectories.
    Forward-filtering backward-simulation implementation - no fixed-lag approximation needed for offline inference.
    :param graph: encodes road network, simplified and projected to UTM
    :param polyline: series of cartesian cooridnates in UTM
    :param n_samps: int
        number of particles
    :param timestamps: seconds
        either float if all times between observations are the same, or a series of timestamps in seconds/UNIX timestamp
    :param mm_model: MapMatchingModel
    :param proposal: either 'optimal' or 'aux_dist'
        defaults to optimal (discretised) proposal
    :param d_refine: metres, resolution of distance discretisation
    :param initial_d_truncate: distance beyond which to assume zero likelihood probability at time zero
        defaults to 5 * mm_model.gps_sd
    :param max_rejections: number of rejections to attempt before doing full fixed-lag stitching
        0 will do full fixed-lag stitching and track ess_stitch
    :param ess_threshold: in [0,1], particle filter resamples if ess < ess_threshold * n_samps
    :param store_norm_quants: if True normalisation quanitities (including gradient evals) returned in out_particles
    :param kwargs: optional parameters to pass to proposal
        i.e. d_max, d_refine or var
        as well as ess_threshold for backward simulation update
    :return: MMParticles object
    """
    proposal_func = get_proposal(proposal)

    num_obs = len(polyline)

    ess_all = max_rejections == 0

    start = tm()

    filter_particles = [None] * num_obs
    filter_weights = np.zeros((num_obs, n_samps))

    # Initiate filter_particles
    filter_particles[0] = initiate_particles(graph, polyline[0], n_samps, mm_model=mm_model,
                                             d_refine=d_refine, d_truncate=initial_d_truncate,
                                             ess_all=ess_all)
    filter_weights[0] = 1 / n_samps
    live_weights = filter_weights[0].copy()

    ess_pf = np.zeros(num_obs)
    ess_pf[0] = n_samps

    print("0 PF ESS: " + str(ess_pf[0]))

    if 'd_refine' in inspect.getfullargspec(proposal_func)[0]:
        kwargs['d_refine'] = d_refine

    time_interval_arr = get_time_interval_array(timestamps, num_obs)

    # Forward filtering, storing x_t-1, x_t ~ p(x_t-1:t|y_t)
    for i in range(num_obs - 1):
        resample = ess_pf[i] < ess_threshold * n_samps
        filter_particles[i + 1], temp_weights, temp_prior_norm = propose_particles(proposal_func,
                                                                                   live_weights if resample else None,
                                                                                   graph,
                                                                                   filter_particles[i],
                                                                                   polyline[i + 1],
                                                                                   time_interval_arr[i],
                                                                                   mm_model,
                                                                                   full_smoothing=False,
                                                                                   store_norm_quants=store_norm_quants,
                                                                                   **kwargs)

        filter_particles[i].prior_norm = temp_prior_norm

        if not resample:
            temp_weights *= live_weights

        temp_weights /= np.sum(temp_weights)
        filter_weights[i + 1] = temp_weights.copy()
        live_weights = temp_weights.copy()
        ess_pf[i + 1] = 1 / np.sum(temp_weights ** 2)

        print(str(filter_particles[i + 1].latest_observation_time) + " PF ESS: " + str(ess_pf[i + 1]))

    # Backward simulation
    out_particles = backward_simulate(graph,
                                      filter_particles,
                                      filter_weights,
                                      time_interval_arr,
                                      mm_model,
                                      max_rejections,
                                      verbose=True,
                                      store_norm_quants=store_norm_quants)
    out_particles.ess_pf = ess_pf

    end = tm()
    out_particles.time = end - start
    return out_particles


def propose_particles(proposal_func: Callable,
                      resample_weights: Union[None, np.ndarray],
                      graph: MultiDiGraph,
                      particles: MMParticles,
                      new_observation: np.ndarray,
                      time_interval: float,
                      mm_model: MapMatchingModel,
                      full_smoothing: bool = True,
                      store_norm_quants: bool = False,
                      **kwargs) -> Tuple[MMParticles, np.ndarray, np.ndarray]:
    """
    Samples a single particle from the (distance discretised) optimal proposal.
    :param proposal_func: function to proposal single particle
    :param resample_weights: weights for resampling, None for no resampling
    :param graph: encodes road network, simplified and projected to UTM
    :param particles: all particles at last observation time
    :param new_observation: cartesian coordinate in UTM
    :param time_interval: time between last observation and newly received observation
    :param mm_model: MapMatchingModel
    :param full_smoothing: if True returns full trajectory
        otherwise returns only x_t-1 to x_t
    :param store_norm_quants: whether to additionally return quantities needed for gradient EM step
        assuming deviation prior is used
    :return: particle, unnormalised weight, prior_norm(_quants)
    """
    n_samps = particles.n
    out_particles = particles.copy()

    if resample_weights is not None:
        resample_inds = np.random.choice(n_samps, n_samps, replace=True, p=resample_weights)
        not_prop_inds = np.arange(n_samps)[~np.isin(np.arange(n_samps), resample_inds)]
    else:
        resample_inds = np.arange(n_samps)
        not_prop_inds = []

    weights = np.zeros(n_samps)
    prior_norms = np.zeros((n_samps, len(mm_model.distance_params) + 2)) if store_norm_quants else np.zeros(n_samps)
    for j in range(n_samps):
        in_particle = particles[resample_inds[j]]
        in_particle = in_particle.copy() if in_particle is not None else None
        out_particles[j], weights[j], prior_norms[resample_inds[j]] = proposal_func(graph,
                                                                                    in_particle,
                                                                                    new_observation,
                                                                                    time_interval,
                                                                                    mm_model,
                                                                                    full_smoothing=full_smoothing,
                                                                                    store_norm_quants=store_norm_quants,
                                                                                    **kwargs)
    for k in not_prop_inds:
        if particles[k] is not None:
            prior_norms[k] = proposal_func(graph,
                                           particles[k],
                                           None,
                                           time_interval,
                                           mm_model,
                                           full_smoothing=False,
                                           store_norm_quants=store_norm_quants,
                                           only_norm_const=True,
                                           **kwargs)
        else:
            prior_norms[k] = np.zeros(len(mm_model.distance_params) + 2) if store_norm_quants else 0

    return out_particles, weights, prior_norms
