########################################################################################################################
# Module: inference/smc.py
# Description: Implementation of sequential Monte Carlo map-matching. Both offline and online.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

from time import time as tm
import inspect

import numpy as np

import tools.edges
from inference.particles import MMParticles
from inference.proposal import optimal_proposal
from inference.resampling import fixed_lag_stitching


def initiate_particles(graph,
                       first_observation,
                       n_samps,
                       gps_sd=7,
                       d_refine=1,
                       truncation_distance=None,
                       ess_all=True):
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
    :param ess_all: boolean
        if true initiate effective sample size for each particle for each observation
        otherwise initiate effective sample size only for each observation
    :return: MMParticles object (from inference.smc)
    """
    if truncation_distance is None:
        truncation_distance = gps_sd * 3

    start = tm()

    # Discretize edges within truncation
    dis_points = tools.edges.get_truncated_discrete_edges(graph, first_observation, d_refine, truncation_distance)

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

    end = tm()
    out_particles.time += end - start

    return out_particles


def update_particles(graph,
                     particles,
                     new_observation,
                     time_interval,
                     proposal,
                     lag=3,
                     gps_sd=7,
                     max_rejections=100,
                     **kwargs):
    """
    Update a MMParticles object in light of a newly received observation.
    Particle filter: propose + reweight then resample.
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
    :param proposal: function
        propagates forward each particle and then reweights
        see inference/proposal
    :param resampling: function
        resamples particles
        converts weighted particles to unweighted
        invokes fixed-lag approximation
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
        out_particles[j], weights[j] = proposal(graph, out_particles[j], new_observation,
                                                time_interval, gps_sd, **kwargs)

    # print(sum([p is None for p in out_particles]))

    # Normalise weights
    weights /= sum(weights)

    # Store ESS
    out_particles.ess_pf = np.append(out_particles.ess_pf, 1 / np.sum(weights ** 2))

    ############################## ADD CATCH FOR ALL NONE (i.e. reinitiate particles at new_observation = start new route)

    # Resample
    out_particles = fixed_lag_stitching(graph, out_particles, weights, lag, max_rejections)

    end = tm()
    out_particles.time += end - start

    return out_particles


def offline_map_match_fl(graph,
                         polyline,
                         n_samps,
                         time_interval,
                         proposal=optimal_proposal,
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
    :param proposal: function
        function that takes previous trajectory and new observation
        and output new trajectory and (unnormalised weight)
        defaults to the optimal (discrete distance) proposal
    :param resampling: function
        resamples particles
        converts weighted particles to unweighted
        invokes fixed-lag approximation
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
    :return: MMParticles object (from inference.smc)
    """
    num_obs = len(polyline)

    ess_all = max_rejections == 0

    # Initiate particles
    particles = initiate_particles(graph, polyline[0], n_samps,
                                   gps_sd=gps_sd, d_refine=d_refine, truncation_distance=initial_truncation,
                                   ess_all=ess_all)

    print(str(particles.latest_observation_time) + " PF ESS: " + str(np.mean(particles.ess_pf[-1])))

    if 'd_refine' in inspect.getfullargspec(proposal)[0]:
        kwargs['d_refine'] = d_refine

    if isinstance(time_interval, (int, float)):
        time_interval_arr = np.ones(num_obs - 1) * time_interval
    elif len(time_interval) == (num_obs - 1):
        time_interval_arr = time_interval
    else:
        raise ValueError("time_interval must be either float or list-like of length one less than polyline")

    # Update particles
    for i in range(num_obs - 1):
        particles = update_particles(graph, particles, polyline[1 + i], time_interval=time_interval_arr[i],
                                     proposal=proposal, lag=lag, gps_sd=gps_sd, max_rejections=max_rejections,
                                     **kwargs)

        print(str(particles.latest_observation_time) + " PF ESS: " + str(np.mean(particles.ess_pf[-1])))

    return particles

