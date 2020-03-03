########################################################################################################################
# Module: inference/smc.py
# Description: Implementation of sequential Monte Carlo map-matching. Both offline and online.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

from time import time

import numpy as np

import tools.edges
from inference.particles import MMParticles
from inference.model import default_d_max
from inference.proposal import optimal_proposal
from inference.resampling import fixed_lag_stitching


def initiate_particles(graph,
                       first_observation,
                       n_samps,
                       gps_sd=7,
                       d_init_refine=1,
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
    :param d_init_refine: float
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

    start = time()

    # Discretize edges within truncation
    dis_points = tools.edges.get_truncated_discrete_edges(graph, first_observation, d_init_refine, truncation_distance)

    # Likelihood weights
    weights = np.exp(-0.5 / gps_sd ** 2 * dis_points[:, 4] ** 2)
    weights /= np.sum(weights)

    # Sample indices according to weights
    sampled_indices = np.random.choice(len(weights), n_samps, replace=True, p=weights)

    # Output
    out_particles = MMParticles(dis_points[sampled_indices, :4])

    # Initiate ESS
    out_particles.ess = np.ones((1, out_particles.n)) * out_particles.n if ess_all else np.array([out_particles.n])
    out_particles.ess_pf = np.array([out_particles.n])


    end = time()
    out_particles.time += end - start

    return out_particles


def update_particles(graph,
                     particles,
                     new_observation,
                     time_interval,
                     proposal,
                     lag=3,
                     gps_sd=7,
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
    :param lag: int
        fixed lag for resampling/stitching
    :param gps_sd: float
        metres
        standard deviation of GPS noise
    :param kwargs:
        any additional arguments to be passed to proposal
        i.e. d_refine or d_max for optimal proposal
    :return: MMParticles object (from inference.smc)
    """
    start = time()

    # Initiate particle output
    out_particles = particles.copy()

    # Initiate weight output
    weights = np.zeros(particles.n)

    # Propose and weight for each particle
    for j in range(particles.n):
        out_particles[j], weights[j] = proposal(graph, particles[j], new_observation,
                                                time_interval, gps_sd, **kwargs)
    # Normalise weights
    weights /= sum(weights)

    out_particles.ess_pf = np.append(out_particles.ess_pf, 1 / np.sum(weights ** 2))

    # Resample
    out_particles = fixed_lag_stitching(graph, out_particles, weights, lag)

    end = time()
    out_particles.time += end - start

    return out_particles


def offline_map_match(graph,
                      polyline,
                      n_samps,
                      time_interval,
                      proposal=optimal_proposal,
                      lag=3,
                      gps_sd=7,
                      d_init_refine=1,
                      initial_truncation=None,
                      **kwargs):
    """
    Runs offline map-matching. I.e. receives a full polyline and refers equal probability trajectory particles.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param polyline: list-like, length M
        UTM projection
        series of 2D coordinates
    :param n_samps: int
        number of particles
    :param time_interval: float ########################################################################## change to variable? i.e. allow array
        seconds
        time between observations
    :param proposal: function
        function that takes previous trajectory and new observation
        and output new trajectory and (unnormalised weight)
        defaults to the optimal (discrete distance) proposal
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
        distance to truncate for sampling initial postition
        defaults to 3 * gps_sd
    :param d_max: float
        metres
        maximum distance vehicle could possibly travel in time_interval
        defaults to 35 * time_interval

    :return: MMParticles object (from inference.smc)
    """

    # Initiate particles
    particles = initiate_particles(graph, polyline[0], n_samps,
                                   gps_sd=gps_sd, d_init_refine=d_init_refine, truncation_distance=initial_truncation,
                                   ess_all=True)

    # Update particles
    for observation in polyline[1:]:
        particles = update_particles(graph, particles, observation, time_interval=time_interval, proposal=proposal,
                                     lag=lag, gps_sd=gps_sd,
                                     **kwargs)

        print(str(particles.latest_observation_time) + " ESS av: " + str(np.mean(particles.ess[-1])))

    return particles


