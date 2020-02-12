########################################################################################################################
# Module: inference/smc.py
# Description: Implementation of sequential Monte Carlo map-matching. Both offline and online.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import numpy as np

import tools.edges
from inference.particles import MMParticles
from inference.resampling import fixed_lag_stitching


def default_d_max(d_max, time_interval, max_speed=35):
    """
    Initiates default value of the maximum distance possibly travelled in the time interval.
    Assumes a maximum possible speed.
    :param d_max: float or None
        metres
        value to be checked
    :param time_interval: float
        seconds
        time between observations
    :param max_speed: float
        metres per second
        assumed maximum possible speed
    :return: float
        defaulted d_max
    """
    return max_speed * time_interval if d_max is None else d_max


def initiate_particles(graph, first_observation, n_samps, gps_sd=7, d_refine=1, truncation_distance=None, ess_all=True):
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
    :return: MMParticles object (from inference.smc)
    """
    if truncation_distance is None:
        truncation_distance = gps_sd * 3

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
    out_particles.ess = np.ones((1, out_particles.n)) * out_particles.n if ess_all else np.array([out_particles.n])

    return out_particles


def update_particles(graph, particles, new_observation, time_interval,
                     proposal, lag=3, gps_sd=7,
                     d_refine=1, d_max=None,
                     **kwargs):
    """
    Update a MMParticles object in light of a newly received observation.
    Particle filter: propose + reweight then resample.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param particles: MMParticles object (from inference.smc)
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
    :param d_refine: float
        metres
        resolution of distance discretisation
        increase of speed, decrease for accuracy
    :param d_max: float
        metres
        maximum distance for vehicle to travel in time_interval
        defaults to time_interval * 35 (35m/s ≈ 78mph)
    :param kwargs:
        any additional arguments to be passed to proposal or resampling functions
        i.e. fixed lag, GPS noise level etc
    :return: MMParticles object (from inference.smc)
    """
    # Default d_max
    d_max = default_d_max(d_max, time_interval)

    # Initiate particle output
    out_particles = particles.copy()

    # Initiate weight output
    weights = np.zeros(particles.n)

    # Propose and weight for each particle
    for j in range(particles.n):
        out_particles[j], weights[j] = proposal(graph, particles[j], new_observation,
                                                time_interval, gps_sd, d_refine, d_max, **kwargs)

    # Normalise weights
    weights /= sum(weights)

    # Resample
    out_particles = fixed_lag_stitching(graph, out_particles, weights, lag)

    return out_particles

