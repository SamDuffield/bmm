########################################################################################################################
# Module: inference/smc.py
# Description: Implementation of sequential Monte Carlo map-matching. Both offline and online.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import copy

import numpy as np

import tools.edges


class MMParticles:
    """
    Class to store trajectories from map-matching algorithm.

    In particular contains the following objects:
    self.n: int
        number of particles
    self.latest_observation_time: float
        time of most recently received observation
    self.particles: list of numpy.ndarrays, length = n_samps
        each numpy.ndarray with shape = (_, 7)
        columns: t, u, v, k, alpha, n_inter, d
            t: seconds, observation time
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
            alpha: in [0,1], position along edge
            n_inter: int, number of options if intersection
            d: metres, distance travelled since previous observation time
    """
    def __init__(self, initial_positions, name="Trajectories"):
        """
        Initiate storage of trajectories with some start positions as input.
        :param initial_positions: list-like, length = n_samps
            each element is list-like of length 4 with elements u, v, k, alpha
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
            alpha: in [0,1], position along edge
        :param name: str
            optional name for trajectories
        """
        self.name = name
        self.n = len(initial_positions)
        self.particles = [np.zeros((1, 7)) for _ in range(self.n)]
        for i in range(self.n):
            self.particles[i][0, 1:5] = initial_positions[i]

    def copy(self):
        return copy.deepcopy(self)

    @property
    def latest_observation_time(self):
        """
        Extracts time of most recent observation.
        :return: float
            time of most recent observation
        """
        return self.particles[0][-1, 0]

    @property
    def observation_times(self):
        """
        Extracts all observation times.
        :return: numpy.ndarray, shape = (m,)
            observation times
        """
        all_times = self.particles[0][:, 0]
        observation_times = all_times[(all_times != 0) | (np.arange(len(all_times)) == 0)]
        return observation_times

    @property
    def m(self):
        """
        Number of observations received.
        :return: int
            number of observations received
        """
        return len(self.observation_times)

    def __getitem__(self, item):
        """
        Extract single particle
        :param item: int
            index of particle to be extracted
        :return: numpy.ndarray, shape = (_, 7)
            columns: t, u, v, k, alpha, n_inter, d
            t: seconds, observation time
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
            alpha: in [0,1], position along edge
            n_inter: int, number of options if intersection
            d: metres, distance travelled since previous observation time
        """
        return self.particles[item]

    def __setitem__(self, key, value):
        """
        Allows editing and replacement of particles
        :param key: int
            index of particle
        :param value: numpy.ndarray
            replacement value(s)
        """
        self.particles[key] = value


def initiate_particles(graph, first_observation, n_samps, d_refine=1, gps_sd=7, truncation_distance=None):
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
    :param d_refine: float
        metres
        resolution of distance discretisation
        increase of speed, decrease for accuracy
    :param gps_sd: float
        metres
        standard deviation of GPS noise
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

    return MMParticles(dis_points[sampled_indices, :4])


def update_particles(graph, particles, time_interval, new_observation,
                     proposal, resampling_scheme,
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
    :param time_interval: float
        time between last observation and newly received observation
    :param new_observation: numpy.ndarray, shape = (2,)
        UTM projection
        coordinate of new observation
    :param proposal: function
        propagates forward each particle and then reweights
        see inference/proposal
    :param resampling_scheme: function
        resamples: weighted particles -> unweighted particles
        see inference/resampling
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

    if d_max is None:
        d_max = time_interval * 35

    # Propose and reweight
    out_particles = particles.copy()
    out_particles, weights = proposal(graph, out_particles, time_interval, new_observation, d_refine, d_max,
                                      **kwargs)

    # Resample
    out_particles = resampling_scheme(out_particles, weights, **kwargs)

    return out_particles



























