########################################################################################################################
# Module: inference/particles.py
# Description: Defines a class to store map-matching particles.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import copy
from itertools import groupby


import numpy as np


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
    def __init__(self, initial_positions):
        """
        Initiate storage of trajectories with some start positions as input.
        :param initial_positions: list-like, length = n_samps
            each element is list-like of length 4 with elements u, v, k, alpha
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
            alpha: in [0,1], position along edge
        """
        self.n = len(initial_positions)
        self.particles = [np.zeros((1, 7)) for _ in range(self.n)]
        for i in range(self.n):
            self.particles[i][0, 1:5] = initial_positions[i]
        self.time_intervals = np.array([])
        self.ess_pf = np.zeros(1)
        self.time = 0

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return self.n

    @property
    def _first_non_none_particle(self):
        try:
            return self.particles[0] if self.particles[0] is not None\
                else next(particle for particle in self.particles if particle is not None)
        except StopIteration:
            raise ValueError("All particles are none")

    @property
    def latest_observation_time(self):
        """
        Extracts time of most recent observation.
        :return: float
            time of most recent observation
        """
        return self._first_non_none_particle[-1, 0]

    @property
    def observation_times(self):
        """
        Extracts all observation times.
        :return: numpy.ndarray, shape = (m,)
            observation times
        """
        all_times = self._first_non_none_particle[:, 0]
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

    def route_nodes(self):
        """
        Returns n series of nodes describing the routes
        :return: length n list of numpy.ndarrays, shape (_,)
        """
        nodes = []
        for p in self.particles:
            edges = p[:, 1:4]
            pruned_edges = np.array([e for i, e in enumerate(edges) if i == 0 or not np.array_equal(e, edges[i-1])])
            nodes += [np.append(pruned_edges[:, 0], pruned_edges[-1, 1])]
        return nodes

