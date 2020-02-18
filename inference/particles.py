########################################################################################################################
# Module: inference/particles.py
# Description: Defines a class to store map-matching particles.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import copy

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
        self.time = 0
        self.ess = None

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return self.n

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

