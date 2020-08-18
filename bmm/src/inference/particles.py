########################################################################################################################
# Module: inference/particles.py
# Description: Class to store map-matching particles.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

import copy

from bmm.src.tools.edges import observation_time_indices


import numpy as np


class MMParticles:
    """
    Class to store trajectories from map-matching algorithm.

    In particular contains the following object:
    self.particles: list, length = n_samps of arrays
        each array with shape = (_, 9)
        columns: t, u, v, k, alpha, x, y, n_inter, d
            t: seconds, observation time
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
            alpha: in [0,1], position along edge
            x: float, metres, cartesian x coordinate
            y: float, metres, cartesian y coordinate
            d: float, metres, distance travelled since previous observation time

    As well as some useful properties:
        self.n: number of particles
        self.m: number of observations
        self.observation_times: np.array of observation times
        self.latest_observation_time: time of most recently received observation
        self.route_nodes list of length n, each element contains the series of nodes traversed for that particle
    """
    def __init__(self, initial_positions):
        """
        Initiate storage of trajectories with some start positions as input.
        :param initial_positions: list-like, length = n_samps
            each element is list-like of length 6 with elements u, v, k, alpha, x, y
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
            alpha: in [0,1], position along edge
            x: float, metres, cartesian x coordinate
            y: float, metres, cartesian y coordinate
        """
        if initial_positions is not None:
            self.n = len(initial_positions)
            self.particles = [np.zeros((1, 8)) for _ in range(self.n)]
            for i in range(self.n):
                self.particles[i][0, 1:7] = initial_positions[i]
            self.time_intervals = np.array([])
            self.ess_pf = np.zeros(1)
            self.time = 0

    def __repr__(self) -> str:
        return 'bmm.MMParticles'

    def copy(self) -> 'MMParticles':
        out_part = MMParticles(None)
        out_part.n = self.n
        out_part.particles = [p.copy() if p is not None else None for p in self.particles]
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                out_part.__dict__[key] = value.copy()
            # elif isinstance(value, list):
            #     out_part.__dict__[key] = [p.copy() if p is not None else None for p in value]
            elif not isinstance(value, list):
                out_part.__dict__[key] = value
        return out_part

    def deepcopy(self) -> 'MMParticles':
        return copy.deepcopy(self)

    def __len__(self) -> int:
        return self.n

    @property
    def _first_non_none_particle(self) -> np.ndarray:
        """
        Finds the first element of self.particles that is not None
        :return: array for single particle
        """
        try:
            return self.particles[0] if self.particles[0] is not None\
                else next(particle for particle in self.particles if particle is not None)
        except StopIteration:
            raise ValueError("All particles are none")

    @property
    def latest_observation_time(self) -> float:
        """
        Extracts most recent observation time.
        :return: time of most recent observation
        """
        return self._first_non_none_particle[-1, 0]

    @property
    def observation_times(self) -> np.ndarray:
        """
        Extracts all observation times.
        :return: array, shape = (m,)
        """
        all_times = self._first_non_none_particle[:, 0]
        return all_times[observation_time_indices(all_times)]

    @property
    def m(self) -> int:
        """
        Number of observations received.
        :return: number of observations received
        """
        return len(self.observation_times)

    def __getitem__(self, item):
        """
        Extract single particle
        :param item: index of particle to be extracted
        :return: single path array, shape = (_, 9)
        """
        return self.particles[item]

    def __setitem__(self, key, value):
        """
        Allows editing and replacement of particles
        :param key: particle(s) to replace
        :param value: replacement value(s)
        """
        self.particles[key] = value

    def route_nodes(self):
        """
        Returns n series of nodes describing the routes
        :return: length n list of arrays, shape (_,)
        """
        nodes = []
        for p in self.particles:
            edges = p[:, 1:4]
            pruned_edges = np.array([e for i, e in enumerate(edges) if i == 0 or not np.array_equal(e, edges[i-1])])
            nodes += [np.append(pruned_edges[:, 0], pruned_edges[-1, 1])]
        return nodes

