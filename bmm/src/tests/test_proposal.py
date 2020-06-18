########################################################################################################################
# Module: tests/test_proposal.py
# Description: Tests for the propagate and reweighting steps of our SMC implementation.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import unittest

import numpy as np

from bmm.src.tests.test_smc import TestWithGraphAndData
from bmm.src.inference import proposal


class TestGetRoutes(TestWithGraphAndData):
    def test_possible_routes(self):
        start_position = np.zeros((1, 7))
        start_position[0, 1:4] = self.gdf.to_numpy()[50, :3]
        start_position[0, 4] = 0.1
        routes = proposal.get_possible_routes(self.graph, start_position.copy(), 100)
        self.assertEqual(routes[-1].shape[1], 7)
        self.assertEqual(routes[0][-1, 6], 100)

    def test_possible_routes_all(self):
        start_position = np.zeros((1, 7))
        start_position[0, 1:4] = self.gdf.to_numpy()[50, :3]
        start_position[0, 4] = 0.1
        routes = proposal.get_possible_routes(self.graph, start_position.copy(), 100, all_routes=True)
        self.assertEqual(routes[-1].shape[1], 7)


class TestDiscretiseRoutes(TestWithGraphAndData):
    def test_discretise_route(self):
        start_position = np.zeros((1, 7))
        start_position[0, 1:4] = self.gdf.to_numpy()[50, :3]
        start_position[0, 4] = 0.1
        routes = proposal.get_possible_routes(self.graph, start_position.copy(), 100, all_routes=True)
        dis_route = proposal.discretise_route(self.graph, routes[-1], np.linspace(0.01, 100, 50))
        self.assertEqual(dis_route.shape[1], 5)


if __name__ == '__main__':
    unittest.main()
