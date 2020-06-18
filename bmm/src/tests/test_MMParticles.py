########################################################################################################################
# Module: tests/test_MMParticles.py
# Description: Tests for MMParticles class.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import unittest

import numpy as np
import numpy.testing as npt

import bmm.src.inference.particles


class TestMMParticles(unittest.TestCase):
    def setUp(self):
        self.mmp = bmm.src.inference.particles.MMParticles(np.zeros((3, 4)))


class TestInit(TestMMParticles):
    def test_initial_n(self):
        self.assertEqual(self.mmp.n, 3)

    def test_initial_latest_observation_time(self):
        self.assertEqual(self.mmp.latest_observation_time, 0)

    def test_initial_observation_times(self):
        npt.assert_array_equal(self.mmp.observation_times, np.array([0]))

    def test_initial_m(self):
        self.assertEqual(self.mmp.m, 1)

    def test_initial_index(self):
        npt.assert_array_equal(self.mmp[0], np.zeros((1, 7)))

    def test_initial_replace(self):
        self.mmp[1] = np.array(np.ones((1, 7)))
        npt.assert_array_equal(self.mmp[1], np.ones((1, 7)))


class TestUpdate(TestMMParticles):
    def setUp(self):
        super().setUp()
        for i in range(self.mmp.n):
            self.mmp.particles[i] = np.append(self.mmp.particles[i], [[4, 0, 0, 0, 0, 0, 0]], axis=0)

    def test_update_particle_shape(self):
        self.assertEqual(self.mmp[0].shape, (2, 7))

    def test_update_n(self):
        self.assertEqual(self.mmp.n, 3)

    def test_update_latest_observation_time(self):
        self.assertEqual(self.mmp.latest_observation_time, 4)

    def test_update_observation_times(self):
        npt.assert_array_equal(self.mmp.observation_times, np.array([0, 4]))

    def test_update_m(self):
        self.assertEqual(self.mmp.m, 2)

    def test_update_index(self):
        npt.assert_array_equal(self.mmp[0], np.array([[0, 0, 0, 0, 0, 0, 0],
                                                      [4, 0, 0, 0, 0, 0, 0]]))


if __name__ == '__main__':

    unittest.main()
