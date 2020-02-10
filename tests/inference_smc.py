########################################################################################################################
# Module: tests/inference_smc.py
# Description: Tests for SMC implementation.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import unittest
import os

import numpy as np
import numpy.testing as npt

from data.utils import read_data
from inference import smc


def load_test_data(test_data_path=None, nrows=None):
    if test_data_path is None:
        test_dir_path = os.path.dirname(os.path.realpath(__file__))
        test_files = os.listdir(test_dir_path)
        test_data_files = [file_name for file_name in test_files if file_name[:8] == 'testdata']
        if len(test_data_files) == 0:
            assert ValueError("Test data not found")
        test_data_file = test_data_files[0]
        test_data_path = test_dir_path + '/' + test_data_file

    if nrows is None:
        return read_data(test_data_path)
    else:
        return read_data(test_data_path, nrows).get_chunk()


class TestMMParticles(unittest.TestCase):
    def setUp(self):
        self.mmp = smc.MMParticles(np.zeros((3, 4)))


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
    test_data = load_test_data()

    unittest.main()
