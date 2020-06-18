########################################################################################################################
# Module: tests/test_resampling.py
# Description: Tests for resampling schemes.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import unittest

import numpy as np
import numpy.testing as npt

import bmm.src.inference.particles
from bmm.src.inference import resampling


class TestMultinomial(unittest.TestCase):
    def test_array(self):
        array = np.arange(10)
        weights = np.zeros(10)
        weights[0] = 1
        npt.assert_array_equal(resampling.multinomial(array, weights), np.zeros(10))

    def test_list(self):
        list = [a for a in range(10)]
        weights = np.zeros(10)
        weights[0] = 1
        self.assertEqual(resampling.multinomial(list, weights), [0 for _ in range(10)])

    def test_mmparticles(self):
        init_array = np.zeros((3, 4))
        init_array += np.arange(3).reshape(3, 1)
        mmp = bmm.src.inference.particles.MMParticles(init_array)
        weights = np.array([0, 1, 0])
        mmp_resampled = resampling.multinomial(mmp, weights)
        for i in range(3):
            npt.assert_array_equal(mmp_resampled[i], np.array([[0, 1, 1, 1, 1, 0, 0]]))


if __name__ == '__main__':
    unittest.main()
