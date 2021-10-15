########################################################################################################################
# Module: tests/test_resampling.py
# Description: Tests for resampling schemes.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import unittest

import numpy as np
import numpy.testing as npt

from bmm.src.inference.particles import MMParticles
from bmm.src.inference import resampling


class TestMultinomial(unittest.TestCase):
    def test_array_trivial(self):
        array = np.arange(10)
        weights = np.zeros(10)
        weights[0] = 1
        npt.assert_array_equal(resampling.multinomial(array, weights), np.zeros(10))

    def test_array_repeated(self):
        array = np.arange(10)
        weights = np.arange(1, 11)
        weights = weights / weights.sum()
        repeated_resample = np.array([resampling.multinomial(array, weights) for _ in range(10000)])
        empirical_weights = np.array([(repeated_resample == i).mean() for i in array])
        npt.assert_array_almost_equal(weights, empirical_weights, decimal=2)

    def test_list_trivial(self):
        tlist = [a for a in range(10)]
        weights = np.zeros(10)
        weights[0] = 1
        self.assertEqual(resampling.multinomial(tlist, weights), [0 for _ in range(10)])

    def test_list_repeated(self):
        tlist = [a for a in range(10)]
        weights = np.arange(1, 11)
        weights = weights / weights.sum()
        repeated_resample = np.array([resampling.multinomial(tlist, weights) for _ in range(10000)])
        empirical_weights = np.array([(repeated_resample == i).mean() for i in tlist])
        npt.assert_array_almost_equal(weights, empirical_weights, decimal=2)

    def test_mmparticles_trivial(self):
        init_array = np.zeros((3, 6))
        init_array += np.arange(3).reshape(3, 1)
        mmp = MMParticles(init_array)
        weights = np.array([0, 1, 0])
        mmp_resampled = resampling.multinomial(mmp, weights)
        for i in range(3):
            npt.assert_array_equal(mmp_resampled[i], np.array([[0, 1, 1, 1, 1, 1, 1, 0]]))

    def test_mmparticles_repeated(self):
        init_array = np.zeros((10, 6))
        init_array += np.arange(10).reshape(10, 1)
        mmp = MMParticles(init_array)
        weights = np.arange(1, 11)
        weights = weights / weights.sum()
        repeated_resample = [resampling.multinomial(mmp, weights) for _ in range(10000)]
        repeated_resample_arr = np.array([p.particles for p in repeated_resample])[:, :, 0, 1]
        empirical_weights = np.array([(repeated_resample_arr == i).mean() for i in np.arange(10)])
        npt.assert_array_almost_equal(weights, empirical_weights, decimal=2)


if __name__ == '__main__':
    unittest.main()
