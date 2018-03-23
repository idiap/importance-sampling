#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

import numpy as np

from importance_sampling.datasets import InMemoryDataset
from importance_sampling.reweighting import UNWEIGHTED, UNBIASED
from importance_sampling.samplers import AdaptiveAdditiveSmoothingSampler, \
    AdditiveSmoothingSampler, ModelSampler, PowerSmoothingSampler, \
    UniformSampler, ConstantVarianceSampler


class MockModel(object):
    def __init__(self, positive_score = 1.0):
        self._score = positive_score

    def score(self, x, y, batch_size=1):
        s = np.ones((len(x),))
        s[y[:, 1] != 0] = self._score

        return s


class TestSamplers(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # Create a toy 2D circle dataset
        X = np.random.rand(1000, 2)
        y = (((X - np.array([[0.5, 0.5]]))**2).sum(axis=1) < 0.1).astype(int)
        self.dataset = InMemoryDataset(
            X[:600],
            y[:600],
            X[600:],
            y[600:]
        )

        # The probability of selecting a point in the circle with
        # uniform sampling
        self.prior = (y[:600] == 1).sum() / 600.

        super(TestSamplers, self).__init__(*args, **kwargs)

    def _get_prob(self, a, b=1.0):
        """Compute the probability of sampling a positive class given the
        relative importances a of positive and b of negative"""
        p = self.prior
        return (p * a) / ((1-p)*b + p*a)

    def _test_sampler(self, sampler, N, expected_ones, error=0.02):
        idxs, xy, w = sampler.sample(100)

        self.assertEqual(len(idxs), 100)
        self.assertEqual(len(idxs), len(xy[0]))
        self.assertEqual(len(idxs), len(xy[1]))
        self.assertTrue(np.all(w == 1.))

        ones = 0
        for i in range(N//100):
            _, (x, y), _ = sampler.sample(100)
            ones += y[:, 1].sum()
        self.assertTrue(
            expected_ones - N*error < ones < expected_ones + N*error,
            "Got %d and expected %d" % (ones, expected_ones)
        )

    def test_uniform_sampler(self):
        N = 10000
        expected_ones = self.prior * N

        self._test_sampler(
            UniformSampler(self.dataset, UNWEIGHTED),
            N,
            expected_ones
        )

    def test_model_sampler(self):
        importance = 4.0
        N = 10000
        expected_ones = N * self._get_prob(importance)

        self._test_sampler(
            ModelSampler(self.dataset, UNWEIGHTED, MockModel(importance)),
            N,
            expected_ones
        )

    def test_additive_smoothing_sampler(self):
        importance = 4.0
        c = 2.0
        N = 10000
        expected_ones = N * self._get_prob(importance + c, 1.0 + c)

        self._test_sampler(
            AdditiveSmoothingSampler(
                ModelSampler(self.dataset, UNWEIGHTED, MockModel(importance)),
                c=c
            ),
            N,
            expected_ones
        )

    def test_adaptive_additive_smoothing_sampler(self):
        importance = 4.0
        c = (self.prior * 4.0 + (1.0 - self.prior) * 1.0) / 2.
        N = 10000
        expected_ones = N * self._get_prob(importance + c, 1.0 + c)

        self._test_sampler(
            AdaptiveAdditiveSmoothingSampler(
                ModelSampler(self.dataset, UNWEIGHTED, MockModel(importance))
            ),
            N,
            expected_ones
        )

    def test_power_smoothing_sampler(self):
        importance = 4.0
        N = 10000
        expected_ones = N * self._get_prob(importance**0.5)

        self._test_sampler(
            PowerSmoothingSampler(
                ModelSampler(self.dataset, UNWEIGHTED, MockModel(importance))
            ),
            N,
            expected_ones
        )

    def test_constant_variance_sampler(self):
        importance = 100.0
        y = np.zeros(1024)
        y[np.random.choice(1024, 10)] = 1.0
        dataset = InMemoryDataset(
            np.random.rand(1024, 2),
            y,
            np.random.rand(100, 2),
            y[:100]
        )
        model = MockModel(importance)
        sampler = ConstantVarianceSampler(dataset, UNBIASED, model)

        idxs1, xy, w = sampler.sample(100)
        sampler.update(idxs1, model.score(xy[0], xy[1]))
        for i in range(30):
            _, xy, _ = sampler.sample(100)
            sampler.update(idxs1, model.score(xy[0], xy[1]))
        idxs2, xy, w = sampler.sample(100)
        sampler.update(idxs2, model.score(xy[0], xy[1]))

        self.assertEqual(len(idxs1), 100)
        self.assertLess(len(idxs2), 100)

    @unittest.skip("Not implemented yet")
    def test_lstm_sampler(self):
        pass

    @unittest.skip("Not implemented yet")
    def test_per_class_gaussian_sampler(self):
        pass


if __name__ == "__main__":
    unittest.main()
