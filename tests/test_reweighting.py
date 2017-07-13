#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from keras.layers import Input
from keras.models import Model
import numpy as np

from importance_sampling.reweighting import BiasedReweightingPolicy, \
    NoReweightingPolicy


class TestReweighting(unittest.TestCase):
    def _test_external_reweighting_layer(self, rw):
        s1, s2 = Input(shape=(1,)), Input(shape=(1,))
        w = rw.weight_layer()([s1, s2])
        m = Model(inputs=[s1, s2], outputs=[w])
        m.compile("sgd", "mse")

        r = np.random.rand(100, 1).astype(np.float32)
        r_hat = m.predict([np.zeros((100, 1)), r])
        self.assertTrue(np.all(r == r_hat))

    def test_biased_reweighting(self):
        rw = BiasedReweightingPolicy(k=1.)
        s = np.random.rand(100)
        i = np.arange(100)
        w = rw.sample_weights(i, s).ravel()

        self.assertEqual(rw.weight_size, 1)
        self.assertAlmostEqual(w.dot(s), s.sum())
        self._test_external_reweighting_layer(rw)

        # Make sure that it is just a normalized version of the same weights
        # raised to k
        rw = BiasedReweightingPolicy(k=0.5)
        w_hat = rw.sample_weights(i, s).ravel()
        scales = w**0.5 / w_hat
        scales_hat = np.ones(100)*scales[0]
        self.assertTrue(np.allclose(scales, scales_hat))

    def test_no_reweighting(self):
        rw = NoReweightingPolicy()
        self.assertTrue(
            np.all(
                rw.sample_weights(np.arange(100), np.random.rand(100)) == 1.0
            )
        )
        self._test_external_reweighting_layer(rw)

    def test_adjusted_biased_reweighting(self):
        self.skipTest("Not implemented yet")

    def test_correcting_reweighting_policy(self):
        self.skipTest("Not implemented yet")


if __name__ == "__main__":
    unittest.main()
