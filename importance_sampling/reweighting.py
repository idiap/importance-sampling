#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from keras import backend as K
from keras.layers import Layer
import numpy as np


class ReweightingPolicy(object):
    """ReweightingPolicy defines how we weigh the samples given their
    importance.
    
    Each policy should provide
        1. A layer implementation for use with Keras models
        2. A python implementation for use with the samplers
    """
    def weight_layer(self):
        """Return a layer that accepts the scores and the return value of the
        sample_weights() method and produces new sample weights."""
        raise NotImplementedError()

    def sample_weights(self, idxs, scores):
        """Given the scores and the chosen indices return whatever is needed by
        the weight_layer to produce the sample weights."""
        raise NotImplementedError()

    @property
    def weight_size(self):
        """Return how many numbers per sample make up the sample weights"""
        raise NotImplementedError()


class AdjustedBiasedReweightingPolicy(ReweightingPolicy):
    """AdjustedBiasedReweightingPolicy adjusts the biased sample weights with
    the importance that has just been computed in the forward-backward pass.

    See AdjustedBiasedReweighting for details.
    """
    def __init__(self, k=1.0):
        self.k = k

    def weight_layer(self):
        return AdjustedBiasedReweighting(self.k)

    def sample_weights(self, idxs, scores):
        N = len(scores)
        S1 = scores[np.setdiff1d(np.arange(N), idxs)].sum()

        return np.tile([float(N), float(S1)], (len(idxs), 1))

    @property
    def weight_size(self):
        return 2


class BiasedReweightingPolicy(ReweightingPolicy):
    """BiasedReweightingPolicy computes the sample weights before  the
    forward-backward pass based on the sampling probabilities. It can introduce
    a bias that focuses on the hard examples when combined with the loss as an
    importance metric."""
    def __init__(self, k=1.0):
        self.k = k

    def weight_layer(self):
        return ExternalReweighting()

    def sample_weights(self, idxs, scores):
        N = len(scores)
        s = scores[idxs]
        w = scores.sum() / N / s
        w_hat = w**self.k
        w_hat *= w.dot(s) / w_hat.dot(s)

        return w_hat[:, np.newaxis]

    @property
    def weight_size(self):
        return 1


class NoReweightingPolicy(ReweightingPolicy):
    """Set all sample weights to 1."""
    def weight_layer(self):
        return ExternalReweighting()

    def sample_weights(self, idxs, scores):
        return np.ones((len(idxs), 1))

    @property
    def weight_size(self):
        return 1


class CorrectingReweightingPolicy(ReweightingPolicy):
    """CorrectingReweightingPolicy aims to scale the sample weights according
    to the mistakes of the importance predictor

    Arguments
    ---------
    k: float
       The bias power used in all other reweighting schemes
    """
    def __init__(self, k=1.0):
        self.k = k
        self._biased_reweighting = BiasedReweightingPolicy(k)

    def weight_layer(self):
        return CorrectingReweighting()

    def sample_weights(self, idxs, scores):
        w = self._biased_reweighting.sample_weights(idxs, scores)

        return np.hstack([w, scores[idxs][:, np.newaxis]])

    @property
    def weight_size(self):
        return 2


class AdjustedBiasedReweighting(Layer):
    """Implement a Keras layer that using the sum of the weights and the number
    of samples it recomputes the weights.
    
    The specifics are the following:

    Given B = {i_1, i_2, ..., i_|B|} the mini-batch idexes, \\hat{s_i} the
    predicted score of sample i and k the max-loss bias constant.

        S_1 = \\sum_{i \\notin B} \\hat{s_i}
        S_2 = \\sum_{i \\in B} s_i
        a_i^{(1)} = \\frac{1}{N} (\\frac{S1 + S2}{s_i})^k \\forall i \\in B
        a_i^{(2)} = \\frac{1}{N} \\frac{S1 + S2}{s_i} \\forall i \\in B
        t = \\frac{\\sum_{i} a_i^{(2)} s_i }{\\sum_{i} a_i^{(1)} s_i}
        a_i = t a_i^{(1)} \\forall i \\in B
    """
    def __init__(self, k=1.0, **kwargs):
        self.k = k

        super(AdjustedBiasedReweighting, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2
        assert input_shape[0][1] == 1
        assert input_shape[1][1] == 2

        super(AdjustedBiasedReweighting, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, x):
        s, s_hat = x

        # Compute the variables defined in the class comment
        S2 = K.sum(s)
        S1 = s_hat[0, 1]
        N = s_hat[0, 0]

        # Compute the unbiased weights
        a2 = (S1 + S2) / N / s

        # Compute the biased weights and the scaling factor t
        a1 = K.pow(a2, self.k)
        sT = K.transpose(s)
        t = K.dot(sT, a2) / K.dot(sT, a1)

        return K.stop_gradient([a1 * t])[0]


class ExternalReweighting(Layer):
    """Use the provided input as sample weights"""
    def build(self, input_shape):
        super(ExternalReweighting, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, x):
        return K.stop_gradient(x[1])


class CorrectingReweighting(Layer):
    """Use the provided weights and the score to correct sample weights that
    were computed with a very wrong predicted score"""
    def __init__(self, min_decrease=0, max_increase=2, **kwargs):
        self.min_decrease = min_decrease
        self.max_increase = max_increase

        super(CorrectingReweighting, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CorrectingReweighting, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, x):
        s, x1 = x
        a = x1[:, :1]
        s_hat = x1[:, 1:2]

        # Rescale the weights, making sure we mostly scale down
        a_hat = a * K.clip(s_hat / s, self.min_decrease, self.max_increase)

        # Scale again so that the reported loss is comparable to the other ones
        t = 1
        #sT = K.transpose(s)
        #t = K.dot(sT, a) / K.dot(sT, a_hat)

        return K.stop_gradient([a_hat * t])[0]


UNWEIGHTED = NoReweightingPolicy()
UNBIASED = BiasedReweightingPolicy(k=1.0)
