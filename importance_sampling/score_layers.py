#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from keras import backend as K
from keras import objectives
from keras.layers import Layer


class LossLayer(Layer):
    """LossLayer outputs the loss per sample
    
    # Arguments
        loss: The loss function to use to combine the model output and the
              target
    """
    def __init__(self, loss, **kwargs):
        self.loss = objectives.get(loss)

        super(LossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass # Nothing to do

        super(LossLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # We need two inputs X and y
        assert len(input_shape) == 2

        # (None, 1) because all losses should be scalar
        return (input_shape[0][0], 1)

    def call(self, x, mask=None):
        # We need two inputs X and y
        assert len(x) == 2

        return K.expand_dims(self.loss(*x))


class GradientNormLayer(Layer):
    """GradientNormLayer aims to output the gradient norm given a list of
    parameters (whose gradient to compute) and a loss function to combine the
    two inputs.
    
    # Arguments
        parameter_list: A list of Keras variables to compute the gradient
                        norm for
        loss: The loss function to use to combine the model output and the
              target into a scalar and then compute the gradient norm
        fast: If set to True it means we know that the gradient with respect to
              each sample only affects one part of the parameter list so we can
              use the batch mode to compute the gradient
    """
    def __init__(self, parameter_list, loss, fast=False, **kwargs):
        self.parameter_list = parameter_list
        self.loss = objectives.get(loss)
        self.fast = fast

        super(GradientNormLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass # Nothing to do

        super(GradientNormLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # We get two inputs
        assert len(input_shape) == 2

        return (input_shape[0][0], 1)

    def call(self, x, mask=None):
        # x should be an output and a target
        assert len(x) == 2

        losses = self.loss(*x)
        if self.fast:
            grads = K.sqrt(sum([
                K.sum(K.square(g), axis=1)
                for g in K.gradients(losses, self.parameter_list)
            ]))
        else:
            nb_samples = K.shape(losses)[0]
            grads = K.map_fn(
                lambda i: self._grad_norm(losses[i]),
                K.arange(0, nb_samples),
                dtype=K.floatx()
            )

        return K.reshape(grads, (-1, 1))

    def _grad_norm(self, loss):
        grads = K.gradients(loss, self.parameter_list)
        return K.sqrt(
            sum([
                K.sum(K.square(g))
                for g in grads
            ])
        )
