#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from functools import partial

from blinker import signal
from keras import backend as K
from keras.layers import Activation, Input, Lambda, multiply
from keras.metrics import categorical_accuracy, binary_accuracy, \
    get as get_metric
import numpy as np
from transparent_keras import TransparentModel

from reweighting import UNBIASED
from score_layers import GradientNormLayer, LossLayer
from utils.functional import compose


def generic_accuracy(y_true, y_pred):
    if K.int_shape(y_pred)[1] == 1:
        return binary_accuracy(y_true, y_pred)
    else:
        return categorical_accuracy(y_true, y_pred)


def MetricLayer(metric_func):
    # Special care for accuracy because keras treats it specially
    if "accuracy" in metric_func:
        metric_func = generic_accuracy
    metric_func = compose(K.expand_dims, get_metric(metric_func))

    return Lambda(lambda inputs: metric_func(*inputs), output_shape=(None, 1))


def _get_scoring_layer(score, y_true, y_pred, loss="categorical_crossentropy",
                       layer=None):
    """Get a scoring layer that computes the score for each pair of y_true,
    y_pred"""
    assert score in ["loss", "gnorm", "acc"]

    if score == "loss":
        return LossLayer(loss)([
            y_true,
            y_pred
        ])
    elif score == "gnorm":
        return GradientNormLayer(
            layer.output,
            loss,
            fast=True
        )([
            y_true,
            y_pred
        ])
    elif score == "acc":
        return LossLayer(categorical_accuracy)([
            y_true,
            y_pred
        ])


class ModelWrapper(object):
    """The goal of the ModelWrapper is to take a NN and add some extra layers
    that produce a score, a loss and the sample weights to perform importance
    sampling."""
    def evaluate(self, x, y, batch_size=128):
        bs = batch_size
        result = np.mean(np.vstack([
            self.evaluate_batch(x[s:s+bs], y[s:s+bs])
            for s in range(0, len(x), bs)
        ]), axis=0)

        signal("is.evaluation").send(result)
        return result

    def score(self, x, y, batch_size=128):
        bs = batch_size
        result = np.hstack([
            self.score_batch(x[s:s+bs], y[s:s+bs]).T
            for s in range(0, len(x), bs)
        ]).T

        signal("is.score").send(result)
        return result

    def set_lr(self, lr):
        """Set the learning rate of the wrapped models.

        We try to set the learning rate on a member variable model and a member
        variable small. If we do not find a member variable model we raise a
        NotImplementedError
        """
        try:
            K.set_value(
                self.model.optimizer.lr,
                lr
            )
        except AttributeError:
            raise NotImplementedError()

        try:
            K.set_value(
                self.small.optimizer.lr,
                lr
            )
        except AttributeError:
            pass

    def evaluate_batch(self, x, y):
        raise NotImplementedError()

    def score_batch(self, x, y):
        raise NotImplementedError()

    def train_batch(self, x, y, w):
        raise NotImplementedError()


class ModelWrapperDecorator(ModelWrapper):
    def __init__(self, model_wrapper, implemented_attributes=set()):
        self.model_wrapper = model_wrapper
        self.implemented_attributes = (
            implemented_attributes | set(["model_wrapper"])
        )

    def __getattribute__(self, name):
        _getattr = object.__getattribute__
        implemented_attributes = _getattr(self, "implemented_attributes")
        if name in implemented_attributes:
            return _getattr(self, name)
        else:
            model_wrapper = _getattr(self, "model_wrapper")
            return getattr(model_wrapper, name)


class OracleWrapper(ModelWrapper):
    AVG_LOSS = 0
    LOSS = 1
    WEIGHTED_LOSS = 2
    SCORE = 3
    METRIC0 = 4

    def __init__(self, model, reweighting, score="loss"):
        self.model = self._augment_model(model, score, reweighting)
        self.reweighting = reweighting

    def _augment_model(self, model, score, reweighting):
        # Extract some info from the model
        loss = model.loss
        optimizer = model.optimizer.__class__(**model.optimizer.get_config())
        output_shape = K.int_shape(model.output)[1:]
        if isinstance(loss, str) and loss.startswith("sparse"):
            output_shape = output_shape[:-1] + (1,)

        # Make sure that some stuff look ok
        assert not isinstance(loss, list)

        # We need to create two more inputs
        #   1. the targets
        #   2. the predicted scores
        y_true = Input(shape=output_shape)
        pred_score = Input(shape=(reweighting.weight_size,))

        # Create a loss layer and a score layer
        loss_tensor = LossLayer(loss)([y_true, model.output])
        last_layer = -2 if isinstance(model.layers[-1], Activation) else -1
        score_tensor = _get_scoring_layer(
            score,
            y_true,
            model.output,
            loss,
            model.layers[last_layer]
        )

        # Create the sample weights
        weights = reweighting.weight_layer()([score_tensor, pred_score])

        # Create the output
        weighted_loss = multiply([loss_tensor, weights])

        # Create the metric layers
        metrics = model.metrics or []
        metrics = [
            MetricLayer(metric)([y_true, model.output])
            for metric in metrics
        ]

        # Finally build, compile, return
        new_model = TransparentModel(
            inputs=[model.input, y_true, pred_score],
            outputs=[weighted_loss],
            observed_tensors=[loss_tensor, weighted_loss, score_tensor] + metrics
        )
        new_model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: y_pred
        )

        return new_model

    def evaluate_batch(self, x, y):
        dummy_weights = np.ones((x.shape[0], self.reweighting.weight_size))
        dummy_target = np.zeros((x.shape[0], 1))
        outputs = self.model.test_on_batch([x, y, dummy_weights], dummy_target)

        signal("is.evaluate_batch").send(outputs)

        return np.hstack([outputs[self.LOSS]] + outputs[self.METRIC0:])

    def score_batch(self, x, y):
        dummy_weights = np.ones((x.shape[0], self.reweighting.weight_size))
        dummy_target = np.zeros((x.shape[0], 1))
        outputs = self.model.test_on_batch([x, y, dummy_weights], dummy_target)

        return outputs[self.SCORE].ravel()

    def train_batch(self, x, y, w):
        # create a dummy target to please keras
        dummy_target = np.zeros((x.shape[0], 1))

        # train on a single batch
        outputs = self.model.train_on_batch([x, y, w], dummy_target)

        # Add the outputs in a tuple to send to whoever is listening
        result = (
            outputs[self.WEIGHTED_LOSS],
            outputs[self.METRIC0:],
            outputs[self.SCORE]
        )
        signal("is.training").send(result)

        return result
