#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from functools import partial

from blinker import signal
from keras import backend as K
from keras.layers import Input, Layer, multiply
from keras.models import Model
import numpy as np

from .layers import GradientNormLayer, LossLayer, MetricLayer
from .reweighting import UNBIASED
from .utils.functional import compose

def _tolist(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return x


def _get_scoring_layer(score, y_true, y_pred, loss="categorical_crossentropy",
                       layer=None, model=None):
    """Get a scoring layer that computes the score for each pair of y_true,
    y_pred"""
    assert score in ["loss", "gnorm", "full_gnorm", "acc"]

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
    elif score == "full_gnorm":
        return GradientNormLayer(
            model.trainable_weights,
            loss,
            fast=False
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
    def _iterate_batches(self, x, y, batch_size):
        bs = batch_size
        for s in range(0, len(y), bs):
            yield [xi[s:s+bs] for xi in _tolist(x)], y[s:s+bs]

    def evaluate(self, x, y, batch_size=128):
        result = np.mean(
            np.vstack([
                self.evaluate_batch(xi, yi)
                for xi, yi in self._iterate_batches(x, y, batch_size)
            ]),
            axis=0
        )

        signal("is.evaluation").send(result)
        return result

    def score(self, x, y, batch_size=128):
        bs = batch_size
        result = np.hstack([
            self.score_batch(xi, yi).T
            for xi, yi in self._iterate_batches(x, y, batch_size)
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
                self.optimizer.lr,
                lr
            )
        except AttributeError:
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

    def __init__(self, model, reweighting, score="loss", layer=None):
        self.reweighting = reweighting
        self.layer = self._gnorm_layer(model, layer)

        # Augment the model with reweighting, scoring etc
        # Save the new model and the training functions in member variables
        self._augment_model(model, score, reweighting)

    def _gnorm_layer(self, model, layer):
        # If we were given a layer then use it directly
        if isinstance(layer, Layer):
            return layer

        # If we were given a layer index extract the layer
        if isinstance(layer, int):
            return model.layers[layer]

        try:
            # Get the last or the previous to last layer depending on wether
            # the last has trainable weights
            skip_one = not bool(model.layers[-1].trainable_weights)
            last_layer = -2 if skip_one else -1

            return model.layers[last_layer]
        except:
            # In case of an error then probably we are not using the gnorm
            # importance
            return None

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
        score_tensor = _get_scoring_layer(
            score,
            y_true,
            model.output,
            loss,
            self.layer,
            model
        )

        # Create the sample weights
        weights = reweighting.weight_layer()([score_tensor, pred_score])

        # Create the output
        weighted_loss = weighted_loss_model = multiply([loss_tensor, weights])
        for l in model.losses:
            weighted_loss += l
        weighted_loss_mean = K.mean(weighted_loss)

        # Create the metric layers
        metrics = model.metrics or []
        metrics = [
            MetricLayer(metric)([y_true, model.output])
            for metric in metrics
        ]

        # Create a model for plotting and providing access to things such as
        # trainable_weights etc.
        new_model = Model(
            inputs=_tolist(model.input) + [y_true, pred_score],
            outputs=[weighted_loss_model]
        )

        # Build separate on_batch keras functions for scoring and training
        updates = optimizer.get_updates(
            weighted_loss_mean,
            new_model.trainable_weights
        )
        metrics_updates = []
        if hasattr(model, "metrics_updates"):
            metrics_updates = model.metrics_updates
        learning_phase = []
        if weighted_loss_model._uses_learning_phase:
            learning_phase.append(K.learning_phase())
        inputs = _tolist(model.input) + [y_true, pred_score] + learning_phase
        outputs = [
            weighted_loss_mean,
            loss_tensor,
            weighted_loss,
            score_tensor
        ] + metrics

        train_on_batch = K.function(
            inputs=inputs,
            outputs=outputs,
            updates=updates + model.updates + metrics_updates
        )
        evaluate_on_batch = K.function(
            inputs=inputs,
            outputs=outputs,
            updates=model.state_updates + metrics_updates
        )

        self.model = new_model
        self.optimizer = optimizer
        self.model.optimizer = optimizer
        self._train_on_batch = train_on_batch
        self._evaluate_on_batch = evaluate_on_batch

    def evaluate_batch(self, x, y):
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
        dummy_weights = np.ones((y.shape[0], self.reweighting.weight_size))
        inputs = _tolist(x) + [y, dummy_weights] + [0]
        outputs = self._evaluate_on_batch(inputs)

        signal("is.evaluate_batch").send(outputs)

        return np.hstack([outputs[self.LOSS]] + outputs[self.METRIC0:])

    def score_batch(self, x, y):
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
        dummy_weights = np.ones((y.shape[0], self.reweighting.weight_size))
        inputs = _tolist(x) + [y, dummy_weights] + [0]
        outputs = self._evaluate_on_batch(inputs)

        return outputs[self.SCORE].ravel()

    def train_batch(self, x, y, w):
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)

        # train on a single batch
        outputs = self._train_on_batch(_tolist(x) + [y, w, 1])

        # Add the outputs in a tuple to send to whoever is listening
        result = (
            outputs[self.WEIGHTED_LOSS],
            outputs[self.METRIC0:],
            outputs[self.SCORE]
        )
        signal("is.training").send(result)

        return result
