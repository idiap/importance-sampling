#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from functools import partial

from blinker import signal
from keras import backend as K
from keras.layers import Activation, Input, Layer, multiply
from keras.metrics import categorical_accuracy, binary_accuracy, \
    get as get_metric, sparse_categorical_accuracy
import numpy as np
from transparent_keras import TransparentModel

from .reweighting import UNBIASED
from .score_layers import GradientNormLayer, LossLayer
from .utils.functional import compose


def _tolist(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return x


class MetricLayer(Layer):
    """Create a layer that computes a metric taking into account masks"""
    def __init__(self, metric_func, **kwargs):
        self.supports_masking = True
        self.metric_func = metric_func

        super(MetricLayer, self).__init__(**kwargs)

    def compute_mask(self, inputs, input_mask):
        return None

    def build(self, input_shape):
        # Special care for accuracy because keras treats it specially
        try:
            if "accuracy" in self.metric_func:
                self.metric_func = self._generic_accuracy
        except TypeError:
            pass # metric_func is not a string
        self.metric_func = compose(K.expand_dims, get_metric(self.metric_func))

        super(MetricLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # We need two inputs y_true, y_pred
        assert len(input_shape) == 2

        return (input_shape[0][0], 1)

    def call(self, inputs, mask=None):
        # Compute the metric
        metric = self.metric_func(*inputs)
        if K.int_shape(metric)[-1] == 1:
            metric = K.squeeze(metric, axis=-1)

        # Apply the mask if needed
        if mask is not None:
            if not isinstance(mask, list):
                mask = [mask]
            mask = [K.cast(m, K.floatx()) for m in mask if m is not None]
            mask = reduce(lambda a, b: a*b, mask)
            metric *= mask
            metric /= K.mean(mask, axis=-1, keepdims=True)

        # Make sure that the tensor returned is (None, 1)
        dims = len(K.int_shape(metric))
        if dims > 1:
            metric = K.mean(metric, axis=list(range(1, dims)))

        return K.expand_dims(metric)

    @staticmethod
    def _generic_accuracy(y_true, y_pred):
        if K.int_shape(y_pred)[1] == 1:
            return binary_accuracy(y_true, y_pred)
        if K.int_shape(y_true)[-1] == 1:
            return sparse_categorical_accuracy(y_true, y_pred)

        return categorical_accuracy(y_true, y_pred)


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
        skip_one = not bool(model.layers[-1].trainable_weights)
        last_layer = -2 if skip_one else -1
        score_tensor = _get_scoring_layer(
            score,
            y_true,
            model.output,
            loss,
            model.layers[last_layer],
            model
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
            inputs=_tolist(model.input) + [y_true, pred_score],
            outputs=[weighted_loss],
            observed_tensors=[loss_tensor, weighted_loss, score_tensor] + metrics
        )
        new_model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: y_pred
        )

        return new_model

    def evaluate_batch(self, x, y):
        dummy_weights = np.ones((y.shape[0], self.reweighting.weight_size))
        dummy_target = np.zeros((y.shape[0], 1))
        inputs = _tolist(x) + [y, dummy_weights]
        outputs = self.model.test_on_batch(inputs, dummy_target)

        signal("is.evaluate_batch").send(outputs)

        return np.hstack([outputs[self.LOSS]] + outputs[self.METRIC0:])

    def score_batch(self, x, y):
        dummy_weights = np.ones((y.shape[0], self.reweighting.weight_size))
        dummy_target = np.zeros((y.shape[0], 1))
        inputs = _tolist(x) + [y, dummy_weights]
        outputs = self.model.test_on_batch(inputs, dummy_target)

        return outputs[self.SCORE].ravel()

    def train_batch(self, x, y, w):
        # create a dummy target to please keras
        dummy_target = np.zeros((y.shape[0], 1))

        # train on a single batch
        outputs = self.model.train_on_batch(_tolist(x) + [y, w], dummy_target)

        # Add the outputs in a tuple to send to whoever is listening
        result = (
            outputs[self.WEIGHTED_LOSS],
            outputs[self.METRIC0:],
            outputs[self.SCORE]
        )
        signal("is.training").send(result)

        return result
