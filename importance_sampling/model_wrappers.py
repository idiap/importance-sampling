#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from functools import partial
import sys

from blinker import signal
from keras import backend as K
from keras.layers import Input, Layer, multiply
from keras.models import Model, clone_model
import numpy as np

from .layers import GradientNormLayer, LossLayer, MetricLayer
from .reweighting import UNBIASED
from .utils.functional import compose


def _tolist(x, acceptable_iterables=(list, tuple)):
    if not isinstance(x, acceptable_iterables):
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

    FUSED_ACTIVATION_WARNING = ("[WARNING]: The last layer has a fused "
                                "activation i.e. Dense(..., "
                                "activation=\"sigmoid\").\nIn order for the "
                                "preactivation to be automatically extracted "
                                "use a separate activation layer (see "
                                "examples).\n")

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

            # If the last layer has trainable weights that means that we cannot
            # automatically extract the preactivation tensor so we have to warn
            # them because they might be missing out or they might not even
            # have noticed
            if last_layer == -1:
                config = model.layers[-1].get_config()
                if config.get("activation", "linear") != "linear":
                    sys.stderr.write(self.FUSED_ACTIVATION_WARNING)

            return model.layers[last_layer]
        except:
            # In case of an error then probably we are not using the gnorm
            # importance
            return None

    def _augment_model(self, model, score, reweighting):
        # Extract some info from the model
        loss = model.loss
        optimizer = model.optimizer.__class__(**model.optimizer.get_config())
        output_shape = model.get_output_shape_at(0)[1:]
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
        loss_tensor = LossLayer(loss)([y_true, model.get_output_at(0)])
        score_tensor = _get_scoring_layer(
            score,
            y_true,
            model.get_output_at(0),
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
            MetricLayer(metric)([y_true, model.get_output_at(0)])
            for metric in metrics
        ]

        # Create a model for plotting and providing access to things such as
        # trainable_weights etc.
        new_model = Model(
            inputs=_tolist(model.get_input_at(0)) + [y_true, pred_score],
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
        inputs = _tolist(model.get_input_at(0)) + [y_true, pred_score] + \
            learning_phase
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


class SVRGWrapper(ModelWrapper):
    """Train using SVRG."""
    def __init__(self, model):
        self._augment(model)

    def _augment(self, model):
        # TODO: There is a lot of overlap with the OracleWrapper, merge some
        #       functionality into a separate function or a parent class

        # Extract info from the model
        loss_function = model.loss
        output_shape = model.get_output_shape_at(0)[1:]

        # Create two identical models one with the current weights and one with
        # the snapshot of the weights
        self.model = model
        self._snapshot = clone_model(model)

        # Create the target variable and compute the losses and the metrics
        inputs = [
            Input(shape=K.int_shape(x)[1:])
            for x in _tolist(model.get_input_at(0))
        ]
        model_output = self.model(inputs)
        snapshot_output = self._snapshot(inputs)
        y_true = Input(shape=output_shape)
        loss = LossLayer(loss_function)([y_true, model_output])
        loss_snapshot = LossLayer(loss_function)([y_true, snapshot_output])
        metrics = self.model.metrics or []
        metrics = [
            MetricLayer(metric)([y_true, model_output])
            for metric in metrics
        ]

        # Make a set of variables that will be holding the batch gradient of
        # the snapshot
        self._batch_grad = [
            K.zeros(K.int_shape(p))
            for p in self.model.trainable_weights
        ]

        # Create an optimizer that computes the variance reduced gradients and
        # get the updates
        loss_mean = K.mean(loss)
        loss_snapshot_mean = K.mean(loss_snapshot)
        optimizer, updates = self._get_updates(
            loss_mean,
            loss_snapshot_mean,
            self._batch_grad
        )

        # Create the training function and gradient computation function
        metrics_updates = []
        if hasattr(self.model, "metrics_updates"):
            metrics_updates = self.model.metrics_updates
        learning_phase = []
        if loss._uses_learning_phase:
            learning_phase.append(K.learning_phase())
        inputs = inputs + [y_true] + learning_phase
        outputs = [loss_mean, loss] + metrics

        train_on_batch = K.function(
            inputs=inputs,
            outputs=outputs,
            updates=updates + self.model.updates + metrics_updates
        )
        evaluate_on_batch = K.function(
            inputs=inputs,
            outputs=outputs,
            updates=self.model.state_updates + metrics_updates
        )
        get_grad = K.function(
            inputs=inputs,
            outputs=K.gradients(loss_mean, self.model.trainable_weights),
            updates=self.model.updates
        )

        self.optimizer = optimizer
        self._train_on_batch = train_on_batch
        self._evaluate_on_batch = evaluate_on_batch
        self._get_grad = get_grad

    def _get_updates(self, loss, loss_snapshot, batch_grad):
        model = self.model
        snapshot = self._snapshot
        class Optimizer(self.model.optimizer.__class__):
            def get_gradients(self, *args):
                grad = K.gradients(loss, model.trainable_weights)
                grad_snapshot = K.gradients(
                    loss_snapshot,
                    snapshot.trainable_weights
                )

                return [
                    g - gs + bg
                    for g, gs, bg in zip(grad, grad_snapshot, batch_grad)
                ]

        optimizer = Optimizer(**self.model.optimizer.get_config())
        return optimizer, \
            optimizer.get_updates(loss, self.model.trainable_weights)

    def evaluate_batch(self, x, y):
        outputs = self._evaluate_on_batch(_tolist(x) + [y, 0])
        signal("is.evaluate_batch").send(outputs)

        return np.hstack(outputs[1:])

    def score_batch(self, x, y):
        raise NotImplementedError()

    def train_batch(self, x, y, w):
        outputs = self._train_on_batch(_tolist(x) + [y, 1])

        result = (
            outputs[0],   # mean loss
            outputs[2:],  # metrics
            outputs[1]    # loss per sample
        )
        signal("is.training").send(result)

        return result

    def update_grad(self, sample_generator):
        sample_generator = iter(sample_generator)
        x, y = next(sample_generator)
        N = len(y)
        gradient_sum = self._get_grad(_tolist(x) + [y, 1])
        for g_sum in gradient_sum:
            g_sum *= N
        for x, y in sample_generator:
            grads = self._get_grad(_tolist(x) + [y, 1])
            n = len(y)
            for g_sum, g in zip(gradient_sum, grads):
                g_sum += g*n
            N += len(y)
        for g_sum in gradient_sum:
            g_sum /= N

        K.batch_set_value(zip(self._batch_grad, gradient_sum))
        self._snapshot.set_weights(self.model.get_weights())


class KatyushaWrapper(SVRGWrapper):
    """Implement Katyusha training on top of plain SVRG."""
    def __init__(self, model, t1=0.5, t2=0.5):
        self.t1 = K.variable(t1, name="tau1")
        self.t2 = K.variable(t2, name="tau2")

        super(KatyushaWrapper, self).__init__(model)

    def _get_updates(self, loss, loss_snapshot, batch_grad):
        optimizer = self.model.optimizer
        t1, t2 = self.t1, self.t2
        lr = optimizer.lr

        # create copies and local copies of the parameters
        shapes = [K.int_shape(p) for p in self.model.trainable_weights]
        x_tilde = [p for p in self._snapshot.trainable_weights]
        z = [K.variable(p) for p in self.model.trainable_weights]
        y = [K.variable(p) for p in self.model.trainable_weights]

        # Get the gradients
        grad = K.gradients(loss, self.model.trainable_weights)
        grad_snapshot = K.gradients(
            loss_snapshot,
            self._snapshot.trainable_weights
        )

        # Collect the updates
        p_plus = [
            t1*zi + t2*x_tildei + (1-t1-t2)*yi
            for zi, x_tildei, yi in
            zip(z, x_tilde, y)
        ]
        vr_grad = [
            gi + bg - gsi
            for gi, bg, gsi in zip(grad, grad_snapshot, batch_grad)
        ]
        updates = [
            K.update(yi, xi - lr * gi)
            for yi, xi, gi in zip(y, p_plus, vr_grad)
        ] + [
            K.update(zi,  zi - lr * gi / t1)
            for zi, xi, gi in zip(z, p_plus, vr_grad)
        ] + [
            K.update(p, xi)
            for p, xi in zip(self.model.trainable_weights, p_plus)
        ]

        return optimizer, updates
