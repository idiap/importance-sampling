#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import sys

from keras.callbacks import BaseLogger, CallbackList, History, ProgbarLogger
from keras.layers import Dropout, BatchNormalization
import numpy as np
from blinker import signal

from .datasets import InMemoryDataset, GeneratorDataset
from .model_wrappers import OracleWrapper, SVRGWrapper
from .samplers import ConditionalStartSampler, VarianceReductionCondition, \
    AdaptiveAdditiveSmoothingSampler, AdditiveSmoothingSampler, ModelSampler, \
    LSTMSampler, ConstantVarianceSampler, ConstantTimeSampler, SCSGSampler
from .reweighting import BiasedReweightingPolicy, NoReweightingPolicy
from .utils.functional import ___, compose, partial


class _BaseImportanceTraining(object):
    def __init__(self, model, score="gnorm", layer=None):
        """Abstract base class for training a model using importance sampling.

        Arguments
        ---------
            model: The Keras model to train
        """
        # Wrap and transform the model so that it outputs the importance scores
        # and can be used in an importance sampling training scheme
        self._check_model(model)
        self.original_model = model
        self.model = OracleWrapper(
            model,
            self.reweighting,
            score=score,
            layer=layer
        )
        signal("is.sample").connect(self._on_sample)
        signal("is.score").connect(self._on_scores)

    def _on_sample(self, event):
        self._latest_sample_event = event

    def _on_scores(self, scores):
        self._latest_scores = scores

    def _check_model(self, model):
        """Check if the model uses Dropout and BatchNorm and warn that it may
        affect the importance calculations."""
        if any(
            isinstance(l, (BatchNormalization, Dropout))
            for l in model.layers
        ):
            sys.stderr.write(("[NOTICE]: You are using BatchNormalization "
                              "and/or Dropout.\nThose layers may affect the "
                              "importance calculations and you are advised "
                              "to exchange them for LayerNormalization or "
                              "BatchNormalization in test mode and L2 "
                              "regularization.\n"))

    @property
    def reweighting(self):
        """The reweighting policy that controls the bias of the estimation when
        using importance sampling."""
        raise NotImplementedError()

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        """Create a new sampler to sample from the given dataset using
        importance sampling."""
        raise NotImplementedError()

    def fit(self, x, y, batch_size=32, epochs=1, verbose=1, callbacks=None,
            validation_split=0.0, validation_data=None, steps_per_epoch=None,
            on_sample=None, on_scores=None):
        """Create an `InMemoryDataset` instance with the given data and train
        the model using importance sampling for a given number of epochs.

        Arguments
        ---------
            x: Numpy array of training data or list of numpy arrays
            y: Numpy array of target data
            batch_size: int, number of samples per gradient update
            epochs: int, number of times to iterate over the entire
                    training set
            verbose: {0, >0}, whether to employ the progress bar Keras
                     callback or not
            callbacks: list of Keras callbacks to be called during training
            validation_split: float in [0, 1), percentage of data to use for
                              evaluation
            validation_data: tuple of numpy arrays, Data to evaluate the
                             trained model on without ever training on them
            steps_per_epoch: int or None, number of gradient updates before
                             considering an epoch has passed
            on_sample: callable that accepts the sampler, idxs, w, scores
            on_scores: callable that accepts the sampler and scores
        Returns
        -------
            A Keras `History` object that contains information collected during
            training.
        """
        # Create two data tuples from the given x, y, validation_*
        if validation_data is not None:
            x_train, y_train = x, y
            x_test, y_test = validation_data

        elif validation_split > 0:
            assert validation_split < 1, "100% of the data used for testing"
            n = int(round(validation_split * len(x)))
            idxs = np.arange(len(x))
            np.random.shuffle(idxs)
            x_train, y_train = x[idxs[n:]], y[idxs[n:]]
            x_test, y_test = x[idxs[:n]], y[idxs[:n]]

        else:
            x_train, y_train = x, y
            x_test, y_test = np.empty(shape=(0, 1)), np.empty(shape=(0, 1))

        # Make the dataset to train on
        dataset = InMemoryDataset(
            x_train,
            y_train,
            x_test,
            y_test,
            categorical=False  # this means use the targets as is
        )

        return self.fit_dataset(
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=verbose,
            callbacks=callbacks,
            on_sample=on_sample,
            on_scores=on_scores
        )

    def fit_generator(self, train, steps_per_epoch, batch_size=32,
                      epochs=1, verbose=1, callbacks=None,
                      validation_data=None, validation_steps=None,
                      on_sample=None, on_scores=None):
        """Create a GeneratorDataset instance and train the model using
        importance sampling for a given number of epochs.

        NOTICE: This method may not be supported by all importance training
        classes and may result in NotImplementedError()

        Arguments
        ---------
            train: A generator that returns tuples (inputs, targets)
            steps_per_epoch: int, number of gradient updates before considering
                             an epoch has passed
            batch_size: int, the number of samples per gradient update (ideally
                             set to the number of items returned by the
                             generator at each call)
            epochs: int, multiplied by steps_per_epoch denotes the total number
                    of gradient updates
            verbose: {0, >0}, whether to use the progress bar Keras callback
            validation_data: generator or tuple (inputs, targets)
            validation_steps: None or int, used only if validation_data is a
                              generator
            on_sample: callable that accepts the sampler, idxs, w, scores
            on_scores: callable that accepts the sampler and scores
        """
        # Create the validation data to pass to the GeneratorDataset
        if validation_data is not None:
            if isinstance(validation_data, (tuple, list)):
                test = validation_data
                test_len = None
            else:
                test = validation_data
                test_len = validation_steps * batch_size
        else:
            test = (np.empty(shape=(0, 1)), np.empty(shape=(0, 1)))
            test_len = None

        dataset = GeneratorDataset(train, test, test_len)

        return self.fit_dataset(
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=verbose,
            callbacks=callbacks,
            on_sample=on_sample,
            on_scores=on_scores
        )

    def fit_dataset(self, dataset, steps_per_epoch=None, batch_size=32,
                    epochs=1, verbose=1, callbacks=None, on_sample=None,
                    on_scores=None):
        """Train the model on the given dataset for a given number of epochs.

        Arguments
        ---------
            dataset: Instance of `BaseDataset` that provides the data
                     to train on.
            steps_per_epoch: int or None, number of gradient updates before
                             considering an epoch has passed. If None it is set
                             to be `len(dataset.train_data) / batch_size`.
            batch_size: int, number of samples per gradient update
            epochs: int, number of times to iterate `steps_per_epoch` times
            verbose: {0, >0}, whether to employ the progress bar Keras
                     callback or not
            callbacks: list of Keras callbacks to be called during training
            on_sample: callable that accepts the sampler, idxs, w, scores
            on_scores: callable that accepts the sampler and scores
        """
        try:
            if len(dataset.train_data) < batch_size:
                raise ValueError(("The model cannot be trained with "
                                  "batch_size > training set"))
        except RuntimeError as e:
            assert "no size" in str(e)

        # Set steps_per_epoch properly
        if steps_per_epoch is None:
            steps_per_epoch = len(dataset.train_data) // batch_size

        # Create the callbacks list
        self.history = History()
        callbacks = [BaseLogger()] + (callbacks or []) + [self.history]
        if verbose > 0:
            callbacks += [ProgbarLogger(count_mode="steps")]
        callbacks = CallbackList(callbacks)
        #TODO: Should we be making it possible to call back a different model
        #      than self.model.model?
        callbacks.set_model(self.model.model)
        callbacks.set_params({
            "epochs": epochs,
            "steps": steps_per_epoch,
            "verbose": verbose,
            "do_validation": len(dataset.test_data) > 0,
            "metrics": self._get_metric_names() + [
                "val_" + n for n in self._get_metric_names()
            ]
        })

        # Create the sampler
        sampler = self.sampler(dataset, batch_size, steps_per_epoch, epochs)

        # Start the training loop
        epoch = 0
        self.model.model.stop_training = False
        callbacks.on_train_begin()
        while epoch < epochs:
            callbacks.on_epoch_begin(epoch)
            for step in range(steps_per_epoch):
                batch_logs = {"batch": step, "size": batch_size}
                callbacks.on_batch_begin(step, batch_logs)

                # Importance sampling is done here
                idxs, (x, y), w = sampler.sample(batch_size)
                # Train on the sampled data
                loss, metrics, scores = self.model.train_batch(x, y, w)
                # Update the sampler
                sampler.update(idxs, scores)

                values = map(lambda x: x.mean(), [loss] + metrics)
                for l, o in zip(self._get_metric_names(), values):
                    batch_logs[l] = o
                callbacks.on_batch_end(step, batch_logs)

                if on_scores is not None:
                    on_scores(
                        sampler,
                        self._latest_scores
                    )

                if on_sample is not None:
                    on_sample(
                        sampler,
                        self._latest_sample_event["idxs"],
                        self._latest_sample_event["w"],
                        self._latest_sample_event["predicted_scores"]
                    )

                if self.model.model.stop_training:
                    break

            # Evaluate now that an epoch passed
            epoch_logs = {}
            if len(dataset.test_data) > 0:
                val = self.model.evaluate(
                    *dataset.test_data[:],
                    batch_size=batch_size
                )
                epoch_logs = {
                    "val_" + l: o
                    for l, o in zip(self._get_metric_names(), val)
                }
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.model.model.stop_training:
                break
            epoch += 1
        callbacks.on_train_end()

        return self.history

    def _get_metric_names(self):
        metrics = self.original_model.metrics or []
        return (
            ["loss"] +
            list(map(str, metrics)) +
            ["score"]
        )


class _UnbiasedImportanceTraining(_BaseImportanceTraining):
    def __init__(self, model, score="gnorm", layer=None):
        self._reweighting = BiasedReweightingPolicy(1.0)  # no bias

        super(_UnbiasedImportanceTraining, self).__init__(model, score, layer)

    @property
    def reweighting(self):
        return self._reweighting


class ImportanceTraining(_UnbiasedImportanceTraining):
    """Train a model with exact importance sampling.

    Arguments
    ---------
        model: The Keras model to train
        presample: float, the number of samples to presample for scoring
                   given as a factor of the batch size
        tau_th: float or None, the variance reduction threshold after which we
                enable importance sampling
        forward_batch_size: int or None, the batch size to use for the forward
                            pass during scoring
        score: {"gnorm", "loss", "full_gnorm"}, the importance metric to use
               for importance sampling
        layer: None or int or Layer, the layer to compute the gnorm with
    """
    def __init__(self, model, presample=3.0, tau_th=None,
                 forward_batch_size=None, score="gnorm", layer=None):
        self._presample = presample
        self._tau_th = tau_th
        self._forward_batch_size = forward_batch_size

        # Call the parent to wrap the model
        super(ImportanceTraining, self).__init__(model, score, layer)

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        # Configure the sampler
        large_batch = int(self._presample * batch_size)
        forward_batch_size = self._forward_batch_size or batch_size
        # Compute the threshold using eq. 29 in
        # https://arxiv.org/abs/1803.00942
        B = large_batch
        b = batch_size
        tau_th = float(B + 3*b) / (3*b)
        tau_th = self._tau_th if self._tau_th is not None else tau_th

        return ConditionalStartSampler(
            ModelSampler(
                dataset,
                self.reweighting,
                self.model,
                large_batch=large_batch,
                forward_batch_size=forward_batch_size
            ),
            VarianceReductionCondition(tau_th)
        )


class ConstantVarianceImportanceTraining(_UnbiasedImportanceTraining):
    """Train a model faster by keeping the per iteration variance constant but
    decreasing the time.

    Arguments
    ---------
        model: The Keras model to train
        score: {"gnorm", "loss", "full_gnorm"}, the importance metric to use
               for importance sampling
        layer: None or int or Layer, the layer to compute the gnorm with
    """
    def __init__(self, model, backward_time=2.0, extra_samples=0.2,
                 score="gnorm", layer=None):
        self._backward_time = backward_time
        self._extra_samples = extra_samples

        super(ConstantVarianceImportanceTraining, self).__init__(
            model,
            score,
            layer
        )

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        return ConstantVarianceSampler(
            dataset,
            self.reweighting,
            self.model,
            backward_time=self._backward_time,
            extra_samples=self._extra_samples
        )


class ConstantTimeImportanceTraining(_UnbiasedImportanceTraining):
    """Train a model faster by keeping the per iteration time constant but
    improving the quality of the gradients.

    Arguments
    ---------
        model: The Keras model to train
        score: {"gnorm", "loss", "full_gnorm"}, the importance metric to use
               for importance sampling
        layer: None or int or Layer, the layer to compute the gnorm with
    """
    def __init__(self, model, backward_time=2.0, score="gnorm", layer=None):
        self._backward_time = backward_time

        super(ConstantTimeImportanceTraining, self).__init__(
            model,
            score,
            layer
        )

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        return ConstantTimeSampler(
            dataset,
            self.reweighting,
            self.model,
            backward_time=self._backward_time
        )


class BiasedImportanceTraining(_BaseImportanceTraining):
    """Train a model with exact importance sampling using the loss as
    importance.

    Arguments
    ---------
        model: The Keras model to train
        k: float in (-oo, 1], controls the bias of the training that focuses
           the network on the hard examples (see paper)
        smooth: float, influences the sampling distribution towards uniform by
                adding `smooth` to all probabilities and renormalizing
        adaptive_smoothing: bool, If True the `smooth` argument is a percentage
                            of the average training loss
        presample: int, the number of samples to presample for scoring
        forward_batch_size: int, the batch size to use for the forward pass
                            during scoring
    """
    def __init__(self, model, k=0.5, smooth=0.0, adaptive_smoothing=False,
                 presample=256, forward_batch_size=128):
        # Create the reweighting policy
        self._reweighting = BiasedReweightingPolicy(k)

        # Call the parent to wrap the model
        super(BiasedImportanceTraining, self).__init__(model)

        # Create the sampler factory, the workhorse of the whole deal :-)
        adaptive_smoothing_factory = partial(
            AdaptiveAdditiveSmoothingSampler, ___, smooth
        )
        additive_smoothing_factory = partial(
            AdditiveSmoothingSampler, ___, smooth
        )
        self._sampler = partial(
            ModelSampler,
            ___,
            self.reweighting,
            self.model,
            large_batch=presample,
            forward_batch_size=forward_batch_size
        )
        if adaptive_smoothing and smooth > 0:
            self._sampler = compose(adaptive_smoothing_factory, self._sampler)
        elif smooth > 0:
            self._sampler = compose(additive_smoothing_factory, self._sampler)

    @property
    def reweighting(self):
        return self._reweighting

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        return self._sampler(dataset)


class ApproximateImportanceTraining(_BaseImportanceTraining):
    """Train a model with importance sampling using an LSTM with a class
    embedding to predict the importance of the training samples.

    Arguments
    ---------
        model: The Keras model to train
        k: float in (-oo, 1], controls the bias of the training that focuses
           the network on the hard examples (see paper)
        smooth: float, influences the sampling distribution towards uniform by
                adding `smooth` to all probabilities and renormalizing
        adaptive_smoothing: bool, If True the `smooth` argument is a percentage
                            of the average training loss
        presample: int, the number of samples to presample for scoring
    """
    def __init__(self, model, k=0.5, smooth=0.0, adaptive_smoothing=False,
                 presample=2048):
        # Create the reweighting policy
        self._reweighting = BiasedReweightingPolicy(k)

        # Call the parent to wrap the model
        super(ApproximateImportanceTraining, self).__init__(model)

        # Create the sampler factory, the workhorse of the whole deal :-)
        adaptive_smoothing_factory = partial(
            AdaptiveAdditiveSmoothingSampler, ___, smooth
        )
        additive_smoothing_factory = partial(
            AdditiveSmoothingSampler, ___, smooth
        )
        self._sampler = partial(
            LSTMSampler,
            ___,
            self.reweighting,
            presample=presample
        )
        if adaptive_smoothing and smooth > 0:
            self._sampler = compose(adaptive_smoothing_factory, self._sampler)
        elif smooth > 0:
            self._sampler = compose(additive_smoothing_factory, self._sampler)

    @property
    def reweighting(self):
        return self._reweighting

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        return self._sampler(dataset)

    def fit_generator(*args, **kwargs):
        raise NotImplementedError("ApproximateImportanceTraining doesn't "
                                  "support generator training")


class SVRG(_BaseImportanceTraining):
    """Train a model with Stochastic Variance Reduced Gradient descent.

    See [1, 2] for what this trainer implements.

    [1] Johnson, R. and Zhang, T., 2013. Accelerating stochastic gradient
        descent using predictive variance reduction. In Advances in neural
        information processing systems (pp. 315-323).
    [2] Lei, L. and Jordan, M., 2017, April. Less than a Single Pass:
        Stochastically Controlled Stochastic Gradient. In Artificial
        Intelligence and Statistics (pp. 148-156).

    Arguments
    ---------
        model: The Keras model to train
        B: float, the number of samples to use for estimating the average
           gradient (whole dataset used in case of 0). Given as a factor of the
           batch_size.
        B_rate: float, multiply B with B_rate after every update
        B_over_b: int, How many updates to perform before recalculating the
                  average gradient
    """
    def __init__(self, model, B=10., B_rate=1.0, B_over_b=128):
        self._B = B
        self._B_rate = B_rate
        self._B_over_b = B_over_b

        self.original_model = model
        self.model = SVRGWrapper(model)

    @property
    def reweighting(self):
        """SVRG does not need sample weights so it returns the
        NoReweightingPolicy()"""
        return NoReweightingPolicy()

    def sampler(self, dataset, batch_size, steps_per_epoch, epochs):
        """Create the SCSG sampler"""
        B = int(self._B * batch_size)
        
        return SCSGSampler(
            dataset,
            self.reweighting,
            self.model,
            B,
            self._B_over_b,
            self._B_rate
        )
