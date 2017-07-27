#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from functools import partial

from blinker import signal
from keras.layers import Dense, Embedding, Flatten, Input, LSTM, Masking, \
    concatenate
from keras.models import Model
import numpy as np


def _get_dataset_length(dset, default=1):
    """Return the dataset's training data length and in case the dataset is
    uncountable return a defalt value."""
    try:
        return len(dset.train_data)
    except RuntimeError:
        return default


class BaseSampler(object):
    """BaseSampler denotes the interface for all the samplers.

    Samplers should provide the rest of the program with data points to train
    on and corresponding relative weights."""
    def __init__(self, dataset, reweighting):
        self.dataset = dataset
        self.reweighting = reweighting

    def _get_samples_with_scores(self, batch_size):
        """Child classes should implement this method.

        Arguments
        ---------
        batch_size: int
                    Return at least that many samples
        
        Return
        ------
        idxs: array
              The indices of some samples in the dataset
        scores: array or None
                The predicted importance scores for the corresponding idxs or
                None for uniform sampling
        xy: tuple or None
            Optionally return the data for the corresponding idxs
        """
        raise NotImplementedError()


    def sample(self, batch_size):
        # Get the importance scores of some samples
        idxs1, scores, xy = self._get_samples_with_scores(batch_size)

        # Sample from the available ones
        p = scores / scores.sum() if scores is not None else None
        idxs2 = np.random.choice(len(idxs1), batch_size, p=p)
        w = self.reweighting.sample_weights(idxs2, scores)

        # Make sure we have the data
        if xy is None:
            xy = self.dataset.train_data[idxs1[idxs2]]
        else:
            x, y = xy
            xy = (x[idxs2], y[idxs2])

        scores = scores[idxs2] if scores is not None else np.ones(batch_size)
        signal("is.sample").send({
            "idxs": idxs1,
            "xy": xy,
            "w": w,
            "predicted_scores": scores
        })
        return idxs1[idxs2], xy, w

    def update(self, idxs, results):
        pass


class UniformSampler(BaseSampler):
    """UniformSampler is the simplest possible sampler which samples the
    dataset uniformly."""
    def __init__(self, dataset, reweighting):
        super(UniformSampler, self).__init__(dataset, reweighting)
        # Basically if we don't know the length the indices don't matter so
        # sample batch_size 0s.
        self.idxs = np.arange(_get_dataset_length(self.dataset, default=1))

    def _get_samples_with_scores(self, batch_size):
        return (
            self.idxs,
            None,
            None
        )


class ModelSampler(BaseSampler):
    """ModelSampler uses a model to score the samples and then performs
    importance sampling based on those scores.

    It can be used to implement several training pipelines where the scoring
    model is separately trained or is sampled from the main model or is the
    main model."""
    def __init__(self, dataset, reweighting, model, large_batch=1024,
                 forward_batch_size=128):
        self.model = model
        self.large_batch = large_batch
        self.forward_batch_size = forward_batch_size
        self.N = _get_dataset_length(dataset, default=1)

        super(ModelSampler, self).__init__(dataset, reweighting)

    def _get_samples_with_scores(self, batch_size):
        assert batch_size < self.large_batch

        # Sample a large number of points in random and score them
        idxs = np.random.choice(self.N, self.large_batch)
        x, y = self.dataset.train_data[idxs]
        scores = self.model.score(x, y, batch_size=self.forward_batch_size)

        return (
            idxs,
            scores,
            (x, y)
        )


class LSTMSampler(BaseSampler):
    """Use an LSTM to predict the loss based on the previous losses of each
    sample

    Arguments
    ---------

    dataset: The dataset we want to sample from
    presample: int
               Presample that many data points using uniform sampling to
               decrease the complexity
    history: int
             How many scores per data point to keep in history
    smooth: float
            Depending on whether we are using adaptive smoothing or additive we
            either add smooth*mean or simply smooth to each importance before
            sampling
    warmup: int
            Sample unformly at first that many times to gather some history
    log: bool
         Do the regression in log space
    adaptive_smooth: bool
                     Smooth adaptively based on the mean of the scores
    forget: float
            A float less than one to used to calculate the mean of the scores
    """
    def __init__(self, dataset, reweighting, presample=2048, history=10, warmup=100,
                 log=False):
        # Initialize the history for every sample
        init = -np.log(1.0 / dataset.output_size)
        if log:
            init = np.log(init)
        self.history = np.zeros((len(dataset.train_data), history, 1))
        self.history[:, 0, 0] = init
        self.cnts = np.ones(len(dataset.train_data), dtype=np.uint8)

        # Keep some member variables
        self.presample = presample
        self.log = log
        self.warmup = warmup
        self.batches_sampled = 0

        # Create our LSTM model
        x00 = Input(shape=(history, 1))
        x10 = Input(shape=(1,))
        x0 = Masking(mask_value=0.0)(x00)
        x0 = LSTM(32, return_sequences=True, unroll=True)(x0)
        x0 = LSTM(32, unroll=True)(x0)
        x1 = Embedding(dataset.output_size, 32)(x10)
        x1 = Flatten()(x1)
        x = concatenate([x0, x1])
        y = Dense(1)(x)
        self.model = Model(inputs=[x00, x10], outputs=y)
        self.model.compile(optimizer="adam", loss="mse")

        super(LSTMSampler, self).__init__(dataset, reweighting)

    def _to_ids(self, y):
        try:
            if y.shape[1] > 1:
                return np.expand_dims(y.argmax(axis=1), -1)
        except:
            return y

    def _get_samples_with_scores(self, batch_size):
        """Use the LSTM to predict the loss of each sample"""
        # Increment our batch counter
        self.batches_sampled += 1

        # if we are still warming up just sample randomly
        if self.batches_sampled <= self.warmup:
            return (
                np.random.choice(len(self.dataset.train_data), self.presample),
                np.ones(self.presample),
                None
            )

        # Presample so that we do not run the LSTM for the whole dataset
        idxs = np.random.choice(len(self.history), self.presample)
        x, y = self.dataset.train_data[idxs]

        # Predict normalize and sample
        scores = self.model.predict(
            [self.history[idxs], self._to_ids(y)],
            batch_size=1024
        ).ravel()

        # Perform the regression in logspace if needed
        if self.log:
            np.exp(scores, scores)
        else:
            np.maximum(scores, 0, scores)

        return (
            idxs,
            scores,
            (x, y)
        )

    def update(self, idxs, x):
        # Fetch the classes for the regression
        _, y = self.dataset.train_data[idxs]

        # If we are doing the regression in logspace
        if self.log:
            x = np.log(x)

        # Train the lstm so that it can predict x given the history
        self.model.train_on_batch([self.history[idxs], self._to_ids(y)], x)

        # Update the history to include x
        full = idxs[self.cnts[idxs] == self.history.shape[1]]
        self.history[full] = np.roll(self.history[full], -1, axis=1)
        self.cnts[full] -= 1
        self.history[idxs, self.cnts[idxs], :1] = x
        self.cnts[idxs] += 1


class PerClassGaussian(BaseSampler):
    """Fit a Gaussian per class to predict the losses"""
    def __init__(self, dataset, reweighting, alpha=0.9, presample=2048):
        self.alpha = alpha
        self.presample = presample
        self.mu = np.ones(dataset.output_size) * np.log(dataset.output_size)
        self.variance = np.ones(dataset.output_size)

        super(PerClassGaussian, self).__init__(dataset, reweighting)

    def _get_samples_with_scores(self, batch_size):
        # Presample so that we do not need to compute everything
        # on the whole dataset
        idxs = np.random.choice(len(self.dataset.train_data), self.presample)
        x, y = self.dataset.train_data[idxs]
        yis = y.ravel() if y.shape[1] == 1 else y.argmax(axis=1)

        # Compute the sampling probs for each of the above presampled
        # data points
        scores = self.mu + np.sqrt(np.maximum(self.variance - self.mu**2, 0))
        scores = scores[yis]

        return (
            idxs,
            scores,
            (x, y)
        )

    def update(self, idxs, x):
        # Fetch the classes in order to model per class information
        _, y = self.dataset.train_data[idxs]
        yis = y.ravel() if y.shape[1] == 1 else y.argmax(axis=1)

        # Update the mean and variance one by one
        # TODO: Improve the following implementation
        for xi, yi in zip(x, yis):
            d = (1.0 - self.alpha) * xi
            self.mu[yi] = self.alpha * self.mu[yi] + d
            self.variance[yi] = self.alpha * self.variance[yi] + d * xi


class LSTMComparisonSampler(BaseSampler):
    """Compare LSTM and Model scores on a fixed presampled subset of the
    training data"""
    def __init__(self, dataset, lstm, model, subset=1024):
        self._idxs = np.random.choice(len(dataset.train_data), subset)
        self._x, self._y= dataset.train_data[self._idxs]
        self.lstm = lstm
        self.model = model

    def _get_samples_with_scores(self, batch_size):
        s1 = self.model.model.score(self._x, self._y)
        s2 = self.lstm.model.predict(
            [self.lstm.history[self._idxs], self.lstm._to_ids(self._y)],
            batch_size=1024
        ).ravel()
        signal("is.lstm_comparison_sampler.scores").send(zip(s1, s2))

        return self.lstm._get_samples_with_scores(batch_size)

    def update(self, idxs, x):
        return self.lstm.update(idxs, x)


class SamplerDecorator(BaseSampler):
    """Just decorate another sampler.

    Arguments
    ---------
    sampler: BaseSampler
             The sampler being decorated
    """
    def __init__(self, sampler):
        self.sampler = sampler

        super(SamplerDecorator, self).__init__(
            sampler.dataset,
            sampler.reweighting
        )

    def _get_samples_with_scores(self, batch_size):
        raise NotImplementedError()

    def update(self, idxs, results):
        self.sampler.update(idxs, results)


class AdditiveSmoothingSampler(SamplerDecorator):
    """Add a constant to all the importance scores in order to smooth them
    towards uniform

    Arguments
    ---------
    sampler: BaseSampler
             The sampler being decorated
    c: float
       A constant to add to every importance score
    """
    def __init__(self, sampler, c=1.0):
        self.c = c
        super(AdditiveSmoothingSampler, self).__init__(sampler)

    def _get_samples_with_scores(self, batch_size):
        idxs, scores, xy = self.sampler._get_samples_with_scores(batch_size)

        return (
            idxs,
            scores + self.c,
            xy
        )


class AdaptiveAdditiveSmoothingSampler(SamplerDecorator):
    """Add a percentage of the moving average of the predicted importance
    scores to smooth them towards uniform.

    Arguments
    ---------
    sampler: BaseSampler
             The sampler being decorated
    percentage: float
                Multiplied by the moving average of the importance scores to
                add to each score to smooth it
    forget: float
            Used to compute the exponential moving average mu = forget * mu +
            (1-forget) * mu_new
    """
    def __init__(self, sampler, percentage=0.5, forget=0.9):
        self.percentage = percentage
        self.forget = forget
        self.mu = 1.0  # it could be 0 it doesn't really matter
        super(AdaptiveAdditiveSmoothingSampler, self).__init__(sampler)

    def _get_samples_with_scores(self, batch_size):
        idxs, scores, xy = self.sampler._get_samples_with_scores(batch_size)

        self.mu = self.forget * self.mu + (1 - self.forget) * scores.mean()

        return (
            idxs,
            scores + self.percentage * self.mu,
            xy
        )


class PowerSmoothingSampler(SamplerDecorator):
    """Raise the importance scores to a power (less than 1) to smooth them
    towards uniform.
    
    Arguments
    ---------
    sampler: BaseSampler
             The sampler being decorated
    power: float
           The power to raise the scores to
    """
    def __init__(self, sampler, power=0.5):
        assert 0 <= power <= 1

        self.power = power
        super(PowerSmoothingSampler, self).__init__(sampler)

    def _get_samples_with_scores(self, batch_size):
        idxs, scores, xy = self.sampler._get_samples_with_scores(batch_size)

        return (
            idxs,
            scores**self.power,
            xy
        )


class OnlineBatchSelectionSampler(ModelSampler):
    """OnlineBatchSelection is the online batch creation method by Loschchilov
    & Hutter.

    See 'Online Batch Selection for Faster Training of Neural Networks'.

    Arguments
    ---------
    dataset: The dataset to sample from
    reweighting: The reweighting scheme
    model: The model to be used for scoring
    steps_per_epoch: int
                     How many batches to create before considering that an
                     epoch has passed
    recompute: int
               Recompute the scores after r minibatches seen
    s_e: tuple
         Used to compute the sampling probabilities from the ranking
    n_epochs: int
              The number of epochs, used to compute the sampling probabilities
    """
    def __init__(self, dataset, reweighting, model, large_batch=1024,
                 forward_batch_size=128, steps_per_epoch=300, recompute=2,
                 s_e=(1, 1), n_epochs=1):
        super(OnlineBatchSelectionSampler, self).__init__(
            dataset,
            reweighting,
            model,
            large_batch=large_batch,
            forward_batch_size=forward_batch_size
        )

        # The configuration of OnlineBatchSelection
        self.steps_per_epoch = steps_per_epoch
        self.recompute = recompute
        self.s_e = s_e
        self.n_epochs = n_epochs

        # Mutable variables to be updated
        self._batch = 0
        self._epoch = 0
        self._raw_scores = np.ones((len(dataset.train_data),))
        self._scores = np.ones_like(self._raw_scores)
        self._ranks = np.arange(len(dataset.train_data))

    def _get_samples_with_scores(self, batch_size):
        return (
            np.arange(len(self._ranks)),
            self._scores,
            None
        )

    def update(self, idxs, results):
        # Compute the current epoch and the current batch
        self._batch += 1
        self._epoch = 1 + self._batch / self.steps_per_epoch

        # Add the new scores to the raw_scores
        self._raw_scores[idxs] = results.ravel()

        # if it is a new epoch
        if self._batch % self.steps_per_epoch == 0:
            # For the very first batch or every 'recompute' epochs compute the
            # loss across the whole dataset
            if self.recompute > 0 and self._epoch % self.recompute == 0:
                # Extra internal batch size so that we do not load too much
                # data into memory
                scores = []
                for i in range(0, len(self.dataset.train_data), 1024*64):
                    x, y = self.dataset.train_data[i:i+1024*64]
                    scores.append(self.model.score(
                        x, y, batch_size=self.forward_batch_size
                    ))
                self._raw_scores[:] = np.hstack(scores)

            # Sort and recompute the ranks
            N = len(self.dataset.train_data)
            self._ranks[self._raw_scores.argsort()] = np.arange(N)[::-1]

            # Recompute the sample scores from the ranks
            s_e0, s_eend = self.s_e
            n_epochs = self.n_epochs
            s = s_e0 * np.exp(np.log(s_eend/s_e0)/n_epochs) ** self._epoch
            s = 1.0 / np.exp(np.log(s)/N)
            self._scores = s**self._ranks
