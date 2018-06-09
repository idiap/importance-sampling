# How to

The `training` module provides several implementations of `ImportanceTraining`
that can wrap a *Keras* model and train it with importance sampling.

```python
from importance_sampling.training import ImportanceTraining, BiasedImportanceTraining

# assuming model is a Keras model
wrapped_model = ImportanceTraining(model)
wrapped_model = BiasedImportanceTraining(model, k=1.0, smooth=0.5)

wrapped_model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)
```

## Sampling probabilites and sample weights

All of the `fit` methods accept two extra keyword arguments `on_sample` and
`on_scores`. They are callables that allow the user of the library to have read
access to the sampling probabilities weights and scores from the performed
importance sampling. Their API is the following,

```python
on_sample(sampler, idxs, w, predicted_scores)
```

**Arguments**

* **sampler**: The instance of BaseSampler being currently used
* **idxs**: A numpy array containing the indices that were sampled
* **w**: A numpy array containing the computed sample weights
* **predicted_scores**: A numpy array containing the unnormalized importance
  scores

```python
on_scores(sampler, scores)
```

**Arguments**

* **sampler**: The instance of BaseSampler being currently used
* **scores**: A numpy array containing all the importance scores from the
  presampled data

## Bias

`BiasedImportanceTraining` and `ApproximateImportanceTraining` classes accept a
constructor parameter \(k \in (-\infty, 1]\). \(k\) biases the gradient
estimator to focus more on hard examples, the smaller the value the closer to
max-loss minimization the algorithm is. By default `k=0.5` which is found to
often improve the generalization performance of the final model.

## Smoothing

Modern deep networks often have innate sources of randomness (e.g. dropout,
batch normalization) that can result in noisy importance predictions. To
alleviate this noise one can smooth the importance using additive smoothing.
The proposed `ImportanceTraining` class does not use smoothing and we propose
to replace *Dropout* and *BatchNormalization* with \(L_2\) regularization and
*LayerNormalization*.

The classes that accept smoothing do so in the following way, the
\(\text{smooth} \in \mathbb{R}\) parameter is added to all importance
predictions before computing the sampling distribution. In addition, they
accept the \(\text{adaptive_smoothing}\) parameter which when set to `True`
multiplies \(\text{smooth}\) with \(\bar{L} \approx
\mathbb{E}\left[\frac{1}{\|B\|} \sum_{i \in B} L(x_i, y_i)\right]\) as computed
by the moving average of the mini-batch losses.

Although, smooth is initialized at `smooth=0.0`, if instability is observed
during training, it can be set to small values (e.g. `[0.05, 0.1, 0.5]`) or one
can use adaptive smoothing with a sane default value for smooth being
`smooth=0.5`.

## Methods

The wrapped models aim to expose the same `fit` methods as the original *Keras*
models in order to make their use as simple as possible. The following is a
list of deviations or additions:

* `class_weights`, `sample_weights` are **not** supported
* `fit_generator` accepts a `batch_size` argument
* `fit_generator` is not supported by all `ImportanceTraining` classes
* `fit_dataset` has been added as a method (see [Datasets](datasets.md))

Below, follows the list of methods with their arguments.

### fit

```
fit(x, y, batch_size=32, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, steps_per_epoch=None, on_sample=None, on_scores=None)
```

**Arguments**

* **x**: Numpy array of training data, lists and dictionaries are not supported
* **y**: Numpy array of target data, lists and dictionaries are not supported
* **batch\_size**: The number of samples per gradient update
* **epochs**: Multiplied by `steps_per_epoch` defines the total number of
  parameter updates
* **verbose**: When set `>0` the *Keras* progress callback is added to the list
  of callbacks
* **callbacks**: A list of *Keras* callbacks for logging, changing training
  parameters, monitoring, etc.
* **validation\_split**: A float in `[0, 1)` that defines the percentage of the
  training data to use for evaluation
* **validation\_data**: A tuple of numpy arrays containing data and targets to
  evaluate the network on
* **steps\_per\_epoch**: The number of gradient updates to do in order to
  assume that an epoch has passed
* **on_sample**: A callable that accepts the sampler, idxs, w, scores
* **on_scores**: A callable that accepts the sampler and scores

**Returns**

A *Keras* `History` instance.

### fit\_generator

```
fit_generator(train, steps_per_epoch, batch_size=32, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, on_sample=None, on_scores=None)
```

**Arguments**

* **train**: A generator yielding tuples of (data, targets)
* **steps\_per\_epoch**: The number of gradient updates to do in order to
  assume that an epoch has passed
* **batch\_size**: The number of samples per gradient update (in contrast to
  *Keras* this can be variable)
* **epochs**: Multiplied by `steps_per_epoch` defines the total number of
  parameter updates
* **verbose**: When set `>0` the *Keras* progress callback is added to the list
  of callbacks
* **callbacks**: A list of *Keras* callbacks for logging, changing training
  parameters, monitoring, etc.
* **validation\_data**: A tuple of numpy arrays containing data and targets to
  evaluate the network on or a generator yielding tuples of (data, targets)
* **validation\_steps**: The number of tuples to extract from the validation
  data generator (if a generator is given)
* **on_sample**: A callable that accepts the sampler, idxs, w, scores
* **on_scores**: A callable that accepts the sampler and scores

**Returns**

A *Keras* `History` instance.

### fit\_dataset

```
fit_dataset(dataset, steps_per_epoch=None, batch_size=32, epochs=1, verbose=1, callbacks=None, on_sample=None, on_scores=None)
```

The calls to the other `fit*` methods are delegated to this one after a
`Dataset` instance has been created. See [Datasets]() for details on how to
create a `Dataset` and what datasets are available by default.

**Arguments**

* **dataset**: Instance of the `Dataset` class
* **steps\_per\_epoch**: The number of gradient updates to do in order to
  assume that an epoch has passed (if not given equals the number of training
  samples)
* **batch\_size**: The number of samples per gradient update (in contrast to
  *Keras* this can be variable)
* **epochs**: Multiplied by `steps_per_epoch` defines the total number of
  parameter updates
* **verbose**: When set `>0` the *Keras* progress callback is added to the list
  of callbacks
* **callbacks**: A list of *Keras* callbacks for logging, changing training
  parameters, monitoring, etc.
* **on_sample**: A callable that accepts the sampler, idxs, w, scores
* **on_scores**: A callable that accepts the sampler and scores

**Returns**

A *Keras* `History` instance.

## ImportanceTraining

```
importance_sampling.training.ImportanceTraining(model, presample=3.0, tau_th=None, forward_batch_size=None, score="gnorm", layer=None)
```

`ImportanceTraining` uses the passed model to compute the importance of the
samples. It computes the variance reduction and enables importance sampling
only when the variance will be reduced more than `tau_th`. When importance sampling is enabled, it
samples uniformly `presample*batch_size` samples, then it runs a
**forward pass** for all of them to compute the `score` and **resamples
according to the importance**.

See our [paper](https://arxiv.org/abs/1803.00942) for a precise definition of
the algorithm.

**Arguments**

* **model**: The Keras model to train
* **presample**: The number of samples to presample for scoring, given as a
  factor of the batch size
* **tau\_th**: The variance reduction threshold after which we enable
  importance sampling, when not given it is computed from eq. 29 (it is given
  in units of batch size increment)
* **forward\_batch\_size**: Define the batch size when running the forward pass
  to compute the importance
* **score**: Choose the importance score among \(\{\text{gnorm}, \text{loss},
  \text{full_gnorm}\}\). `gnorm` computes an upper bound to the full gradient norm
  that requires only one forward pass.
* **layer**: Defines which layer will be used to compute the upper bound (if
  not given it is automatically inferred). It can also be given as an index in
  the model's layers property.

## BiasedImportanceTraining

```
importance_sampling.training.BiasedImportanceTraining(model, k=0.5, smooth=0.0, adaptive_smoothing=False, presample=256, forward_batch_size=128)
```

`BiasedImportanceTraining` uses the model and the loss to compute the per
sample importance. `presample` data points are sampled uniformly and after a
forward pass on all of them the importance distribution is calculated and we
resample the mini batch.

See the corresponding [paper](https://arxiv.org/abs/1706.00043) for details.

**Arguments**

* **model**: The Keras model to train
* **k**: Controls the bias of the sampling that focuses the network on the hard
  examples
* **smooth**: Influences the sampling distribution towards uniform by additive
  smoothing
* **adaptive\_smoothing**: When set to `True` multiplies `smooth` with the
  average training loss
* **presample**: Defines the number of samples to compute the importance for
  before creating each batch
* **forward\_batch\_size**: Define the batch size when running the forward pass
  to compute the importance

## ApproximateImportanceTraining

```
importance_sampling.training.ApproximateImportanceTraining(model, k=0.5, smooth=0.0, adaptive_smoothing=False, presample=2048)
```

`ApproximateImportanceTraining` creates a small model that uses the per sample
history of the loss and the class to predict the importance for each sample. It
can be faster than `BiasedImportanceTraining` but less effective.

See the corresponding [paper](https://arxiv.org/abs/1706.00043) for details.

**Arguments**

* **model**: The Keras model to train
* **k**: Controls the bias of the sampling that focuses the network on the hard
  examples
* **smooth**: Influences the sampling distribution towards uniform by additive
  smoothing
* **adaptive\_smoothing**: When set to `True` multiplies `smooth` with the
  average training loss
* **presample**: Defines the number of samples to compute the importance for
  before creating each batch
