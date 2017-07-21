# How to

The `training` module provides several implementations of `ImportanceTraining`
that can wrap a *Keras* model and train it with importance sampling.

```python
from importance_sampling.training import ImportanceTraining

# assuming model is a Keras model
wrapped_model = ImportanceTraining(model)
wrapped_model = ImportanceTraining(model, k=1.0, smooth=0.5)

wrapped_model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)
```

## Bias

All importance training classes accept a constructor parameter \(k \in (-\infty,
1]\). \(k\) biases the gradient estimator to focus more on hard examples, the
smaller the value the closer to max-loss minimization the algorithm is. By
default `k=0.5` which is found to often improve the generalization performance
of the final model.

## Smoothing

Modern deep networks often have innate sources of randomness (e.g. dropout,
batch normalization) that can result in noisy importance predictions. To
alleviate this noise one can smooth the importance using additive smoothing.

The \(\text{smooth} \in \mathbb{R}\) parameter is added to all importance
predictions before computing the sampling distribution. In addition, all
classes accept the \(\text{adaptive_smoothing}\) parameter which when set to
`True` multiplies \(\text{smooth}\) with \(\bar{L} \approx
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
fit(x, y, batch_size=32, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, steps_per_epoch=None)
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

**Returns**

A *Keras* `History` instance.

### fit\_generator

```
fit_generator(train, steps_per_epoch, batch_size=32, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None)
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

**Returns**

A *Keras* `History` instance.

### fit\_dataset

```
fit_dataset(dataset, steps_per_epoch=None, batch_size=32, epochs=1, verbose=1, callbacks=None)
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

**Returns**

A *Keras* `History` instance.

## ImportanceTraining

```
importance_sampling.training.ImportanceTraining(model, k=0.5, smooth=0.0, adaptive_smoothing=False, presample=1024, forward_batch_size=128)
```

`ImportanceTraining` uses the passed model to compute the importance of the
samples. Initially, it samples uniformly `presample` samples, then it runs a
**forward pass** for all of them to compute the loss (which is used as the
importance in this case) and **resamples according to the importance**.

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
can be faster than `ImportanceTraining` but less effective.

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
