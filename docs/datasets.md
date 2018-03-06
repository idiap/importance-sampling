# Datasets

Importance sampling entails the process of accessing random samples from a
dataset in a specific manner. To achieve this we introduce an interface for a
random access `Dataset` in `importance_sampling.datasets`.

Initially, we will present the Dataset interface and subsequently several
implementations both reusable for your own datasets and wrapping some well
known benchmark datasets.

## BaseDataset

`BaseDataset` provides the interface that the rest of the components rely on to
perform importance sampling. The main idea is that it provides two properties
`train_data` and `test_data` that return a proxy object that can be accessed as
a random access container returning tuples of (data, targets). The concept will
be better illustrated by the following code example.

```python
# Let's assume dataset is an instance of BaseDataset

x, y = dataset.train_data[0]           # Just the first sample
x, y = dataset.train_data[::5]         # Every fifth sample
x, y = dataset.train_data[[0, 33, 1]]  # 1st, 33rd and 2nd samples
N = len(dataset.train_data)            # How many samples are there?
```

To implement this behaviour, subclasses of `BaseDataset` have to extend 4
functions, `_train_data`, `_train_size`, `_test_data` and `_test_size`, see
`importance_sampling.datasets.InMemoryDataset` for a simple implementation of
this API. The complete API of BaseDataset is given below:

```python
class BaseDataset(object):
    def _train_data(self, idxs=slice(None)):
        raise NotImplementedError()

    def _train_size(self):
        raise NotImplementedError()

    def _test_data(self, idxs=slice(None)):
        raise NotImplementedError()

    def _test_size(self):
        raise NotImplementedError()

    @property
    def shape(self):
        """Return the shape of the data without the batch axis"""
        raise NotImplementedError()

    @property
    def output_size(self):
        """Return the dimensions of the targets"""
        raise NotImplementedError()
```

## InMemoryDataset

```python
importance_sampling.dataset.InMemoryDataset(X_train, y_train, X_test, y_test, categorical=True)
```

`InMemoryDataset` simply wraps 4 Numpy arrays and implements the interface. If
`categorical` is True `y_*` are transformed to one-hot dense vectors to
indicate the class.

**Arguments**

* **X_train:** At least 2-dimensional Numpy array that contains the training
  data
* **y_train:** Numpy array that contains the training targets
* **X_test:** At least 2-dimensional Numpy array that contains the
  testing/validation data
* **y_test:** Numpy array that contains the testing/validation targets
* **categorical:** Controls whether the targets will be transformed into
  one-hot dense vectors

**Static methods**

```python
from_loadable(dataset)
```

Creates a dataset from an object that returns the four Numpy arrays with a
`load_data()` method. Used, for instance, with the *Keras* datasets.

**Example**

```python
from importance_sampling.datasets import InMemoryDataset
import numpy as np

dset = InMemoryDataset(
    np.random.rand(100, 10),
    np.random.rand(100, 1),
    np.random.rand(100, 10),
    np.random.rand(100, 1),
    categorical=False
)

assert dset.shape == (10,)
assert dset.output_size == 1
assert len(dset.train_data) == 100
```


## InMemoryImageDataset

```python
importance_sampling.dataset.InMemoryImageDataset(X_train, y_train, X_test, y_test)
```

`InMemoryImageDataset` asserts that the passed arrays are 4-dimensional and
normalizes them as `float32` in the range `[0, 1]`.

**Arguments**

* **X_train:** At least 2-dimensional Numpy array that contains the training
  data
* **y_train:** Numpy array that contains the training targets
* **X_test:** At least 2-dimensional Numpy array that contains the
  testing/validation data
* **y_test:** Numpy array that contains the testing/validation targets

**Static methods**

```python
from_loadable(dataset)
```

Creates a dataset from an object that returns the four Numpy arrays with a
`load_data()` method. Used, for instance, with the *Keras* datasets.

**Example**

```python
from keras.datasets import mnist
from importance_sampling.datasets import InMemoryImageDataset

dset = InMemoryImageDataset.from_loadable(mnist)

assert dset.shape == (28, 28, 1)
assert dset.output_size == 10
assert len(dset.train_data) == 60000
```

## OntheflyAugmentedImages

```python
importance_sampling.dataset.OntheflyAugmentedImages(dataset, augmentation_params, N=None, random_state=0, cache_size=None)
```

`OntheflyAugmentedImages` uses *Keras* `ImageDataGenerator` to augment an image
dataset deterministically producing `N` images without explicitly storing them
in memory.

**Arguments**

* **dataset:** Another instance of `BaseDataset` that this class will decorate
* **augmentation\_params:** A dictionary of keyword arguments to pass to
  `ImageDataGenerator`
* **N:** The size of the augmented dataset, if not given it defaults to 10
  times the decorated dataset
* **random\_state:** A seed for the pseudo random number generator so that the
  augmented datasets are reproducible
* **cache\_size:** The number of samples to cache using an LRU policy in order
  to reduce the time spent augmenting the same images (defaults to
  `len(dataset.train_data)`)

**Example**

```python
from keras.datasets import cifar10
from importance_sampling.datasets import InMemoryImageDataset, \
    OntheflyAugmentedImages

dset = OntheflyAugmentedImages(
    InMemoryImageDataset.from_loadable(cifar10),
    dict(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
)

assert dset.shape == (32, 32, 3)
assert dset.output_size == 10
assert len(dset.train_data) == 10 * 50000
```

## GeneratorDataset

```python
importance_sampling.dataset.GeneratorDataset(train_data, test_data=None, test_data_length=None, cache_size=5)
```

The `GeneratorDataset` wraps one or two generators and partially implements the
`BaseDataset` interface. The `test_data` can be a generator or in memory data.
The generators are consumed in background threads and at most `cache_size`
return values are saved from each at any given time.

**Arguments**

* **train\_data**: A normal Keras compatible data generator. It should be infinite and
  return both inputs and targets
* **test\_data**: Either a Keras compatible data generator or a list, numpy
  array etc.
* **test\_data\_length**: When `test_data` is a generator then the number of
  points in the test set should be given.
* **cach\_size**: The maximum return values cached in the backgound threads
  from the generators, equivalent to Keras's `max_queue_size`

**Example**

```python
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from importance_sampling.datasets import GeneratorDataset

# Load cifar into x_train, y_train, x_test, y_test
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Create a data augmentation pipeline
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

dset = GeneratorDataset(
    datagen.flow(x_train, y_train, batch_size=32),
    (x_test, y_test)
)

assert dset.shape == (32, 32, 3)
assert dset.output_size == 10
assert len(dset.test_data) == 10000
```

## Provided dataset classes

`MNIST`, `CIFAR10` and `CIFAR100` are already provided as dataset classes with
no constructor parameters.

### CanevetICML2016

```python
importance_sampling.datasets.CanevetICML2016(N=8192, test_split=0.33, smooth=40)
```

This dataset is an artificial 2-dimensional binary classification dataset that
is suitable for importance sampling and was introduced by [Canevet et al. ICML
2016][canevet_et_al].

<div class="fig col-3">
<img alt="Canevet dataset with smooth 10"
     src="../img/canevet_icml2016_smooth10.png" />
<img alt="Canevet dataset with smooth 40"
     src="../img/canevet_icml2016_smooth40.png" />
<img alt="Canevet dataset with smooth 100"
     src="../img/canevet_icml2016_smooth100.png" />
<span>The effect of the smooth argument on the artificial dataset. From left to
right smooth is 10, 40, 100.</span>
</div>

**Arguments**

* **N:** The dataset is going to have N^2 points
* **test\_split:** The percentage of the points to keep as a test/validation
  set
* **smooth:** A jitter controlling parameter whose effects are seen in the
  previous figure.

[canevet_et_al]: http://fleuret.org/papers/canevet-et-al-icml2016.pdf
