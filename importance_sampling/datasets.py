#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from bisect import bisect_left
from collections import OrderedDict, deque
from functools import partial
import gzip
from itertools import islice
import os
from os import path
import pickle
import sys
from tempfile import TemporaryFile
from threading import Condition, Lock, Thread

from keras.applications.resnet50 import preprocess_input
from keras.datasets import cifar10, cifar100, mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
import numpy as np

from .utils.functional import compose

try:
    from PIL import Image as pil_image
except ImportError:
    pass


class cifar_sanity_check(object):
    """Create a dataset from a subset of CIFAR10 that is designed to be ideal
    for importance sampling by having a sample repeated thousands of times"""
    def __init__(self, classes=(3, 5), replicate=30000, replicate_idx=42):
        assert len(classes) > 1
        self.classes = classes
        self.replicate = replicate
        self.replicate_idx = replicate_idx

    def load_data(self):
        # Load the original data
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # Get only the classes given
        idxs_train = np.arange(len(y_train))[
            np.logical_or(*[y_train == c for c in self.classes]).ravel()
        ]
        idxs_test = np.arange(len(y_test))[
            np.logical_or(*[y_test == c for c in self.classes]).ravel()
        ]
        X_train = X_train[idxs_train]
        y_train = y_train[idxs_train]
        X_test = X_test[idxs_test]
        y_test = y_test[idxs_test]
        for i, c in enumerate(self.classes):
            y_train[y_train == c] = i
            y_test[y_test == c] = i

        # Replicate on document in the training set
        x, y = X_train[self.replicate_idx], y_train[self.replicate_idx]
        x = np.tile(x, (self.replicate, 1, 1, 1))
        y = np.tile(y, (self.replicate, 1))

        return (
            (np.vstack([X_train, x]), np.vstack([y_train, y])),
            (X_test, y_test)
        )


class canevet_icml2016_nn(object):
    """Approximates the artifical dataset used in [1] section 4.2.

    [1] Can\'evet, Olivier, et al. "Importance Sampling Tree for Large-scale
        Empirical Expectation." Proceedings of the International Conference on
        Machine Learning (ICML). No. EPFL-CONF-218848. 2016.
    """
    def __init__(self, N=8192, test_split=0.33, smooth=40):
        self.N = N
        self.test_split = test_split
        self.smooth = smooth

    def load_data(self):
        # Create the data using magic numbers to approximate the figure in
        # canevet_icml2016
        x = np.linspace(0, 1, self.N).astype(np.float32)
        ones = np.ones_like(x).astype(int)
        boundary = np.sin(4*(x + 0.5)**5)/3 + 0.5

        data = np.empty(shape=[self.N, self.N, 3], dtype=np.float32)
        data[:, :, 0] = 1-x
        for i in range(self.N):
            data[i, :, 1] = 1-x[i]
            data[i, :, 2] = 1 / (1 + np.exp(self.smooth*(x - boundary[i])))
            data[i, :, 2] = np.random.binomial(ones, data[i, :, 2])
        data = data.reshape(-1, 3)
        np.random.shuffle(data)

        # Create train and test arrays
        split = int(len(data)*self.test_split)
        X_train = data[:-split, :2]
        y_train = data[:-split, 2]
        X_test = data[-split:, :2]
        y_test = data[-split:, 2]

        return (X_train, y_train), (X_test, y_test)


class BaseDataset(object):
    class _DataProxy(object):
        def __init__(self, dataset, subset):
            self.data = getattr(dataset, "_%s_data" % (subset,))
            self.size = getattr(dataset, "_%s_size" % (subset,))

        def __getitem__(self, idxs):
            return self.data(idxs)

        def __len__(self):
            return self.size()

    def _train_data(self, idxs=slice(None)):
        """Return the training data in the form (x, y)"""
        raise NotImplementedError()

    def _train_size(self):
        """Training data length"""
        x, y = self._train_data()
        if isinstance(x, (list, tuple)):
            return len(x[0])
        else:
            return len(x)

    def _test_data(self, idxs=slice(None)):
        """Return the testing data in the form (x, y)"""
        raise NotImplementedError()

    def _test_size(self):
        """Test data length"""
        x, y = self._test_data()
        if isinstance(x, (list, tuple)):
            return len(x[0])
        else:
            return len(x)

    def _slice_data(self, x, y, idxs):
        if isinstance(x, (list, tuple)):
            return [xi[idxs] for xi in x], y[idxs]
        else:
            return x[idxs], y[idxs]

    def _extract_shape(self, x):
        if isinstance(x, (list, tuple)):
            return [xi.shape[1:] for xi in x]
        return x.shape[1:]

    @property
    def train_data(self):
        return self._DataProxy(self, "train")

    @property
    def test_data(self):
        return self._DataProxy(self, "test")

    @property
    def shape(self):
        """Return the shape of the samples"""
        raise NotImplementedError()

    @property
    def output_size(self):
        """Return the number of outputs"""
        raise NotImplementedError()

    @property
    def output_shape(self):
        """Return the shape of the output (it could differ for seq models for
        instance)."""
        return (self.output_size,)


class InMemoryDataset(BaseDataset):
    """A dataset that fits in memory and is simply 4 numpy arrays (x, y) *
    (train, test) where x can also be a list of numpy arrays"""
    def __init__(self, X_train, y_train, X_test, y_test, categorical=True):
        self._x_train = X_train
        self._x_test = X_test

        # are the targets to be made one hot vectors
        if categorical:
            self._y_train = np_utils.to_categorical(y_train)
            self._y_test = np_utils.to_categorical(y_test)
            self._output_size = self._y_train.shape[1]

        # handle sparse output classification
        elif issubclass(y_train.dtype.type, np.integer):
            self._y_train = y_train
            self._y_test = y_test
            self._output_size = self._y_train.max() + 1  # assume 0 based idxs

        # not classification, just copy them
        else:
            self._y_train = y_train
            self._y_test = y_test
            self._output_size = self._y_train.shape[1]

    def _train_data(self, idxs=slice(None)):
        return self._slice_data(self._x_train, self._y_train, idxs)

    def _test_data(self, idxs=slice(None)):
        return self._slice_data(self._x_test, self._y_test, idxs)

    @property
    def shape(self):
        return self._extract_shape(self._x_train)

    @property
    def output_size(self):
        return self._output_size

    @classmethod
    def from_loadable(cls, dataset):
        (a, b), (c, d) = dataset.load_data()
        return cls(a, b, c, d)


class InMemoryImageDataset(InMemoryDataset):
    """Make sure that the in memory dataset has 4 dimensions and is normalized
    to [0, 1]"""
    def __init__(self, X_train, y_train, X_test, y_test, categorical=True):
        # Expand the dims and make sure the shapes are correct image shapes
        if len(X_train.shape) < 4:
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)
        assert X_train.shape[1:] == X_test.shape[1:]
        assert len(X_train.shape) == 4

        # Normalize to [0, 1]
        X_train = X_train.astype(np.float32) / X_train.max()
        X_test = X_test.astype(np.float32) / X_test.max()

        super(InMemoryImageDataset, self).__init__(
            X_train,
            y_train,
            X_test,
            y_test,
            categorical=categorical
        )


CIFAR10 = partial(InMemoryImageDataset.from_loadable, cifar10)
CIFAR100 = partial(InMemoryImageDataset.from_loadable, cifar100)
MNIST = partial(InMemoryImageDataset.from_loadable, mnist)
CIFARSanityCheck = compose(
    InMemoryImageDataset.from_loadable,
    cifar_sanity_check
)
CanevetICML2016 = compose(InMemoryDataset.from_loadable, canevet_icml2016_nn)


class GeneratorDataset(BaseDataset):
    """GeneratorDataset wraps a generator (or two) and partially implements the
    BaseDataset interface."""
    def __init__(self, train_data, test_data=None, test_data_length=None,
                 cache_size=5):
        self._train_data_gen = train_data
        self._test_data_gen = test_data
        self._test_data_len = test_data_length
        self._cache_size = cache_size

        # Determine the shapes and sizes
        x, y = next(self._train_data_gen)
        self._shape = self._extract_shape(x)
        self._output_size = y.shape[1] if len(y.shape) > 1 else 1

        # Create the queues
        self._train_cache = deque()
        self._train_lock = Lock()
        self._train_cv = Condition(self._train_lock)
        self._test_cache = deque()
        self._test_lock = Lock()
        self._test_cv = Condition(self._test_lock)

        # Start the threads
        self._train_thread = Thread(
            name="train_thread",
            target=self._generator_thread,
            args=(
                self._train_data_gen,
                self._train_cache,
                self._train_lock,
                self._train_cv,
                self._cache_size
            )
        )
        self._test_thread = Thread(
            name="test_thread",
            target=self._generator_thread,
            args=(
                self._test_data_gen,
                self._test_cache,
                self._test_lock,
                self._test_cv,
                self._cache_size
            )
        )
        self._train_thread.daemon = True
        self._test_thread.daemon = True
        self._train_thread.start()
        self._test_thread.start()

    @staticmethod
    def _generator_thread(gen, cache, lock, cv, max_size):
        if gen is None:
            return
        if isinstance(gen, (tuple, list, np.ndarray)):
            return
        while True:
            xy = next(gen)
            with lock:
                while len(cache) >= max_size:
                    cv.wait()
                cache.append(xy)
                cv.notify()

    def _get_count(self, idxs):
        if isinstance(idxs, slice):
            # Use 2**32 as infinity
            start, stop, step = idxs.indices(2**32)
            return (stop - start) // step
        elif isinstance(idxs, (list, np.ndarray)):
            return len(idxs)
        elif isinstance(idxs, int):
            return 1
        else:
            raise IndexError("Invalid indices passed to dataset")

    def _get_at_least_n(self, cache, lock, cv, n):
        cnt = 0
        batches = []
        with lock:
            while cnt < n:
                while len(cache) <= 0:
                    cv.wait()
                batch = cache.popleft()
                cv.notify()
                cnt += len(batch[1])
                if isinstance(batch[0], (list, tuple)):
                    batches.append(list(batch[0]) + [batch[1]])
                else:
                    batches.append(batch)

        xy = tuple(map(np.vstack, zip(*batches)))
        if len(xy) > 2:
            return list(xy[:-1]), xy[-1]
        else:
            return xy

    def _train_data(self, idxs=slice(None)):
        N = self._get_count(idxs)
        x, y = self._get_at_least_n(
            self._train_cache,
            self._train_lock,
            self._train_cv,
            N
        )

        return self._slice_data(x, y, slice(N))

    def _train_size(self):
        raise RuntimeError("This dataset has no size")

    def _test_data(self, idxs=slice(None)):
        # No test data
        if self._test_data_gen is None:
            raise RuntimeError("This dataset has no test data")

        # Test data are all in memory
        if isinstance(self._test_data_gen, (tuple, list, np.ndarray)):
            x, y = self._test_data_gen
            return self._slice_data(x, y, idxs)

        # Test data are provided via a generator
        N = min(self._test_data_len, self._get_count(idxs))
        x, y = self._get_at_least_n(
            self._test_cache,
            self._test_lock,
            self._test_cv,
            N
        )

        return self._slice_data(x, y, slice(N))

    def _test_size(self):
        # No test data
        if self._test_data_gen is None:
            raise RuntimeError("This dataset has no test data")

        # Test data are all in memory
        if isinstance(self._test_data_gen, (tuple, list, np.ndarray)):
            x, y = self._test_data_gen
            return len(x)

        # Test data are provided via a generator
        return self._test_data_len

    @property
    def shape(self):
        return self._shape

    @property
    def output_size(self):
        return self._output_size


class ZCAWhitening(InMemoryImageDataset):
    """Make a whitened copy of the decorated dataset in memory."""
    def __init__(self, dataset):
        # Get the data in memory
        x_train, y_train = dataset.train_data[:]
        x_test, y_test = dataset.test_data[:]

        # Make the whitener and train it
        gen = ImageDataGenerator(zca_whitening=True, featurewise_center=True)
        gen.fit(x_train)

        batches_train = list(islice(
            gen.flow(x_train, y_train, 32),
            int(np.ceil(len(x_train) / 32.))
        ))
        batches_test = list(islice(
            gen.flow(x_test, y_test, 32),
            int(np.ceil(len(x_test) / 32.))
        ))

        super(ZCAWhitening, self).__init__(
            np.vstack([b[0] for b in batches_train]),
            np.vstack([b[1] for b in batches_train]),
            np.vstack([b[0] for b in batches_test]),
            np.vstack([b[1] for b in batches_test]),
            categorical=False
        )


class AugmentedImages(BaseDataset):
    def __init__(self, dataset, augmentation_params, N=None):
        # Initialize member variables
        self.dataset = dataset
        self.augmentation_params = augmentation_params
        self.N = len(self.dataset.train_data) * 10 if N is None else N
        assert len(self.dataset.shape) == 3

        # Allocate space for the augmented data
        self._x = np.memmap(
            TemporaryFile(),
            dtype=np.float32,
            shape=(self.N,) + self.dataset.shape
        )
        self._y = np.zeros((self.N, self.dataset.output_size))

        # Train a generator and generate all the data
        generator = ImageDataGenerator(**self.augmentation_params)
        x, y = self.dataset.train_data[:]
        generator.fit(x)
        start = 0
        for bx, by in generator.flow(x, y, batch_size=128):
            end = min(self.N, start+len(bx))
            self._x[start:end] = bx[:end-start]
            self._y[start:end] = by[:end-start]

            start = end
            if start >= self.N:
                break

    def _train_data(self, idxs=slice(None)):
        return self._x[idxs], self._y[idxs]

    def _test_data(self, idxs=slice(None)):
        return self.dataset.test_data[idxs]

    @property
    def shape(self):
        return self.dataset.shape

    @property
    def output_size(self):
        return self.dataset.output_size


class OntheflyAugmentedImages(BaseDataset):
    """Use a Keras ImageDataGenerator to augment images on the fly in a
    determenistic way."""
    def __init__(self, dataset, augmentation_params, N=None, random_state=0,
                 cache_size=None):
        # Initialize some member variables
        self.dataset = dataset
        self.generator = ImageDataGenerator(**augmentation_params)
        self.N = N or (len(self.dataset.train_data) * 10)
        self.random_state = random_state
        assert len(self.dataset.shape) == 3

        # Figure out the base images for each of the augmented ones
        self.idxs = np.random.choice(
            len(self.dataset.train_data),
            self.N
        )

        # Fit the generator
        self.generator.fit(self.dataset.train_data[:][0])

        # Standardize the test data
        self._x_test = np.copy(self.dataset.test_data[:][0])
        self._x_test = self.generator.standardize(self._x_test)
        self._y_test = self.dataset.test_data[:][1]

        # Create an LRU cache to speed things up a bit for the transforms
        cache_size = cache_size or len(self.dataset.train_data)
        self.cache = OrderedDict([(-i, i) for i in range(cache_size)])
        self.cache_data = np.empty(
            shape=(cache_size,) + self.dataset.shape,
            dtype=np.float32
        )

    def _transform(self, idx, x):
        # if it is not cached add it
        if idx not in self.cache:
            # Remove the first in and add the new idx (i is the offset in
            # cache_data)
            _, i = self.cache.popitem(last=False)
            self.cache[idx] = i

            # Do the transformation and add it to the data
            np.random.seed(idx + self.random_state)
            x = self.generator.random_transform(x)
            x = self.generator.standardize(x)
            self.cache_data[i] = x

        # and if it is update it as the most recently used
        else:
            self.cache[idx] = self.cache.pop(idx)

        return self.cache_data[self.cache[idx]]

    def _train_data(self, idxs=slice(None)):
        # Make sure we accept everything that numpy accepts as indices
        idxs = np.arange(self.N)[idxs]

        # Get the original images and then transform them
        x, y = self.dataset.train_data[self.idxs[idxs]]
        x_hat = np.copy(x)
        random_state = np.random.get_state()
        for i, idx in enumerate(idxs):
            x_hat[i] = self._transform(idx, x_hat[i])
        np.random.set_state(random_state)

        return x_hat, y

    def _test_data(self, idxs=slice(None)):
        return self._x_test[idxs], self._y_test[idxs]

    def _train_size(self):
        return self.N

    @property
    def shape(self):
        return self.dataset.shape

    @property
    def output_size(self):
        return self.dataset.output_size


class PennTreeBank(BaseDataset):
    """Load the PennTreebank from Tomas Mikolov's format expected in the
    default Keras directory."""
    def __init__(self, context, ptb_path=None, val=True, verbose=True,
                 cache=True):
        if ptb_path is None:
            ptb_path = path.expanduser("~/.keras/datasets/ptb")

        if val:
            test = "valid"
        else:
            test = "test"

        # Cache the dataset for faster subsequent loads
        cache_path = path.join(ptb_path, "ptb.train-%s.pickle.gz" % (test,))

        if not path.exists(cache_path):
            with open(path.join(ptb_path, "ptb.train.txt")) as f:
                train = [l.split() + ['<EOS>'] for l in f]
            with open(path.join(ptb_path, "ptb.%s.txt" % (test,))) as f:
                test = [l.split() + ['<EOS>'] for l in f]
            V = np.array(sorted({w for l in train for w in l}))
            N = max(max(map(len, train)), max(map(len, test)))

            # No need to have context bigger than the biggest sentence
            context = min(context, N)

            # Allocate memory
            x_train = np.empty((0, context), dtype=np.int32)
            y_train = np.empty((0, 1), dtype=np.int32)
            x_test = np.empty_like(x_train)
            y_test = np.empty_like(y_train)

            # Encode the strings to numbers
            if verbose:
                prog = Progbar(len(train) + len(test))
            for i, s in enumerate(train):
                xi, yi = self._encode(s, V, context)
                x_train = np.vstack([x_train, xi])
                y_train = np.vstack([y_train, yi])
                if verbose and i % 100 == 0:
                    prog.update(i)
            for i, s in enumerate(test):
                xi, yi = self._encode(s, V, context)
                x_test = np.vstack([x_test, xi])
                y_test = np.vstack([y_test, yi])
                if verbose and i % 100 == 0:
                    prog.update(len(train) + i)
            if verbose:
                prog.update(len(train) + len(test))

            with gzip.open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "train": (x_train, y_train),
                        "test": (x_test, y_test),
                        "vocab": V
                    },
                    f,
                    protocol=2
                )

        # Read the dataset from the cached binary file
        with gzip.open(cache_path) as f:
            data = pickle.load(f)
            self._x_train, self._y_train = data["train"]
            self._x_test, self._y_test = data["test"]
            self.V = data["vocab"]

    def _encode(self, s, V, context):
        """
        Arguments
        ----------
            s: Sentence as a list of strings
            V: Vocabulary as a np array of strings
            context: The maximum length of previous words to include
        """
        idxs = np.searchsorted(V, s)
        x = np.zeros((len(s)-1, context), dtype=np.int32)
        y = np.zeros((len(s)-1, 1), np.int32)
        for i in range(1, len(s)):
            x[i-1, :i] = idxs[:i][-context:] + 1  # 0 means missing value
            y[i-1] = idxs[i]

        return x, y

    def _train_data(self, idxs=slice(None)):
        return self._x_train[idxs], self._y_train[idxs]

    def _test_data(self, idxs=slice(None)):
        return self._x_test[idxs], self._y_test[idxs]

    @property
    def shape(self):
        return self._x_train.shape[1:]

    @property
    def output_size(self):
        return len(self.V)


class ImageNetDownsampled(BaseDataset):
    """Dataset interface to the downsampled ImageNet [1].

    The data are expected in the following format:
        _ base-path/
         \_ imagenet-16x16
         |_ imagenet-32x32
         |_ imagenet-64x64
          \_ mean.npy
          |_ train_data.npy
          |_ train_labels.npy
          |_ val_data.npy
          |_ val_labels.npy

    1: A Downsampled Variant of ImageNet as an Alternative to the CIFAR
       datasets (https://arxiv.org/abs/1707.08819v2)
    """
    def __init__(self, basepath, size=32, mmap=False):
        basepath = path.join(basepath, "imagenet-%dx%d" % (size, size))
        self.mean = np.load(path.join(basepath, "mean.npy"))
        self._y_train = np.load(path.join(basepath, "train_labels.npy"))
        self._y_val = np.load(path.join(basepath, "val_labels.npy"))
        self._x_train = np.load(
            path.join(basepath, "train_data.npy"),
            mmap_mode="r" if mmap else None
        )
        self._x_val = np.load(
            path.join(basepath, "val_data.npy"),
            mmap_mode="r" if mmap else None
        )

    def _get_batch(self, X, Y, idxs):
        """Preprocess the batch by subtracting the mean and normalizing."""
        if isinstance(idxs, slice):
            idxs = np.arange(2*len(X))[idxs]

        N = len(idxs)
        x = np.zeros((N,) + self.shape, dtype=np.float32)
        y = np.zeros((N, 1000), dtype=np.float32)

        # Fill in the class information
        y[np.arange(N), Y[idxs % len(X)]-1] = 1.

        # Fill in the images
        d = self.shape[0]
        x[:] = X[idxs % len(X)]
        flip = (idxs / len(X)) == 1  # if idx > len(X) flip horizontally
        x[flip] = x[flip, :, ::-1]
        x -= self.mean
        x /= 255

        return x, y

    def _train_data(self, idxs=slice(None)):
        return self._get_batch(self._x_train, self._y_train, idxs)

    def _test_data(self, idxs=slice(None)):
        return self._get_batch(self._x_val, self._y_val, idxs)

    def _train_size(self):
        return 2*len(self._x_train)

    def _test_size(self):
        return len(self._x_val)

    @property
    def shape(self):
        return self._x_train.shape[1:]

    @property
    def output_size(self):
        return 1000


class TIMIT(InMemoryDataset):
    """Load the TIMIT dataset [1] from a custom pickled format.

    The format is the following:
        [
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        ]

        Each X_* is a list of numpy arrays that contain the full utterance
        features.

        Each y_* is a list of numpy arrays that contain the per phoneme label
        for the full utterance.

    [1] Garofolo, John S., et al. TIMIT Acoustic-Phonetic Continuous Speech
        Corpus LDC93S1. Web Download. Philadelphia: Linguistic Data Consortium,
        1993
    """
    def __init__(self, context, path, val=False):
        # Read the data
        data = pickle.load(open(path))
        train = data[:2]
        test = data[2:4] if val else data[4:]

        x_train, y_train = self._create_xy(train, context)
        x_test, y_test = self._create_xy(test, context)

        super(TIMIT, self).__init__(
            x_train, y_train,
            x_test, y_test,
            categorical=False
        )

    def _create_xy(self, data, context):
        X = []
        y = []
        for xi, yi in zip(*data):
            for j in range(context-1, len(xi)):
                X.append(xi[j-context+1:j+1])
                y.append(yi[j:j+1])  # slice so that y.shape == (?, 1)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


class MIT67(BaseDataset):
    """Dataset interface to the MIT67 Indoor Scenes dataset [1].

    The dataset is expected to be in the following format:
        - base-path/
         \_ images
           \_ airport_inside
           |_ artstudio
           |_ ...
         |_ TrainImages.txt
         |_ TestImages.txt

    1: Quattoni, Ariadna, and Antonio Torralba. "Recognizing indoor scenes."
       Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE
       Conference on. IEEE, 2009.
    """
    def __init__(self, basepath):
        self._base = basepath

        # Read the file paths
        self._train_set = np.array([
            l.strip() for l in
            open(path.join(basepath, "TrainImages.txt"))
        ])
        self._test_set = np.array([
            l.strip() for l in
            open(path.join(basepath, "TestImages.txt"))
        ])

        # Create masks to lazily read the data in memory
        self._unread_train = np.ones(len(self._train_set), dtype=np.bool)
        self._unread_test = np.ones(len(self._test_set), dtype=np.bool)

        # Allocate space for the images
        self._x_train = np.zeros(
            (len(self._train_set), 224, 224, 3),
            dtype=np.float32
        )
        self._x_test = np.zeros(
            (len(self._test_set), 224, 224, 3),
            dtype=np.float32
        )

        # Create the target classes
        class_set = np.array(sorted(set([
            filepath.split("/")[0]
            for filepath in self._train_set
        ])))
        self._y_train = self._to_classes(self._train_set, class_set)
        self._y_test = self._to_classes(self._test_set, class_set)

    def _to_classes(self, files, class_set):
        y = np.zeros((len(files), len(class_set)), dtype=np.float32)
        for i, f in enumerate(files):
            yi = f.split("/")[0]
            y[i, np.searchsorted(class_set, yi)] = 1.
        return y

    def _read_and_return(self, unread, files, data, idxs):
        if np.any(unread[idxs]):
            for i in np.arange(len(files))[idxs]:
                if not unread[i]:
                    continue
                data[i] = self._read_image(files[i])
                unread[i] = False
        return data[idxs]

    def _train_size(self):
        return len(self._x_train)

    def _train_data(self, idxs=slice(None)):
        return self._read_and_return(
            self._unread_train,
            self._train_set,
            self._x_train,
            idxs
        ), self._y_train[idxs]

    def _test_size(self):
        return len(self._x_test)

    def _test_data(self, idxs=slice(None)):
        return self._read_and_return(
            self._unread_test,
            self._test_set,
            self._x_test,
            idxs
        ), self._y_test[idxs]

    def _read_image(self, image_path):
        """Read an image from disk, resize, crop and preprocess it using the
        ImageNet stats."""
        image_path = path.join(self._base, "images", image_path)
        img = pil_image.open(image_path).convert("RGB")
        s = max(224./img.width, 224./img.height)
        nw, nh = int(s * img.width), int(s * img.height)
        pw, ph = int((nw - 224)/2), int((nh - 224)/2)
        dims = nw, nh
        box = (pw, ph, pw+224, ph+224)
        img = img.resize(dims, pil_image.BILINEAR).crop(box)

        return preprocess_input(np.array(img, dtype=np.float32))

    @property
    def shape(self):
        return (224, 224, 3)

    @property
    def output_size(self):
        return 67


class CASIAWebFace(BaseDataset):
    """Provide a BaseDataset interface to CASIAWebFace.

    The interface is created for training with a triplet loss which is a bit
    unorthodox.

    Arguments
    ---------
        basepath: The path to the dataset
        alpha: The margin for the triplet loss (returned as target)
        validation: Consider as validation set all the person ids that %
                    validation == 0
    """
    def __init__(self, basepath, alpha=0.2, embedding=128, validation=5,
                 cache=4096):
        # Save the configuration in member variables
        self._alpha = alpha
        self._embedding = embedding

        # Load the paths for the images
        self._basepath = basepath
        ids = [x for x in os.listdir(basepath) if "." not in x]
        self._train = np.array([
            x for x in ids if int(x.replace("0", "")) % validation > 0
        ])
        self._n_images = np.array([
            len([
                img for img in os.listdir(path.join(basepath, x))
                if img.endswith("jpg")
            ]) for x in self._train
        ])
        self._idxs = np.arange(self._n_images.sum())

        # Create the necessary variables for the cache
        self._cache = np.zeros((cache, 3, 224, 224, 3), dtype=np.float32)
        self._cache_lock = Lock()
        self._cache_cv = Condition(self._cache_lock)

        # Start a thread to load images and wait for the cache to be filled
        self._producer_thread = Thread(target=self._update_images)
        self._producer_thread.daemon = True
        with self._cache_lock:
            self._producer_thread.start()
            while np.all(self._cache[-1] == 0):
                self._cache_cv.wait()

    def _get_batch_memory(self, N):
        if not hasattr(self, "_batch_xa") or len(self._batch_xa) < N:
            self._batch_xa = np.zeros((N, 224, 224, 3), dtype=np.float32)
            self._batch_xp = np.zeros_like(self._batch_xa)
            self._batch_xn = np.zeros_like(self._batch_xa)

        return self._batch_xa[:N], self._batch_xp[:N], self._batch_xn[:N]

    def _train_data(self, idxs=slice(None)):
        N = len(self._idxs[idxs])
        y = np.ones((N, 1), dtype=np.float32)*self._alpha
        xa, xp, xn = self._get_batch_memory(N)
        samples = np.random.choice(len(self._cache), N)

        with self._cache_lock:
            xa[:] = self._cache[samples, 0]
            xp[:] = self._cache[samples, 1]
            xn[:] = self._cache[samples, 2]

        return [xa, xp, xn], y

    def _train_size(self):
        return self._n_images.sum()

    def _test_data(self, idxs=slice(None)):
        return [np.random.rand(1, 224, 224, 3)]*3, np.zeros((1, 1))

    def _test_size(self):
        return 1

    @property
    def shape(self):
        return [(224, 224, 3)]*3

    @property
    def output_size(self):
        return self._embedding

    def _update_images(self):
        try:
            with self._cache_lock:
                for i in range(len(self._cache)):
                    triplet = self._read_random_triplet()
                    self._cache[i, 0] = triplet[0]
                    self._cache[i, 1] = triplet[1]
                    self._cache[i, 2] = triplet[2]
                self._cache_cv.notifyAll()

            while True:
                triplet = self._read_random_triplet()
                i = np.random.choice(len(self._cache))
                with self._cache_lock:
                    self._cache[i, 0] = triplet[0]
                    self._cache[i, 1] = triplet[1]
                    self._cache[i, 2] = triplet[2]
        except:
            if sys is not None:
                sys.stderr.write("Producer thread tear down by exception\n")

    def _read_random_triplet(self):
        pos = np.random.choice(len(self._train))
        neg = np.random.choice(len(self._train))
        if pos == neg:
            return self._read_random_triplet()

        anchor_image, pos_image = np.random.choice(self._n_images[pos], 2)
        if anchor_image == pos_image:
            return self._read_random_triplet()

        neg_image = np.random.choice(self._n_images[neg])

        # Now we have our triplet
        return (
            self._read_image(self._train[pos], anchor_image),
            self._read_image(self._train[pos], pos_image),
            self._read_image(self._train[neg], neg_image)
        )

    def _read_image(self, person, image):
        image_path = path.join(
            self._basepath,
            person,
            "{:03d}.jpg".format(image+1)
        )
        img = pil_image.open(image_path).convert("RGB")
        img = img.resize((224, 224), pil_image.BILINEAR)

        return preprocess_input(np.array(img, dtype=np.float32))


class LFW(BaseDataset):
    """BaseDataset interface to Labeled Faces in the Wild dataset.

    The dataset provides both images and indexes for the lfw folds.

    Arguments
    ---------
        basepath: The path to the dataset
        fold: [1,10] or None
              Choose a fold to evaluate on or all the images
        idxs: bool
              Whether to load images or just indexes for the fold
    """
    def __init__(self, basepath, fold=1, idxs=False):
        self._basepath = basepath
        self._fold = fold
        self._idxs = idxs
        self._collect_images()
        self._collect_pairs()

    def _image_path(self, name, img):
        return path.join(name, "{}_{:04d}".format(name, img))

    def _get_person(self, image):
        return bisect_left(
            self._names,
            image.split(os.sep)[0]
        )

    def _get_idx(self, name, img):
        return bisect_left(self._images, self._image_path(name, img))

    def _collect_images(self):
        """Collect all the image paths into a sorted list."""
        image_path = path.join(self._basepath, "all_images")
        self._names = np.array(sorted(set([
            name for name in os.listdir(image_path)
            if path.isdir(path.join(image_path, name))
        ])))
        self._images = np.array(sorted([
            path.join(name, img)
            for name in self._names
            for img in os.listdir(path.join(image_path, name))
            if img.endswith(".jpg")
        ]))

    def _collect_pairs(self):
        if self._fold is None:
            return

        with open(path.join(self._basepath, "view2", "pairs.txt")) as f:
            folds, n = map(int, next(f).split())
            assert 1 <= self._fold <= folds
            idxs = np.zeros((n*2*folds, 2), dtype=np.int32)
            matches = np.zeros((n*2*folds, 1), dtype=np.float32)
            for i, l in enumerate(f):
                parts = l.split()
                matches[i] = float(len(parts) == 3)
                if matches[i]:
                    idxs[i] = [
                        self._get_idx(parts[0], int(parts[1])),
                        self._get_idx(parts[0], int(parts[2]))
                    ]
                else:
                    idxs[i] = [
                        self._get_idx(parts[0], int(parts[1])),
                        self._get_idx(parts[2], int(parts[3]))
                    ]
            idxs_2 = np.arange(len(idxs))
            train = np.logical_or(
                idxs_2 < 2*n*(self._fold - 1),
                idxs_2 >= 2*n*self._fold
            )
            test = np.logical_and(
                idxs_2 >= 2*n*(self._fold - 1),
                idxs_2 < 2*n*self._fold
            )
            self._idxs_train = idxs[train]
            self._idxs_test = idxs[test]
            self._y_train = matches[train]
            self._y_test = matches[test]

    def _read_image(self, image):
        full_img_path = path.join(self._basepath, "all_images", image)
        img = pil_image.open(full_img_path).convert("RGB")
        img = img.resize((224, 224), pil_image.BILINEAR)

        return preprocess_input(np.array(img, dtype=np.float32))

    def _get_data(self, pairs):
        if self._idxs:
            return pairs
        else:
            x1 = np.stack(map(self._read_image, self._images[pairs[:, 0]]))
            x2 = np.stack(map(self._read_image, self._images[pairs[:, 1]]))
            return [x1, x2]

    def _train_data(self, idxs=slice(None)):
        if self._fold is None:
            images = self._images[idxs]
            x = np.stack(map(self._read_image, images))
            y = np.array(map(self._get_person, images))
        else:
            x = self._get_data(self._idxs_train[idxs])
            y = self._y_train[idxs]

        return x, y

    def _train_size(self):
        if self._fold is None:
            return len(self._images)
        return len(self._idxs_train)

    def _test_data(self, idxs=slice(None)):
        if self._fold is None:
            raise NotImplementedError()
        x = self._get_data(self._idxs_test[idxs])
        y = self._y_test[idxs]

        return x, y

    def _test_size(self):
        return 0 if self._fold is None else len(self._idxs_test)

    @property
    def shape(self):
        if self._fold is None:
            return (224, 224, 3)
        else:
            if self._idxs:
                return (2,)
            else:
                return [(224, 224, 3)]*2

    @property
    def output_size(self):
        return 1


class CASIAWebFace2(BaseDataset):
    """Provide a classification interface to CASIAWebFace."""
    def __init__(self, basepath):
        self._basepath = basepath
        ids = [x for x in os.listdir(basepath) if "." not in x]
        self._output_size = len(ids)
        self._images = [
            (path.join(basepath, x, img), i) for i, x in enumerate(ids)
            for img in os.listdir(path.join(basepath, x))
            if img.endswith("jpg")
        ]
        self._idxs = np.arange(len(self._images))

    def _read_image(self, image_path):
        img = pil_image.open(image_path).convert("RGB")
        img = img.resize((224, 224), pil_image.BILINEAR)

        return preprocess_input(np.array(img, dtype=np.float32))

    def _train_data(self, idxs=slice(None)):
        idxs = self._idxs[idxs]
        y = np.array([self._images[i][1] for i in idxs])[:, np.newaxis]
        x = np.stack([
            self._read_image(self._images[i][0])
            for i in idxs
        ])

        return x, y

    def _train_size(self):
        return len(self._images)

    def _test_data(self, idxs=slice(None)):
        return np.random.rand(1, 224, 224, 3), np.zeors((1, 1))

    def _test_size(self):
        return 1

    @property
    def shape(self):
        return (224, 224, 3)

    @property
    def output_size(self):
        return self._output_size


class PixelByPixelMNIST(InMemoryDataset):
    """Transform MNIST into a sequence classification problem."""
    def __init__(self, permutation_seed=0):
        # Load the data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        dims = np.prod(x_train.shape[1:])

        # Generate the permutation
        state = np.random.get_state()
        np.random.seed(permutation_seed)
        permutation = np.random.permutation(dims)
        np.random.set_state(state)

        # Permutate, preprocess
        x_train = x_train.reshape(-1, dims)[:, permutation, np.newaxis]
        x_test = x_test.reshape(-1, dims)[:, permutation, np.newaxis]
        x_train = x_train.astype(np.float32) / 255.
        x_test = x_test.astype(np.float32) / 255.

        super(PixelByPixelMNIST, self).__init__(
            x_train,
            y_train,
            x_test,
            y_test
        )
