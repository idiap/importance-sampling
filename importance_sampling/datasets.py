#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from collections import OrderedDict
from functools import partial
import gzip
from itertools import islice
from os import path
import pickle
from tempfile import TemporaryFile

from keras.datasets import cifar10, cifar100, mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
import numpy as np

from .utils.functional import compose


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
        return len(self._train_data()[0])

    def _test_data(self, idxs=slice(None)):
        """Return the testing data in the form (x, y)"""
        raise NotImplementedError()

    def _test_size(self):
        """Test data length"""
        return len(self._test_data()[0])

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



class InMemoryDataset(BaseDataset):
    """A dataset that fits in memory and is simply 4 numpy arrays (x, y) *
    (train, test)"""
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
            self._output_size = self._y_train.max() + 1  # assume 0 based indexes

        # not classification, just copy them
        else:
            self._y_train = y_train
            self._y_test = y_test
            self._output_size = self._y_train.shape[1]

    def _train_data(self, idxs=slice(None)):
        return self._x_train[idxs], self._y_train[idxs]

    def _test_data(self, idxs=slice(None)):
        return self._x_test[idxs], self._y_test[idxs]

    @property
    def shape(self):
        return self._x_train.shape[1:]

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
    def __init__(self, X_train, y_train, X_test, y_test):
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
            y_test
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
    def __init__(self, train_data, test_data=None, test_data_length=None):
        self._train_data_gen = train_data
        self._test_data_gen = test_data
        self._test_data_len = test_data_length

        # Determine the shapes and sizes
        x, y = next(self._train_data_gen)
        self._shape = x.shape[1:]
        self._output_size = y.shape[1] if len(y.shape) > 1 else 1

    def _get_count(self, idxs):
        if isinstance(idxs, slice):
            # Use 2**32 as infinity
            start, stop, step = idxs.indices(2**32)
            return (stop - start)/step
        elif isinstance(idxs, (list, np.ndarray)):
            return len(idxs)
        elif isinstance(idxs, int):
            return 1
        else:
            raise IndexError("Invalid indices passed to dataset")

    def _get_n_batches(self, generator, n_batches):
        batches = list(islice(generator, n_batches))

        return tuple(map(np.vstack, zip(*batches)))

    def _get_at_least_n(self, generator, n):
        cnt = 0
        batches = []
        while cnt < n:
            batch = next(generator)
            cnt += len(batch[1])
            batches.append(batch)

        return tuple(map(np.vstack, zip(*batches)))

    def _train_data(self, idxs=slice(None)):
        N = self._get_count(idxs)
        x, y = self._get_at_least_n(self._train_data_gen, N)

        return x[:N], y[:N]

    def _train_size(self):
        raise RuntimeError("This dataset has no size")

    def _test_data(self, idxs=slice(None)):
        # No test data
        if self._test_data_gen is None:
            raise RuntimeError("This dataset has no test data")

        # Test data are all in memory
        if isinstance(self._test_data_gen, (tuple, list, np.ndarray)):
            x, y = self._test_data_gen
            return x[idxs], y[idxs]

        # Test data are provided via a generator
        N = min(self._test_data_len, self._get_count(idxs))
        x, y = self._get_at_least_n(self._test_data_gen, N)

        return x[:N], y[:N]

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
        self.cache = OrderedDict([(-i,i) for i in range(cache_size)])
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
