#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from functools import reduce
from math import ceil

import h5py
from keras import backend as K
from keras.utils.data_utils import Sequence
import numpy as np


def weights_from_hdf5(f):
    """Extract all the weights from an h5py File or Group"""
    if "weight_names" in f.attrs:
        for n in f.attrs["weight_names"]:
            yield n, f[n]
    else:
        for k in f.keys():
            for n, w in weights_from_hdf5(f[k]):
                yield n, w


def possible_weight_names(name, n=10):
    name = name.decode()
    yield name
    parts = name.split("/")
    for i in range(1, n+1):
        yield str("{}_{}/{}".format(parts[0], i, parts[1]))


def load_weights_by_name(f, layers):
    """Load the weights by name from the h5py file to the model"""
    # If f is not an h5py thing try to open it
    if not isinstance(f, (h5py.File, h5py.Group)):
        with h5py.File(f, "r") as h5f:
            return load_weights_by_name(h5f, layers)

    # Extract all the weights from the layers/model
    if not isinstance(layers, list):
        layers = layers.layers
    weights = dict(reduce(
        lambda a, x: a + [(w.name, w) for w in x.weights],
        layers,
        []
    ))

    # Loop through all the possible layer weights in the file and make a list
    # of updates
    updates = []
    updated = []
    for name, weight in weights_from_hdf5(f):
        for n in possible_weight_names(name):
            if n in weights:
                updates.append((weights[n], weight))
                updated.append(n)
                break
    K.batch_set_value(updates)

    return updated


class DatasetSequence(Sequence):
    """Implement the Keras Sequence interface from a BaseDataset interface."""
    def __init__(self, dataset, train=True, part=slice(None), batch_size=32):
        self._data = dataset.train_data if train else dataset.test_data
        self._idxs = np.arange(len(self._data))[part]
        self._batch_size = batch_size

    def __len__(self):
        return int(ceil(float(len(self._idxs)) / self._batch_size))

    def __getitem__(self, idx):
        batch = self._idxs[self._batch_size*idx:self._batch_size*(idx+1)]
        return self._data[batch]
