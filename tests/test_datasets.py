#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from functools import partial
import unittest

import numpy as np

from importance_sampling.datasets import CIFAR10, CIFARSanityCheck, MNIST, \
    CanevetICML2016, OntheflyAgumentedImages, PennTreeBank
from importance_sampling.utils.functional import compose


class TestDatasets(unittest.TestCase):
    def _test_dset(self, dset, n_train, n_test, shape, output_size):
        dset = dset()
        self.assertEqual(len(dset.train_data), n_train)
        self.assertEqual(len(dset.test_data), n_test)
        self.assertEqual(dset.shape, shape)
        self.assertEqual(dset.train_data[[0]][0][0].shape, shape)
        self.assertEqual(dset.output_size, output_size)

    def test_datasets(self):
        datasets = [
            (CIFAR10, 50000, 10000, (32, 32, 3), 10),
            (CIFARSanityCheck, 40000, 2000, (32, 32, 3), 2),
            (MNIST, 60000, 10000, (28, 28, 1), 10),
            (
                partial(CanevetICML2016, N=256), int(256**2 - 256**2 * 0.33) + 1,
                    int(256**2 * 0.33), (2,), 2
            ),
            (
                compose(
                    partial(OntheflyAgumentedImages, augmentation_params=dict(
                        featurewise_center=False,
                        samplewise_center=False,
                        featurewise_std_normalization=False,
                        samplewise_std_normalization=False,
                        zca_whitening=False,
                        rotation_range=0,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        horizontal_flip=True,
                        vertical_flip=False
                    )), CIFAR10), 500000, 10000, (32, 32, 3), 10
            ),
            (
                partial(PennTreeBank, 20), 887521, 70390, (20,), 10000
            )
        ]

        for args in datasets:
            self._test_dset(*args)

    def test_image_augmentation(self):
        dset = OntheflyAgumentedImages(
            CIFAR10(),
            dict(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False
            )
        )

        idxs = np.random.choice(len(dset.train_data), 100)
        x_r, y_r = dset.train_data[idxs]
        for i in range(10):
            x, y = dset.train_data[idxs]
            self.assertTrue(np.all(x_r == x))
            self.assertTrue(np.all(y_r == y))

        dset = OntheflyAgumentedImages(
            CIFAR10(),
            dict(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=True,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False
            )
        )

        idxs = np.random.choice(len(dset.train_data), 100)
        x_r, y_r = dset.train_data[idxs]
        for i in range(10):
            x, y = dset.train_data[idxs]
            self.assertTrue(np.all(x_r == x))
            self.assertTrue(np.all(y_r == y))


if __name__ == "__main__":
    unittest.main()
