#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from os import path
import shutil
import tempfile
import unittest

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

from importance_sampling.training import ImportanceTraining


class TestSaveLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def test_checkpoint(self):
        m = Sequential([
            Dense(10, activation="relu", input_shape=(2,)),
            Dense(2)
        ])
        m.compile("sgd", "mse")
        x = np.random.rand(32, 2)
        y = np.random.rand(32, 2)
        print(m.loss)
        ImportanceTraining(m).fit(
            x, y,
            epochs=1,
            callbacks=[ModelCheckpoint(
                path.join(self.tmpdir, "model.h5")
            )]
        )


if __name__ == "__main__":
    unittest.main()
