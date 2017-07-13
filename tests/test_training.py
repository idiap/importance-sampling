#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from keras.layers import Dense
from keras.models import Sequential
import numpy as np

from importance_sampling.training import ImportanceTraining, \
    ApproximateImportanceTraining


class TestTraining(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.model = Sequential([
            Dense(10, activation="relu", input_shape=(2,)),
            Dense(10, activation="relu"),
            Dense(2)
        ])
        self.model.compile("sgd", "mse", metrics=["mae"])

        super(TestTraining, self).__init__(*args, **kwargs)

    def test_simple_training(self):
        for Training in [ImportanceTraining, ApproximateImportanceTraining]:
            model = Training(self.model)
            x = np.random.rand(128, 2)
            y = np.random.rand(128, 2)

            history = model.fit(x, y, epochs=5)
            self.assertTrue("loss" in history.history)
            self.assertEqual(len(history.history["loss"]), 5)


if __name__ == "__main__":
    unittest.main()
