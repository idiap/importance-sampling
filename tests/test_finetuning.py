#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from keras.applications import Xception
from keras.layers import Dense
from keras.models import Model
import numpy as np

from importance_sampling.training import ImportanceTraining


class TestFinetuning(unittest.TestCase):
    def _generate_images(self, batch_size=16):
        while True:
            xi = np.random.rand(batch_size, 71, 71, 3)
            yi = np.zeros((batch_size, 10))
            yi[np.arange(batch_size), np.random.choice(10, batch_size)] = 1.0
            yield xi, yi

    def test_simple_cifar(self):
        base = Xception(
            input_shape=(71, 71, 3),
            include_top=False,
            pooling="avg"
        )
        y = Dense(10, activation="softmax")(base.output)
        model = Model(base.input, y)
        model.compile("sgd", "categorical_crossentropy", metrics=["accuracy"])

        history = ImportanceTraining(model, presample=16).fit_generator(
            self._generate_images(batch_size=8),
            steps_per_epoch=10,
            epochs=1,
            batch_size=8
        )
        self.assertTrue("loss" in history.history)
        self.assertTrue("accuracy" in history.history)
        self.assertEqual(len(history.history["loss"]), 1)
        self.assertEqual(len(history.history["accuracy"]), 1)


if __name__ == "__main__":
    unittest.main()
