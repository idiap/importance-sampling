#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from keras.layers import Dense, Input, dot
from keras.models import Model, Sequential
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

        x1 = Input(shape=(10,))
        x2 = Input(shape=(10,))
        y = dot([
            Dense(10)(x1),
            Dense(10)(x2)
        ], axes=1)
        self.model2 = Model(inputs=[x1, x2], outputs=y)
        self.model2.compile(loss="mse", optimizer="adam")

        super(TestTraining, self).__init__(*args, **kwargs)

    def test_simple_training(self):
        for Training in [ImportanceTraining, ApproximateImportanceTraining]:
            model = Training(self.model)
            x = np.random.rand(128, 2)
            y = np.random.rand(128, 2)

            history = model.fit(x, y, epochs=5)
            self.assertTrue("loss" in history.history)
            self.assertEqual(len(history.history["loss"]), 5)

    def test_generator_training(self):
        def gen():
            while True:
                yield np.random.rand(16, 2), np.random.rand(16, 2)

        def gen2():
            while True:
                yield (np.random.rand(16, 10), np.random.rand(16, 10)), \
                    np.random.rand(16, 1)
        x_val1, y_val1 = np.random.rand(32, 2), np.random.rand(32, 2)
        x_val2, y_val2 = (np.random.rand(32, 10), np.random.rand(32, 10)), \
            np.random.rand(32, 1)

        for Training in [ImportanceTraining]:
            model = Training(self.model)
            history = model.fit_generator(
                gen(), validation_data=(x_val1, y_val1),
                steps_per_epoch=8, epochs=5
            )
            self.assertTrue("loss" in history.history)
            self.assertEqual(len(history.history["loss"]), 5)

            model = Training(self.model2)
            history = model.fit_generator(
                gen2(), validation_data=(x_val2, y_val2),
                steps_per_epoch=8, epochs=5
            )
            self.assertTrue("loss" in history.history)
            self.assertEqual(len(history.history["loss"]), 5)

        with self.assertRaises(NotImplementedError):
            ApproximateImportanceTraining(self.model).fit_generator(
                gen(), validation_data=(x_val1, y_val1),
                steps_per_epoch=8, epochs=5
            )

    def test_multiple_inputs(self):
        x1 = np.random.rand(64, 10)
        x2 = np.random.rand(64, 10)
        y = np.random.rand(64, 1)

        for Training in [ImportanceTraining]:
            model = Training(self.model2)

            history = model.fit([x1, x2], y, epochs=5, batch_size=16)
            self.assertTrue("loss" in history.history)
            self.assertEqual(len(history.history["loss"]), 5)


if __name__ == "__main__":
    unittest.main()
