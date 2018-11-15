#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from keras.models import Model
import numpy as np

from importance_sampling.models import wide_resnet
from importance_sampling.pretrained import ResNet50


class TestModels(unittest.TestCase):
    def _test_model(self, model, input_shape, output_shape,
                    loss="categorical_crossentropy"):
        B = 10
        X = np.random.rand(B, *input_shape).astype(np.float32)
        y = np.random.rand(B, *output_shape).astype(np.float32)

        # It is indeed a model
        self.assertTrue(isinstance(model, Model))

        # It can predict
        y_hat = model.predict_on_batch(X)
        self.assertEqual(y_hat.shape[1:], output_shape)

        # It can evaluate
        model.compile("sgd", loss)
        loss = model.train_on_batch(X, y)

    def test_wide_resnet(self):
        self._test_model(
            wide_resnet(28, 2)((32, 32, 3), 10),
            (32, 32, 3),
            (10,)
        )
        self._test_model(
            wide_resnet(18, 5)((32, 32, 3), 10),
            (32, 32, 3),
            (10,)
        )

    def test_pretrained_resnet50(self):
        self._test_model(
            ResNet50(),
            (224, 224, 3),
            (1000,)
        )
        self._test_model(
            ResNet50(input_shape=(200, 200, 3), output_size=10),
            (200, 200, 3),
            (10,)
        )


if __name__ == "__main__":
    unittest.main()
