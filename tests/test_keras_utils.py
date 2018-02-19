
from tempfile import NamedTemporaryFile
import unittest

from keras.layers import Dense, Input
from keras.models import Model, Sequential
import numpy as np

from importance_sampling.utils import keras_utils


class TestKerasUtils(unittest.TestCase):
    def test_simple_save_load(self):
        f = NamedTemporaryFile()
        m = Sequential([
            Dense(10, input_dim=2),
            Dense(2)
        ])
        m.save(f.name)

        w = m.get_weights()
        updates = keras_utils.load_weights_by_name(f.name, m)

        self.assertEqual(4, len(updates))
        for w1, w2 in zip(w, m.get_weights()):
            self.assertTrue(np.allclose(w1, w2))

    def test_nested_save_load(self):
        f = NamedTemporaryFile()
        m = Sequential([
            Dense(10, input_dim=2),
            Dense(2)
        ])
        x1, x2 = Input(shape=(2,)), Input(shape=(2,))
        y1, y2 = m(x1), m(x2)
        model = Model([x1, x2], [y1, y2])
        model.save(f.name)

        w = m.get_weights()
        updates = keras_utils.load_weights_by_name(f.name, m)

        self.assertEqual(4, len(updates))
        for w1, w2 in zip(w, m.get_weights()):
            self.assertTrue(np.allclose(w1, w2))


if __name__ == "__main__":
    unittest.main()
