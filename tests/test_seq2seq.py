#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

from keras.layers import Activation, Embedding, LSTM
from keras.models import Sequential
import numpy as np

from importance_sampling.training import ImportanceTraining


class TestSeq2Seq(unittest.TestCase):
    def test_simple_seq2seq(self):
        model = Sequential([
            Embedding(100, 32, mask_zero=True, input_length=10),
            LSTM(32, return_sequences=True),
            LSTM(10, return_sequences=True),
            Activation("softmax")
        ])
        model.compile("adam", "categorical_crossentropy")

        x = (np.random.rand(10, 10)*100).astype(np.int32)
        y = np.random.rand(10, 10, 10)
        y /= y.sum(axis=-1, keepdims=True)
        ImportanceTraining(model).fit(x, y, batch_size=10)


if __name__ == "__main__":
    unittest.main()
