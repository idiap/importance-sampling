#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from keras import backend as K
from keras.layers import    \
    Activation,             \
    BatchNormalization,     \
    Convolution2D,          \
    Dense,                  \
    Dropout,                \
    ELU,                    \
    Embedding,              \
    Flatten,                \
    GlobalAveragePooling2D, \
    Input,                  \
    LSTM,                   \
    MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD


def build_small_nn(input_shape, output_size):
    model = Sequential([
        Dense(40, activation="tanh", input_shape=input_shape),
        Dense(40, activation="tanh"),
        Dense(output_size),
        Activation("softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


def build_cnn(input_shape, output_size):
    kwargs = {
        "kernel_size": 3,
        "activation": "relu",
        "padding": "same"
    }
    model = Sequential([
        # conv1_*
        Convolution2D(64, input_shape=input_shape, **kwargs),
        BatchNormalization(),
        Convolution2D(64, **kwargs),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # conv2_*
        Convolution2D(128, **kwargs),
        BatchNormalization(),
        Convolution2D(128, **kwargs),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # conv3_*
        Convolution2D(256, **kwargs),
        BatchNormalization(),
        Convolution2D(256, **kwargs),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Fully connected
        Flatten(),
        Dense(1024),
        Activation("relu"),
        Dropout(0.5),
        Dense(512),
        Activation("relu"),
        Dropout(0.5),
        Dense(output_size),
        Activation("softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


def build_small_cnn(input_shape, output_size):
    model = Sequential([
        # conv1_*
        Convolution2D(32, kernel_size=3, padding="same",
                      input_shape=input_shape),
        Activation("relu"),
        Convolution2D(32, kernel_size=3, padding="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # conv2_*
        Convolution2D(64, kernel_size=3, padding="same"),
        Activation("relu"),
        Convolution2D(64, kernel_size=3, padding="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Fully connected
        Flatten(),
        Dense(512),
        Activation("relu"),
        Dropout(0.5),
        Dense(512),
        Activation("relu"),
        Dropout(0.5),
        Dense(output_size),
        Activation("softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


def build_lr(input_shape, output_size):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(output_size),
        Activation("softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


def build_all_conv_nn(input_shape, output_size):
    """Build a small variation of the best performing network from
    'Springenberg, Jost Tobias, et al. "Striving for simplicity: The all
     convolutional net." arXiv preprint arXiv:1412.6806 (2014)' which should
     achieve approximately 91% in CIFAR-10.
    """
    kwargs = {
        "activation": "relu",
        "border_mode": "same"
    }
    model = Sequential([
        # conv1
        Convolution2D(96, 3, 3, input_shape=input_shape, **kwargs),
        BatchNormalization(),
        Convolution2D(96, 3, 3, **kwargs),
        BatchNormalization(),
        Convolution2D(96, 3, 3, subsample=(2, 2), **kwargs),
        BatchNormalization(),
        Dropout(0.25),

        # conv2
        Convolution2D(192, 3, 3, **kwargs),
        BatchNormalization(),
        Convolution2D(192, 3, 3, **kwargs),
        BatchNormalization(),
        Convolution2D(192, 3, 3, subsample=(2, 2), **kwargs),
        BatchNormalization(),
        Dropout(0.25),

        # conv3
        Convolution2D(192, 1, 1, **kwargs),
        BatchNormalization(),
        Dropout(0.25),
        Convolution2D(output_size, 1, 1, **kwargs),
        GlobalAveragePooling2D(),
        Activation("softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(momentum=0.9),
        metrics=["accuracy"]
    )

    return model


def build_elu_cnn(input_shape, output_size):
    """Build a variation of the CNN implemented in the ELU paper.

    https://arxiv.org/abs/1511.07289
    """
    def layers(n, channels, kernel):
        return sum(
            (
                [
                    Convolution2D(
                        channels,
                        kernel_size=kernel,
                        padding="same"
                    ),
                    ELU()
                ]
                for i in range(n)
            ),
            []
        )

    model = Sequential(
        [
            Convolution2D(384, kernel_size=3, padding="same",
                          input_shape=input_shape)
        ] +
        layers(1, 384, 3) +
        [MaxPooling2D(pool_size=(2, 2))] +
        layers(1, 384, 1) + layers(1, 384, 2) + layers(2, 640, 2) +
        [MaxPooling2D(pool_size=(2, 2))] +
        layers(1, 640, 1) + layers(3, 768, 2) +
        [MaxPooling2D(pool_size=(2, 2))] +
        layers(1, 768, 1) + layers(2, 896, 2) +
        [MaxPooling2D(pool_size=(2, 2))] +
        layers(1, 896, 3) + layers(2, 1024, 2) +
        [
            MaxPooling2D(pool_size=(2, 2)),
            Convolution2D(output_size, kernel_size=1, padding="same"),
            GlobalAveragePooling2D(),
            Activation("softmax")
        ]
    )

    model.compile(
        optimizer=SGD(momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_lstm_lm(input_shape, output_size):
    # LM datasets will report the vocab_size as output_size
    vocab_size = output_size

    model = Sequential([
        Embedding(vocab_size + 1, 64, mask_zero=True,
                  input_length=input_shape[0]),
        LSTM(256, unroll=True, return_sequences=True),
        Dropout(0.5),
        LSTM(256, unroll=True),
        Dropout(0.5),
        Dense(output_size),
        Activation("softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_lstm_lm2(input_shape, output_size):
    # LM datasets will report the vocab_size as output_size
    vocab_size = output_size

    model = Sequential([
        Embedding(vocab_size + 1, 128, mask_zero=True,
                  input_length=input_shape[0]),
        LSTM(384, unroll=True, return_sequences=True),
        Dropout(0.5),
        LSTM(384, unroll=True),
        Dropout(0.5),
        Dense(output_size),
        Activation("softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_lstm_lm3(input_shape, output_size):
    # LM datasets will report the vocab_size as output_size
    vocab_size = output_size

    model = Sequential([
        Embedding(vocab_size + 1, 128, mask_zero=True,
                  input_length=input_shape[0]),
        LSTM(650, unroll=True, return_sequences=True),
        Dropout(0.5),
        LSTM(650, unroll=True),
        Dropout(0.5),
        Dense(output_size),
        Activation("softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_small_cnn_squared(input_shape, output_size):
    def squared_categorical_crossent(y_true, y_pred):
        return K.square(K.categorical_crossentropy(y_pred, y_true))

    model = build_small_cnn(input_shape, output_size)
    model.compile(
        optimizer=model.optimizer,
        loss=squared_categorical_crossent,
        metrics=model.metrics
    )

    return model


def get(name):
    models = {
        "small_nn": build_small_nn,
        "small_cnn": build_small_cnn,
        "small_cnn_sq": build_small_cnn_squared,
        "cnn": build_cnn,
        "all_conv": build_all_conv_nn,
        "elu_cnn": build_elu_cnn,
        "lstm_lm": build_lstm_lm,
        "lstm_lm2": build_lstm_lm2,
        "lstm_lm3": build_lstm_lm3,
    }
    return models[name]
