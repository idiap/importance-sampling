#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

from keras import backend as K
from keras.applications import ResNet50
from keras.layers import    \
    Activation,             \
    AveragePooling2D,       \
    Convolution2D,          \
    Dense,                  \
    Dropout,                \
    ELU,                    \
    Embedding,              \
    Flatten,                \
    GlobalAveragePooling2D, \
    Input,                  \
    LSTM,                   \
    MaxPooling2D,           \
    Masking,                \
    TimeDistributed,        \
    add
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.regularizers import l2

from .layers import       \
    BatchRenormalization, \
    Bias,                 \
    LayerNormalization,   \
    StatsBatchNorm


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
        BatchRenormalization(),
        Convolution2D(64, **kwargs),
        BatchRenormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # conv2_*
        Convolution2D(128, **kwargs),
        BatchRenormalization(),
        Convolution2D(128, **kwargs),
        BatchRenormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # conv3_*
        Convolution2D(256, **kwargs),
        BatchRenormalization(),
        Convolution2D(256, **kwargs),
        BatchRenormalization(),
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

        # conv2_*
        Convolution2D(64, kernel_size=3, padding="same"),
        Activation("relu"),
        Convolution2D(64, kernel_size=3, padding="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),

        # Fully connected
        Flatten(),
        Dense(512),
        Activation("relu"),
        Dense(512),
        Activation("relu"),
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
        BatchRenormalization(),
        Convolution2D(96, 3, 3, **kwargs),
        BatchRenormalization(),
        Convolution2D(96, 3, 3, subsample=(2, 2), **kwargs),
        BatchRenormalization(),
        Dropout(0.25),

        # conv2
        Convolution2D(192, 3, 3, **kwargs),
        BatchRenormalization(),
        Convolution2D(192, 3, 3, **kwargs),
        BatchRenormalization(),
        Convolution2D(192, 3, 3, subsample=(2, 2), **kwargs),
        BatchRenormalization(),
        Dropout(0.25),

        # conv3
        Convolution2D(192, 1, 1, **kwargs),
        BatchRenormalization(),
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
        LSTM(256, unroll=True),
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


def build_lstm_timit(input_shape, output_size):
    """Build a simple LSTM to classify the phonemes in the TIMIT dataset"""
    model = Sequential([
        LSTM(256, unroll=True, input_shape=input_shape),
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


def wide_resnet(L, k, drop_rate=0.0):
    """Implement the WRN-L-k from 'Wide Residual Networks' BMVC 2016"""
    def wide_resnet_impl(input_shape, output_size):
        def conv(channels, strides,
                 params=dict(padding="same", use_bias=False,
                             kernel_regularizer=l2(5e-4))):
            def inner(x):
                x = LayerNormalization()(x)
                x = Bias()(x)
                x = Activation("relu")(x)
                x = Convolution2D(channels, 3, strides=strides, **params)(x)
                x = Dropout(drop_rate)(x) if drop_rate > 0 else x
                x = LayerNormalization()(x)
                x = Bias()(x)
                x = Activation("relu")(x)
                x = Convolution2D(channels, 3, **params)(x)
                return x
            return inner

        def resize(x, shape):
            if K.int_shape(x) == shape:
                return x
            channels = shape[3 if K.image_data_format() == "channels_last" else 1]
            strides = K.int_shape(x)[2] // shape[2]
            return Convolution2D(
                channels, 1, padding="same", use_bias=False, strides=strides
            )(x)

        def block(channels, k, n, strides):
            def inner(x):
                for i in range(n):
                    x2 = conv(channels*k, strides if i==0 else 1)(x)
                    x = add([resize(x, K.int_shape(x2)), x2])
                return x
            return inner

        # According to the paper L = 6*n+4
        n = (L-4)/6

        group0 = Convolution2D(16, 3, padding="same", use_bias=False,
                                      kernel_regularizer=l2(5e-4))
        group1 = block(16, k, n, 1)
        group2 = block(32, k, n, 2)
        group3 = block(64, k, n, 2)

        x_in = x = Input(shape=input_shape)
        x = group0(x)
        x = group1(x)
        x = group2(x)
        x = group3(x)

        x = LayerNormalization()(x)
        x = Bias()(x)
        x = Activation("relu")(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(output_size, kernel_regularizer=l2(5e-4))(x)
        y = Activation("softmax")(x)

        model = Model(inputs=x_in, outputs=y)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        return model
    return wide_resnet_impl


def resnet_50(norm_layer=StatsBatchNorm, weights="imagenet", flatten=False):
    def resnet_impl(input_shape, output_size):
        def block(x_in, kernel, filters, strides, stage, block, resize=False):
            name = "%d%s_branch" % (stage, block)
            x = Convolution2D(filters[0], 1, strides=strides,
                              name="res"+name+"2a")(x_in)
            x = norm_layer(name="bn"+name+"2a")(x)
            x = Activation("relu")(x)
            x = Convolution2D(filters[1], kernel, padding="same",
                              name="res"+name+"2b")(x)
            x = norm_layer(name="bn"+name+"2b")(x)
            x = Activation("relu")(x)
            x = Convolution2D(filters[2], 1, name="res"+name+"2c")(x)
            x = norm_layer(name="bn"+name+"2c")(x)

            if resize:
                shortcut = Convolution2D(filters[2], 1, strides=strides,
                              name="res"+name+"1")(x_in)
                shortcut = norm_layer(name="bn"+name+"1")(shortcut)
            else:
                shortcut = x_in

            return Activation("relu")(add([x, shortcut]))

        x_in = Input(shape=input_shape)

        x = Convolution2D(64, 7, strides=2, padding="same", name="conv1")(x_in)
        x = norm_layer(name="bn_conv1")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = block(x, 3, [64, 64, 256], 1, 2, 'a', True)
        x = block(x, 3, [64, 64, 256], 1, 2, 'b')
        x = block(x, 3, [64, 64, 256], 1, 2, 'c')

        x = block(x, 3, [128, 128, 512], 2, 3, 'a', True)
        x = block(x, 3, [128, 128, 512], 1, 3, 'b')
        x = block(x, 3, [128, 128, 512], 1, 3, 'c')
        x = block(x, 3, [128, 128, 512], 1, 3, 'd')

        x = block(x, 3, [256, 256, 1024], 2, 4, 'a', True)
        x = block(x, 3, [256, 256, 1024], 1, 4, 'b')
        x = block(x, 3, [256, 256, 1024], 1, 4, 'c')
        x = block(x, 3, [256, 256, 1024], 1, 4, 'd')
        x = block(x, 3, [256, 256, 1024], 1, 4, 'e')
        x = block(x, 3, [256, 256, 1024], 1, 4, 'f')

        x = block(x, 3, [512, 512, 2048], 2, 5, 'a', True)
        x = block(x, 3, [512, 512, 2048], 1, 5, 'b')
        x = block(x, 3, [512, 512, 2048], 1, 5, 'c')

        x = AveragePooling2D((7, 7), name="avg_pool")(x)

        if flatten:
            x = Flatten()(x)
        else:
            x = GlobalAveragePooling2D()(x)

        x = Dense(output_size, name="fc"+str(output_size))(x)
        y = Activation("softmax")(x)

        model = Model(x_in, y, name="resnet50")
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy", "top_k_categorical_accuracy"]
        )

        if weights == "imagenet":
            nn = ResNet50()
            model.set_weights(
                nn.get_weights()[:-2] + model.get_weights()[-2:]
            )

        return model

    return resnet_impl


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
        "lstm_timit": build_lstm_timit,
        "wide_resnet_16_4": wide_resnet(16, 4),
        "wide_resnet_16_4_dropout": wide_resnet(16, 4, 0.3),
        "wide_resnet_28_2": wide_resnet(28, 2),
        "wide_resnet_28_10": wide_resnet(28, 10),
        "wide_resnet_28_10_dropout": wide_resnet(28, 10, 0.3),
        "pretrained_resnet50": resnet_50(flatten=True)
    }
    return models[name]
