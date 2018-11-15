#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Replace the models provided by the Keras applications module"""

from keras.layers import \
    Activation,          \
    AveragePooling2D,    \
    Conv2D,              \
    Dense,               \
    Flatten,             \
    Input,               \
    MaxPooling2D,        \
    add
from keras.models import Model
from keras.utils.data_utils import get_file

from .layers import StatsBatchNorm


RESNET50_WEIGHTS_PATH = ("https://github.com/fchollet/deep-learning-models/"
                         "releases/download/v0.2/"
                         "resnet50_weights_tf_dim_ordering_tf_kernels.h5")


def ResNet50(weights="imagenet", input_shape=(224, 224, 3), output_size=1000,
             softmax=False, norm_layer=StatsBatchNorm):
    def block(x_in, kernel, filters, strides, stage, block, shortcut=False):
        conv_name = "res" + str(stage) + block + "_branch"
        bn_name = "bn" + str(stage) + block + "_branch"

        x = Conv2D(filters[0], 1, strides=strides, name=conv_name+"2a")(x_in)
        x = norm_layer(name=bn_name+"2a")(x)
        x = Activation("relu")(x)
        x = Conv2D(filters[1], kernel, padding="same", name=conv_name+"2b")(x)
        x = norm_layer(name=bn_name+"2b")(x)
        x = Activation("relu")(x)
        x = Conv2D(filters[2], 1, name=conv_name+"2c")(x)
        x = norm_layer(name=bn_name+"2c")(x)

        if shortcut:
            s = Conv2D(filters[2], 1, strides=strides, name=conv_name+"1")(x_in)
            s = norm_layer(name=bn_name+"1")(s)
        else:
            s = x_in

        return Activation("relu")(add([x, s]))

    x_in = Input(shape=input_shape)
    x = Conv2D(64, 7, strides=2, padding="same", name="conv1")(x_in)
    x = norm_layer(name="bn_conv1")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = block(x, 3, [64, 64, 256], 1, 2, "a", shortcut=True)
    x = block(x, 3, [64, 64, 256], 1, 2, "b")
    x = block(x, 3, [64, 64, 256], 1, 2, "c")

    x = block(x, 3, [128, 128, 512], 2, 3, "a", shortcut=True)
    x = block(x, 3, [128, 128, 512], 1, 3, "b")
    x = block(x, 3, [128, 128, 512], 1, 3, "c")
    x = block(x, 3, [128, 128, 512], 1, 3, "d")

    x = block(x, 3, [256, 256, 1024], 2, 4, "a", shortcut=True)
    x = block(x, 3, [256, 256, 1024], 1, 4, "b")
    x = block(x, 3, [256, 256, 1024], 1, 4, "c")
    x = block(x, 3, [256, 256, 1024], 1, 4, "d")
    x = block(x, 3, [256, 256, 1024], 1, 4, "e")
    x = block(x, 3, [256, 256, 1024], 1, 4, "f")

    x = block(x, 3, [512, 512, 2048], 2, 5, "a", shortcut=True)
    x = block(x, 3, [512, 512, 2048], 1, 5, "b")
    x = block(x, 3, [512, 512, 2048], 1, 5, "c")

    x = AveragePooling2D((7, 7), name="avg_pool")(x)
    x = Flatten()(x)
    x = Dense(output_size, name="fc"+str(output_size))(x)
    if softmax:
        x = Activation("softmax")(x)

    model = Model(x_in, x, name="resnet50")

    if weights == "imagenet":
        weights_path = get_file(
            "resnet50_weights_tf_dim_ordering_tf_kernels.h5",
            RESNET50_WEIGHTS_PATH,
            cache_subdir="models",
            md5_hash="a7b3fe01876f51b976af0dea6bc144eb"
        )
        model.load_weights(weights_path, by_name=True)

    return model


def DenseNet121(*args, **kwargs):
    raise NotImplementedError()
