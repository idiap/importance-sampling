"""Trains a ResNet on the CIFAR10 dataset.

ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf

ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function

import time

from keras import backend as K
from keras.callbacks import LearningRateScheduler, Callback
from keras.datasets import cifar10
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, \
    GlobalAveragePooling2D, Input, add
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import to_categorical
import numpy as np

from importance_sampling.layers import LayerNormalization
from importance_sampling.training import ImportanceTraining
from example_utils import get_parser


class TrainingSchedule(Callback):
    """Implement the training schedule for training a resnet on CIFAR10 for a
    given time budget."""
    def __init__(self, total_time):
        self._total_time = total_time
        self._lr = self._get_lr(0.0)

    def _get_lr(self, progress):
        if progress > 0.8:
            return 1e-4
        elif progress > 0.6:
            return 1e-3
        else:
            return 1e-2

    def on_train_begin(self, logs={}):
        self._start = time.time()
        self._lr = self._get_lr(0.0)
        K.set_value(self.model.optimizer.lr, self._lr)

    def on_batch_end(self, batch, logs):
        t = time.time() - self._start

        if t >= self._total_time:
            self.model.stop_training = True

        lr = self._get_lr(t / self._total_time)
        if lr != self._lr:
            self._lr = lr
            K.set_value(self.model.optimizer.lr, self._lr)

    @property
    def lr(self):
        return self._lr


def resnet(depth, batch_norm=False, l2_reg=1e-4):
    def layer(x_in, filters, kernel_size, strides, norm, activation):
        conv = Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(l2_reg)
        )

        x = conv(x_in)
        if norm:
            if batch_norm:
                x = BatchNormalization()(x)
            else:
                x = LayerNormalization()(x)
        if activation:
            x = Activation("relu")(x)

        return x

    def implementation(input_shape, output_size):
        # How many blocks per stage
        blocks = int((depth - 2) / 6)
        # Initial number of filters
        filters = 64

        # Build the layers
        x = x_in = Input(shape=input_shape)
        x = layer(x, filters, 3, 1, True, True)
        for stage in range(3):
            for block in range(blocks):
                strides = 1
                if stage > 0 and block == 0:
                    strides = 2
                y = layer(x, filters, 3, strides, True, True)
                y = layer(y, filters, 3, 1, True, False)
                if strides > 1:
                    x = layer(x, filters, 1, strides, False, False)

                x = add([x, y])
                x = Activation("relu")(x)
            filters *= 2

        # Add the classifier
        y = GlobalAveragePooling2D()(x)
        y = Dense(output_size, kernel_initializer="he_normal")(y)
        y = Activation("softmax")(y)

        # Create the model
        return Model(x_in, y)
    return implementation


if __name__ == "__main__":
    parser = get_parser("Train a ResNet on CIFAR10")
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Choose the depth of the resnet"
    )
    parser.add_argument(
        "--presample",
        type=float,
        default=3.0,
        help="Presample that many times the batch size for importance sampling"
    )
    parser.add_argument(
        "--bn",
        action="store_true",
        help="Use BatchNormalization instead of LayerNormalization"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Choose the size of the minibatch"
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        default=3600*3,
        help="How many seconds to train for"
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1e-4,
        help="Define the L2 regularization for the network"
    )
    args = parser.parse_args()

    # Load the data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255
    mu = x_train.mean(axis=0)
    x_train -= mu
    x_test -= mu
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build the model
    training_schedule = TrainingSchedule(args.time_budget)
    model = resnet(args.depth, args.bn, args.l2)(x_train.shape[1:], 10)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(lr=training_schedule.lr, momentum=0.9),
        metrics=["accuracy"]
    )
    model.summary()

    # Create the data augmentation generator
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False)
    datagen.fit(x_train)

    # Train the model
    if args.importance_training:
        ImportanceTraining(model).fit_generator(
            datagen.flow(x_train, y_train, batch_size=args.batch_size),
            validation_data=(x_test, y_test),
            epochs=10**6,
            verbose=1,
            callbacks=[training_schedule],
            batch_size=args.batch_size,
            steps_per_epoch=int(np.ceil(float(len(x_train)) / args.batch_size))
        )
    else:
        model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=args.batch_size),
            validation_data=(x_test, y_test),
            epochs=10**6,
            verbose=1,
            callbacks=[training_schedule]
        )

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
