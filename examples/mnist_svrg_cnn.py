"""Trains a simple convnet on the MNIST dataset."""

from __future__ import print_function

import time

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras import backend as K

from importance_sampling.training import ConstantTimeImportanceTraining, SVRG
from example_utils import get_parser

if __name__ == "__main__":
    batch_size = 128
    num_classes = 10
    epochs = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     kernel_regularizer=l2(1e-5),
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1e-5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-5)))
    model.add(Dense(num_classes, kernel_regularizer=l2(1e-5)))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
                  metrics=['accuracy'])

    # Keep the initial weights to compare
    W = model.get_weights()

    # Train with SVRG
    s_svrg = time.time()
    model.set_weights(W)
    SVRG(model, B=0, B_over_b=len(x_train) // batch_size).fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    e_svrg = time.time()
    score_svrg = model.evaluate(x_test, y_test, verbose=0)

    # Train with uniform
    s_uniform = time.time()
    model.set_weights(W)
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    e_uniform = time.time()
    score_uniform = model.evaluate(x_test, y_test, verbose=0)

    # Train with IS
    s_is = time.time()
    model.set_weights(W)
    ConstantTimeImportanceTraining(model).fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    e_is = time.time()
    score_is = model.evaluate(x_test, y_test, verbose=0)

    # Print the results
    print("SVRG: ", score_svrg[1], " in ", e_svrg - s_svrg, "s")
    print("Uniform: ", score_uniform[1], " in ", e_uniform - s_uniform, "s")
    print("IS: ", score_is[1], " in ", e_is - s_is, "s")
