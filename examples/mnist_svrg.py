"""Trains a simple logistic regression on the MNIST dataset."""

from __future__ import print_function

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import SGD

from importance_sampling.training import SVRG
from example_utils import get_parser

if __name__ == "__main__":
    parser = get_parser("Train logistic regression with SVRG on MNIST")
    args = parser.parse_args()

    batch_size = 16
    num_classes = 10
    epochs = 100

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(10, input_shape=(784,)))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.1, momentum=0),
                  metrics=['accuracy'])

    sgd_epochs = epochs
    svrg_epochs = 0
    svrg_wrapped = SVRG(model, B=0, B_over_b=300)
    if args.importance_training:
        sgd_epochs = 20
        svrg_epochs = (epochs-20) // (2 + len(x_train)//(batch_size*300))
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=sgd_epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    history = svrg_wrapped.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=svrg_epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
