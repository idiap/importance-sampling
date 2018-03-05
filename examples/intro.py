
# Import what is needed to build the Keras model
from keras import backend as K
from keras.layers import Activation, Dense
from keras.models import Sequential

# Import a toy dataset and the importance training
from importance_sampling.datasets import CanevetICML2016
from importance_sampling.training import ImportanceTraining


def create_nn():
    """Build a simple fully connected NN"""
    model = Sequential([
        Dense(40, activation="tanh", input_shape=(2,)),
        Dense(40, activation="tanh"),
        Dense(1),
        Activation("sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    # Load the data
    dataset = CanevetICML2016(N=1024)
    x_train, y_train = dataset.train_data[:]
    x_test, y_test = dataset.test_data[:]
    y_train, y_test = y_train.argmax(axis=1), y_test.argmax(axis=1)

    # Create the NN and keep the initial weights
    model = create_nn()
    weights = model.get_weights()

    # Train with uniform sampling
    K.set_value(model.optimizer.lr, 0.01)
    model.fit(
        x_train, y_train,
        batch_size=64, epochs=10,
        validation_data=(x_test, y_test)
    )

    # Train with biased importance sampling
    model.set_weights(weights)
    K.set_value(model.optimizer.lr, 0.01)
    ImportanceTraining(model, presample=10, tau_th=2.5).fit(
        x_train, y_train,
        batch_size=64, epochs=10,
        validation_data=(x_test, y_test)
    )
