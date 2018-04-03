Importance Sampling
====================

This python package provides a library that accelerates the training of
arbitrary neural networks created with `Keras <http://keras.io>`__ using
**importance sampling**.

.. code:: python

    # Keras imports

    from importance_sampling.training import ImportanceTraining

    x_train, y_train, x_val, y_val = load_data()
    model = create_keras_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    ImportanceTraining(model).fit(
        x_train, y_train,
        batch_size=32,
        epochs=10,
        verbose=1,
        validation_data=(x_val, y_val)
    )

    model.evaluate(x_val, y_val)

Importance sampling for Deep Learning is an active research field and this
library is undergoing development so your mileage may vary.

Relevant Research
-----------------

**Ours**

* Not All Samples Are Created Equal: Deep Learning with Importance Sampling [`preprint <https://arxiv.org/abs/1803.00942>`__]
* Biased Importance Sampling for Deep Neural Network Training [`preprint <https://arxiv.org/abs/1706.00043>`__]

**By others**

* Stochastic optimization with importance sampling for regularized loss
  minimization [`pdf <http://www.jmlr.org/proceedings/papers/v37/zhaoa15.pdf>`__]
* Variance reduction in SGD by distributed importance sampling [`pdf <https://arxiv.org/pdf/1511.06481>`__]

Dependencies & Installation
---------------------------

Normally if you already have a functional Keras installation you just need to
``pip install keras-importance-sampling``.

* ``Keras`` > 2
* A Keras backend among *Tensorflow*, *Theano* and *CNTK*
* ``blinker``
* ``numpy``
* ``matplotlib``, ``seaborn``, ``scikit-learn`` are optional (used by the plot
  scripts)

Documentation
-------------

The module has a dedicated `documentation site
<http://idiap.ch/~katharas/importance-sampling/>`__ but you can also read the
`source code <https://github.com/idiap/importance-sampling>`__ and the `examples
<https://github.com/idiap/importance-sampling/tree/master/examples>`__ to get an
idea of how the library should be used and extended.

Examples
---------

In the ``examples`` folder you can find some Keras examples that have been edited
to use importance sampling.

Code examples
*************

In this section we will showcase part of the API that can be used to train
neural networks with importance sampling.

.. code:: python

    # Import what is needed to build the Keras model
    from keras import backend as K
    from keras.layers import Dense, Activation
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
        ImportanceTraining(model, forward_batch_size=1024).fit(
            x_train, y_train,
            batch_size=64, epochs=3,
            validation_data=(x_test, y_test)
        )

Using the script
****************

The following terminal commands train a small VGG-like network to ~0.65% error
on MNIST (the numbers are from a CPU).
.. code::

    $ # Train a small cnn with mnist for 500 mini-batches using importance
    $ # sampling with bias to achieve ~ 0.65% error (on the CPU).
    $ time ./importance_sampling.py \
    >   small_cnn \
    >   oracle-gnorm \
    >   model \
    >   predicted \
    >   mnist \
    >   /tmp/is \
    >   --hyperparams 'batch_size=i128;lr=f0.003;lr_reductions=I10000' \
    >   --train_for 500 --validate_every 500
    real    1m41.985s
    user    8m14.400s
    sys     0m35.900s
    $
    $ # And with uniform sampling to achieve ~ 0.9% error.
    $ time ./importance_sampling.py \
    >   small_cnn \
    >   oracle-loss \
    >   uniform \
    >   unweighted \
    >   mnist \
    >   /tmp/uniform \
    >   --hyperparams 'batch_size=i128;lr=f0.003;lr_reductions=I10000' \
    >   --train_for 3000 --validate_every 3000
    real    9m23.971s
    user    47m32.600s
    sys     3m4.188s
