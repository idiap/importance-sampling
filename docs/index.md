# Importance Sampling for Keras

Deep learning models spend countless GPU/CPU cycles on trivial, correctly
classified examples that do not individually affect the parameters. For
instance, even a very simple neural network achieves ~98% accuracy on MNIST
after a single epoch.

Importance sampling focuses the computation to informative/important samples
(by sampling mini-batches from a distribution other than uniform) thus
accelerating the convergence.

This library:

* wraps Keras models requiring just **one line changed** to try out *Importance Sampling*
* comes with modified Keras examples for quick and dirty comparison
* is the result of ongoing research which means that *your mileage may vary*

## Quick-start

The main API that is provided is that of
`importance_sampling.training.ImportanceTraining`. The library uses composition
to seamlessly wrap your Keras models and perform importance sampling behind the
scenes.

*The following example can also be found in the [examples][github_examples]
folder (mnist_mlp.py).*

```python
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

from importance_sampling.training import ImportanceTraining

# Load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255
y_train = np.eye(10).astype(np.float32)[y_train]
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255
y_test = np.eye(10).astype(np.float32)[y_test]

# Build your NN normally
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10))
model.add(Activation("softmax"))
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

# Train with importance sampling
history = ImportanceTraining(model).fit(
    x_train, y_train,
    batch_size=128, epochs=5,
    verbose=1,
    validation_data=(x_test, y_test)
)
```

## Installation

Importance sampling has the following dependencies:

* Keras >= 2
* numpy
* blinker

You can install it from PyPI with:

```bash
pip install --user keras-importance-sampling
```

## Research

In case you want theoretical and empirical evidence regarding Importance
Sampling and Deep Learning we encourage you to follow our research.

1. [Not All Samples Are Created Equal: Deep Learning with Importance Sampling (2018)][nasace]
2. [Biased Importance Sampling for Deep Neural Network Training (2017)][biased_is]

```bibtex
@article{katharopoulos2018is,
    Author = {Katharopoulos, Angelos and Fleuret, Fran\c{c}ois},
    Journal = {arXiv preprint arXiv:1803.00942},
    Title = {Not All Samples Are Created Equal: Deep Learning with Importance
        Sampling},
    Year = {2018}
}
```

Moreover we suggest you look into the following highly related and influential papers:

* Stochastic optimization with importance sampling for regularized loss
  minimization [[pdf][zhao_zhang]]
* Variance reduction in SGD by distributed importance sampling [[pdf][distributed_is]]

## Support, License and Copyright

This software is distributed with the **MIT** license which pretty much means
that you can use it however you want and for whatever reason you want. All the
information regarding support, copyright and the license can be found in the
[LICENSE][lic] file in the repository.

[github_examples]: https://github.com/idiap/importance-sampling/tree/master/examples
[nasace]: https://arxiv.org/abs/1803.00942
[biased_is]: https://arxiv.org/abs/1706.00043
[zhao_zhang]: http://www.jmlr.org/proceedings/papers/v37/zhaoa15.pdf
[distributed_is]: https://arxiv.org/pdf/1511.06481
[lic]: https://github.com/idiap/importance-sampling/blob/master/LICENSE
