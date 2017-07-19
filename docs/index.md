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
* will *not necessarily* result in wall clock speedup

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
from keras.layers import Dense, Dropout

from importance_sampling.training import ImportanceTraining

# Load mnist
...
...

# Build your NN normally
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

# Train with importance sampling
history = ImportanceTraining(model).fit(
    x_train, y_train,
    batch_size=128, epochs=3,
    verbose=1,
    validation_data=(x_test, y_test)
)
```

## Installation

Importance sampling has the following dependencies:

* Keras >= 2
* numpy
* transparent-keras
* blinker

You can install it from PyPI with:

```bash
pip install --user keras-importance-sampling
```

## Research

In case you want theoretical and empirical evidence regarding Importance
Sampling and Deep Learning we encourage you to read our paper [Biased
Importance Sampling for Deep Neural Network Training][our_paper] and cite it if
you want using the following bibtex entry.

```bibtex
@article{katharopoulos2017is,
    Author = {Katharopoulos, Angelos and Fleuret, Fran\c{c}ois},
    Journal = {arXiv preprint arXiv:1706.00043},
    Title = {Biased Importance Sampling for Deep Neural Network Training},
    Year = {2017}
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
[our_paper]: https://arxiv.org/abs/1706.00043
[zhao_zhang]: http://www.jmlr.org/proceedings/papers/v37/zhaoa15.pdf
[distributed_is]: https://arxiv.org/pdf/1511.06481
[lic]: https://github.com/idiap/importance-sampling/blob/master/LICENSE
