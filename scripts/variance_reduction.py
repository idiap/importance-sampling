#!/usr/bin/env python
#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import argparse
import os
from os import path

from keras import backend as K
from keras.losses import get as get_loss
from keras.utils.generic_utils import Progbar
import numpy as np

from importance_sampling import models
from importance_sampling.datasets import CIFAR10, CIFAR100, MNIST, \
    OntheflyAugmentedImages, ImageNetDownsampled
from importance_sampling.model_wrappers import OracleWrapper
from importance_sampling.reweighting import BiasedReweightingPolicy
from importance_sampling.utils import tf_config
from importance_sampling.utils.functional import compose, partial, ___


def build_grad(network):
    """Return the gradient of the network."""
    x = network.input
    y = network.output
    y_true = K.placeholder(shape=K.int_shape(y))
    sample_weights = K.placeholder(shape=(None,))

    l = K.mean(sample_weights * get_loss(network.loss)(y_true, y))
    grads = network.optimizer.get_gradients(l, network.trainable_weights)
    grad = K.concatenate([
        K.reshape(g, (-1,))
        for g in grads
    ])

    return K.function(
        [x, y_true, sample_weights],
        [grad]
    )


def build_grad_batched(network, batch_size):
    """Compute the average gradient by splitting the inputs in batches of size
    'batch_size' and averaging."""
    grad = build_grad(network)
    def inner(inputs):
        X, y, w = inputs
        N = len(X)
        g = 0
        for i in range(0, N, batch_size):
            g = g + w[i:i+batch_size].sum() * grad([
                X[i:i+batch_size],
                y[i:i+batch_size],
                w[i:i+batch_size]
            ])[0]
        return [g / w.sum()]

    return inner


def load_dataset(dataset):
    datasets = {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "cifar10-augmented": compose(
            partial(OntheflyAugmentedImages, ___, dict(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False
            )),
            CIFAR10
        ),
        "cifar100-augmented": compose(
            partial(OntheflyAugmentedImages, ___, dict(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False
            )),
            CIFAR100
        ),
        "imagenet-32x32": partial(
            ImageNetDownsampled,
            os.getenv("IMAGENET"),
            size=32
        ),
    }

    return datasets[dataset]()


def uniform_score(x, y, batch_size=None):
    return np.ones((len(x),))


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the variance reduction achieved by different "
                     "importance sampling methods")
    )

    parser.add_argument(
        "model",
        choices=[
            "small_cnn", "cnn", "wide_resnet_28_2"
        ],
        help="Choose the type of the model"
    )
    parser.add_argument(
        "weights",
        help="The file containing the model weights"
    )
    parser.add_argument(
        "dataset",
        choices=[
            "mnist", "cifar10", "cifar100", "cifar10-augmented",
            "cifar100-augmented", "imagenet-32x32"
        ],
        help="Choose the dataset to compute the loss"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="How many samples to choose"
    )
    parser.add_argument(
        "--score",
        choices=["gnorm", "full_gnorm", "loss", "ones"],
        nargs="+",
        default="loss",
        help="Choose a score to perform sampling with"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The batch size for computing the loss"
    )
    parser.add_argument(
        "--inner_batch_size",
        type=int,
        default=32,
        help=("The batch size to use for gradient computations "
              "(to decrease memory usage)")
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1024,
        help="The sample size to compute the variance reduction"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="A seed for the PRNG (mainly used for dataset generation)"
    )
    parser.add_argument(
        "--save_scores",
        help="Directory to save the scores in"
    )

    args = parser.parse_args(argv)

    np.random.seed(args.random_seed)

    dataset = load_dataset(args.dataset)
    network = models.get(args.model)(dataset.shape, dataset.output_size)
    network.load_weights(args.weights)
    grad = build_grad_batched(network, args.inner_batch_size)
    reweighting = BiasedReweightingPolicy()

    # Compute the full gradient
    idxs = np.random.choice(len(dataset.train_data), args.sample_size)
    x, y = dataset.train_data[idxs]
    full_grad = grad([x, y, np.ones(len(x))])[0]

    # Sample and approximate
    for score_metric in args.score:
        if score_metric != "ones":
            model = OracleWrapper(network, reweighting, score=score_metric)
            score = model.score
        else:
            score = uniform_score
        gs = np.zeros(shape=(10,) + full_grad.shape, dtype=np.float32)
        print "Calculating %s..." % (score_metric,)
        scores = score(x, y, batch_size=1)
        p = scores/scores.sum()
        pb = Progbar(args.samples)
        for i in range(args.samples):
            pb.update(i)
            idxs = np.random.choice(args.sample_size, args.batch_size, p=p)
            w = reweighting.sample_weights(idxs, scores).ravel()
            gs[i] = grad([x[idxs], y[idxs], w])[0]
        pb.update(args.samples)
        norms = np.sqrt(((full_grad - gs)**2).sum(axis=1))
        alignment = gs.dot(full_grad[:, np.newaxis]) / np.sqrt(np.sum(full_grad**2))
        alignment /= np.sqrt((gs**2).sum(axis=1, keepdims=True))
        print "Mean of norms of diff", np.mean(norms)
        print "Variance of norms of diff", np.var(norms)
        print "Mean of alignment", np.mean(alignment)
        print "Variance of alignment", np.var(alignment)
        if args.save_scores:
            np.savetxt(
                path.join(args.save_scores, score_metric+".txt"),
                scores
            )


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
