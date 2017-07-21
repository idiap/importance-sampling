#!/usr/bin/env python
#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import argparse

import numpy as np

from importance_sampling import models
from importance_sampling.datasets import CIFAR10, CIFAR100, MNIST, \
    OntheflyAugmentedImages, PennTreeBank
from importance_sampling.model_wrappers import OracleWrapper
from importance_sampling.utils.functional import compose, partial, ___



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
        "ptb": partial(PennTreeBank, 20)
    }

    return datasets[dataset]()



def main(argv):
    parser = argparse.ArgumentParser(
        description="Plot the loss distribution of a model and dataset pair"
    )

    parser.add_argument(
        "model",
        choices=[
            "small_cnn", "cnn", "lstm_lm", "lstm_lm2", "lstm_lm3",
            "small_cnn_sq"
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
            "cifar100-augmented", "ptb"
        ],
        help="Choose the dataset to compute the loss"
    )
    parser.add_argument(
        "--score",
        choices=["gnorm", "loss"],
        default="loss",
        help="Choose a score to plot"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The batch size for computing the loss"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="A seed for the PRNG (mainly used for dataset generation)"
    )

    args = parser.parse_args(argv)

    np.random.seed(args.random_seed)
    
    dataset = load_dataset(args.dataset)
    network = models.get(args.model)(dataset.shape, dataset.output_size)
    model = OracleWrapper(network, score=args.score)
    model.model.load_weights(args.weights)

    for i in range(0, dataset.train_size, args.batch_size):
        idxs = slice(i, i+args.batch_size)
        for s in model.score_batch(*dataset.train_data(idxs)):
            print s


if __name__ == "__main__":
   import sys
   main(sys.argv[1:])
