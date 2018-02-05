#!/usr/bin/env python

"""Evaluate the embedding performance in the lfw dataset"""

import argparse
import os

from importance_sampling.datasets import LFW
import numpy as np


def compute_distances(representations, pairs):
    X = representations[pairs[:, 0], :]
    Y = representations[pairs[:, 1], :]

    X = X / np.sqrt((X**2).sum(axis=1, keepdims=True))
    Y = Y / np.sqrt((Y**2).sum(axis=1, keepdims=True))

    return np.sum((X - Y)**2, axis=1, keepdims=True)


def compute_threshold(distances, matches):
    a, b = 0.0, distances.max()
    N = len(distances)
    while b-a > 1e-5:
        m = (a + b)/2
        t1 = (a + m)/2
        t2 = (m + b)/2

        n0 = np.sum((distances < m).astype(float) == matches)
        n1 = np.sum((distances < t1).astype(float) == matches)
        n2 = np.sum((distances < t2).astype(float) == matches)

        if n0 > n1 and n0 > n2:
            a, b = t1, t2
        elif n1 > n0 and n0 >= n2:
            b = m
        elif n2 > n0 and n0 >= n1:
            a = m
        else:
            return m

    return (a+b)/2


def evaluate(representations, dataset):
    pairs_train, matches_train = dataset.train_data[:]
    distances = compute_distances(representations, pairs_train)
    t = compute_threshold(distances, matches_train)

    pairs_test, matches_test = dataset.test_data[:]
    distances = compute_distances(representations, pairs_test)

    return ((distances < t).astype(float) == matches_test).astype(float).mean()


def main(argv):
    parser = argparse.ArgumentParser(
        description="Evaluate a representation on the lfw dataset"
    )

    parser.add_argument(
        "representations",
        help="The representations of the faces of LFW"
    )

    parser.add_argument(
        "--embedding",
        type=int,
        default=128,
        help="Choose the dimensionality of the representation"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="How many folds does the dataset have"
    )
    parser.add_argument(
        "--dataset_path",
        default=os.getenv("LFW", ""),
        help="The basepath of the LFW dataset"
    )

    args = parser.parse_args(argv)

    print "Loading representations..."
    representations = np.fromfile(args.representations, dtype=np.float32)
    representations = representations.reshape(-1, args.embedding)

    for fold in range(1, args.folds+1):
        dataset = LFW(args.dataset_path, fold=fold, idxs=True)
        print "Fold {}: {}".format(fold, evaluate(representations, dataset))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
