#!/usr/bin/env python
#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import argparse
from itertools import product
import os
from os import path


LINKS = set([".", ".."])


class Experiment(object):
    def __init__(self, root, dataset, network, sampler, score, reweighting,
                 params):
        self.root = root
        self.dataset = dataset
        self.network = network
        self.sampler = sampler
        self.score = score
        self.reweighting = reweighting
        self.params = params

    @property
    def path(self):
        return path.join(
            self.root,
            self.dataset,
            self.network,
            self.sampler,
            self.score,
            self.reweighting,
            self.params
        )

    @property
    def exists(self):
        """Return if this experiment is a valid combination of dataset,
        network, ..."""
        return path.exists(self.path)

    @property
    def started(self):
        """Has the experiment started running?"""
        return path.exists(path.join(self.path, "stdout"))

    @property
    def updated(self):
        """Check when was the last time train.txt was updated"""
        try:
            return path.getmtime(path.join(self.path, "train.txt"))
        except OSError:
            return path.getmtime(self.path)

    @property
    def epochs(self):
        """Return the number of epochs that this experiment has run"""
        cnt = 0
        try:
            with open(path.join(self.path, "val_eval.txt")) as f:
                for l in f:
                    cnt += 1
        except IOError:
            pass
        return max(cnt - 1, 0)

    def __repr__(self):
        return "Experiment(%r, %r, %r, %r, %r, %r, %r)" % (
            self.root,
            self.dataset,
            self.network,
            self.sampler,
            self.score,
            self.reweighting,
            self.params
        )


def _is_existing_dir(x):
    return path.exists(x) and path.isdir(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List the importance sampling experiments"
    )
    parser.add_argument(
        "--expdir",
        default="."
    )
    parser.add_argument(
        "--latest", "-t",
        action="store_true",
        help="Sort based on time last updated"
    )
    parser.add_argument(
        "--pending", "-p",
        action="store_true",
        help="Show only pending (not started) experiments"
    )

    args = parser.parse_args()

    root = args.expdir
    datasets = set(
        x
        for x in os.listdir(root)
        if _is_existing_dir(path.join(root, x))
    ) - LINKS
    networks = set(sum([
        os.listdir(path.join(root, *x))
        for x in product(datasets)
        if _is_existing_dir(path.join(root, *x))
    ], [])) - LINKS
    samplers = set(sum([
        os.listdir(path.join(root, *x))
        for x in product(datasets, networks)
        if _is_existing_dir(path.join(root, *x))
    ], [])) - LINKS
    scores = set(sum([
        os.listdir(path.join(root, *x))
        for x in product(datasets, networks, samplers)
        if _is_existing_dir(path.join(root, *x))
    ], [])) - LINKS
    reweightings = set(sum([
        os.listdir(path.join(root, *x))
        for x in product(datasets, networks, samplers, scores)
        if _is_existing_dir(path.join(root, *x))
    ], [])) - LINKS
    params = set(sum([
        os.listdir(path.join(root, *x))
        for x in product(datasets, networks, samplers, scores, reweightings)
        if _is_existing_dir(path.join(root, *x))
    ], [])) - LINKS

    experiments = filter(
        lambda e: e.exists and e.started != args.pending,
        [
            Experiment(root, *x)
            for x in product(datasets, networks, samplers, scores,
                             reweightings, params)
        ]
    )

    if args.latest:
        experiments.sort(key=lambda e: e.updated)
    else:
        experiments.sort(key=lambda e: e.path)

    print "Dataset\tNetwork\tSampler\tScore\tReweighting\tParams\tEpochs"
    for e in experiments:
        print "\t".join([
            e.dataset,
            e.network,
            e.sampler,
            e.score,
            e.reweighting,
            e.params,
            str(e.epochs)
        ])
