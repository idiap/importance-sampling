#!/usr/bin/env python

"""Run the forward pass on the lfw images so that we can then run the
evaluation"""

import argparse
import os

from importance_sampling.datasets import LFW
from importance_sampling.models import get as get_model
from importance_sampling.utils import tf_config, keras_utils


def make_slice(x):
    def int_or_none(x):
        try:
            return int(x)
        except ValueError:
            return None
    return slice(*list(map(int_or_none, x.split(":"))))


def batch_gen(dataset, part, batch_size):
    idxs = np.arange(len(dataset.train_data))[part]
    # Keras annoyingly requires infinite iterators
    while True:
        for i in range(0, len(idxs), batch_size):
            yield dataset.train_data[idxs[i:i+batch_size]][0]


def main(argv):
    parser = argparse.ArgumentParser(
        description="Compute representations for the lfw dataset"
    )

    parser.add_argument(
        "model",
        choices=["pretrained_resnet50"],
        help="Choose the architecture to load"
    )
    parser.add_argument(
        "weights",
        help="Load those weights"
    )
    parser.add_argument(
        "output",
        help="Save the produced representations to this file"
    )

    parser.add_argument(
        "--embedding",
        type=int,
        default=128,
        help="Choose the dimensionality of the representation"
    )
    parser.add_argument(
        "--slice",
        type=make_slice,
        default=":",
        help="Slice the dataset to get a part to transform"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="The batch size used for the forward pass"
    )
    parser.add_argument(
        "--dataset_path",
        default=os.getenv("LFW", ""),
        help="The basepath of the LFW dataset"
    )

    args = parser.parse_args(argv)
    print "Loading dataset..."
    dataset = LFW(args.dataset_path, fold=None)
    print "Loading model..."
    model = get_model(args.model)(
        dataset.shape,
        args.embedding
    )
    keras_utils.load_weights_by_name(args.weights, model)

    print "Transforming..."
    representations = model.predict_generator(
        keras_utils.DatasetSequence(
            dataset,
            part=args.slice,
            batch_size=args.batch_size
        ),
        verbose=1
    )

    print "Saving {} representations...".format(len(representations))
    representations.tofile(args.output)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
