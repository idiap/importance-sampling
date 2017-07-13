#!/usr/bin/env python
#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import argparse
import time

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import seaborn as sns


def lines(f, delim):
    while True:
        line = f.readline()
        if line == "":
            break
        yield map(float, line.strip().split(delim))


def update(data, ax, xlim, ylim, vl):
    ax.clear()
    sns.distplot(data, ax=ax)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if vl is not None:
        ax.plot([vl, vl], ax.get_ylim(), "k--")

    return ax


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(
        description="Plot the distribution of values in each line"
    )
    parser.add_argument(
        "--delimiter", "-d",
        default=" ",
        help="Define the field delimiter"
    )
    parser.add_argument(
        "--xlim",
        type=lambda x: None if not x else map(float, x.split(",")),
        default="",
        help="Specific limits for the x axis"
    )
    parser.add_argument(
        "--ylim",
        type=lambda x: None if not x else map(float, x.split(",")),
        default="",
        help="Specific limits for the y axis"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames expected"
    )
    parser.add_argument(
        "--to_file",
        help="Save the animation as a video to that file"
    )
    parser.add_argument(
        "--vline",
        type=float,
        help="Plot a vertical line"
    )

    args = parser.parse_args(sys.argv[1:])

    data_gen = lines(sys.stdin, args.delimiter)

    fig, ax = plt.subplots()
    anim = FuncAnimation(
        fig,
        update,
        data_gen,
        fargs=(ax, args.xlim, args.ylim, args.vline),
        interval=100,
        save_count=args.frames
    )
    if args.to_file:
        anim.save(args.to_file)
    else:
        plt.show()
