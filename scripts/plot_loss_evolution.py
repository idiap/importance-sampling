#!/usr/bin/env python
#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers as animation_writers
from matplotlib.cm import ScalarMappable
import numpy as np
from sklearn.linear_model import LinearRegression


def maybe_int(x):
    try:
        return int(x)
    except:
        return None


def file_or_stdin(x):
    if x == "-":
        return sys.stdin
    else:
        return x


def colors(x):
    x = np.array(x)
    x -= x.min()
    x /= x.max()
    x *= 255
    x = np.round(x).astype(int)
    return [
        plt.cm.viridis.colors[xx]
        for xx in x
    ]


def main(argv):
    parser = argparse.ArgumentParser(
        description="Plot the loss evolving through time"
    )

    parser.add_argument(
        "metrics",
        type=file_or_stdin,
        help="The file containing the loss"
    )

    parser.add_argument(
        "--to_file",
        help="Save the animation to a video file"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1000,
        help="Change that many datapoints in between frames"
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=10000,
        help="That many points in each frame"
    )
    parser.add_argument(
        "--frames",
        type=lambda x: slice(*map(maybe_int, x.split(":"))),
        default=":",
        help="Choose only those frames"
    )
    parser.add_argument(
        "--lim",
        type=lambda x: map(float, x.split(",")),
        help="Define the limits of the axes"
    )

    args = parser.parse_args(argv)
    loss = np.loadtxt(args.metrics)

    fig, ax = plt.subplots()
    lr = LinearRegression()
    sc = ax.scatter(loss[:args.n_points, 0], loss[:args.n_points, 1], c=colors(loss[:args.n_points, 2]))
    lims = args.lim if args.lim else [0, loss[:, 0].max()]
    ln, = ax.plot(lims, lims, "--", color="black", label="linear fit")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("$L(\cdot)$")
    ax.set_ylabel("$\hat{L}(\cdot)$")
    mappable = ScalarMappable(cmap="viridis")
    mappable.set_array(loss[:10000, 2])
    plt.colorbar(mappable)

    STEP = args.step
    N_POINTS = args.n_points
    def update(i):
        s = i*STEP
        e = s + N_POINTS
        lr.fit(loss[s:e, :1], loss[s:e, 1].ravel())
        ln.set_ydata([
            lr.intercept_.ravel(),
            lr.intercept_.ravel() + lims[1]*lr.coef_.ravel()
        ])
        ax.set_title("Showing samples %d to %d" % (s, e))
        sc.set_facecolor(colors(loss[s:e, 2]))
        sc.set_offsets(loss[s:e, :2])
        return ax, sc, ln

    anim = FuncAnimation(
        fig, update,
        interval=100,
        frames=np.arange(len(loss) / STEP)[args.frames],
        blit=False, repeat=False
    )
    if args.to_file:
        writer = animation_writers["ffmpeg"](fps=15)
        anim.save(args.to_file, writer=writer)
    else:
        plt.show()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
