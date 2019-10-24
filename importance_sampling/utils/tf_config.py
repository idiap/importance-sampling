#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""This module aims to configure the Keras tensorflow session based on
environment variables or other sources"""

from multiprocessing import cpu_count
import os

from keras import backend as K
from .tf import tf


if K.backend() == "tensorflow":
    TF_THREADS = int(os.environ.get("TF_THREADS", cpu_count()))

    config = tf.ConfigProto(
        intra_op_parallelism_threads=TF_THREADS,
        inter_op_parallelism_threads=TF_THREADS,
        device_count={"CPU": TF_THREADS}
    )
    session = tf.Session(config=config)
    K.set_session(session)


def with_tensorflow(f):
    def inner(*args, **kwargs):
        if K.backend() == "tensorflow":
            return f(tf, *args, **kwargs)
    return inner


@with_tensorflow
def set_random_seed(tf, seed):
    return tf.set_random_seed(seed)
