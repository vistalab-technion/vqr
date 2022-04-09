from typing import Tuple

import numpy as np
import pylab as pl
from numpy import ndarray


def generate_heart() -> Tuple[ndarray, ndarray]:
    """
    Generates independent X and Y.
    X is sampled i.i.d from a uniform distribution [0, 1]
    Y is an r.v whose distribution is heart-shaped.
    :return: X, Y
    """
    # generates 2k-3k points heart
    image = 1 - pl.imread("../data/heart.png")[:, :, 2]
    image = image[::2, ::2]
    image = image / np.sum(image)
    idces = image.nonzero()
    Y = np.zeros([len(idces[0]), 2])
    Y[:, 0] = idces[0] / idces[0].max()
    Y[:, 1] = idces[1] / idces[1].max()

    rng = np.random.default_rng(None)
    X = rng.uniform(size=(Y.shape[0], 1))
    return X, Y


def generate_star() -> Tuple[ndarray, ndarray]:
    """
    Generates independent X and Y.
    X is sampled i.i.d from a uniform distribution [0, 1]
    Y is an r.v whose distribution is star-shaped.
    :return: X, Y
    """
    image = 1 - pl.imread("../data/star.jpg") / 255
    image = image[::7, ::7]
    image = image / np.sum(image)
    idces = image.nonzero()
    Y = np.zeros([len(idces[0]), 2])
    Y[:, 0] = idces[0] / idces[0].max()
    Y[:, 1] = idces[1] / idces[1].max()
    rng = np.random.default_rng(None)
    X = rng.uniform(size=(Y.shape[0], 1))
    return X, Y
