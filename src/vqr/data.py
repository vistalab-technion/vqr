from typing import Dict, Tuple, Optional

import numpy as np
import pylab as pl
from numpy import array, ndarray
from numpy.random import Generator, shuffle


def _gen_random_mvn_data(n: int, d: int, rng, random_cov: bool = False) -> ndarray:
    if (not random_cov) and d == 2:
        S = np.array([[1.0, -0.7], [-0.7, 1.0]])

    else:
        # Generate random orthonormal matrix Q
        Q, R = np.linalg.qr(rng.normal(size=(d, d)))

        # Generate positive eigenvalues
        eigs = rng.uniform(size=(d,))

        # PSD Covariance matrix, zero mean
        S = Q.T @ np.diag(eigs) @ Q

    mu = np.zeros(d)

    # Generate multivariate normal data
    Y = rng.multivariate_normal(mean=mu, cov=S, size=(n,))

    return Y


def generate_mvn_data(
    n: int, d: int, k: int, seed: Optional[int] = None
) -> Tuple[ndarray, ndarray]:
    """
    Generates a dataset of independent features and targets.
    Features are sampled i.i.d from a standard normal distribution. In case of >1
    dimensional features, each sample in the vector-valued feature is drawn i.i.d
    from standard normal distribution.
    Targets are sampled i.i.d from a multivariate normal distribution.
    :param n: Number of samples.
    :param d: Dimension of targets.
    :param k: Dimension of features.
    :param seed: Random seed to use for generation. None means don't set.
    :return: A tuple (X, Y), where X is of shape (n, k) and contains the features and Y
        is of shape (n, d) and contains the responses.
    """
    rng = np.random.default_rng(seed)

    # Generate correlated targets (Y) and uncorrelated features (X)
    Y = _gen_random_mvn_data(n, d, rng)
    X = rng.normal(size=(n, k))

    return X, Y


def generate_linear_x_y_mvn_data(
    n: int, d: int, k: int, seed: Optional[int] = None
) -> Tuple[ndarray, ndarray]:
    """
    Generates a dataset of linearly dependent features and targets.
    Features are sampled uniformly [0, 1]. Targets are obtained via
        Y = A X + N
    where A is a random matrix whose entries are sampled i.i.d from
    standard normal distribution. N is a random vector drawn from a
    multivariate normal distribution with a random covariance matrix.

    :param n: Number of samples.
    :param d: Dimension of targets.
    :param k: Dimension of features.
    :param seed: Random seed to use for generation. None means don't set.
    :return: A tuple (X, Y), where X is of shape (n, k) and contains the features and Y
        is of shape (n, d) and contains the responses.
    """
    rng = np.random.default_rng(seed)

    X = rng.uniform(size=(n, k)) - 0.5
    A = rng.random(size=(k, d))
    N = _gen_random_mvn_data(n, d, rng)

    Y = X @ A + N
    return X, Y


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


def split_train_calib_test(
    X: array, Y: array, split_ratios: Tuple[float, float], seed=42
) -> Dict[str, Tuple[array, array]]:
    """
    Randomly splits X and Y into train, calib, and test sets.

    Train set contains `split_ratio[0]*n` points.
    Calibration set contains `(split_ratio[1] - split_ratio[0])*n` points.
    Test set contains `(1-split_ratio[1])*n` points.

    :param X: features sized (n, k)
    :param Y: targets sized (n, d)
    :param split_ratios: Ratios by which to split the dataset (refer to the docstring).
    :param seed: seed the shuffle operation.
    :return: Dictionary with keys ('train', 'calib', 'test') -
    each key consists of X and Y corresponding to train, calibration and test,
    respectively.
    """
    assert X.shape[0] == Y.shape[0]
    N, _ = X.shape
    idxes = np.arange(0, N)
    shuffle(idxes)
    X_train = X[idxes[0 : int(N * split_ratios[0])], :]
    Y_train = Y[idxes[0 : int(N * split_ratios[0])], :]
    X_calib = X[idxes[int(N * split_ratios[0]) : int(N * split_ratios[1])], :]
    Y_calib = Y[idxes[int(N * split_ratios[0]) : int(N * split_ratios[1])], :]
    X_test = X[idxes[int(N * split_ratios[1]) :], :]
    Y_test = Y[idxes[int(N * split_ratios[1]) :], :]
    return {
        "train": (X_train, Y_train),
        "calib": (X_calib, Y_calib),
        "test": (X_test, Y_test),
    }
