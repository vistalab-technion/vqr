from typing import Dict, Tuple, Optional

import numpy as np
from numpy import array, ndarray
from numpy.random import Generator, shuffle


def _gen_random_mvn_data(n: int, d: int, rng) -> ndarray:
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

    X = rng.uniform(size=(n, k))
    A = rng.random(size=(k, d))
    N = _gen_random_mvn_data(n, d, rng)

    Y = X @ A + N
    return X, Y


def split_train_calib_test(
    X: array, Y: array, split_ratios: Tuple[float, float]
) -> Dict[str, Tuple[array, array]]:
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
