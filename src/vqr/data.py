from typing import Tuple, Optional

import numpy as np
from numpy import ndarray
from numpy.random import Generator


def generate_mvn_data(
    n: int, d: int, k: int, seed: Optional[int] = None
) -> Tuple[ndarray, ndarray]:
    """
    Generates a dataset of random uncorrelated features and multivariate normal
    targets.
    :param n: Number of samples.
    :param d: Dimension of targets.
    :param k: Dimension of features.
    :param seed: Random seed to use for generation. None means don't set.
    :return: A tuple (X, Y), where X is of shape (n, k) and contains the features and Y
        is of shape (n, d) and contains the responses.
    """
    rng = np.random.default_rng(seed)

    # Generate random orthonormal matrix Q
    Q, R = np.linalg.qr(rng.normal(size=(d, d)))

    # Generate positive eigenvalues
    eigs = rng.uniform(size=(d,))

    # PSD Covariance matrix, zero mean
    S = Q.T @ np.diag(eigs) @ Q
    mu = np.zeros(d)

    # Generate correlated targets (Y) and uncorrelated features (X)
    Y = rng.multivariate_normal(mean=mu, cov=S, size=(n,))
    X = rng.normal(size=(n, k))

    return X, Y
