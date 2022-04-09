from typing import Dict, Tuple

import numpy as np
from numpy import array
from numpy.random import shuffle


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
