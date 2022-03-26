from typing import List, Sequence

import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt

integrated_quantiles = np.load("src/double_int.npy")


def decode_quantile_values(T: int, d: int, Q: ndarray) -> Sequence[ndarray]:
    """
    Decodes the regression coefficients of a VQR solution into vector quantile values.
    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param Q: The regression coefficients, of shape (T**d, 1).
    :return: A sequence of length d of vector quantile values. Each element j in the
        sequence is a d-dimensional array of shape (T, T, ..., T) containing the
        vector quantiles values of j-th variable in Y, i.e., the quantiles of Y_j|Y_{-j}
        where Y_{-j} means all the variables in Y except the j-th.
    """
    Q = np.reshape(Q, newshape=(T,) * d)

    Q_functions: List[ndarray] = [np.array([np.nan])] * d
    for axis in reversed(range(d)):
        # Calculate derivative along this axis
        dQ_du = (1 / T) * np.diff(Q, axis=axis)

        # Duplicate first "row" along axis and insert it first
        pad_with = [
            (0, 0),
        ] * d
        pad_with[axis] = (1, 0)
        dQ_du = np.pad(dQ_du, pad_width=pad_with, mode="edge")

        Q_functions[d - 1 - axis] = dQ_du * T ** 2

    return tuple(Q_functions)


T = 20

dQ_du1, dQ_du2 = decode_quantile_values(T=T, d=2, Q=integrated_quantiles)
dQ_du1 = dQ_du1 / T
dQ_du2 = dQ_du2 / T
Q_ = np.zeros([T, T])
Q_target = np.reshape(integrated_quantiles, newshape=[T, T])
Q_[:, 0] = Q_target[:, 0]
Q_[0, :] = Q_target[0, :]

int_Q_du1 = np.zeros([T, T])
int_Q_du1[0, :] = Q_[0, :]

int_Q_du2 = np.zeros([T, T])


expected_integral = np.concatenate(Q_, axis=0)
plt.plot(integrated_quantiles, label="original")
plt.plot(expected_integral, label="reconstructed")
plt.legend()
plt.show()
expectedt_Q1, expected_Q2 = decode_quantile_values(T=20, d=2, Q=expected_integral)
