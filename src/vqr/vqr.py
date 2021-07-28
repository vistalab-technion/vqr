from typing import Any, Dict, List, Union, Callable, Optional, Sequence

import cvxpy as cp
import numpy as np
from numpy import ndarray
from scipy.spatial.distance import cdist

DEFAULT_METRIC = lambda x, y: np.dot(x, y)


def vqr_ot(
    Y: ndarray,
    X: Optional[ndarray] = None,
    n_levels: int = 50,
    metric: Union[str, Callable] = DEFAULT_METRIC,
    solver_opts: Optional[Dict[str, Any]] = None,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Solves the Optimal Transport formulation of Vector Quantile Regression.

    See:
        Carlier, Chernozhukov, Galichon. Vector quantile regression:
        An optimal transport approach,
        Annals of Statistics, 2016

    :param Y: The regression target variable, of shape (N, d) where N is the number
        of samples and d is the dimension of the target, which is also the
        dimension of the quantiles which will be estimated.
    :param X: The regression input (features) variable, of shape (N, k), where k is
        the number of features. Note that X may be None, in which case the problem
        becomes quantile estimation (estimating the quantiles of Y) instead of
        quantile regression.
    :param n_levels: Number of quantile levels to estimate along each of the
        d dimensions. The quantile level will be spaced uniformly between 0 to 1.
    :param metric: The metric to use in order to compute pairwise distances between
        vectors. Should accept to vectors in dimension d and return a scalar.
    :param solver_opts: Extra arguments for CVXPY's solve().
    :return: A tuple of three ndarrays:
        - U, of shape (T**d, d): This contains the d-dimensional grid on which the
            vector quantiles are defined. If can be decoded back into a
            meshgrid-style sequence of arrays using :obj:`decode_quantile_grid`.
        - A, of shape (T**d, 1): This contains the "alpha" Lagrange multiplier
            which contains the regression intercept variable. This can be decoded
            into the d vector quantiles of Y using the
            :obj:`decode_quantile_values` function.
        - B, of shape (T**d, k): This contains the "beta" Lagrange multiplier
            which contains the regression coefficients.

        The outputs should be used as follows:
        Given a sample x of shape (1, k), the conditional quantiles Y|X=x are given by
        Y_hat = B @ x.T  + A.
        This is an array of shape (T**d, 1). In order to obtain the d vector quantiles
        (one for each dimension of Y), use the :obj:`decode_quantile_values`
        function on Y_hat. These surfaces can be visualized over the grid defined in U.
    """

    N = len(Y)
    Y = np.reshape(Y, (N, -1))

    ones = np.ones(shape=(N, 1))
    if X is None:
        X = ones
    else:
        X = np.reshape(X, (N, -1))
        X = np.concatenate([ones, X], axis=1)

    k: int = X.shape[1] - 1  # Number of features (can be zero)
    d: int = Y.shape[1]  # number or target dimensions

    X_bar = np.mean(X, axis=0, keepdims=True)  # (1, k+1)

    # All quantile levels
    T: int = n_levels
    Td: int = T ** d
    u: ndarray = quantile_levels(T)

    # Quantile levels grid: list of grid coordinate matrices, one per dimension
    U_grids: Sequence[ndarray] = np.meshgrid(*([u] * d))
    # Stack all nd-grid coordinates into one long matrix, of shape (T**d, d)
    U: ndarray = np.stack([U_grid.reshape(-1) for U_grid in U_grids], axis=1)
    assert U.shape == (Td, d)

    # Pairwise distances (similarity)
    S: ndarray = cdist(U, Y, metric)

    # Optimization problem definition: optimal transport formulation
    one_N = np.ones([N, 1])
    one_T = np.ones([Td, 1])
    Pi = cp.Variable(shape=(Td, N))
    Pi_S = cp.sum(cp.multiply(Pi, S))
    constraints = [
        Pi @ X == 1 / Td * one_T @ X_bar.T,
        Pi >= 0,
        one_T.T @ Pi == 1 / N * one_N.T,
    ]
    problem = cp.Problem(objective=cp.Maximize(Pi_S), constraints=constraints)

    # Solve the problem
    problem.solve(**solver_opts)

    # Obtain the lagrange multipliers Alpha (A) and Beta (B)
    AB: ndarray = constraints[0].dual_value
    AB = np.reshape(AB, newshape=[Td, k + 1])
    A = AB[:, [0]]  # A is (T**d, 1)
    if k == 0:
        B = None
    else:
        B = AB[:, 1:]  # B is (T**d, k)

    return U, A, B


def quantile_levels(n_levels: int) -> ndarray:
    """
    Creates a vector of evenly-spaced quantile levels between zero and one.
    :param n_levels: Number of
    :return: An ndarray of shape (n_levels,).
    """
    T = n_levels
    return (np.arange(T) + 1) * (1 / T)


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

        Q_functions[d - 1 - axis] = dQ_du

    return tuple(Q_functions)


def decode_quantile_grid(T: int, d: int, U: ndarray) -> Sequence[ndarray]:
    """
    Decodes the U variable of a VQR solution into the grid of the
    evaluation points for the vector quantile functions.
    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param U: The encoded grid U, of shape (T**d, d).
    :return: A sequence length d. Each ndarray in the sequence is a d-dimensional
        array of shape (T, T, ..., T), which together represent the d-dimentional grid
        on which the vector quantiles were evaluated.
    """
    return tuple(np.reshape(U[:, dim], newshape=(T,) * d) for dim in range(d))
