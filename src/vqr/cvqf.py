from __future__ import annotations

from typing import List, Tuple, Optional, Sequence
from itertools import repeat

import ot
import numpy as np
from numpy import ndarray as Array


class QuantileFunction:
    """
    Represents a discretized conditional vector-valued quantile function Q_{Y|X=x}(u)
    of the variable Y|X, where:

    - Y is a d-dimensional target variable
    - X is a k-dimensional covariates vector
    - u is a d-dimensional quantile level.
    - Q_{Y|X=x}(u) is the d-dimensional quantile of Y|X=x at level u.

    This instances of this class expose both the d quantile surfaces of the function,
    and also allow evaluating the function at a given quantile level u.

    - Iterating over an instance or indexing it yields the d quantile surfaces,
      e.g. list(q) or q[i].
    - Using an instance as a function evaluates the quantile function, eg. q(u).
    """

    def __init__(
        self,
        T: int,
        d: int,
        Qs: Sequence[Array],
        Us: Sequence[Array],
        X: Optional[Array] = None,
    ):
        """
        :param T: The number of quantile levels (in each dimension).
        :param d: The dimension of the target variable.
        :param Qs: Quantile surfaces of the quantile function: d arrays,
        each d-dimensional with shape (T, T, ..., T).
        :param Us: Quantile levels per dimension of Y. A sequence of length d,
        where each element is of shape (T, T, ..., T).
        :param X: The covariates on which this quantile function is conditioned.
        """
        if len(Qs) != d:
            raise ValueError(
                f"Expecting {d=} quantile surfaces in quantile function, got {len(Qs)}"
            )

        if not all(Q.shape == tuple([T] * d) for Q in Qs):
            raise ValueError(
                f"Expecting {T=} levels in each dimension of each quantile surface"
            )

        if not (len(Us) == len(Qs) and all(U.shape == Q.shape for Q, U in zip(Qs, Us))):
            raise ValueError(f"Expecting Us and Qs to match in number and shape")

        if X is not None:
            if np.ndim(X) > 2 or (np.ndim(X) == 2 and X.shape[0] > 1):
                raise ValueError(
                    f"Unexpected shape of X, must be (1, k), got {X.shape}"
                )

            X = np.reshape(X, (1, -1))

        self._Qs = np.stack(Qs, axis=0)  # Stack into shape (d,T,T,...,T)
        assert self._Qs.shape == tuple([d, *[T] * d])

        self._Us = np.stack(Us, axis=0)  # Stack into shape (d,T,T,...,T)
        assert self._Us.shape == self._Qs.shape

        self.T = T
        self.d = d
        self.X = X
        self.k = 0 if X is None else X.shape[1]

    @property
    def values(self) -> Array:
        """
        :return: All the discrete values of this quantile function.
        A (d+1)-dimensional array of shape (d, T, T, ... T), where the first axis
        indexes different quantile surfaces.
        """
        return self._Qs

    @property
    def levels(self) -> Array:
        """
        :return: All the discrete quantile levels of this quantile function.
        A (d+1)-dimensional array of shape (d, T, T, ... T), where the first axis
        indexes different quantile surfaces.
        """
        return self._Us

    def __iter__(self):
        """
        :return: An iterator over the quantile surfaces (Qs) of this quantile function.
        Yields d elements, each a d-dimensional array of shape (T, T, ..., T).
        """
        for d_idx in range(self.d):
            yield self._Qs[d_idx]

    def __len__(self):
        """
        :return: Number of dimensions of this quantile function.
        """
        return self.d

    def __getitem__(self, d_idx: int) -> Array:
        """
        :param d_idx: An index of a dimension of the target variable, in range [0, d).
        :return: Quantile surface for that dimension.
        """
        return self._Qs[d_idx]

    def __call__(self, u: Array) -> Array:
        """
        :param u: d-dimensional quantile level represented as the integer index of
        the level in each dimension. Each entry of u must be an integer in [0, T-1].
        For example if d=2, T=10 then u=[3, 9] represents the quantile level [0.4, 1.0]
        :return: The vector-quantile value at level u, i.e. Q_{Y|X=x}(u).
        """
        u = u.astype(np.int)

        if not np.ndim(u) == 1 or not len(u) == self.d:
            raise ValueError(f"u must be of shape (d,), got shape {u.shape}")

        if not np.all((u >= 0) & (u < self.T)):
            raise ValueError(
                f"u must contain indices of quantile levels, each in [0, T-1]"
            )

        q_idx = (slice(None), *u)
        q = self._Qs[q_idx]
        assert q.shape == (self.d,)
        return q


def quantile_levels(T: int) -> Array:
    """
    Creates a vector of evenly-spaced quantile levels between zero and one.
    :param T: Number of levels to create.
    :return: An array of shape (T,).
    """
    return (np.arange(T) + 1) * (1 / T)


def vector_quantile_levels(T: int, d: int) -> Array:
    """
    Constructs an array of vector-quantile levels. Each level is a d-dimensional
    vector with entries in [0,1].

    The quantile levels are returned stacked along the first axis of the result.
    The output of this function can be converted into a "meshgrid" with
    decode_quantile_grid().

    :param T: Number of levels in each dimension.
    :param d: Dimension of target variable.
    :return: A (T^d, d) array with a different level at each "row".
    """
    # Number of all quantile levels
    Td: int = T**d

    # Quantile levels grid: list of grid coordinate matrices, one per dimension
    # d arrays of shape (T,..., T)
    U_grids: Sequence[Array] = np.meshgrid(*([quantile_levels(T)] * d))

    # Stack all nd-grid coordinates into one long matrix, of shape (T**d, d)
    U: Array = np.stack([U_grid.reshape(-1) for U_grid in U_grids], axis=1)
    assert U.shape == (Td, d)

    return U


def decode_quantile_values(T: int, d: int, Y_hat: Array) -> Sequence[Array]:
    """
    Decodes the regression coefficients of a VQR solution into vector quantile values.
    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param Y_hat: The regression output, of shape (T**d, 1).
    :return: A sequence of length d of vector quantile values. Each element j in the
        sequence is a d-dimensional array of shape (T, T, ..., T) containing the
        vector quantiles values of j-th variable in Y, i.e., the quantiles of Y_j|Y_{-j}
        where Y_{-j} means all the variables in Y except the j-th.
    """
    Q = np.reshape(Y_hat, newshape=(T,) * d)

    Q_functions: List[Array] = [np.array([np.nan])] * d
    for axis in reversed(range(d)):
        # Calculate derivative along this axis
        dQ_du = (1 / T) * np.diff(Q, axis=axis)

        # Duplicate first "row" along axis and insert it first
        pad_with = [
            (0, 0),
        ] * d
        pad_with[axis] = (1, 0)
        dQ_du = np.pad(dQ_du, pad_width=pad_with, mode="edge")

        Q_functions[d - 1 - axis] = dQ_du * T**2

    return tuple(Q_functions)


def decode_quantile_grid(T: int, d: int, U: Array) -> Sequence[Array]:
    """
    Decodes stacked vector quantile levels (the output of the vector_quantile_level()
    function) into a "meshgrid" of the evaluation points of the vector quantile
    function.

    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
    problem.
    :param U: The encoded grid U, of shape (T**d, d).
    :return: A sequence length d. Each array in the sequence is a d-dimensional
    array of shape (T, T, ..., T), which together represent the d-dimensional grid
    on which the vector quantiles were evaluated.
    """
    return tuple(np.reshape(U[:, dim], newshape=(T,) * d) for dim in range(d))


def get_d_T(Qs: Sequence[Array]) -> Tuple[int, int]:
    """
    Helper for obtaining d and T parameters from VQF surfaces.

    :param Qs: Quantile surfaces per dimension of Y. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :return: d and T. Will raise if shapes are not consistent with the above.
    """
    shapes = tuple(Q.shape for Q in Qs)

    d = len(shapes)
    T = shapes[0][0]

    # Validate
    expected_shape = (T,) * d
    if not all(shape == expected_shape for shape in shapes):
        raise ValueError(f"Quantile surfaces have unexpected shapes: {shapes}")

    return d, T


def inversion_sampling(n: int, Qs: Sequence[Array]):
    """
    Generates samples from the variable Y based on it's fitted
    quantile function, using inversion-transform sampling.

    :param n: Number of samples to generate.
    :param Qs: Quantile surfaces per dimension of Y. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :return: Samples obtained from this quantile function, of shape (n, d).
    """

    d, T = get_d_T(Qs)

    # Samples of points on the quantile-level grid
    Us = np.random.randint(0, T, size=(n, d))

    # Sample from Y|X=x
    Y_samp = np.empty(shape=(n, d))
    for i, U_i in enumerate(Us):
        # U_i is a vector-quantile level, of shape (d,)
        Y_samp[i, :] = np.array([Q_d[tuple(U_i)] for Q_d in Qs])

    return Y_samp


def quantile_contour(Qs: Sequence[Array], alpha: float = 0.05) -> Tuple[Array, Array]:
    """
    Creates a contour of points in d-dimensional space which surround the region in
    which 100*(1-2*alpha)^d percent of the distribution (that was corresponds to a
    given quantile function) is contained.

    :param Qs: Quantile surfaces per dimension of Y. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :param alpha: Confidence level for the contour.
    :return: A tuple of two arrays, each of shape (n, d).
    The first contains the points along the d-dimensional
    contour, and the second contains the d-dimensional index of each point.
    """

    if not 0 < alpha < 0.5:
        raise ValueError(f"Got {alpha=}, but must be in (0, 0.5)")

    d, T = get_d_T(Qs)
    lo = int(np.round(T * alpha))
    hi = int(min(np.round(T * (1 - alpha)), T - 1))

    # Will contain d lists of points, each element will be (d,)
    contour_points_list = [[] for _ in range(d)]
    contour_index_list = [[] for _ in range(d)]

    for i, Q in enumerate(Qs):
        for j in range(d):
            for lo_hi, _ in zip([lo, hi], range(d)):
                # The zip here is just to prevent a loop when d==1.
                # This keeps all dims fixed on either lo or hi, while the j-th
                # dimension is sweeped from lo to hi.
                # idx: List[Union[int, slice]] = [lo_hi] * d
                # idx[j] = slice(lo, hi)

                idxs = [repeat(lo_hi)] * d
                idxs[j] = range(lo, hi)
                zipped_idxs = tuple(zip(*idxs))  # list of tuples e.g. [(u1, u2), ...]
                # d lists, e.g. [(u1, u1, ...), (u2, u2, ...)]
                grouped_indices = tuple(zip(*zipped_idxs))

                contour_points_list[i].extend(Q[grouped_indices])
                contour_index_list[i].extend(zipped_idxs)

    # Contours generated by accessing the same indices in each quantile surface
    assert all(
        np.allclose(contour_index_list[0], contour_index_list[i]) for i in range(1, d)
    )
    contour_points = np.array(contour_points_list).T  # (N, d)
    contour_indices = np.array(contour_index_list[0])  # (N, d)
    return contour_points, contour_indices


def vector_monotone_rearrangement(
    Qs: Sequence[Array], max_iters: int = 2e6
) -> Sequence[Array]:
    """
    Performs vector monotone rearrangement. Can be interpreted as "vector sorting".

    A vector-extension for the quantile rearrangement idea proposed in

    Victor Chernozhukov, Iván Fernández‐Val, Alfred Galichon.
    Quantile and probability curves without crossing.
    Econometrica, 2010.

    Solves an exact OT problem using POT's emd solver with U @ Y.T as the cost matrix.
    Rearrangement is performed by multiplication with the permutation matrix Pi.

    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param Qs: Quantile surfaces per dimension of Y. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :param max_iters: Maximum number of iterations for the emd solver.
    :return: Rearranged quantile surfaces per dimension of Y. A sequence of length d
    where each element is of shape (T, T, ..., T).
    """
    d, T = get_d_T(Qs)
    U: Array = vector_quantile_levels(T, d)
    Y: Array = np.stack([Q.ravel() for Q in Qs], axis=-1)
    pi = ot.emd(M=-U @ Y.T, a=[], b=[], numItermax=max_iters)
    rearranged_Y = (T**d) * pi @ Y
    rearranged_Qs = [rearranged_Y[:, i].reshape((T,) * d) for i in range(d)]
    return rearranged_Qs


def check_comonotonicity(Qs: Sequence[Array], Us: Sequence[Array]) -> Array:
    """
    Measures co-monotonicity defined as (u_i - u_j).T @ (Q(u_i) - Q(u_j)) for all i, j.
    Results in a T^d x T^d symmetric matrix.  If co-monotonicity is satisfied,
    all entries in this matrix are positive. Negative entries in this matrix
    represent the quantile crossing problem (and its analogue in higher dimensions).

    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param Qs: Quantile surfaces per dimension of Y. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :param Us: Quantile levels per dimension of Y. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :return: A T^d x T^d symmetric matrix that measures co-monotonicity between all
    pairs of quantile levels and quantile values. The (i, j)^th entry in the matrix
    measures co-monotonicity between quantile levels  u_i and u_j and the quantile
    values Q(u_i) and Q(u_j).
    """
    d, T = get_d_T(Qs)
    levels = np.stack([level.ravel() for level in Us])
    quantiles = np.stack([Q.ravel() for Q in Qs])

    pairwise_diff_levels = (
        levels.reshape(d, 1, T**d) - levels.reshape(d, T**d, 1)
    ).reshape(d, T ** (2 * d))

    pairwise_diff_quantiles = (
        quantiles.reshape(d, 1, T**d) - quantiles.reshape(d, T**d, 1)
    ).reshape(d, T ** (2 * d))

    all_pairs_inner_prod = np.sum(
        pairwise_diff_levels * pairwise_diff_quantiles, axis=0
    ).reshape(T**d, T**d)

    return all_pairs_inner_prod
