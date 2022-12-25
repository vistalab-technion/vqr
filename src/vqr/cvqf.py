from __future__ import annotations

from typing import Any, Dict, List, Tuple, Callable, Optional, Sequence
from itertools import repeat

import ot
import numpy as np
from numpy import ndarray as Array
from sklearn.utils import check_array


class CVQF:
    """
    Represents a conditional vector quantile function, Q_{Y|X}(u;x).

    Y is assumed to be a d-dimensional variable (d>=1), and X a k-dimensional variable.
    """

    def __init__(self):
        pass

    def evaluate(self, u: Array, x: Array) -> Array:
        """
        Evaluates the CVQF at quantile level u for covariates x.

        :param u: d-dimensional vector quantile level of shape (d,). Each value
        should be in [0, 1].
        :param x: k-dimensional covariates (features) vector, of shape (k,) or (1,k).
        :return: A d-dimensional vector quantile of shape (d,).
        """
        pass

    def condition(self, x: Array) -> VQF:
        """
        Conditions the CVQF on a specific covariate vector, x.

        :param x: The covariates on which to condition of shape (k,) or (1,k).
        :return: A quantile function which then only depends on the level u of shape
        (d,).
        """
        pass

    __call__ = evaluate


class VQF:
    """
    Represents a vector quantile function, Q_{Y}(u) or a conditional vector quantile
    function conditioned on a specific value of X=x, i.e. Q_{Y|X=x}(u).

    Y is assumed to be a d-dimensional variable (d>=1).
    """

    def __init__(self, x: Optional[Array] = None):
        pass

    def evaluate(self, u: Array) -> Array:
        """
        Evaluates the VQF at quantile level u.

        :param u: d-dimensional vector quantile level of shape (d,). Each value
        should be in [0, 1].
        :return: A d-dimensional vector quantile of shape (d,).
        """
        pass

    __call__ = evaluate


class DiscreteCVQF(CVQF):
    """
    Encapsulates the solution to a VQR problem. Contains the vector quantiles and
    regression coefficients of the solution, and provides useful methods for
    interacting with them.

    Should only be constructed by :class:`VQRSolver`s, not manually.

    Given a sample x of shape (1, k), the conditional d-dimensional vector quantiles
    Y|X=x are given by d/du [ B @ x.T  + A ].
    We first calculate the regression targets Y_hat = B @ x.T  + A.
    This is an array of shape (T**d, 1).
    In order to obtain the d vector quantile surfaces (one for each dimension of Y),
    use the :obj:`decode_quantile_values` function on Y_hat.
    These surfaces can be visualized over the grid defined in U.
    The combination of these surfaces comprise the conditional quantile function,
    Q_{Y|X=x}(u) which has a d-dimensional input and output.
    """

    def __init__(
        self,
        T: int,
        d: int,
        U: Array,
        A: Array,
        B: Optional[Array] = None,
        X_transform: Optional[Callable[[Array], Array]] = None,
        k_in: Optional[int] = None,
        solution_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        :param U: Array of shape (T**d, d). Contains the d-dimensional grid on
        which the vector quantiles are defined. If can be decoded back into a
        meshgrid-style sequence of arrays using :obj:`decode_quantile_grid`
        :param A: Array of shape (T**d, 1). Contains the  the regression intercept
        variable. This can be decoded into the d vector quantiles of Y using the
        :obj:`decode_quantile_values` function.
        :param B: Array of shape (T**d, k). Contains the  regression coefficients.
        Will be None if the input was an estimation problem (X=None) instead of a
        regression problem. The k is the dimension of the covariates (X). If an
        X_transform is provided, k corresponds to the dimension AFTER the
        transformation.
        :param X_transform: Transformation to apply to covariates (X) for non-linear
        VQR. Must be provided together with k_in. The transformation is assumed to
        take input of shape (N, k_in) and return output of shape (N, k).
        :param k_in: Covariates input dimension of the X_transform.
        If X_transform is None, must be None or zero.
        :param solution_metrics: Optional key-value pairs which can contain any
        metric values which are tracked by the solver, such as losses, runtimes, etc.
        The keys in this dict are solver-specific and should be specified in the
        documentation of the corresponding solver.
        """
        # Validate dimensions
        assert all(x is not None for x in [T, d, U, A])
        assert U.ndim == 2 and A.ndim == 2
        assert U.shape[0] == A.shape[0]
        assert U.shape[0] == T**d
        assert A.shape[1] == 1
        assert B is None or (B.ndim == 2 and B.shape[0] == T**d)
        assert (X_transform is not None and k_in) or (X_transform is None and not k_in)
        assert solution_metrics is None or isinstance(solution_metrics, dict)

        # TODO: Remove support for B=None

        self._T = T
        self._d = d
        self._U = U
        self._A = A
        self._B = B
        self._k = B.shape[1] if B is not None else 0
        self._X_transform = X_transform
        self._k_in = k_in
        self._solution_metrics = solution_metrics or {}

    @property
    def is_conditional(self) -> bool:
        return self._B is not None

    def evaluate(self, u: Array, x: Array) -> Array:
        pass

    def condition(self, x: Array, refine: bool = False) -> DiscreteVQF:
        """
        Conditions the CVQF on a specific covariate vector, x.

        :param x: The covariates on which to condition, shape (1,k) or (k,).
        :param refine: Whether to refine the conditional quantile function using vector
        monotone rearrangement.
        :return: A DiscreteVQF instance corresponding to the given covariates X.
        """

        # TODO: Remove support for x=None

        if not self.is_conditional:
            if x is not None:
                raise ValueError(f"VQE was fitted but covariates were supplied")

            Y_hat = self._A
        else:
            if x is None:
                raise ValueError(f"VQR was fitted but no covariates were supplied")

            x = x.reshape(1, -1)
            check_array(x, ensure_2d=True, allow_nd=False)

            z = x  # z represents the transformed x
            if self._X_transform is not None:
                _, k_in = x.shape
                if k_in != self._k_in:
                    raise ValueError(
                        f"VQR model was trained with X_transform expecting k_in"
                        f"={self._k_in}, but got covariates with {k_in=} features."
                    )

                z = self._X_transform(x)

            _, k = z.shape
            if k != self._k:
                raise ValueError(
                    f"VQR model was fitted with k={self._k}, "
                    f"but got data with {k=} features."
                )

            B = self._B  # (T**d, k)
            A = self._A  # (T**d, 1)
            Y_hat = B @ z.T + A  # result is (T**d, 1)
            Y_hat = Y_hat  # (1, T**d)

        refine_fn = lambda Qs: (vector_monotone_rearrangement(Qs) if refine else Qs)
        return DiscreteVQF(
            T=self._T,
            d=self._d,
            Qs=refine_fn(decode_quantile_values(self._T, self._d, Y_hat)),
            Us=decode_quantile_grid(self._T, self._d, self._U),
            X=x,
        )

    @property
    def quantile_grid(self) -> Sequence[Array]:
        """
        :return: A sequence of quantile level grids as ndarrays. This is a
            d-dimensional meshgrid (see np.meshgrid) where d is the dimension of the
            target variable Y.
        """
        return decode_quantile_grid(self._T, self._d, self._U)

    @property
    def quantile_levels(self) -> Array:
        """
        :return: An array containing the levels at which the vector quantiles were
            estimated along each target dimension.
        """
        return quantile_levels(self._T)

    @property
    def dim_y(self) -> int:
        """
        :return: The dimension d, of the target variable (Y).
        """
        return self._d

    @property
    def dim_x(self) -> int:
        """
        :return: The dimension k, of the covariates (X) which are expected to be
        passed in to obtain conditional quantiles.
        """
        # If there was an X_transform, the input dimension of that is what we expect to
        # receive.
        return self._k_in or self._k

    @property
    def metrics(self):
        """
        :return: A dict containing solver-specific metrics about the solution.
        """
        return self._solution_metrics.copy()


class DiscreteVQF(VQF):
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
