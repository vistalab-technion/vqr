from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Callable, Optional, Sequence

import numpy as np
from numpy import ndarray as Array
from sklearn.utils import check_array


class VectorQuantiles:
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
        assert U.shape[0] == A.shape[0] == T**d
        assert A.shape[1] == 1
        assert B is None or (B.ndim == 2 and B.shape[0] == T**d)
        assert (X_transform is not None and k_in) or (X_transform is None and not k_in)
        assert solution_metrics is None or isinstance(solution_metrics, dict)

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

    def vector_quantiles(self, X: Optional[Array] = None) -> Sequence[Sequence[Array]]:
        """
        :param X: Covariates, of shape (N, k). Should be None if the fitted solution
            was for a VQE (un conditional quantiles).
        :return: A sequence of sequence of arrays.
        The outer sequence corresponds to the number of samples, and it's length is N.
        The inner sequences contain the vector-quantile values.
        Each inner sequence is of length d, where d is the dimension of the target
        variable (Y). The j-th inner array is the d-dimensional vector-quantile of
        the j-th variable in Y given the other variables of Y.
        It is of shape (T, T, ... T).
        """

        if not self.is_conditional:
            if X is not None:
                raise ValueError(f"VQE was fitted but covariates were supplied")

            Y_hats = [self._A]
        else:
            if X is None:
                raise ValueError(f"VQR was fitted but no covariates were supplied")

            check_array(X, ensure_2d=True, allow_nd=False)

            if self._X_transform is not None:
                N, k_in = X.shape
                if k_in != self._k_in:
                    raise ValueError(
                        f"VQR model was trained with X_transform expecting k_in"
                        f"={self._k_in}, but got covariates with {k_in=} features."
                    )

                X = self._X_transform(X)

            N, k = X.shape
            if k != self._k:
                raise ValueError(
                    f"VQR model was fitted with k={self._k}, "
                    f"but got data with {k=} features."
                )

            B = self._B  # (T**d, k)
            A = self._A  # (T**d, 1) -> will be broadcast to (T**d, N)
            Y_hat = B @ X.T + A  # result is (T**d, N)
            Y_hats = Y_hat.T  # (N, T**d)

        return tuple(
            decode_quantile_values(self._T, self._d, Y_hat) for Y_hat in Y_hats
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


class VQRSolver(ABC):
    """
    Abstraction of a method for solving the Vector Quantile Regression (VQR) problem.

    For an overview of the VQR problem and it's solutions, refer to:
        Carlier G, Chernozhukov V, De Bie G, Galichon A.
        Vector quantile regression and optimal transport, from theory to numerics.
        Empirical Econometrics, 2020.
    """

    @classmethod
    @abstractmethod
    def solver_name(cls) -> str:
        """
        :return: An identifier for this solver.
        """
        pass

    @abstractmethod
    def solve_vqr(
        self,
        T: int,
        Y: Array,
        X: Optional[Array] = None,
    ) -> VectorQuantiles:
        """
        Solves the provided VQR problem in an implementation-specific way.

        :param T: Number of quantile levels to estimate along each of the d
        dimensions. The quantile level will be spaced uniformly between 0 to 1.
        :param Y: The regression target variable, of shape (N, d) where N is the
        number of samples and d is the dimension of the target, which is also the
        dimension of the quantiles which will be estimated.
        :param X: The regression input (features, covariates), of shape (N, k),
        where k is the number of features. Note that X may be None, in which case the
        problem becomes quantile estimation (estimating the quantiles of Y) instead
        of quantile regression.
        :return: A VQRSolution containing the vector quantiles and regression coefficients.
        """
        pass


def quantile_levels(T: int) -> Array:
    """
    Creates a vector of evenly-spaced quantile levels between zero and one.
    :param T: Number of levels to create.
    :return: An array of shape (T,).
    """
    return (np.arange(T) + 1) * (1 / T)


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
    Decodes the U variable of a VQR solution into the grid of the
    evaluation points for the vector quantile functions.
    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
    problem.
    :param U: The encoded grid U, of shape (T**d, d).
    :return: A sequence length d. Each array in the sequence is a d-dimensional
    array of shape (T, T, ..., T), which together represent the d-dimentional grid
    on which the vector quantiles were evaluated.
    """
    return tuple(np.reshape(U[:, dim], newshape=(T,) * d) for dim in range(d))


def inversion_sampling(T: int, d: int, n: int, Qs: Sequence[Array]):
    """
    Generates samples from the variable Y based on it's fitted
    quantile function, using inversion-transform sampling.
    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param n: Number of samples to generate.
    :param Qs: Quantile functions per dimension of Y. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :return: Samples obtained from this quantile function, of shape (n, d).
    """

    # Samples of points on the quantile-level grid
    Us = np.random.randint(0, T, size=(n, d))

    # Sample from Y|X=x
    Y_samp = np.empty(shape=(n, d))
    for i, U_i in enumerate(Us):
        # U_i is a vector-quantile level, of shape (d,)
        Y_samp[i, :] = np.array([Q_d[tuple(U_i)] for Q_d in Qs])

    return Y_samp


def quantile_contour(T: int, d: int, Qs: Sequence[Array], alpha: float = 0.05) -> Array:
    """
    Creates a contour of points in d-dimensional space which surround the region in
    which 1-2*alpha of the distribution (that was corresponds to a given quantile
    function) is contained.

    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param Qs: Quantile functions per dimension of Y. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :param alpha: Confidence level for the contour.
    :return: An array of shape (n, d) containing points along the d-dimensional contour.
    """

    if not 0 < alpha < 0.5:
        raise ValueError(f"Got {alpha=}, but must be in (0, 0.5)")

    lo = int(np.round(T * alpha))
    hi = int(min(np.round(T * (1 - alpha)), T - 1))

    # Will contain d lists of points, each element will be (d,)
    contour_points_list = [[] for _ in range(d)]

    for i, Q in enumerate(Qs):
        for j in range(d):
            for lo_hi, _ in zip([lo, hi], range(d)):
                # The zip here is just to prevent a loop when d==1.
                # This keeps all dims fixed on either lo or hi, while the j-th
                # dimension is sweeped from lo to hi.
                idx: List[Union[int, slice]] = [lo_hi] * d
                idx[j] = slice(lo, hi)
                contour_points_list[i].extend(Q[tuple(idx)])

    contour_points = np.array(contour_points_list).T  # (N, d)
    return contour_points
