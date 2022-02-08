from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Callable, Optional, Sequence

import cvxpy as cp
import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike as Array
from scipy.spatial.distance import cdist

SIMILARITY_FN_INNER_PROD = lambda x, y: np.dot(x, y)


class VQRSolution:
    """
    Encapsulates the solution to a VQR problem. Contains the vector quantiles and
    regression coefficients of the solution, and provides useful methods for
    interacting with them.

    Should only be constructed by :class:`VQRSolver`s, not manually.

    Given a sample x of shape (1, k), the conditional d-dimensional vector quantiles
    Y|X=x are given by
    Y_hat = B @ x.T  + A.
    This is an array of shape (T**d, 1).
    In order to obtain the d vector quantile surfaces (one for each dimension of Y),
    use the :obj:`decode_quantile_values` function on Y_hat.
    These surfaces can be visualized over the grid defined in U.
    """

    def __init__(self, U: Array, A: Array, B: Optional[Array]):
        """
        :param U: Array of shape (T**d, d). Contains the d-dimensional grid on
        which the vector quantiles are defined. If can be decoded back into a
        meshgrid-style sequence of arrays using :obj:`decode_quantile_grid`
        :param A: Array of shape (T**d, 1). Contains the  the regression intercept
        variable. This can be decoded into the d vector quantiles of Y using the
        :obj:`decode_quantile_values` function.
        :param B: Array of shape (T**d, k). Contains the  regression coefficients.
        Will be None if the input was an estimation problem (X=None) instead of a
        regression problem.
        """
        self.U = U
        self.A = A
        self.B = B


class VQRSolver(ABC):
    """
    Abstraction of a method for solving the Vector Quantile Regression (VQR) problem.

    For an overview of the VQR problem and it's solutions, refer to:
        Carlier G, Chernozhukov V, De Bie G, Galichon A.
        Vector quantile regression and optimal transport, from theory to numerics.
        Empirical Econometrics, 2020.
    """

    def __init__(
        self,
        similarity_fn: Union[str, Callable] = SIMILARITY_FN_INNER_PROD,
        **solver_opts,
    ):
        """
        :param similarity_fn: A scalar function to use in order to compute pairwise
            similarity/distance between the data (Y) and the quantile-grid (U).
            Should accept to vectors in dimension d and return a scalar.
        :param solver_opts: Kwargs for underlying solver. Implementation specific.
        """
        self._similarity_fn = similarity_fn
        self._solver_opts = solver_opts

    @abstractmethod
    def solve_vqr(
        self,
        T: int,
        Y: Array,
        X: Optional[Array] = None,
    ) -> VQRSolution:
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


class CVXVQRSolver(VQRSolver):
    """
    Solves the Optimal Transport formulation of Vector Quantile Regression using
    CVXPY as a solver backend.

    See:
        Carlier, Chernozhukov, Galichon. Vector quantile regression:
        An optimal transport approach,
        Annals of Statistics, 2016
    """

    def __init__(self, **cvx_solver_opts):
        super().__init__(similarity_fn=SIMILARITY_FN_INNER_PROD, **cvx_solver_opts)

    def solve_vqr(self, T: int, Y: Array, X: Optional[Array] = None) -> VQRSolution:
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
        Td: int = T ** d
        u: Array = quantile_levels(T)

        # Quantile levels grid: list of grid coordinate matrices, one per dimension
        U_grids: Sequence[Array] = np.meshgrid(
            *([u] * d)
        )  # d arrays of shape (T,..., T)
        # Stack all nd-grid coordinates into one long matrix, of shape (T**d, d)
        U: Array = np.stack([U_grid.reshape(-1) for U_grid in U_grids], axis=1)
        assert U.shape == (Td, d)

        # Pairwise distances (similarity)
        S: Array = cdist(U, Y, self._similarity_fn)  # (Td, d) and (N, d)

        # Optimization problem definition: optimal transport formulation
        one_N = np.ones([N, 1])
        one_T = np.ones([Td, 1])
        Pi = cp.Variable(shape=(Td, N))
        Pi_S = cp.sum(cp.multiply(Pi, S))
        constraints = [
            Pi @ X == 1 / Td * one_T @ X_bar,
            Pi >= 0,
            one_T.T @ Pi == 1 / N * one_N.T,
        ]
        problem = cp.Problem(objective=cp.Maximize(Pi_S), constraints=constraints)

        # Solve the problem
        problem.solve(**self._solver_opts)

        # Obtain the lagrange multipliers Alpha (A) and Beta (B)
        AB: Array = constraints[0].dual_value
        AB = np.reshape(AB, newshape=[Td, k + 1])
        A = AB[:, [0]]  # A is (T**d, 1)
        if k == 0:
            B = None
        else:
            B = AB[:, 1:]  # B is (T**d, k)

        return VQRSolution(U, A, B)


def vqr_ot(
    T: int,
    Y: ndarray,
    X: Optional[ndarray] = None,
    metric: Union[str, Callable[[Array, Array], float]] = SIMILARITY_FN_INNER_PROD,
    solver_opts: Optional[Dict[str, Any]] = None,
) -> tuple[ndarray, ndarray, Optional[ndarray]]:
    solver = CVXVQRSolver(**solver_opts)
    solution = solver.solve_vqr(T, Y, X)
    return solution.U, solution.A, solution.B


def quantile_levels(T: int) -> ndarray:
    """
    Creates a vector of evenly-spaced quantile levels between zero and one.
    :param T: Number of levels to create.
    :return: An ndarray of shape (T,).
    """
    T = T
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

        Q_functions[d - 1 - axis] = dQ_du * T ** 2

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
