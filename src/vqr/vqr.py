from typing import Any, Dict, List, Union, Callable, Optional, Sequence

import cvxpy as cp
import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from scipy.spatial.distance import cdist
from sklearn.utils.validation import check_is_fitted

DEFAULT_METRIC = lambda x, y: np.dot(x, y)


def _slice(
    a: ndarray,
    axis: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
):
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]


class VectorQuantileRegressor(RegressorMixin, BaseEstimator):
    """
    Performs vector quantile regression and estimation.
    """

    def __init__(
        self,
        n_levels: int = 50,
        metric: Union[str, Callable] = DEFAULT_METRIC,
        solver_opts: Dict[str, Any] = {},
    ):
        """
        :param n_levels: Number of quantile levels. The range of possible
            levels (between 0 and 1) will be divided into this number of levels.
            For example, if n_levels=4 then the quantiles [.25, .50, .75, 1.0] will be
            evaluation in each dimension.
        :param metric: Distance metric for pairwise distance calculation between
            points. Can be a metric name accepted by scipy.spatial.distance.cdist or
            a custom callable which accepts two ndarrays and returns a scalar distance.
            Default metric is an inner product.
        :param solver_opts: Solver options for CVXPY's solve().
        """
        if n_levels < 2:
            raise ValueError("n_levels must be >= 2")

        self.n_levels = n_levels
        self.metric = metric
        self.solver_opts = solver_opts

    def fit(self, X: ndarray, y: ndarray):
        """
        TODO
        :param X:
        :param y:
        :return:
        """

        # Input validation.
        X, y = check_X_y(X, y, multi_output=True, ensure_2d=True)
        N = len(X)
        Y: ndarray = np.reshape(y, (N, -1))

        k: int = X.shape[1]
        d: int = Y.shape[1]  # number or target dimensions

        # All quantile levels
        T: int = self.n_levels
        Td: int = T ** d
        u: ndarray = (np.arange(T) + 1) * (1 / T)

        # Quantile levels grid: list of grid coordinate matrices, one per dimension
        U_grids: Sequence[ndarray] = np.meshgrid(
            *[
                u,
            ]
            * d
        )
        # Stack all nd-grid coordinates into one long matrix, of shape (T**d, d)
        U: ndarray = np.stack([U_grid.reshape(-1) for U_grid in U_grids], axis=1)
        assert U.shape == (Td, d)

        # Pairwise distances (similarity)
        S: ndarray = cdist(U, Y, self.metric)

        # Optimization problem definition: optimal transport formulation
        one_N = np.ones([N, 1])
        one_T = np.ones([Td, 1])
        Pi = cp.Variable(shape=(Td, N))
        Pi_S = cp.sum(cp.multiply(Pi, S))
        constraints = [
            Pi @ one_N == 1 / Td * one_T,
            Pi >= 0,
            one_T.T @ Pi == 1 / N * one_N.T,
        ]
        problem = cp.Problem(objective=cp.Maximize(Pi_S), constraints=constraints)

        # Solve the problem
        problem.solve(**self.solver_opts)

        # Decode the quantile surfaces from the lagrange multipliers
        B: ndarray = constraints[0].dual_value
        B = np.reshape(B, (T,) * d)

        Q_surfaces: List[ndarray] = [
            np.array([np.nan]),
        ] * d
        for axis in reversed(range(d)):
            # Calculate derivative along this axis
            dB_du = (1 / T) * np.diff(B, axis=axis)

            # Duplicate first "row" along axis and insert it first
            pad_with = [
                (0, 0),
            ] * d
            pad_with[axis] = (1, 0)
            dB_du = np.pad(dB_du, pad_width=pad_with, mode="edge")

            Q_surfaces[d - 1 - axis] = dB_du

        # Save fitted model
        self.T_: int = T
        self.d_: int = d
        self.k_: int = k
        self.u_: ndarray = u
        self.U_grids_: Sequence[ndarray] = tuple(U_grids)
        self.Q_surfaces_: Sequence[ndarray] = tuple(Q_surfaces)

    def predict(self, X: ndarray):
        """
        TODO
        :param X:
        :return:
        """
        check_is_fitted(self)
        pass

    @property
    def quantile_values(self) -> Sequence[ndarray]:
        """
        :return: A sequence of quantile value ndarrays. The sequence is of length d,
            where d is the dimension of the target variable (Y). The j-th ndarray is
            the d-dimensional vector quantile of the j-th variable in Y.
        """
        check_is_fitted(self)
        return self.Q_surfaces_

    @property
    def quantile_grid(self) -> Sequence[ndarray]:
        """
        :return: A sequence of quantile level grids as ndarrays. This is a
            d-dimensional meshgrid (see np.meshgrid) where d is the dimension of the
            target variable Y.
        """
        check_is_fitted(self)
        return self.U_grids_

    @property
    def quantile_levels(self) -> Sequence[ndarray]:
        """
        :return: An ndarray containing the levels at which the vector quantiles were
            estimated along each target dimension.
        """
        check_is_fitted(self)
        return self.u_
