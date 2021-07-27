from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Sequence

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.base import BaseEstimator, RegressorMixin
from matplotlib.cm import ScalarMappable
from sklearn.utils import check_X_y
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.utils.validation import check_is_fitted

DEFAULT_METRIC = lambda x, y: np.dot(x, y)


class VectorQuantileRegressor(RegressorMixin, BaseEstimator):
    """
    Performs vector quantile regression and estimation.
    """

    def __init__(
        self,
        n_levels: int = 50,
        metric: Union[str, Callable] = DEFAULT_METRIC,
        solver_opts: Optional[Dict[str, Any]] = None,
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
        self.solver_opts = solver_opts or {}

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
        U_grids: Sequence[ndarray] = np.meshgrid(*([u] * d))
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
    def quantile_dimension(self) -> int:
        """
        :return: The dimension of the fitted vector quantiles.
        """
        check_is_fitted(self)
        return self.d_

    @property
    def quantile_levels(self) -> ndarray:
        """
        :return: An ndarray containing the levels at which the vector quantiles were
            estimated along each target dimension.
        """
        check_is_fitted(self)
        return self.u_

    def plot_quantiles(self, figsize: Optional[Tuple[int, int]] = None) -> Figure:
        if self.quantile_dimension > 2:
            raise RuntimeError("Can't plot quantiles with dimension greater than 2")

        fig: Figure
        _axes: ndarray
        fig, _axes = plt.subplots(
            nrows=1,
            ncols=self.quantile_dimension,
            figsize=figsize,
            squeeze=False,
        )
        # _axes is (1, d), take first row to get (d,)
        axes: Sequence[Axes] = list(_axes[0])

        tick_labels = [f"{t:.2f}" for t in self.quantile_levels]

        U = self.quantile_grid
        for i, (ax, Q) in enumerate(zip(axes, self.quantile_values)):

            if self.quantile_dimension == 1:
                ax.plot(*U, Q)
                ax.set_xticks(self.quantile_levels)
                ax.set_xticklabels(
                    tick_labels, rotation=90, ha="right", rotation_mode="anchor"
                )

            elif self.quantile_dimension == 2:
                m = ax.imshow(Q, aspect="equal", interpolation="none", origin="lower")

                ticks = self.quantile_levels * self.n_levels - 1
                ax.set_title(f"$Q_{{{i+1}}}(u_1, u_2)$")
                ax.set_yticks(ticks)
                ax.set_yticklabels(tick_labels)
                ax.set_ylabel("$u_1$")
                ax.xaxis.set_ticks_position("bottom")
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    tick_labels, rotation=90, ha="right", rotation_mode="anchor"
                )
                ax.set_xlabel("$u_2$")

                fig.colorbar(m, ax=[ax], shrink=0.2)

            ax.locator_params(axis="both", tight=True, nbins=20)
        return fig

    def plot_quantiles_3d(self, figsize: Optional[Tuple[int, int]] = None) -> Figure:
        if not 1 < self.quantile_dimension < 4:
            raise RuntimeError("Can't plot 3d quantiles with dimension other than 2, 3")

        fig: Figure
        _axes: ndarray
        fig, _axes = plt.subplots(
            nrows=1,
            ncols=self.quantile_dimension,
            figsize=figsize,
            squeeze=False,
            subplot_kw={"projection": "3d"},
        )
        axes: Sequence[Axes3D] = list(_axes[0])

        tick_labels = [f"{t:.2f}" for t in self.quantile_levels]

        U = self.quantile_grid
        for i, (ax, Q) in enumerate(zip(axes, self.quantile_values)):
            if self.quantile_dimension == 2:
                ticks = self.quantile_levels
                m = ax.plot_surface(*U, Q, cmap="viridis")
                fig.colorbar(m, ax=[ax], shrink=0.2)

            if self.quantile_dimension == 3:
                ticks = self.quantile_levels * self.n_levels - 1
                cmap = plt.get_cmap("viridis")
                norm = plt.Normalize(Q.min(), Q.max())
                ax.voxels(np.ones_like(Q), facecolors=cmap(norm(Q)), edgecolors="black")
                fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=[ax], shrink=0.2)
                ax.set_zticks(ticks)
                ax.set_zticklabels(tick_labels)
                ax.set_zlabel("$u_3$")

            ax.set_title(f"$Q_{{{i+1}}}(u_1, u_2, u_3)$")
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels)
            ax.set_ylabel("$u_1$")
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel("$u_2$")

            ax.locator_params(axis="both", tight=True, nbins=10)

        return fig
