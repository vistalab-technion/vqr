from typing import Any, Dict, Tuple, Union, Callable, Optional, Sequence

import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from matplotlib.cm import ScalarMappable
from sklearn.utils import check_X_y
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.exceptions import NotFittedError
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils.validation import check_is_fitted

from vqr.vqr import (
    DEFAULT_METRIC,
    vqr_ot,
    quantile_levels,
    decode_quantile_grid,
    decode_quantile_values,
)


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
        Fits a quantile regression model to the given data. In case the response
        variable y is high-dimentional, a vector-quantile model will be fitted.
        :param X: Features, (n, k). Currently ignored (TODO: Handle X)
        :param y: Responses, (n, d).
        :return: self.
        """

        # Input validation.
        X, y = check_X_y(X, y, multi_output=True, ensure_2d=True)
        N = len(X)
        Y: ndarray = np.reshape(y, (N, -1))

        self.k_: int = X.shape[1]  # number of features
        self.d_: int = Y.shape[1]  # number or target dimensions
        self.u_: ndarray = quantile_levels(self.n_levels)

        self.U_, self.A_, self.B_ = vqr_ot(
            # TODO: X
            Y,
            None,
            self.n_levels,
            self.metric,
            self.solver_opts,
        )

        self.U_grids_: Sequence[ndarray] = decode_quantile_grid(
            self.n_levels, self.d_, self.U_
        )
        self.Q_surfaces_: Sequence[ndarray] = decode_quantile_values(
            self.n_levels, self.d_, self.A_
        )
        return self

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
        """
        Plots 1d or 2d quantiles. A new figure will be created. 1d quantiles will be
        plotted using a simple line plot, while 2d quantiles will be plottes as an
        image, where the pixel colors correspont to quantile value.
        :param figsize: Size of figure to create. Will be passed to plt.subplots.
        :return: The created figure.
        """
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
        """
        Plots 2d or 3d quantiles. A new figure will be created. 2d quantiles will be
        plotted as surfaces, while 3d quantiles will be plotted as voxels, where the
        color of the quantile corresponds to the value of the quantile.
        :param figsize: Size of figure to create. Will be passed to plt.subplots.
        :return: The created figure.
        """
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

    def __repr__(self):
        cls = self.__class__.__name__
        fitted = True
        try:
            check_is_fitted(self)
        except NotFittedError:
            fitted = False

        fields_strs = [f"{fitted=}", f"n_levels={self.n_levels}"]
        if fitted:
            fields_strs.append(f"d={self.quantile_dimension}")

        return f"{cls}({str.join(', ', fields_strs)})"
