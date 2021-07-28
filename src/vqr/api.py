from typing import Any, Dict, Tuple, Union, Callable, Optional, Sequence

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from matplotlib.figure import Figure
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from vqr.vqr import (
    DEFAULT_METRIC,
    vqr_ot,
    quantile_levels,
    decode_quantile_grid,
    decode_quantile_values,
)
from vqr.plot import plot_quantiles, plot_quantiles_3d


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

    def plot_quantiles(
        self, surf_2d: bool = False, figsize: Optional[Tuple[int, int]] = None
    ) -> Figure:
        """
        Plots scalar (d=1) or vector quantiles (d=2 and d=3).
        A new figure will be created.
        - Scalar quantiles (d=1) will be plotted using a simple line plot
        - Vector d=2 quantiles will be plotted either as images (surf_2d=False) or as
            surface plots (surf_2d=True).
        - Vector d=3 quantiles will be plotted as voxels.
        :param surf_2d: Whether for d=2 the quantiles should be plotted as a surface
            or an image.
        :param figsize: Size of figure to create. Will be passed to plt.subplots.
        :return: The created figure.
        """
        check_is_fitted(self)

        plot_kwargs = dict(
            T=self.n_levels,
            d=self.quantile_dimension,
            U=self.U_,
            A=self.A_,
            figsize=figsize,
        )

        if self.quantile_dimension == 3 or self.quantile_dimension == 2 and surf_2d:
            plot_fn = plot_quantiles_3d
        else:
            plot_fn = plot_quantiles

        return plot_fn(**plot_kwargs)

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
