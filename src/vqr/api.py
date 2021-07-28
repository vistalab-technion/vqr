from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union, Callable, Optional, Sequence

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from matplotlib.figure import Figure
from sklearn.utils.validation import check_array, check_is_fitted

from vqr.vqr import (
    DEFAULT_METRIC,
    vqr_ot,
    quantile_levels,
    decode_quantile_grid,
    decode_quantile_values,
)
from vqr.plot import plot_quantiles, plot_quantiles_3d


class VectorQuantileBase(BaseEstimator, ABC):
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

    @property
    def quantile_values(self) -> Sequence[ndarray]:
        """
        :return: A sequence of quantile value ndarrays. The sequence is of length d,
            where d is the dimension of the target variable (Y). The j-th ndarray is
            the d-dimensional vector quantile of the j-th variable in Y.
        """
        check_is_fitted(self)
        return decode_quantile_values(
            self.n_levels, self.quantile_dimension, self.vqr_A
        )

    @property
    def quantile_grid(self) -> Sequence[ndarray]:
        """
        :return: A sequence of quantile level grids as ndarrays. This is a
            d-dimensional meshgrid (see np.meshgrid) where d is the dimension of the
            target variable Y.
        """
        check_is_fitted(self)
        return decode_quantile_grid(self.n_levels, self.quantile_dimension, self.vqr_U)

    @property
    def quantile_levels(self) -> ndarray:
        """
        :return: An ndarray containing the levels at which the vector quantiles were
            estimated along each target dimension.
        """
        check_is_fitted(self)
        return quantile_levels(self.n_levels)

    @property
    @abstractmethod
    def quantile_dimension(self) -> int:
        """
        :return: The dimension of the fitted vector quantiles.
        """
        pass

    @property
    @abstractmethod
    def vqr_U(self) -> ndarray:
        """
        :return: Encoded quantile function evaluation grid of shape (T**d, d),
            for which the VQR problem was solved.
        """
        pass

    @property
    @abstractmethod
    def vqr_A(self) -> ndarray:
        """
        :return: VQR regression coefficient Alpha of shape (T**d, 1), obtained by
            solving the VQR problem.
        """
        pass

    @property
    @abstractmethod
    def vqr_B(self) -> Optional[ndarray]:
        """
        :return: VQR regression coefficient Beta of shape (T**d, k), obtained by
            solving the VQR problem.
        """
        pass

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
            U=self.vqr_U,
            A=self.vqr_A,
            figsize=figsize,
        )

        if self.quantile_dimension == 3 or self.quantile_dimension == 2 and surf_2d:
            plot_fn = plot_quantiles_3d
        else:
            plot_fn = plot_quantiles

        return plot_fn(**plot_kwargs)


class VectorQuantileEstimator(VectorQuantileBase):
    """
    Performs vector quantile estimation.
    """

    def __init__(
        self,
        n_levels: int = 50,
        metric: Union[str, Callable] = DEFAULT_METRIC,
        solver_opts: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(n_levels, metric, solver_opts)

    def fit(self, X: ndarray):
        """
        Fits a quantile estimation model to the given data. In case the data
        is high-dimensional, a vector-quantile model will be fitted.
        :param X: Data of shape (n, d). Note that this is called X here to conform to
            sklearn's API. This is the target data, denoted as Y in the VQR
            problem, and we're ignoring the X in that formulation (thus making this
            estimation and not regression).
        :return: self.
        """

        # Input validation.
        N = len(X)
        Y: ndarray = np.reshape(X, (N, -1))

        self.d_: int = Y.shape[1]  # number or target dimensions
        self.U_, self.A_, B_ = vqr_ot(
            T=self.n_levels,
            Y=Y,
            X=None,
            metric=self.metric,
            solver_opts=self.solver_opts,
        )
        assert self.U_ is not None and self.A_ is not None
        assert B_ is None
        return self

    @property
    def quantile_dimension(self) -> int:
        check_is_fitted(self)
        return self.d_

    @property
    def vqr_U(self) -> ndarray:
        check_is_fitted(self)
        return self.U_

    @property
    def vqr_A(self) -> ndarray:
        check_is_fitted(self)
        return self.A_

    @property
    def vqr_B(self) -> Optional[ndarray]:
        check_is_fitted(self)
        return None


class VectorQuantileRegressor(RegressorMixin, VectorQuantileBase):
    """
    Performs vector quantile regression.
    """

    def __init__(
        self,
        n_levels: int = 50,
        metric: Union[str, Callable] = DEFAULT_METRIC,
        solver_opts: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(n_levels, metric, solver_opts)

    def fit(self, X: ndarray, y: ndarray):
        """
        Fits a quantile regression model to the given data. In case the target data
        is high-dimensional, a vector-quantile model will be fitted.
        :param X: Features/covariates of shape (n, k).
        :param y: Targets/responses of shape (n, d).
        :return: self.
        """

        # Input validation.
        X, y = check_X_y(X, y, multi_output=True, ensure_2d=True)
        N = len(X)
        Y: ndarray = np.reshape(y, (N, -1))

        self.k_: int = X.shape[1]  # number of features
        self.d_: int = Y.shape[1]  # number or target dimensions

        vqr_solution = vqr_ot(
            T=self.n_levels, Y=Y, X=X, metric=self.metric, solver_opts=self.solver_opts
        )
        assert all(x is not None for x in vqr_solution)
        self.U_, self.A_, self.B_ = vqr_solution

        return self

    def predict(self, X: ndarray):
        """
        TODO
        :param X:
        :return:
        """
        check_is_fitted(self)
        X = check_array(X)
        raise NotImplementedError("Not yet implemented")

    @property
    def features_dimension(self) -> int:
        """
        :return: The dimension of the feature vector for regression.
        """
        check_is_fitted(self)
        return self.k_

    @property
    def quantile_dimension(self) -> int:
        check_is_fitted(self)
        return self.d_

    @property
    def vqr_U(self) -> ndarray:
        check_is_fitted(self)
        return self.U_

    @property
    def vqr_A(self) -> ndarray:
        check_is_fitted(self)
        return self.A_

    @property
    def vqr_B(self) -> Optional[ndarray]:
        check_is_fitted(self)
        return self.B_
