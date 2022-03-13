from abc import ABC
from typing import Any, Dict, Type, Tuple, Union, Optional, Sequence

import numpy as np
from numpy import ndarray, quantile
from numpy.typing import ArrayLike as Array
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from vqr.vqr import (
    VQRSolver,
    CVXVQRSolver,
    VectorQuantiles,
    RVQRDualLSESolver,
    quantile_levels,
)
from vqr.plot import plot_quantiles, plot_quantiles_3d

SOLVER_TYPES: Dict[str, Type[VQRSolver]] = {
    "cvx": CVXVQRSolver,
    "rvqr_dual_lse": RVQRDualLSESolver,
}

DEFAULT_SOLVER = "rvqr_dual_lse"


class VectorQuantileBase(BaseEstimator, ABC):
    """
    Base class for vector quantile estimation (VQE) and regression (VQR).
    Compatible with the sklearn API.
    """

    def __init__(
        self,
        n_levels: int = 50,
        solver: Union[str, VQRSolver] = DEFAULT_SOLVER,
        solver_opts: Optional[Dict[str, Any]] = None,
    ):
        """
        :param n_levels: Number of quantile levels. The range of possible
            levels (between 0 and 1) will be divided into this number of levels.
            For example, if n_levels=4 then the quantiles [.25, .50, .75, 1.0] will be
            evaluation in each dimension.
        :param solver: Either a supported solver name (see keys of
            :obj:`SOLVER_TYPES`) or an instance of a :class:`VQRSolver`.
        :param solver_opts: If solver is a string, these kwargs will be passed to the
            constructor of the corresponding solver type.
        """
        if n_levels < 2:
            raise ValueError("n_levels must be >= 2")

        solver_instance: VQRSolver
        if isinstance(solver, str):
            if solver not in SOLVER_TYPES:
                raise ValueError(
                    f"solver must be one of {[*SOLVER_TYPES.keys()]}, got {solver=}"
                )
            solver_opts = solver_opts or {}
            solver_instance = SOLVER_TYPES[solver](**solver_opts)
        elif isinstance(solver, VQRSolver):
            solver_instance = solver
        else:
            raise ValueError(
                f"solver must be either a string or an instance of VQRSolver"
            )

        self.solver_opts = solver_opts
        self.solver = solver_instance
        self._n_levels = n_levels
        self._fitted_solution: Optional[VectorQuantiles] = None

    def __sklearn_is_fitted__(self):
        return self._fitted_solution is not None

    @property
    def n_levels(self) -> int:
        """
        :return: Number of quantile levels which this estimator calculates (in every
        dimension of the target variable).
        """
        return self._n_levels

    @property
    def quantile_grid(self) -> Sequence[Array]:
        """
        :return: A sequence of quantile level grids as arrays. This is a
            d-dimensional meshgrid (see np.meshgrid) where d is the dimension of the
            target variable Y.
        """
        check_is_fitted(self)
        return self._fitted_solution.quantile_grid

    @property
    def quantile_levels(self) -> Array:
        """
        :return: An array containing the levels at which the vector quantiles were
            estimated along each target dimension.
        """
        check_is_fitted(self)
        return self._fitted_solution.quantile_levels

    @property
    def dim_y(self) -> int:
        """
        :return: The dimension of the target variable (Y).
        """
        check_is_fitted(self)
        return self._fitted_solution.dim_y

    @property
    def dim_x(self) -> int:
        """
        :return: The dimension k, of the covariates (X).
        """
        check_is_fitted(self)
        return self._fitted_solution.dim_x

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

        # TODO: Refactor plot to take the solution
        plot_kwargs = dict(
            T=self.n_levels,
            d=self._fitted_solution.dim_y,
            U=self._fitted_solution._U,
            A=self._fitted_solution._A,
            figsize=figsize,
        )

        if self.dim_y == 3 or self.dim_y == 2 and surf_2d:
            plot_fn = plot_quantiles_3d
        else:
            plot_fn = plot_quantiles

        return plot_fn(**plot_kwargs)

    def _inversion_sampling(self, n: int, Qs: Sequence[Array]):
        """
        Generates samples from the variable Y based on it's fitted
        quantile function, using inversion-transform sampling.
        :param n: Number of samples to generate.
        :param Qs: Quantile functions per dimension of Y. A sequence of length d,
        where each element is of shape (T, T, ..., T).
        :return: Samples obtained from this quantile function, of shape (n, d).
        """

        # Samples of points on the quantile-level grid
        Us = np.random.randint(0, self._n_levels, size=(n, self.dim_y))

        # Sample from Y|X=x
        Y_samp = np.empty(shape=(n, self.dim_y))  # (n, d)
        for i, U_i in enumerate(Us):
            # U_i is a vector-quantile level, of shape (d,)
            Y_samp[i, :] = np.array([Q_d[tuple(U_i)] for Q_d in Qs])

        return Y_samp


class VectorQuantileEstimator(VectorQuantileBase):
    """
    Performs vector quantile estimation.
    """

    def __init__(
        self,
        n_levels: int = 50,
        solver: Union[str, VQRSolver] = DEFAULT_SOLVER,
        solver_opts: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(n_levels, solver, solver_opts)

    def vector_quantiles(self) -> Sequence[Array]:
        """
        :return: A sequence of arrays containing the vector-quantile values.
        The sequence is of length d, where d is the dimension of the target
        variable (Y). The j-th array is the d-dimensional vector-quantile of
        the j-th variable of Y given the other variables of Y.
        It is of shape (T, T, ... T).
        """
        check_is_fitted(self)
        return self._fitted_solution.vector_quantiles(X=None)[0]

    def fit(self, X: Array):
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

        self._fitted_solution = self.solver.solve_vqr(T=self.n_levels, Y=Y, X=None)

        return self

    def sample(self, n: int) -> Array:
        """
        Sample from Y based on the fitted vector quantile function Q(u).
        Uses the approach of Inverse transform sampling.
        https://en.wikipedia.org/wiki/Inverse_transform_sampling

        :param n: Number of samples to draw from Y|X=x.
        :return: An array containing the Sampled Y values, of shape (n, d).
        """
        check_is_fitted(self)

        # Calculate vector quantiles
        # d x (T, T, ..., T) where each is d-dimensional
        Qs = self.vector_quantiles()

        # Sample from the quantile function
        return self._inversion_sampling(n, Qs)


class VectorQuantileRegressor(RegressorMixin, VectorQuantileBase):
    """
    Performs vector quantile regression.
    """

    def __init__(
        self,
        n_levels: int = 50,
        solver: Union[str, VQRSolver] = DEFAULT_SOLVER,
        solver_opts: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(n_levels, solver, solver_opts)
        self._scaler = StandardScaler(with_mean=True, with_std=True)

    def fit(self, X: Array, y: Array):
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

        # Scale features to zero-mean
        X_scaled = self._scaler.fit_transform(X)

        self._fitted_solution = self.solver.solve_vqr(T=self.n_levels, Y=Y, X=X_scaled)

        return self

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
        check_is_fitted(self)

        if X is None:
            N = 1
        else:
            check_array(X, ensure_2d=True, allow_nd=False)
            N, k = X.shape
            if k != self.dim_x:
                raise ValueError(
                    f"VQR model was fitted with k={self.dim_x}, "
                    f"but got data with {k=} features."
                )

            # Scale X with the fitted transformation before predicting
            X = self._scaler.transform(X)

        return self._fitted_solution.vector_quantiles(X)

    def predict(self, X: Array) -> Array:
        """
        Estimates conditional quantiles of Y|X=x.
        :param X: Samples at which to estimate. Should have shape (N, k).
        :return: An array containing the vector-quantile values.
        Returns the same as vector_quantiles, but stacked into one array.
        The result will be of shape [N, d, T, T, ..., T].
        """
        check_is_fitted(self)

        vq_samples: Sequence[Sequence[Array]] = self.vector_quantiles(X)

        # Stack the vector quantiles for each sample into one tensor
        vqs = np.stack(
            # vq_sample is a Sequence[Array] of length d
            [np.stack(vq_sample, axis=0) for vq_sample in vq_samples],
            axis=0,
        )

        N = X.shape[0] if X is not None else 1
        assert vqs.shape == (N, self.dim_y, *[self.n_levels] * self.dim_y)
        return vqs

    def sample(self, n: int, x: Optional[Array] = None) -> Array:
        """
        Sample from Y|X=x based on the fitted vector quantile function Q(u;x).
        Uses the approach of Inverse transform sampling.
        https://en.wikipedia.org/wiki/Inverse_transform_sampling

        :param n: Number of samples to draw from Y|X=x.
        :param x: One sample of covariates on which to condition Y.
        Should have shape  (k,) or (1, k).
        If None, then samples will be drawn from the unconditional Y.
        :return: An array containing the Sampled Y values, of shape (n, d).
        """
        check_is_fitted(self)

        if x is not None:
            if np.ndim(x) == 1:
                # reshape to (1,k)
                x = np.reshape(x, (1, -1))
            elif np.ndim(x) != 2 or x.shape[0] != 1:
                raise ValueError(f"x must be (k,) or (1,k), got {x.shape=}")

        # Calculate vector quantiles given sample X=x
        # 1 x d x (T, T, ..., T) where each is d-dimensional
        Qs = self.vector_quantiles(X=x)[0]

        # Sample from the quantile function
        return self._inversion_sampling(n, Qs)


class ScalarQuantileEstimator:
    def __init__(
        self,
        n_levels: int = 50,
    ):
        if n_levels < 2:
            raise ValueError("n_levels must be >= 2")

        self.n_levels = n_levels

    def fit(self, X: ndarray):
        N = len(X)
        Y = np.reshape(X, (N, -1))
        q = quantile(Y, q=quantile_levels(self.n_levels), axis=0)
        assert q is not None
        assert q.shape[0] == len(quantile_levels(self.n_levels))
        self._alpha = q
        return self

    @property
    def sqr_A(self) -> ndarray:
        return self._alpha

    @property
    def quantile_levels(self) -> ndarray:
        """
        :return: An ndarray containing the levels at which the vector quantiles were
            estimated along each target dimension.
        """
        return quantile_levels(self.n_levels)

    @property
    def quantile_values(self) -> ndarray:
        return self.sqr_A
