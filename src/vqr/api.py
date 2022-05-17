from abc import ABC
from typing import Any, Dict, Type, Union, Optional, Sequence

import numpy as np
from numpy import ndarray as Array
from numpy import quantile
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from vqr.vqr import (
    VQRSolver,
    VQRSolution,
    QuantileFunction,
    quantile_levels,
    quantile_contour,
    inversion_sampling,
)
from vqr.coverage import measure_coverage
from vqr.solvers.primal.cvx import CVXVQRSolver
from vqr.solvers.primal.pot import POTVQESolver
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)

SOLVER_TYPES: Dict[str, Type[VQRSolver]] = {
    solver_class.solver_name(): solver_class
    for solver_class in [
        CVXVQRSolver,
        RegularizedDualVQRSolver,
        MLPRegularizedDualVQRSolver,
        POTVQESolver,
    ]
}

DEFAULT_SOLVER_NAME = RegularizedDualVQRSolver.solver_name()


class VectorQuantileBase(BaseEstimator, ABC):
    """
    Base class for vector quantile estimation (VQE) and regression (VQR).
    Compatible with the sklearn API.
    """

    def __init__(
        self,
        n_levels: int = 50,
        solver: Union[str, VQRSolver] = DEFAULT_SOLVER_NAME,
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
        self._fitted_solution: Optional[VQRSolution] = None

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

    @property
    def solution_metrics(self) -> Dict[str, Any]:
        """
        :return: A dict containing solver-specific metrics about the solution.
        """
        check_is_fitted(self)
        return self._fitted_solution.metrics

    @property
    def fitted_solution(self) -> VQRSolution:
        """
        :return: The low-level VQR solution object.
        Usually it is recommended to use the high level API on this class instead.
        """
        check_is_fitted(self)
        return self._fitted_solution


class VectorQuantileEstimator(VectorQuantileBase):
    """
    Performs vector quantile estimation.
    """

    def __init__(
        self,
        n_levels: int = 50,
        solver: Union[str, VQRSolver] = DEFAULT_SOLVER_NAME,
        solver_opts: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(n_levels, solver, solver_opts)

    def vector_quantiles(self, refine: bool = False) -> QuantileFunction:
        """
        :param refine: Refine the quantile function using vector monotone rearrangement.
        :return: A QuantileFunction instance, representing a discretized version of
        the quantile function Q_{Y}(u).
        """
        check_is_fitted(self)
        qf = self._fitted_solution.vector_quantiles(X=None, refine=refine)[0]
        return qf

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
        Y: Array = np.reshape(X, (N, -1))

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

        # Calculate vector quantile function
        vqf = self.vector_quantiles()
        q_surfaces = tuple(vqf)  # d x (T, T, ..., T)

        # Sample from the quantile function
        return inversion_sampling(T=self.n_levels, d=self.dim_y, n=n, Qs=q_surfaces)

    def coverage(self, Y: Array, alpha: float = 0.05) -> float:
        """
        Calculates the coverage of given data points using the quantiles fitted by
        this model.

        First, it creates a contour of points in d-dimensional space which surround
        the region in which 1-2*alpha of the distribution (that was corresponds to a
        given quantile function) is contained. Then, the proportion of points
        contained within this contour is calculated.

        :param Y: Points to measure coverage for. Shape should be (N, d).
        :param alpha: Confidence level for the contour.
        :return: The coverage level, between zero and one.
        """
        check_is_fitted(self)

        # Calculate vector quantile functions
        vqf = self.vector_quantiles()
        q_surfaces = tuple(vqf)  # d x (T, T, ..., T)

        return measure_coverage(
            quantile_contour=quantile_contour(
                T=self.n_levels, d=self.dim_y, Qs=q_surfaces, alpha=alpha
            )[0],
            data=Y,
        )


class VectorQuantileRegressor(RegressorMixin, VectorQuantileBase):
    """
    Performs vector quantile regression.
    """

    def __init__(
        self,
        n_levels: int = 50,
        solver: Union[str, VQRSolver] = DEFAULT_SOLVER_NAME,
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
        Y: Array = np.reshape(y, (N, -1))

        # Scale features to zero-mean
        X_scaled = self._scaler.fit_transform(X)

        self._fitted_solution = self.solver.solve_vqr(T=self.n_levels, Y=Y, X=X_scaled)

        return self

    def vector_quantiles(
        self, X: Optional[Array] = None, refine: bool = False
    ) -> Sequence[QuantileFunction]:
        """
        :param X: Covariates, of shape (N, k). Should be None if the fitted solution
        was for a VQE (un conditional quantiles).
        :param refine: Refine the conditional quantile function using vector monotone
        rearrangement.
        :return: A sequence of length N, containing QuantileFunction instances.
        Each element of the sequence corresponds to one of the covariates in X,
        and contains the discretized conditional quantile function Q_{Y|X=x}(u).
        """
        check_is_fitted(self)
        X = self._validate_X_(X, single=False)

        if X is not None:
            # Scale X with the fitted transformation before predicting
            X = self._scaler.transform(X)

        # Get the conditional quantiles
        cqfs = self._fitted_solution.vector_quantiles(X, refine=refine)
        return cqfs

    def predict(self, X: Array) -> Array:
        """
        Estimates conditional quantiles of Y|X=x.
        :param X: Samples at which to estimate. Should have shape (N, k).
        :return: An array containing the vector-quantile values.
        Returns the same as vector_quantiles, but stacked into one array.
        The result will be of shape [N, d, T, T, ..., T].
        """
        check_is_fitted(self)

        vqfs: Sequence[QuantileFunction] = self.vector_quantiles(X)

        # Stack the vector quantiles for each sample into one tensor
        vqs = np.stack(
            # Iterating over vqf produces the quantile surfaces
            [np.stack(vqf, axis=0) for vqf in vqfs],
            axis=0,
        )

        N = X.shape[0] if X is not None else 1
        assert vqs.shape == (N, self.dim_y, *[self.n_levels] * self.dim_y)
        return vqs

    def sample(self, n: int, x: Array) -> Array:
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
        x = self._validate_X_(X=x, single=True)

        # Calculate vector quantiles given sample X=x
        vqf: QuantileFunction = self.vector_quantiles(X=x)[0]
        q_surfaces = tuple(vqf)  # d x (T, T, ..., T) where each is d-dimensional

        # Sample from the quantile function
        return inversion_sampling(T=self.n_levels, d=self.dim_y, n=n, Qs=q_surfaces)

    def coverage(self, Y: Array, x: Array, alpha: float = 0.05) -> float:
        """
        Calculates the conditional coverage of given data points using the quantiles
        fitted by this model, conditioned on X=x.

        First, it creates a contour of points in d-dimensional space which surround
        the region in which 1-2*alpha of the distribution (that was corresponds to a
        given quantile function) is contained. Then, the proportion of points
        contained within this contour is calculated.

        :param Y: Points to measure coverage for. Shape should be (N, d).
        :param x: One sample of covariates on which to condition Y.
        Should have shape  (k,) or (1, k).
        :param alpha: Confidence level for the contour.
        :return: The coverage level, between zero and one.
        """
        check_is_fitted(self)
        x = self._validate_X_(X=x, single=True)

        # Calculate vector quantiles given sample X=x
        vqf: QuantileFunction = self.vector_quantiles(X=x)[0]
        q_surfaces = tuple(vqf)  # d x (T, T, ..., T) where each is d-dimensional

        return measure_coverage(
            quantile_contour=quantile_contour(
                T=self.n_levels, d=self.dim_y, Qs=q_surfaces, alpha=alpha
            )[0],
            data=Y,
        )

    def _validate_X_(self, X: Optional[Array], single: bool = False) -> Optional[Array]:
        """
        Validates the shape of a covariates array X.
        :param X: A covariates array, either (n, k) or just (k,). Can be None,
        in which case no validation happens.
        :param single: Whether to validate that n=1.
        :return: X after reshaping to (n,k).
        """
        if X is None:
            raise ValueError("Must provide covariates (X) for VQR")

        error_msg = f"X must be (k,) or ({'1' if single else 'n'}, k), got {X.shape=}"
        if np.ndim(X) == 1 and len(X) == self.dim_x:
            # reshape to (1 ,k)
            X = np.reshape(X, (1, -1))

        elif np.ndim(X) != 2:
            raise ValueError(error_msg)

        if X.shape[1] != self.dim_x:
            raise ValueError(error_msg)

        if single and X.shape[0] != 1:
            # Only a single x is supported by this method
            raise ValueError(error_msg)

        return X


class ScalarQuantileEstimator:
    def __init__(
        self,
        n_levels: int = 50,
    ):
        if n_levels < 2:
            raise ValueError("n_levels must be >= 2")

        self.n_levels = n_levels

    def fit(self, X: Array):
        N = len(X)
        Y = np.reshape(X, (N, -1))
        q = quantile(Y, q=quantile_levels(self.n_levels), axis=0)
        assert q is not None
        assert q.shape[0] == len(quantile_levels(self.n_levels))
        self._alpha = q
        return self

    @property
    def sqr_A(self) -> Array:
        return self._alpha

    @property
    def quantile_levels(self) -> Array:
        """
        :return: An array containing the levels at which the vector quantiles were
            estimated along each target dimension.
        """
        return quantile_levels(self.n_levels)

    @property
    def quantile_values(self) -> Array:
        return self.sqr_A
