from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, Optional, Sequence

from numpy import ndarray as Array
from sklearn.utils import check_array

from vqr.cvqf import (
    QuantileFunction,
    quantile_levels,
    decode_quantile_grid,
    decode_quantile_values,
    vector_monotone_rearrangement,
)


class VQRSolver(ABC):
    """
    Abstraction of a method for solving the Vector Quantile Regression (VQR) problem.
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
        Y: Array,
        X: Optional[Array] = None,
    ) -> VQRSolution:
        """
        Solves the provided VQR problem in an implementation-specific way.

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


class VQRSolution:
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

    def vector_quantiles(
        self, X: Optional[Array] = None, refine: bool = False
    ) -> Sequence[QuantileFunction]:
        """
        :param X: Covariates, of shape (N, k). Should be None if the fitted solution
        was for a VQE (un conditional quantiles).
        :param refine: Refine the conditional quantile function using vector monotone
        rearrangement.
        :return: A sequence of length N containing QuantileFunction instances
        corresponding to the given covariates X. If X is None, will be a sequence of
        length one.
        """

        if not self.is_conditional:
            if X is not None:
                raise ValueError(f"VQE was fitted but covariates were supplied")

            Y_hats = [self._A]
        else:
            if X is None:
                raise ValueError(f"VQR was fitted but no covariates were supplied")

            check_array(X, ensure_2d=True, allow_nd=False)

            Z = X  # Z represents the transformed X
            if self._X_transform is not None:
                N, k_in = X.shape
                if k_in != self._k_in:
                    raise ValueError(
                        f"VQR model was trained with X_transform expecting k_in"
                        f"={self._k_in}, but got covariates with {k_in=} features."
                    )

                Z = self._X_transform(X)

            N, k = Z.shape
            if k != self._k:
                raise ValueError(
                    f"VQR model was fitted with k={self._k}, "
                    f"but got data with {k=} features."
                )

            B = self._B  # (T**d, k)
            A = self._A  # (T**d, 1) -> will be broadcast to (T**d, N)
            Y_hat = B @ Z.T + A  # result is (T**d, N)
            Y_hats = Y_hat.T  # (N, T**d)

        refine_fn = lambda Qs: (
            vector_monotone_rearrangement(self._T, self._d, Qs) if refine else Qs
        )
        return tuple(
            QuantileFunction(
                T=self._T,
                d=self._d,
                Qs=refine_fn(decode_quantile_values(self._T, self._d, Y_hat)),
                Us=decode_quantile_grid(self._T, self._d, self._U),
                X=X[[i], :] if X is not None else None,
            )
            for i, Y_hat in enumerate(Y_hats)
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
