from __future__ import annotations

from typing import Optional

import cvxpy as cp
import numpy as np
from numpy import ndarray as Array
from scipy.spatial.distance import cdist

from vqr.cvqf import DiscreteCVQF, vector_quantile_levels
from vqr.utils import get_kwargs
from vqr.solvers.base import VQRDiscreteSolver

SIMILARITY_FN_INNER_PROD = lambda x, y: np.dot(x, y)


class CVXVQRSolver(VQRDiscreteSolver):
    """
    Solves the Optimal Transport formulation of Vector Quantile Regression using
    CVXPY as a solver backend.
    """

    def __init__(self, T: int = 50, verbose: bool = False, **cvxpy_kwargs):
        """

        :param T: Number of quantile levels to estimate along each of the d
        dimensions. The quantile level will be spaced uniformly between 0 and 1.
        :param verbose: Whether to be verbose.
        :param cvxpy_kwargs: Any kwargs supported by CVXPY's Problem.solve().
        """
        super().__init__()
        self.T = T
        self.verbose = verbose
        cvxpy_kwargs["verbose"] = verbose
        self.cvxpy_kwargs = cvxpy_kwargs
        self._solver_opts = get_kwargs()

    @classmethod
    def solver_name(cls) -> str:
        return "cvx_primal"

    @property
    def solver_opts(self) -> dict:
        return self._solver_opts

    @property
    def levels_per_dim(self) -> int:
        return self.T

    def solve_vqr(self, Y: Array, X: Optional[Array] = None) -> DiscreteCVQF:
        T = self.T
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
        Td: int = T**d
        U: Array = vector_quantile_levels(T, d)
        assert U.shape == (Td, d)

        # Pairwise distances (similarity)
        S: Array = cdist(U, Y, SIMILARITY_FN_INNER_PROD)  # (Td, d) and (N, d)

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
        problem.solve(**self.cvxpy_kwargs)

        # Obtain the lagrange multipliers Alpha (A) and Beta (B)
        AB: Array = constraints[0].dual_value
        AB = np.reshape(AB, newshape=[Td, k + 1])
        A = AB[:, [0]]  # A is (T**d, 1)
        if k == 0:
            B = None
        else:
            B = AB[:, 1:]  # B is (T**d, k)

        return DiscreteCVQF(T, d, U, A, B)
