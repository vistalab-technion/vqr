from __future__ import annotations

from typing import Optional, Sequence

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike as Array
from scipy.spatial.distance import cdist

from vqr import VQRSolver, VectorQuantiles
from vqr.vqr import quantile_levels

SIMILARITY_FN_INNER_PROD = lambda x, y: np.dot(x, y)


class CVXVQRSolver(VQRSolver):
    """
    Solves the Optimal Transport formulation of Vector Quantile Regression using
    CVXPY as a solver backend.

    See:
        Carlier, Chernozhukov, Galichon. Vector quantile regression:
        An optimal transport approach,
        Annals of Statistics, 2016
    """

    def __init__(self, verbose: bool = False, **cvx_solver_opts):
        super().__init__()
        self._verbose = verbose
        cvx_solver_opts["verbose"] = verbose
        self._solver_opts = cvx_solver_opts

    def solve_vqr(self, T: int, Y: Array, X: Optional[Array] = None) -> VectorQuantiles:
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
        problem.solve(**self._solver_opts)

        # Obtain the lagrange multipliers Alpha (A) and Beta (B)
        AB: Array = constraints[0].dual_value
        AB = np.reshape(AB, newshape=[Td, k + 1])
        A = AB[:, [0]]  # A is (T**d, 1)
        if k == 0:
            B = None
        else:
            B = AB[:, 1:]  # B is (T**d, k)

        return VectorQuantiles(T, d, U, A, B)