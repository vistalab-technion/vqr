from typing import Optional, Sequence

import ot
import numpy as np
from numpy import ndarray as Array

from vqr import VQRSolution
from vqr.vqr import VQRSolver, vector_quantile_levels
from vqr.solvers.primal.cvx import SIMILARITY_FN_INNER_PROD


class POTVQESolver(VQRSolver):
    """
    Solves the VQE problem using the POT library.
    """

    def __init__(self, **pot_solver_opts):
        self._pot_solver_opts = pot_solver_opts

    @classmethod
    def solver_name(cls) -> str:
        return "vqe_pot"

    def solve_vqr(self, T: int, Y: Array, X: Optional[Array] = None) -> VQRSolution:
        # Can't deal with X's
        if X is not None:
            raise AssertionError(
                f"{self.__class__.__name__} can't work with X. It solves only "
                f"the VQE problem."
            )
        N: int = Y.shape[0]
        d: int = Y.shape[1]  # number or target dimensions

        # All quantile levels
        Td: int = T**d

        U: Array = vector_quantile_levels(T, d)
        assert U.shape == (Td, d)

        # Pairwise distances (similarity)
        S: Array = SIMILARITY_FN_INNER_PROD(U, Y.T)  # (Td, d) and (N, d)
        _, log = ot.emd2(
            M=-S,
            a=np.ones([Td]) / Td,
            b=np.ones([N]) / N,
            log=True,
            **self._pot_solver_opts,
        )

        # Obtain the lagrange multipliers Alpha (A) and Beta (B)
        A = np.reshape(-log["u"][:, None], newshape=(Td, 1))
        return VQRSolution(T, d, U, A, None)
