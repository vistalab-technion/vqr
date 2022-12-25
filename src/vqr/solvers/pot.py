from typing import Optional

import ot
import numpy as np
from numpy import ndarray as Array

from vqr.cvqf import VQRSolution, vector_quantile_levels
from vqr.utils import get_kwargs
from vqr.solvers.cvx import SIMILARITY_FN_INNER_PROD
from vqr.solvers.base import VQRDiscreteSolver


class POTVQESolver(VQRDiscreteSolver):
    """
    Solves the VQE problem as a Wasserstein2 (W2) distance between uniform measures
    on U and Y,  with an inner-product ground metric. Uses the POT library's
    implementation of W2.
    """

    def __init__(self, T: int = 50, **emd2_kwargs):
        """
        :param T: Number of quantile levels to estimate along each of the d
        dimensions. The quantile level will be spaced uniformly between 0 and 1.
        :param emd2_kwargs:  Any kwargs supported by pot.emd2().
        """
        self.T = T
        self.emd2_kwargs = emd2_kwargs
        self._solver_opts = get_kwargs()

    @classmethod
    def solver_name(cls) -> str:
        return "vqe_pot"

    @property
    def solver_opts(self) -> dict:
        return self._solver_opts

    @property
    def levels_per_dim(self) -> int:
        return self.T

    def solve_vqr(self, Y: Array, X: Optional[Array] = None) -> VQRSolution:
        # Can't deal with X's
        if X is not None:
            raise AssertionError(
                f"{self.__class__.__name__} can't work with X. It solves only "
                f"the VQE problem."
            )
        T = self.T
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
            **self.emd2_kwargs,
        )

        # Obtain the lagrange multipliers Alpha (A) and Beta (B)
        A = np.reshape(-log["u"][:, None], newshape=(Td, 1))
        return VQRSolution(T, d, U, A, None)
