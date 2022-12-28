from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, TypeVar, Optional

from numpy import ndarray as Array

from vqr.cvqf import VQF, CVQF, DiscreteCVQF

TSolver = TypeVar("TSolver", bound="Solver")


class Solver(ABC):
    @classmethod
    @abstractmethod
    def solver_name(cls) -> str:
        """
        :return: An identifier for this solver.
        """
        pass

    @property
    @abstractmethod
    def solver_opts(self) -> dict:
        """
        :return: Implementation-specific options use to configure this solver.
        """
        pass

    def copy(self: Type[TSolver], **solver_opts) -> TSolver:
        """
        Creates a copy of this solver, optionally with some parameters overridden.

        :param solver_opts: Solver options which will override the existing options (
        can be a subset).
        :return: A new solver of the same type as self, initialized with the current
        options and given overrides.
        """
        new_opts = {**self.solver_opts, **solver_opts}
        return type(self)(**new_opts)


class VQESolver(Solver, ABC):
    """
    Abstraction of a method for solving the Vector Quantile Estimation (VQE) problem.
    """

    @abstractmethod
    def solve_vqe(self, Y: Array) -> VQF:
        """
        Solves the VQE problem in an implementation-specific way.

        :param Y: The regression target variable, of shape (N, d) where N is the
        number of samples and d is the dimension of the target, which is also the
        dimension of the quantiles which will be estimated.
        :return: A VQF representing Q_{Y}.
        """
        pass


class VQRSolver(VQESolver, ABC):
    """
    Abstraction of a method for solving the Vector Quantile Regression (VQR) problem.
    """

    @abstractmethod
    def solve_vqr(
        self,
        Y: Array,
        X: Array,
    ) -> CVQF:
        """
        Solves the provided VQR problem in an implementation-specific way.

        :param Y: The regression target variable, of shape (N, d) where N is the
        number of samples and d is the dimension of the target, which is also the
        dimension of the quantiles which will be estimated.
        :param X: The regression input (features, covariates), of shape (N, k),
        where k is the number of features.
        :return: A CVQF representing Q_{Y|X}.
        """
        pass


class DiscreteSolver(Solver, ABC):
    """
    Represents a VQR solver which solves the discrete problem, i.e., obtains an
    estimate of the conditional vector quantile function discretized at T quantile
    levels per dimension.
    """

    @property
    @abstractmethod
    def levels_per_dim(self) -> int:
        """
        :return: T, the number of quantile levels to estimate along each of the d
        dimensions. The quantile level will be spaced uniformly between 0 and 1.
        """
        pass


class VQRDiscreteSolver(VQRSolver, DiscreteSolver, ABC):
    """
    Represents a VQR solver which solves the discrete problem, i.e., obtains an
    estimate of the conditional vector quantile function discretized at T quantile
    levels per dimension.
    """

    pass


class VQEDiscreteSolver(VQESolver, DiscreteSolver, ABC):
    """
    Represents a VQE solver which solves the discrete problem, i.e., obtains an
    estimate of the vector quantile function discretized at T quantile
    levels per dimension.
    """

    pass
