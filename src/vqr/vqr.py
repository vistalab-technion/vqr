from __future__ import annotations

from abc import ABC, abstractmethod
from time import time
from typing import Any, Dict, List, Union, Callable, Optional, Sequence

import cvxpy as cp
import numpy as np
import torch
from numpy import array, ndarray
from torch import Tensor, nn, eye, diag
from torch import ones as ones_th
from torch import tensor
from numpy.typing import ArrayLike as Array
from sklearn.utils import check_array
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import ReduceLROnPlateau

SIMILARITY_FN_INNER_PROD = lambda x, y: np.dot(x, y)


class VectorQuantiles:
    """
    Encapsulates the solution to a VQR problem. Contains the vector quantiles and
    regression coefficients of the solution, and provides useful methods for
    interacting with them.

    Should only be constructed by :class:`VQRSolver`s, not manually.

    Given a sample x of shape (1, k), the conditional d-dimensional vector quantiles
    Y|X=x are given by
    Y_hat = B @ x.T  + A.
    This is an array of shape (T**d, 1).
    In order to obtain the d vector quantile surfaces (one for each dimension of Y),
    use the :obj:`decode_quantile_values` function on Y_hat.
    These surfaces can be visualized over the grid defined in U.
    """

    def __init__(
        self,
        T: int,
        d: int,
        U: Array,
        A: Array,
        B: Optional[Array] = None,
        X_transform: Optional[Callable[[Array], Array]] = None,
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
        regression problem.
        :param X_transform: Transformation to apply to covariates (X) for non-linear
        VQR.
        """
        # Validate dimensions
        assert all(x is not None for x in [T, d, U, A])
        assert U.ndim == 2 and A.ndim == 2
        assert U.shape[0] == A.shape[0] == T ** d
        assert A.shape[1] == 1
        assert B is None or (B.ndim == 2 and B.shape[0] == T ** d)

        self._T = T
        self._d = d
        self._U = U
        self._A = A
        self._B = B
        self._k = B.shape[1] if B is not None else 0
        self._X_transform = X_transform

    @property
    def is_conditional(self) -> bool:
        return self._B is not None

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
        if X is not None and not self.is_conditional:
            raise ValueError(f"VQR not conditional but covariates were supplied")

        if not self.is_conditional:
            Y_hats = [self._A]
        else:
            if X is None:
                X = np.zeros(shape=(1, self._k))

            check_array(X, ensure_2d=True, allow_nd=False)

            if self._X_transform is not None:
                X = self._X_transform(X)

            N, k = X.shape
            if k != self._k:
                raise ValueError(
                    f"VQR model was fitted with k={self._k}, "
                    f"but got data with {k=} features."
                )

            B = self._B  # (T**d, k)
            A = self._A  # (T**d, 1) -> will be broadcast to (T**d, N)
            Y_hat = B @ X.T + A  # result is (T**d, N)
            Y_hats = Y_hat.T  # (N, T**d)

        return tuple(
            decode_quantile_values(self._T, self._d, Y_hat) for Y_hat in Y_hats
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
        :return: An ndarray containing the levels at which the vector quantiles were
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
        :return: The dimension k, of the covariates (X).
        """
        return self._k


class VQRSolver(ABC):
    """
    Abstraction of a method for solving the Vector Quantile Regression (VQR) problem.

    For an overview of the VQR problem and it's solutions, refer to:
        Carlier G, Chernozhukov V, De Bie G, Galichon A.
        Vector quantile regression and optimal transport, from theory to numerics.
        Empirical Econometrics, 2020.
    """

    def __init__(
        self,
        similarity_fn: Union[str, Callable] = SIMILARITY_FN_INNER_PROD,
        verbose: bool = False,
        **solver_opts,
    ):
        """
        :param similarity_fn: A scalar function to use in order to compute pairwise
            similarity/distance between the data (Y) and the quantile-grid (U).
            Should accept to vectors in dimension d and return a scalar.
        :param solver_opts: Kwargs for underlying solver. Implementation specific.
        """
        self._similarity_fn = similarity_fn
        self._solver_opts = solver_opts
        self._verbose = verbose

    @abstractmethod
    def solve_vqr(
        self,
        T: int,
        Y: Array,
        X: Optional[Array] = None,
    ) -> VectorQuantiles:
        """
        Solves the provided VQR problem in an implementation-specific way.

        :param T: Number of quantile levels to estimate along each of the d
        dimensions. The quantile level will be spaced uniformly between 0 to 1.
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


class RVQRDualLSESolver(VQRSolver):
    """
    Solves the Regularized Dual formulation of Vector Quantile Regression using
    torch with SGD as a solver backend.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        num_epochs: int = 1000,
        learning_rate: float = 0.9,
        verbose: bool = False,
        **solver_opts,
    ):
        super().__init__(
            similarity_fn=SIMILARITY_FN_INNER_PROD, verbose=verbose, **solver_opts
        )
        self._epsilon = epsilon
        self._num_epochs = num_epochs
        self._lr = learning_rate

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
        S: Array = cdist(U, Y, self._similarity_fn)  # (Td, d) and (N, d)

        one_N = np.ones([N, 1])
        one_T = np.ones([Td, 1])

        #####
        dtype = torch.float32
        Y_th = tensor(Y, dtype=dtype)
        U_th = tensor(U, dtype=dtype)
        mu = tensor(one_T / Td, dtype=dtype)
        nu = tensor(one_N / N, dtype=dtype)
        X_th = tensor(X, dtype=dtype)
        b = torch.zeros(*(Td, X.shape[-1] - 1), dtype=dtype, requires_grad=True)
        psi_init = 0.1 * torch.ones(N, dtype=dtype)
        psi = psi_init.clone().detach().requires_grad_(True)
        epsilon = self._epsilon
        num_epochs = self._num_epochs

        optimizer = torch.optim.SGD(
            params=[b, psi],
            lr=self._lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0,
        )

        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=100,
            threshold=5 * 0.01,  # loss needs to decrease by x% every patience epochs
            threshold_mode="rel",
            min_lr=self._lr * 0.5 ** 10,
            verbose=self._verbose,
        )
        UY = U_th @ Y_th.T

        def _forward():
            pass

        total_time, last_print_time = 0, time()
        for epoch_idx in range(num_epochs):
            epoch_start_time = time()

            optimizer.zero_grad()
            bX = b @ X_th[:, 1:].T
            max_arg = UY - bX - psi.reshape(1, -1)
            phi = (
                epsilon
                * torch.log(
                    torch.sum(
                        torch.exp(
                            (max_arg - torch.max(max_arg, dim=1)[0][:, None]) / epsilon
                        ),
                        dim=1,
                    )
                )
                + torch.max(max_arg, dim=1)[0]
            )
            obj = psi @ nu + phi @ mu
            obj.backward()
            optimizer.step()
            total_loss = obj.item()
            constraint_loss = (phi @ mu).item()

            scheduler.step(total_loss)

            epoch_elapsed_time = time() - epoch_start_time
            total_time += epoch_elapsed_time
            if self._verbose and (epoch_idx % 100 == 0 or epoch_idx == num_epochs - 1):
                elapsed = time() - last_print_time
                print(
                    f"{epoch_idx=}, {total_loss=:.6f} {constraint_loss=:.6f}, "
                    f"{elapsed=:.2f}s"
                )
                last_print_time = time()

        max_arg = UY - bX - psi.reshape(1, -1)
        phi = (
            epsilon
            * torch.log(
                torch.sum(
                    torch.exp(
                        (max_arg - torch.max(max_arg, dim=1)[0][:, None]) / epsilon
                    ),
                    dim=1,
                )
            )
            + torch.max(max_arg, dim=1)[0]
        )

        A = phi.detach().numpy()[:, None]
        if k == 0:
            B = None
        else:
            B = b.detach().numpy()

        if self._verbose:
            print(f"{total_time=:.2f}s")
        return VectorQuantiles(T, d, U, A, B)


class Network(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.C1 = nn.Parameter(eye(k, k, dtype=torch.float32, requires_grad=True))
        self.C2 = nn.Parameter(
            ones_th(k, dtype=torch.float32, requires_grad=True),
        )
        self.C3 = nn.Parameter(
            torch.tensor(array([0.0]), dtype=torch.float32, requires_grad=True)
        )
        self.batch_norm = nn.BatchNorm1d(
            num_features=k, affine=False, track_running_stats=True
        )

    def forward(self, X: Tensor):
        return self.batch_norm(diag(X @ self.C1 @ X.T)[:, None] + self.C2 * X + self.C3)


class DeepNet(nn.Module):
    def __init__(self, hidden_width=500, depth=1, k=2):
        super().__init__()
        self.nl = nn.ReLU()
        self.fc_first = nn.Linear(k, hidden_width)
        self.fc_last = nn.Linear(hidden_width, k)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_width, hidden_width) for _ in range(depth)]
        )
        self.bn_last = nn.BatchNorm1d(
            num_features=k, affine=False, track_running_stats=True
        )
        self.bn_hidden = nn.BatchNorm1d(
            num_features=hidden_width, affine=False, track_running_stats=True
        )

    def forward(self, X_in):
        X = self.nl(self.fc_first(X_in))
        for hidden in self.fc_hidden:
            X_hidden = self.nl(hidden(X))
            X = self.bn_hidden(diag(X_hidden @ X_hidden.T)[:, None] + X_hidden + X)
        X = self.bn_last(self.fc_last(X) + X_in)
        return X


class NonlinearRVQRDualLSESolver(VQRSolver):
    """
    Solves the Regularized Dual formulation of Vector Quantile Regression using
    torch with SGD as a solver backend.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        num_epochs: int = 1000,
        learning_rate: float = 0.9,
        verbose: bool = False,
        k: int = 2,
        **solver_opts,
    ):
        super().__init__(
            similarity_fn=SIMILARITY_FN_INNER_PROD, verbose=verbose, **solver_opts
        )
        self._epsilon = epsilon
        self._num_epochs = num_epochs
        self._lr = learning_rate

        # def g(x_):
        #     # Q = array([[2.0, 1.0], [1.0, 2.0]])
        #     # Q = tensor(Q, dtype=torch.float32)
        #     return diag(x_ @ self._Q @ x_.T)[:, None] + x_

        # self._net = g   # Oracle
        self._net = DeepNet(depth=1, k=k)  # DeepNet approx
        # self._net = Network(k=2)   # Parametric approx

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
        S: Array = cdist(U, Y, self._similarity_fn)  # (Td, d) and (N, d)

        one_N = np.ones([N, 1])
        one_T = np.ones([Td, 1])

        #####
        dtype = torch.float32
        Y_th = tensor(Y, dtype=dtype)
        U_th = tensor(U, dtype=dtype)
        mu = tensor(one_T / Td, dtype=dtype)
        nu = tensor(one_N / N, dtype=dtype)
        X_th = tensor(X, dtype=dtype)
        b = torch.zeros(*(Td, X.shape[-1] - 1), dtype=dtype, requires_grad=True)
        psi_init = 0.1 * torch.ones(N, dtype=dtype)
        psi = psi_init.clone().detach().requires_grad_(True)
        epsilon = self._epsilon
        num_epochs = self._num_epochs
        optimizer = torch.optim.SGD(
            [
                dict(params=[*self._net.parameters()]),
                dict(
                    params=[b, psi],
                ),
            ],
            lr=self._lr,
            momentum=0.9,
            nesterov=True,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, "min", factor=0.9, patience=50, verbose=True, threshold=1e-2
        )
        UY = U_th @ Y_th.T

        for epoch_idx in range(num_epochs):
            optimizer.zero_grad()
            bX = b @ self._net(X_th[:, 1:]).T
            max_arg = UY - bX - psi.reshape(1, -1)
            phi = (
                epsilon
                * torch.log(
                    torch.sum(
                        torch.exp(
                            (max_arg - torch.max(max_arg, dim=1)[0][:, None]) / epsilon
                        ),
                        dim=1,
                    )
                )
                + torch.max(max_arg, dim=1)[0]
            )
            obj = psi @ nu + phi @ mu
            obj.backward()
            optimizer.step()
            scheduler.step(obj)
            total_loss = obj.item()
            constraint_loss = (phi @ mu).item()

            if self._verbose and epoch_idx % 1 == 0:
                print(f"{epoch_idx=}, {total_loss=:.6f} {constraint_loss=:.6f}")
                if total_loss < -10:
                    break

        max_arg = UY - bX - psi.reshape(1, -1)
        phi = (
            epsilon
            * torch.log(
                torch.sum(
                    torch.exp(
                        (max_arg - torch.max(max_arg, dim=1)[0][:, None]) / epsilon
                    ),
                    dim=1,
                )
            )
            + torch.max(max_arg, dim=1)[0]
        )
        self._net.eval()

        A = phi.detach().numpy()[:, None]
        if k == 0:
            B = None
        else:
            B = b.detach().numpy()

        return VectorQuantiles(T, d, U, A, B)


class CVXVQRSolver(VQRSolver):
    """
    Solves the Optimal Transport formulation of Vector Quantile Regression using
    CVXPY as a solver backend.

    See:
        Carlier, Chernozhukov, Galichon. Vector quantile regression:
        An optimal transport approach,
        Annals of Statistics, 2016
    """

    def __init__(self, **cvx_solver_opts):
        super().__init__(similarity_fn=SIMILARITY_FN_INNER_PROD, **cvx_solver_opts)

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
        S: Array = cdist(U, Y, self._similarity_fn)  # (Td, d) and (N, d)

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


def quantile_levels(T: int) -> ndarray:
    """
    Creates a vector of evenly-spaced quantile levels between zero and one.
    :param T: Number of levels to create.
    :return: An ndarray of shape (T,).
    """
    return (np.arange(T) + 1) * (1 / T)


def decode_quantile_values(T: int, d: int, Y_hat: ndarray) -> Sequence[ndarray]:
    """
    Decodes the regression coefficients of a VQR solution into vector quantile values.
    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param Y_hat: The regression output, of shape (T**d, 1).
    :return: A sequence of length d of vector quantile values. Each element j in the
        sequence is a d-dimensional array of shape (T, T, ..., T) containing the
        vector quantiles values of j-th variable in Y, i.e., the quantiles of Y_j|Y_{-j}
        where Y_{-j} means all the variables in Y except the j-th.
    """
    Q = np.reshape(Y_hat, newshape=(T,) * d)

    Q_functions: List[ndarray] = [np.array([np.nan])] * d
    for axis in reversed(range(d)):
        # Calculate derivative along this axis
        dQ_du = (1 / T) * np.diff(Q, axis=axis)

        # Duplicate first "row" along axis and insert it first
        pad_with = [
            (0, 0),
        ] * d
        pad_with[axis] = (1, 0)
        dQ_du = np.pad(dQ_du, pad_width=pad_with, mode="edge")

        Q_functions[d - 1 - axis] = dQ_du * T ** 2

    return tuple(Q_functions)


def decode_quantile_grid(T: int, d: int, U: ndarray) -> Sequence[ndarray]:
    """
    Decodes the U variable of a VQR solution into the grid of the
    evaluation points for the vector quantile functions.
    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
    problem.
    :param U: The encoded grid U, of shape (T**d, d).
    :return: A sequence length d. Each ndarray in the sequence is a d-dimensional
    array of shape (T, T, ..., T), which together represent the d-dimentional grid
    on which the vector quantiles were evaluated.
    """
    return tuple(np.reshape(U[:, dim], newshape=(T,) * d) for dim in range(d))


def inversion_sampling(T: int, d: int, n: int, Qs: Sequence[Array]):
    """
    Generates samples from the variable Y based on it's fitted
    quantile function, using inversion-transform sampling.
    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param n: Number of samples to generate.
    :param Qs: Quantile functions per dimension of Y. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :return: Samples obtained from this quantile function, of shape (n, d).
    """

    # Samples of points on the quantile-level grid
    Us = np.random.randint(0, T, size=(n, d))

    # Sample from Y|X=x
    Y_samp = np.empty(shape=(n, d))
    for i, U_i in enumerate(Us):
        # U_i is a vector-quantile level, of shape (d,)
        Y_samp[i, :] = np.array([Q_d[tuple(U_i)] for Q_d in Qs])

    return Y_samp


def quantile_contour(T: int, d: int, Qs: Sequence[Array], alpha: float = 0.05) -> Array:
    """
    Creates a contour of points in d-dimensional space which surround the region in
    which 1-2*alpha of the distribution (that was corresponds to a given quantile
    function) is contained.

    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param Qs: Quantile functions per dimension of Y. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :param alpha: Confidence level for the contour.
    :return: An array of shape (n, d) containing points along the d-dimensional contour.
    """

    if not 0 < alpha < 0.5:
        raise ValueError(f"Got {alpha=}, but must be in (0, 0.5)")

    lo = int(np.round(T * alpha))
    hi = int(min(np.round(T * (1 - alpha)), T - 1))

    # Will contain d lists of points, each element will be (d,)
    contour_points_list = [[] for _ in range(d)]

    for i, Q in enumerate(Qs):
        for j in range(d):
            for lo_hi, _ in zip([lo, hi], range(d)):
                # The zip here is just to prevent a loop when d==1.
                # This keeps all dims fixed on either lo or hi, while the j-th
                # dimension is sweeped from lo to hi.
                idx: List[Union[int, slice]] = [lo_hi] * d
                idx[j] = slice(lo, hi)
                contour_points_list[i].extend(Q[tuple(idx)])

    contour_points = np.array(contour_points_list).T  # (N, d)
    return contour_points
