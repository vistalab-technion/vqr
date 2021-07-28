from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Sequence

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.base import BaseEstimator, RegressorMixin
from matplotlib.cm import ScalarMappable
from sklearn.utils import check_X_y
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.exceptions import NotFittedError
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.utils.validation import check_is_fitted

DEFAULT_METRIC = lambda x, y: np.dot(x, y)


def quantile_levels(n_levels: int) -> ndarray:
    """
    Creates a vector of evenly-spaced quantile levels between zero and one.
    :param n_levels: Number of
    :return: An ndarray of shape (n_levels,).
    """
    T = n_levels
    return (np.arange(T) + 1) * (1 / T)


def vqr_ot(
    Y: ndarray,
    X: Optional[ndarray] = None,
    n_levels: int = 50,
    metric: Union[str, Callable] = DEFAULT_METRIC,
    solver_opts: Optional[Dict[str, Any]] = None,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Solves the Optimal Transport formulation of Vector Quantile Regression.

    See:
        Carlier, Chernozhukov, Galichon. Vector quantile regression:
        An optimal transport approach,
        Annals of Statistics, 2016

    :param Y: The regression target variable, of shape (N, d) where N is the number
        of samples and d is the dimension of the target, which is also the
        dimension of the quantiles which will be estimated.
    :param X: The regression input (features) variable, of shape (N, k), where k is
        the number of features. Note that X may be None, in which case the problem
        becomes quantile estimation (estimating the quantiles of Y) instead of
        quantile regression.
    :param n_levels: Number of quantile levels to estimate along each of the
        d dimensions. The quantile level will be spaced uniformly between 0 to 1.
    :param metric: The metric to use in order to compute pairwise distances between
        vectors. Should accept to vectors in dimension d and return a scalar.
    :param solver_opts: Extra arguments for CVXPY's solve().
    :return: A tuple of three ndarrays:
        - U, of shape (T**d, d): This contains the d-dimensional grid on which the
            vector quantiles are defined. If can be decoded back into a
            meshgrid-style sequence of arrays using :obj:`decode_quantile_grid`.
        - A, of shape (T**d, 1): This contains the "alpha" Lagrange multiplier
            which contains the regression intercept variable. This can be decoded
            into the d vector quantiles of Y using the
            :obj:`decode_quantile_values` function.
        - B, of shape (T**d, k): This contains the "beta" Lagrange multiplier
            which contains the regression coefficients.

        The outputs should be used as follows:
        Given a sample x of shape (1, k), the conditional quantiles Y|X=x are given by
        Y_hat = B @ x.T  + A.
        This is an array of shape (T**d, 1). In order to obtain the d vector quantiles
        (one for each dimension of Y), use the :obj:`decode_quantile_values`
        function on Y_hat. These surfaces can be visualized over the grid defined in U.
    """

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
    T: int = n_levels
    Td: int = T ** d
    u: ndarray = quantile_levels(T)

    # Quantile levels grid: list of grid coordinate matrices, one per dimension
    U_grids: Sequence[ndarray] = np.meshgrid(*([u] * d))
    # Stack all nd-grid coordinates into one long matrix, of shape (T**d, d)
    U: ndarray = np.stack([U_grid.reshape(-1) for U_grid in U_grids], axis=1)
    assert U.shape == (Td, d)

    # Pairwise distances (similarity)
    S: ndarray = cdist(U, Y, metric)

    # Optimization problem definition: optimal transport formulation
    one_N = np.ones([N, 1])
    one_T = np.ones([Td, 1])
    Pi = cp.Variable(shape=(Td, N))
    Pi_S = cp.sum(cp.multiply(Pi, S))
    constraints = [
        Pi @ X == 1 / Td * one_T @ X_bar.T,
        Pi >= 0,
        one_T.T @ Pi == 1 / N * one_N.T,
    ]
    problem = cp.Problem(objective=cp.Maximize(Pi_S), constraints=constraints)

    # Solve the problem
    problem.solve(**solver_opts)

    # Obtain the lagrange multipliers Alpha (A) and Beta (B)
    AB: ndarray = constraints[0].dual_value
    AB = np.reshape(AB, newshape=[Td, k + 1])
    A = AB[:, [0]]  # A is (T**d, 1)
    if k == 0:
        B = None
    else:
        B = AB[:, 1:]  # B is (T**d, k)

    return U, A, B


def decode_quantile_values(T: int, d: int, Q: ndarray) -> Sequence[ndarray]:
    """
    Decodes the regression coefficients of a VQR solution into vector quantile values.
    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param Q: The regression coefficients, of shape (T**d, 1).
    :return: A sequence of length d of vector quantile values. Each element j in the
        sequence is a d-dimensional array of shape (T, T, ..., T) containing the
        vector quantiles values of j-th variable in Y, i.e., the quantiles of Y_j|Y_{-j}
        where Y_{-j} means all the variables in Y except the j-th.
    """
    Q = np.reshape(Q, newshape=(T,) * d)

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

        Q_functions[d - 1 - axis] = dQ_du

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


class VectorQuantileRegressor(RegressorMixin, BaseEstimator):
    """
    Performs vector quantile regression and estimation.
    """

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

    def fit(self, X: ndarray, y: ndarray):
        """
        Fits a quantile regression model to the given data. In case the response
        variable y is high-dimentional, a vector-quantile model will be fitted.
        :param X: Features, (n, k). Currently ignored (TODO: Handle X)
        :param y: Responses, (n, d).
        :return: self.
        """

        # Input validation.
        X, y = check_X_y(X, y, multi_output=True, ensure_2d=True)
        N = len(X)
        Y: ndarray = np.reshape(y, (N, -1))

        self.k_: int = X.shape[1]  # number of features
        self.d_: int = Y.shape[1]  # number or target dimensions
        self.u_: ndarray = quantile_levels(self.n_levels)

        self.U_, self.A_, self.B_ = vqr_ot(
            # TODO: X
            Y,
            None,
            self.n_levels,
            self.metric,
            self.solver_opts,
        )

        self.U_grids_: Sequence[ndarray] = decode_quantile_grid(
            self.n_levels, self.d_, self.U_
        )
        self.Q_surfaces_: Sequence[ndarray] = decode_quantile_values(
            self.n_levels, self.d_, self.A_
        )
        return self

    def predict(self, X: ndarray):
        """
        TODO
        :param X:
        :return:
        """
        check_is_fitted(self)
        pass

    @property
    def quantile_values(self) -> Sequence[ndarray]:
        """
        :return: A sequence of quantile value ndarrays. The sequence is of length d,
            where d is the dimension of the target variable (Y). The j-th ndarray is
            the d-dimensional vector quantile of the j-th variable in Y.
        """
        check_is_fitted(self)
        return self.Q_surfaces_

    @property
    def quantile_grid(self) -> Sequence[ndarray]:
        """
        :return: A sequence of quantile level grids as ndarrays. This is a
            d-dimensional meshgrid (see np.meshgrid) where d is the dimension of the
            target variable Y.
        """
        check_is_fitted(self)
        return self.U_grids_

    @property
    def quantile_dimension(self) -> int:
        """
        :return: The dimension of the fitted vector quantiles.
        """
        check_is_fitted(self)
        return self.d_

    @property
    def quantile_levels(self) -> ndarray:
        """
        :return: An ndarray containing the levels at which the vector quantiles were
            estimated along each target dimension.
        """
        check_is_fitted(self)
        return self.u_

    def plot_quantiles(self, figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Plots 1d or 2d quantiles. A new figure will be created. 1d quantiles will be
        plotted using a simple line plot, while 2d quantiles will be plottes as an
        image, where the pixel colors correspont to quantile value.
        :param figsize: Size of figure to create. Will be passed to plt.subplots.
        :return: The created figure.
        """
        if self.quantile_dimension > 2:
            raise RuntimeError("Can't plot quantiles with dimension greater than 2")

        fig: Figure
        _axes: ndarray
        fig, _axes = plt.subplots(
            nrows=1,
            ncols=self.quantile_dimension,
            figsize=figsize,
            squeeze=False,
        )
        # _axes is (1, d), take first row to get (d,)
        axes: Sequence[Axes] = list(_axes[0])

        tick_labels = [f"{t:.2f}" for t in self.quantile_levels]

        U = self.quantile_grid
        for i, (ax, Q) in enumerate(zip(axes, self.quantile_values)):

            if self.quantile_dimension == 1:
                ax.plot(*U, Q)
                ax.set_xticks(self.quantile_levels)
                ax.set_xticklabels(
                    tick_labels, rotation=90, ha="right", rotation_mode="anchor"
                )

            elif self.quantile_dimension == 2:
                m = ax.imshow(Q, aspect="equal", interpolation="none", origin="lower")

                ticks = self.quantile_levels * self.n_levels - 1
                ax.set_title(f"$Q_{{{i+1}}}(u_1, u_2)$")
                ax.set_yticks(ticks)
                ax.set_yticklabels(tick_labels)
                ax.set_ylabel("$u_1$")
                ax.xaxis.set_ticks_position("bottom")
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    tick_labels, rotation=90, ha="right", rotation_mode="anchor"
                )
                ax.set_xlabel("$u_2$")

                fig.colorbar(m, ax=[ax], shrink=0.2)

            ax.locator_params(axis="both", tight=True, nbins=20)
        return fig

    def plot_quantiles_3d(self, figsize: Optional[Tuple[int, int]] = None) -> Figure:
        """
        Plots 2d or 3d quantiles. A new figure will be created. 2d quantiles will be
        plotted as surfaces, while 3d quantiles will be plotted as voxels, where the
        color of the quantile corresponds to the value of the quantile.
        :param figsize: Size of figure to create. Will be passed to plt.subplots.
        :return: The created figure.
        """
        if not 1 < self.quantile_dimension < 4:
            raise RuntimeError("Can't plot 3d quantiles with dimension other than 2, 3")

        fig: Figure
        _axes: ndarray
        fig, _axes = plt.subplots(
            nrows=1,
            ncols=self.quantile_dimension,
            figsize=figsize,
            squeeze=False,
            subplot_kw={"projection": "3d"},
        )
        axes: Sequence[Axes3D] = list(_axes[0])

        tick_labels = [f"{t:.2f}" for t in self.quantile_levels]

        U = self.quantile_grid
        for i, (ax, Q) in enumerate(zip(axes, self.quantile_values)):
            if self.quantile_dimension == 2:
                ticks = self.quantile_levels
                m = ax.plot_surface(*U, Q, cmap="viridis")
                fig.colorbar(m, ax=[ax], shrink=0.2)

            if self.quantile_dimension == 3:
                ticks = self.quantile_levels * self.n_levels - 1
                cmap = plt.get_cmap("viridis")
                norm = plt.Normalize(Q.min(), Q.max())
                ax.voxels(np.ones_like(Q), facecolors=cmap(norm(Q)), edgecolors="black")
                fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=[ax], shrink=0.2)
                ax.set_zticks(ticks)
                ax.set_zticklabels(tick_labels)
                ax.set_zlabel("$u_3$")

            ax.set_title(f"$Q_{{{i+1}}}(u_1, u_2, u_3)$")
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels)
            ax.set_ylabel("$u_1$")
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel("$u_2$")

            ax.locator_params(axis="both", tight=True, nbins=10)

        return fig

    def __repr__(self):
        cls = self.__class__.__name__
        fitted = True
        try:
            check_is_fitted(self)
        except NotFittedError:
            fitted = False

        fields_strs = [f"{fitted=}", f"n_levels={self.n_levels}"]
        if fitted:
            fields_strs.append(f"d={self.quantile_dimension}")

        return f"{cls}({str.join(', ', fields_strs)})"
