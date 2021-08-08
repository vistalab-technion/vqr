from typing import Tuple, Optional, Sequence

import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from vqr.vqr import quantile_levels, decode_quantile_grid, decode_quantile_values


def plot_quantiles(
    T: int, d: int, U: ndarray, A: ndarray, figsize: Optional[Tuple[int, int]] = None
) -> Figure:
    """
    Plots scalar quantiles (d=1) or vector quantiles of d=2 obtained from the
    solution of the VQR optimal transport problem.

    A new figure will be created. Scalar quantiles will be
    plotted using a simple line plot, while vector quantiles with d=2 will be plotted
    as an image, where the pixel colors correspond to quantile value.

    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param U: The encoded grid U, of shape (T**d, d).
    :param A: The regression coefficients, of shape (T**d, 1).
    :param figsize: Size of figure to create. Will be passed to plt.subplots.
    :return: The created figure.
    """
    if d > 2:
        raise RuntimeError("Can't plot quantiles with dimension greater than 2")

    fig: Figure
    _axes: ndarray
    fig, _axes = plt.subplots(
        nrows=1,
        ncols=d,
        figsize=figsize,
        squeeze=False,
    )
    # _axes is (1, d), take first row to get (d,)
    axes: Sequence[Axes] = list(_axes[0])

    levels: ndarray = quantile_levels(T)
    tick_labels = [f"{t:.2f}" for t in levels]

    U_grid = decode_quantile_grid(T, d, U)
    Q_values = decode_quantile_values(T, d, A)
    for i, (ax, Q) in enumerate(zip(axes, Q_values)):
        if d == 1:
            ax.plot(*U_grid, Q)
            ax.set_xticks(levels)
            ax.set_xticklabels(
                tick_labels, rotation=90, ha="right", rotation_mode="anchor"
            )

        elif d == 2:
            m = ax.imshow(Q, aspect="equal", interpolation="none", origin="lower")

            ticks = levels * T - 1
            ax.set_title(f"$Q_{{{i + 1}}}(u_1, u_2)$")
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


def plot_quantiles_3d(T, d, U, A, figsize: Optional[Tuple[int, int]] = None) -> Figure:
    """
    Plots vector quantiles of d=2 or d-3 obtained from the solution of the VQR optimal
    transport problem.

    A new figure will be created. A three-dimensional plot will be created, where for d=2
    the quantiles will be plotted as surfaces, and for d=3 the quantiles will be
    plotted as voxels, where the color of the quantile corresponds to the value of the
    quantile.

    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param U: The encoded grid U, of shape (T**d, d).
    :param A: The regression coefficients, of shape (T**d, 1).
    :param figsize: Size of figure to create. Will be passed to plt.subplots.
    :return: The created figure.
    """
    if not 1 < d < 4:
        raise RuntimeError("Can't plot 3d quantiles with dimension other than 2, 3")

    fig: Figure
    _axes: ndarray
    fig, _axes = plt.subplots(
        nrows=1,
        ncols=d,
        figsize=figsize,
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )
    axes: Sequence[Axes3D] = list(_axes[0])

    levels: ndarray = quantile_levels(T)
    tick_labels = [f"{t:.2f}" for t in levels]

    U_grid = decode_quantile_grid(T, d, U)
    Q_values = decode_quantile_values(T, d, A)
    for i, (ax, Q) in enumerate(zip(axes, Q_values)):
        if d == 2:
            ticks = levels
            m = ax.plot_surface(*U_grid, Q, cmap="viridis")
            fig.colorbar(m, ax=[ax], shrink=0.2)
            ax.set_title(f"$Q_{{{i + 1}}}(u_1, u_2)$")

        if d == 3:
            ticks = levels * T - 1
            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(np.min(Q), np.max(Q))
            ax.voxels(np.ones_like(Q), facecolors=cmap(norm(Q)), edgecolors="black")
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=[ax], shrink=0.2)
            ax.set_zticks(ticks)
            ax.set_zticklabels(tick_labels)
            ax.set_zlabel("$u_3$")
            ax.set_title(f"$Q_{{{i + 1}}}(u_1, u_2, u_3)$")

        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels)
        ax.set_ylabel("$u_1$")
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("$u_2$")

        ax.locator_params(axis="both", tight=True, nbins=10)

    return fig
