from typing import Tuple, Optional, Sequence

import numpy as np
from numpy import ndarray
from numpy import ndarray as Array
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from scipy.spatial import ConvexHull
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from vqr.vqr import quantile_levels, quantile_contour
from vqr.coverage import point_in_hull


def _level_label(t: float) -> str:
    """
    Generates labels for quantile levels in plots
    :param t: quantile level
    :return: plot label
    """
    s = f"{t:.2f}"
    if s.startswith("0"):
        s = s[1:]
    return s


def plot_quantiles(
    T: int,
    d: int,
    Qs: Sequence[Array],
    Us: Sequence[Array],
    figsize: Optional[Tuple[int, int]] = None,
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
    :param Qs: Quantile surfaces per dimension of Y. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :param Us: Quantile levels per dimension of U. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :param figsize: Size of figure to create. Will be passed to plt.subplots.
    :return: The created figure.
    """
    if d > 2:
        raise RuntimeError("Can't plot quantiles with dimension greater than 2")

    assert d == len(Qs) == len(Us)

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
    tick_labels = [_level_label(t) for t in levels]

    for i, (ax, Q) in enumerate(zip(axes, Qs)):
        if d == 1:
            ax.plot(*Us, Q)
            ax.set_title(f"$Q_{{{i + 1}}}(u_1)$")
            ax.set_xlabel(f"$u_1$")
            ax.set_xticks(levels)
            ax.set_xticklabels(
                tick_labels, rotation=90, ha="right", rotation_mode="anchor"
            )
            ax.grid(True)

        elif d == 2:
            Q = Q.T  # to match the axes
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


def plot_quantiles_3d(
    T,
    d,
    Qs: Sequence[Array],
    Us: Sequence[Array],
    colorbar: bool = True,
    cmap: str = "viridis",
    alpha: float = 1.0,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[Figure, Sequence[Axes]]:
    """
    Plots vector quantiles of d=2 or d=3 obtained from the solution of the VQR optimal
    transport problem.

    A new figure will be created. A three-dimensional plot will be created, where for d=2
    the quantiles will be plotted as surfaces, and for d=3 the quantiles will be
    plotted as voxels, where the color of the quantile corresponds to the value of the
    quantile.

    :param T: The number of quantile levels that was used for solving the problem.
    :param d: The dimension of the target data (Y) that was used for solving the
        problem.
    :param Qs: Quantile surfaces per dimension of Y. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :param Us: Quantile levels per dimension of U. A sequence of length d,
    where each element is of shape (T, T, ..., T).
    :param colorbar: whether to add a colorbar.
    :param cmap: Colormap name.
    :param alpha: Color alpha value (transparency).
    :param figsize: Size of figure to create. Will be passed to plt.subplots.
    :return: The created figure.
    """
    if not 1 < d < 4:
        raise RuntimeError("Can't plot 3d quantiles with dimension other than 2, 3")

    # Transpose to match axes
    # TODO: Need to check why this is needed
    axes_perm = [*range(1, d), 0]
    Qs = [np.transpose(Q, axes_perm) for Q in Qs]
    Us = [np.transpose(U, axes_perm) for U in Us]

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
    tick_labels = [_level_label(t) for t in levels]

    for i, (ax, Q) in enumerate(zip(axes, Qs)):
        if d == 2:
            ticks = levels
            m = ax.plot_surface(*Us, Q, cmap=cmap, alpha=alpha)
            ax.set_title(f"$Q_{{{i + 1}}}(u_1, u_2)$")
            ax.set_xlabel("$u_1$")
            ax.set_ylabel("$u_2$")
            if colorbar:
                fig.colorbar(m, ax=[ax], shrink=0.2)

        if d == 3:
            ticks = levels * T - 1
            cm = plt.get_cmap(cmap)
            norm = plt.Normalize(np.min(Q), np.max(Q))
            active_voxesls = np.ones_like(Q)
            ax.voxels(
                active_voxesls,
                facecolors=cm(norm(Q)),
                edgecolors=None,
                alpha=alpha,
            )
            ax.set_zticks(ticks)
            ax.set_zticklabels(tick_labels)
            ax.set_title(f"$Q_{{{i + 1}}}(u_1, u_2, u_3)$")
            ax.set_xlabel("$u_1$")
            ax.set_ylabel("$u_2$")
            ax.set_zlabel("$u_3$")

            if colorbar:
                fig.colorbar(ScalarMappable(norm=norm, cmap=cm), ax=[ax], shrink=0.2)

        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)

        ax.locator_params(axis="both", tight=True, nbins=10)

    return fig, axes


def plot_contour_2d(
    Qs: Sequence[Array],
    Y: Array = None,
    alphas: Sequence[float] = (0.1,),
    alpha_labels: Sequence[str] = None,
    alpha_colors: Sequence[str] = None,
    ax: Axes = None,
):
    """
    Plots two-dimensional coverage plots.
    Can show both training and validation data, and calculate coverage.

    :param Qs: The quantile surfaces
    :param Y: optional data to plot
    :param alphas: Contour levels to plot.
    :param ax: Optional axes object to plot to. If not provided, will create a new
    figure.
    :return: A tuple with the (Figure, Axis).
    """
    assert len(Qs) == 2
    Q1, Q2 = Qs

    T = Q1.shape[0]
    assert Q1.shape == Q2.shape == (T, T)
    assert all(0.0 < alpha < 0.5 for alpha in alphas)

    if alpha_labels:
        assert len(alpha_labels) == len(alphas)
    if alpha_colors:
        assert len(alpha_colors) == len(alphas)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = None, ax

    if Y is not None:
        assert np.ndim(Y) == 2 and Y.shape[1] == 2
        ax.scatter(
            Y[:, 0],
            Y[:, 1],
            alpha=0.4,
            color="k",
            marker=".",
            edgecolor="white",
            label="Y",
        )

    for i, alpha in enumerate(alphas):
        Q_contour = quantile_contour(T, d=2, Qs=[Q1, Q2], alpha=alpha)[0]

        label = alpha_labels[i] if alpha_labels else rf"$\alpha$={alpha:.2f}"
        color = alpha_colors[i] if alpha_colors else f"C{i}"
        surf_kws = dict(alpha=0.5, color=color, s=200, marker=".", label=label)
        ax.scatter(
            *Q_contour.T,
            **surf_kws,
        )

    ax.legend()
    return fig, ax


def plot_coverage_2d(
    Q1: Array,
    Q2: Array,
    Y_valid: Array = None,
    Y_train: Array = None,
    alpha: float = 0.1,
    ax: Axes = None,
    title: str = None,
    xylim: str = None,
    xlabel: str = None,
    ylabel: str = None,
    contour_color: str = "k",
    contour_label: str = None,
):
    """
    Plots two-dimensional coverage plots.
    Can show both training and validation data, and calculate coverage.

    :param Q1: The first 2d quantile surface.
    :param Q2: The second 2d quantile surface.
    :param Y_valid: Validation data points (N, 2).
    :param Y_train: Training data points (N', 2)
    :param alpha: Quantile level for coverage calculation. Should be <0.5.
    E.g., 0.05 means the contour will correspond to 0.95 of the data in each dimension.
    :param ax: Optional axes object to plot to. If not provided, will create a new
    figure.
    :param title: Title to give the axes.
    :param xylim: Limits for x and y axes.
    :param xlabel: Label for x axis.
    :param ylabel: Label for y axis.
    :param contour_color: Color for contour line.
    :param contour_label: Legend label for contour line.
    :return: A tuple with the (Figure, Axis, validation coverage).
    """
    T = Q1.shape[0]
    assert Q1.shape == Q2.shape == (T, T)
    assert 0.0 < alpha < 0.5

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = None, ax

    Q_surface = quantile_contour(T, d=2, Qs=[Q1, Q2], alpha=alpha)[0]

    surf_kws = dict(alpha=0.5, color=contour_color, s=200, marker="v")
    ax.scatter(*Q_surface.T, **surf_kws)

    # Plot convex hull
    hull = ConvexHull(Q_surface)
    for i, simplex in enumerate(hull.simplices):
        label = None
        if i == len(hull.simplices) - 1:
            label = contour_label or rf"Quantile Contour ($\alpha$={alpha})"
        ax.plot(
            Q_surface[simplex, 0],
            Q_surface[simplex, 1],
            color=contour_color,
            label=label,
        )

    # Plot training data
    train_coverage = None
    if Y_train is not None:
        is_in_hull = np.array([point_in_hull(p, hull) for p in Y_train])
        train_coverage = np.mean(is_in_hull).item()
        ax.scatter(
            *Y_train.T,
            color="k",
            alpha=0.3,
            label=f"training data (cov={train_coverage * 100:.2f})",
        )

    val_coverage = None
    # Plot validation data
    if Y_valid is not None:
        is_in_hull = np.array([point_in_hull(p, hull) for p in Y_valid])
        val_coverage = np.mean(is_in_hull).item()
        ax.scatter(
            *Y_valid[is_in_hull, :].T,
            marker="x",
            color="g",
            label=f"validation (cov={val_coverage * 100:.2f})",
        )
        ax.scatter(
            *Y_valid[~is_in_hull, :].T,
            marker="x",
            color="m",
            label="validation outliers",
        )

    ax.set_xlim(xylim)
    ax.set_ylim(xylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig, ax, val_coverage
