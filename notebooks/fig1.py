# %%
import os

# %%
import sys

if ".." not in sys.path:
    sys.path.append("..")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
from vqr import VectorQuantileEstimator
from vqr.vqr import quantile_contour, check_comonotonicity

# %%
from vqr.plot import plot_quantiles, plot_contour_2d, plot_quantiles_3d
from experiments import EXPERIMENTS_DATA_DIR
from experiments.data.shapes import (
    StarDataProvider,
    HeartDataProvider,
    generate_star,
    generate_heart,
)

os.makedirs("figs/fig1", exist_ok=True)


HEART_IMG = EXPERIMENTS_DATA_DIR / "heart.png"

mpl.rcParams["font.size"] = 14


# %%

# %% [markdown]
# # Figure 1

# %% [markdown]
# ## Shape datasets

# %%
heart_dp = HeartDataProvider(
    initial_rotation_deg=90, noise_std=0.01, x_max_deg=10, x_discrete=True
)

X, Y = heart_dp.sample(n=10000, x=10)

fig = plt.figure(figsize=(8, 8))
plt.scatter(Y[:, 0], Y[:, 1])

# %%
star_dp = StarDataProvider(
    initial_rotation_deg=0,
    noise_std=0.01,
    x_discrete=True,
)

X, Y = star_dp.sample(n=1000, x=20)

fig = plt.figure(figsize=(8, 8))
plt.scatter(Y[:, 0], Y[:, 1])

# %% [markdown]
# ## Fitting VQE

# %%
star_dp = StarDataProvider(
    initial_rotation_deg=90, noise_std=0.01, x_max_deg=20, x_discrete=True
)

_, Y_star = star_dp.sample(n=10000, x=0)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(Y_star[:, 0], Y_star[:, 1])

# %%

solver = "vqe_pot"
solver_opts = {"numItermax": int(2e6)}

# solver = "regularized_dual"
# solver_opts = {'epsilon': 1e-5, "verbose": True}

T = 50
vq2 = VectorQuantileEstimator(
    n_levels=T,
    solver=solver,
    solver_opts=solver_opts,
)
vq2.fit(Y_star)


# %% [markdown]
# ## Figure 1A: Vector Quantiles
#
# An $\alpha$-contour contains $100\cdot(1-2\alpha)^d$ percent of the data and can be
# seen as the vector quantile equivalent of a confidence interval.
#
# Each point on these contours is the value of a vector quantile, i.e. $Q_{Y}(\mathbf{u})
# $ for some $\mathbf{u}$.


# %%
alphas = [0.02, 0.1, 0.2, 0.3, 0.4]
refines = [True]
xylim = [-0.5, 0.5]

fig, axes = plt.subplots(
    1, len(refines), figsize=(10 * len(refines), 10), squeeze=False
)

for i, (ax, refine) in enumerate(zip(axes.reshape(-1), refines)):
    Qs = list(vq2.vector_quantiles(refine=refine))
    Us = vq2.quantile_grid

    plot_contour_2d(
        Qs=Qs,
        alphas=alphas,
        ax=ax,
        Y=Y_star,
    )
    # ax.set_title(f"{refine=}")
    ax.set_xlim(xylim)
    ax.set_ylim(xylim)
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")

    violations = check_comonotonicity(T, d=2, Qs=Qs, Us=Us) < 0
    print(f"violations ({refine=}): {np.sum(violations) / np.prod(violations.shape)}")

fig.savefig("figs/fig1/fig1a.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## Figure 1B: Vector quantile component surfaces
#
# Each surface represents a component of the vector quantile function.
# On Q1, walking along u1 with a fixed u2 yields a monotonically increasing quantile
# curve, and vice versa for Q2.


# %%


Qs = vq2.vector_quantiles(refine=True)
Us = vq2.quantile_grid

fig, axes = plot_quantiles_3d(
    T,
    d=2,
    Qs=Qs,
    Us=Us,
    figsize=(25, 20),
    colorbar=False,
    alpha=0.5,
    cmap="viridis",
)

# Plot contours on the 3d surface
for alpha in alphas:
    QC_Qvals, QC_idx = quantile_contour(T, d=2, Qs=Qs, alpha=alpha)
    QC_Uvals = np.array([(Us[0][u1, u2], Us[1][u1, u2]) for (u1, u2) in QC_idx])
    axes[0].scatter3D(
        xs=QC_Uvals[:, 0], ys=QC_Uvals[:, 1], zs=QC_Qvals[:, 0], zorder=10
    )
    axes[1].scatter3D(
        xs=QC_Uvals[:, 0], ys=QC_Uvals[:, 1], zs=QC_Qvals[:, 1], zorder=10
    )


axes[0].view_init(axes[0].elev, -150)

fig.savefig("figs/fig1/fig1b.png", dpi=150, bbox_inches="tight")


# %% [markdown]
# ## Figure 1C: Conditional quantiles
#
# Here quantiles function of the data depend nonlinearly on $x$: the distribution of
# $Y|X=x$ is the distribution of $Y$ rotated by $x$ degrees.
# We show contours of different CVQFs, i.e. given different values of $x$.

# %%

X = np.array([[10], [25], [40]])

fig, ax = plt.subplots(1, 1, figsize=(10, 10), squeeze=True)
xylim = [-0.5, 0.5]

for i, x in enumerate(X):
    _, Y_star = star_dp.sample(n=10000, x=x)

    # Using VQE per x instead of VQR on all the data since it's faster and it's just
    # used for the plot.
    vq2 = VectorQuantileEstimator(
        n_levels=T,
        solver=solver,
        solver_opts=solver_opts,
    )
    vq2.fit(Y_star)

    plot_contour_2d(
        Qs=vq2.vector_quantiles(),
        alphas=(0.02, 0.22),
        ax=ax,
        alpha_labels=(f"$x$={x.item():.1f}", ""),
        alpha_colors=(f"C{i}", f"C{i}"),
    )
    ax.set_xlim(xylim)
    ax.set_ylim(xylim)
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")

fig.savefig("figs/fig1/fig1c.png", dpi=150, bbox_inches="tight")


# %%
