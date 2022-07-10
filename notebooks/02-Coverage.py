# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# %%
from vqr.api import ScalarQuantileEstimator, VectorQuantileEstimator

# %%
from vqr.plot import plot_quantiles_3d
from experiments.data.mvn import IndependentDataProvider
from experiments.data.shapes import generate_star, generate_heart

d = 2
n = 2000
T = 50

dataset = "heart"

if dataset == "mvn":
    X, Y = IndependentDataProvider(d=d, k=0).sample(n=n)
elif dataset == "heart":
    X, Y = generate_heart()
elif dataset == "star":
    X, Y = generate_star()
else:
    X, Y = None, None
    NotImplementedError("Unrecognized dataset.")

# %%
fig = plt.figure(figsize=(8, 8))
plt.scatter(Y[:, 0], Y[:, 1])
# _ = plt.title("Y")
plt.tight_layout()
fig.savefig(f"{dataset}_Y.png")


T = 50
vqe = VectorQuantileEstimator(T, solver="vqe_pot", solver_opts={"numItermax": int(2e6)})
fitted_vqe = vqe.fit(Y)


fig = plot_quantiles_3d(
    T,
    d,
    Qs=list(fitted_vqe.vector_quantiles(refine=True)),
    Us=fitted_vqe.quantile_grid,
    figsize=(20, 20),
)
plt.tight_layout()
fig.savefig(f"{dataset}_quantile_surfaces_plot.png")

# %%
Q1, Q2 = vqe.vector_quantiles()
q_10_90 = [Q1[1, -2], Q2[1, -2]]
q_90_90 = [Q1[-2, -2], Q2[-2, -2]]
q_90_10 = [Q1[-2, 1], Q2[-2, 1]]
q_10_10 = [Q1[1, 1], Q2[1, 1]]

fig = plt.figure(figsize=(8, 8))
plt.scatter(Y[:, 0], Y[:, 1])
plt.scatter(q_10_90[0], q_10_90[1], alpha=0.5, color="k", s=200, marker="v")
plt.scatter(q_90_90[0], q_90_90[1], alpha=0.5, color="k", s=200, marker="v")
plt.scatter(q_90_10[0], q_90_10[1], alpha=0.5, color="k", s=200, marker="v")
_ = plt.scatter(q_10_10[0], q_10_10[1], alpha=0.5, color="k", s=200, marker="v")
plt.tight_layout()
fig.savefig(f"{dataset}_vector_quantiles_on_Y.png")

# %%

surface = np.array(
    [
        [*Q1[1:-2, -2], *Q1[1, 1:-2], *Q1[1:-2, 1], *Q1[-2, 1:-2]],
        [*Q2[1:-2, -2], *Q2[1, 1:-2], *Q2[1:-2, 1], *Q2[-2, 1:-2]],
    ]
).T


def point_in_hull(point, hull, tolerance=1e-12):
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)


selected_Y = surface
hull = ConvexHull(selected_Y)

fig = plt.figure(figsize=(8, 8))
for simplex in hull.simplices:
    plt.plot(selected_Y[simplex, 0], selected_Y[simplex, 1])

plt.scatter(*selected_Y.T, alpha=0.5, color="k", s=200, marker="v")

coverage = []
for p in Y:
    point_is_in_hull = point_in_hull(p, hull)
    coverage.append(point_is_in_hull)
    marker = "x" if point_is_in_hull else "d"
    color = "g" if point_is_in_hull else "m"
    plt.scatter(p[0], p[1], marker=marker, color=color)
plt.tight_layout()
fig.savefig(f"{dataset}_Y_inliers_outliers_vector.png")

# %%
coverage_final = (sum(coverage) / len(coverage)) * 100
print(f"Coverage for vector quantiles: {coverage_final}.")

# %%
sqe = ScalarQuantileEstimator(T)
fitted_sqe = sqe.fit(Y)
Q = fitted_sqe.vector_quantiles()
Q1_sep = Q[:, 0]
Q2_sep = Q[:, 1]

q_10_90 = [Q1_sep[1], Q2_sep[-2]]
q_90_90 = [Q1_sep[-2], Q2_sep[-2]]
q_90_10 = [Q1_sep[-2], Q2_sep[1]]
q_10_10 = [Q1_sep[1], Q2_sep[1]]

fig = plt.figure(figsize=(8, 8))
plt.scatter(Y[:, 0], Y[:, 1])
plt.scatter(q_10_90[0], q_10_90[1], alpha=0.5, color="k", s=200, marker="v")
plt.scatter(q_90_90[0], q_90_90[1], alpha=0.5, color="k", s=200, marker="v")
plt.scatter(q_90_10[0], q_90_10[1], alpha=0.5, color="k", s=200, marker="v")
plt.scatter(q_10_10[0], q_10_10[1], alpha=0.5, color="k", s=200, marker="v")
plt.tight_layout()
fig.savefig(f"{dataset}_separable_quantiles_on_Y.png")

# %%

surface_sep = np.array(
    [
        [
            *Q1_sep[1:-2],
            *Q1_sep[1:-2],
            *[Q1_sep[1]] * len(Q2_sep[1:-2]),
            *[Q1_sep[-2]] * len(Q2_sep[1:-2]),
        ],
        [
            *[Q2_sep[-2]] * len(Q1_sep[1:-2]),
            *[Q2_sep[1]] * len(Q1_sep[1:-2]),
            *Q2_sep[1:-2],
            *Q2_sep[1:-2],
        ],
    ]
).T


selected_Y = surface_sep
hull = ConvexHull(selected_Y)
fig = plt.figure(figsize=(8, 8))
for simplex in hull.simplices:
    plt.plot(selected_Y[simplex, 0], selected_Y[simplex, 1])

plt.scatter(*selected_Y.T, alpha=0.5, color="k", s=200, marker="v")

coverage_sep = []
for p in Y:
    point_is_in_hull = point_in_hull(p, hull)
    coverage_sep.append(point_is_in_hull)
    marker = "x" if point_is_in_hull else "d"
    color = "g" if point_is_in_hull else "m"
    plt.scatter(p[0], p[1], marker=marker, color=color)
plt.tight_layout()
fig.savefig(f"{dataset}_Y_inliers_outliers_separable.png")

# %%
coverage_sep_final = (sum(coverage_sep) / len(coverage_sep)) * 100
print(f"Coverage for separable quantiles: {coverage_sep_final}.")

# %%
fig = plt.figure(figsize=(8, 8))
plt.scatter(Y[:, 0], Y[:, 1])
plt.scatter(Q1[1:-2, -2], Q2[1:-2, -2], alpha=0.5, color="k", s=200, marker="v")
plt.scatter(Q1[1, 1:-2], Q2[1, 1:-2], alpha=0.5, color="k", s=200, marker="v")
plt.scatter(Q1[1:-2, 1], Q2[1:-2, 1], alpha=0.5, color="k", s=200, marker="v")
plt.scatter(Q1[-2, 1:-2], Q2[-2, 1:-2], alpha=0.5, color="k", s=200, marker="v")
# _ = plt.title(f"Vector quantile surface of Y. Coverage of cvx hull: {coverage_final}")
plt.tight_layout()
fig.savefig(f"{dataset}_vector_quantile_surface_on_Y.png")

# %%
fig = plt.figure(figsize=(8, 8))

plt.scatter(Y[:, 0], Y[:, 1])
plt.scatter(
    Q1_sep[1:-2],
    [Q2_sep[-2]] * len(Q1_sep[1:-2]),
    alpha=0.5,
    color="k",
    s=200,
    marker="v",
)
plt.scatter(
    Q1_sep[1:-2],
    [Q2_sep[1]] * len(Q1_sep[1:-2]),
    alpha=0.5,
    color="k",
    s=200,
    marker="v",
)
plt.scatter(
    [Q1_sep[1]] * len(Q2_sep[1:-2]),
    Q2_sep[1:-2],
    alpha=0.5,
    color="k",
    s=200,
    marker="v",
)
plt.scatter(
    [Q1_sep[-2]] * len(Q2_sep[1:-2]),
    Q2_sep[1:-2],
    alpha=0.5,
    color="k",
    s=200,
    marker="v",
)
# _ = plt.title(f"Separable quantile surface of Y. Coverage of cvx hull: {coverage_sep_final}")
plt.tight_layout()
fig.savefig(f"{dataset}_separable_quantile_surface_on_Y.png")

# %%
print(coverage_final, coverage_sep_final)

# %%
