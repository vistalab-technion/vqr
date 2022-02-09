import numpy as np
import matplotlib.pyplot as plt
from numpy import ones, array

from vqr import VectorQuantileEstimator, VectorQuantileRegressor
from vqr.vqr import decode_quantile_values
from vqr.data import (
    generate_star,
    generate_heart,
    generate_mvn_data,
    generate_linear_x_y_mvn_data,
)
from vqr.data_banana import get_syn_data

T = 100
N = 20000
d = 2

# X, Y = generate_mvn_data(n=N, d=d, k=3)
# X, Y = generate_linear_x_y_mvn_data(n=N, d=d, k=1, seed=21)
X, Y = get_syn_data(dataset_name="cond_banana", is_nonlinear=False, n=N, k=2)

# _, Y = generate_heart()
# _, Y = generate_star()
# vq = VectorQuantileEstimator(n_levels=T, solver_opts={"verbose": True})
vq = VectorQuantileRegressor(n_levels=T, solver_opts={"verbose": True})
vq.fit(X, Y)
fig = plt.figure(figsize=(10, 10))
quantile_grid = np.stack(
    [vq.quantile_grid[0].reshape(-1), vq.quantile_grid[1].reshape(-1)], axis=1
)
quantile_values = np.stack(
    [vq.quantile_values[0].reshape(-1), vq.quantile_values[1].reshape(-1)], axis=1
)

fig = plt.figure(figsize=(10, 10))
plt.scatter(quantile_grid[:, 0], quantile_grid[:, 1])
plt.tight_layout()
plt.savefig("./quantile_grid.png", dpi=200)

fig = plt.figure(figsize=(10, 10))
plt.scatter(Y[:, 0], Y[:, 1])
plt.tight_layout()
plt.savefig("./quantile_values.png", dpi=200)

fig = plt.figure(figsize=(10, 10))
plt.scatter(quantile_grid[:, 0], quantile_grid[:, 1], c="r")
plt.scatter(Y[:, 0], Y[:, 1], c="c")
for i, (base_sample, generated_sample) in enumerate(
    zip(quantile_grid, quantile_values)
):
    if (
        ((i // T == 1) and 1 < i % T < T - 1)
        or ((i // T == T - 2) and 1 < i % T < T - 1)
        or ((1 < i // T < T - 1) and i % T == 1)
        or ((1 < i // T < T - 1) and i % T == T - 2)
    ):
        plt.plot(
            [base_sample[0], generated_sample[0]],
            [base_sample[1], generated_sample[1]],
            linestyle="--",
            color="blue",
            linewidth=1,
        )
        plt.scatter(generated_sample[0], generated_sample[1], c="b")
plt.tight_layout()
plt.savefig("./correspondence.png", dpi=200)
plt.show()


conditional_quantiles = decode_quantile_values(
    T, d, vq.predict(X=np.array([[1.5, 1.5]]))
)
conditional_quantiles = np.stack(
    [conditional_quantiles[0].reshape(-1), conditional_quantiles[1].reshape(-1)], axis=1
)


fig = plt.figure(figsize=(10, 10))
plt.scatter(quantile_grid[:, 0], quantile_grid[:, 1], c="r")
plt.scatter(Y[:, 0], Y[:, 1], c="c")
for i, (base_sample, conditional_quantile) in enumerate(
    zip(quantile_grid, conditional_quantiles)
):
    if (
        ((i // T == 1) and 1 < i % T < T - 1)
        or ((i // T == T - 2) and 1 < i % T < T - 1)
        or ((1 < i // T < T - 1) and i % T == 1)
        or ((1 < i // T < T - 1) and i % T == T - 2)
    ):
        plt.plot(
            [base_sample[0], conditional_quantile[0]],
            [base_sample[1], conditional_quantile[1]],
            linestyle="--",
            color="blue",
            linewidth=1,
        )
        plt.scatter(conditional_quantile[0], conditional_quantile[1], c="b")
plt.tight_layout()
plt.savefig("./conditional_correspondence.png", dpi=200)
plt.show()
