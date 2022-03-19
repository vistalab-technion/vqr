import pickle

import ot
import torch
from numpy import array, zeros
from torch import tensor, float32
from matplotlib import pyplot as plt
from numpy.random import randint

from vqr.api import VectorQuantileRegressor
from vqr.vqr import (
    RVQRDualLSESolver,
    NonlinearRVQRDualLSESolver,
    decode_quantile_values,
)
from vqr.nonlinear_data import get_k_dim_banana


def w2(Y_gt_, Y_est_):
    return ot.emd2(
        a=[], b=[], M=ot.dist(Y_gt_, Y_est_), numItermax=200_000, numThreads=4
    )


n = 12000
d = 2
k = 1
T = 30
Y1, Y2, X, _ = get_k_dim_banana(n=n, k=k, d=d, is_nonlinear=True)
Y = torch.stack([Y1, Y2], dim=1)
#
Y_15_1, Y_15_2, _, _ = get_k_dim_banana(
    n=n, k=k, d=d, is_nonlinear=True, X=tensor(array([[1.5]]), dtype=torch.float32)
)
Y_20_1, Y_20_2, _, _ = get_k_dim_banana(
    n=n, k=k, d=d, is_nonlinear=True, X=tensor(array([[2.0]]), dtype=torch.float32)
)
Y_25_1, Y_25_2, _, _ = get_k_dim_banana(
    n=n, k=k, d=d, is_nonlinear=True, X=tensor(array([[2.5]]), dtype=torch.float32)
)

nonlinear_vqr_est = VectorQuantileRegressor(
    n_levels=T,
    solver=NonlinearRVQRDualLSESolver(
        verbose=True, num_epochs=1500, epsilon=0.0001, learning_rate=2.9, k=k
    ),
)
# nonlinear_vqr_est = VectorQuantileRegressor(
#     n_levels=T,
#     solver=RVQRDualLSESolver(
#         verbose=True, num_epochs=1500, epsilon=0.001, learning_rate=0.9
#     ),
# )
nonlinear_vqr_est = nonlinear_vqr_est.fit(X, Y)
B_nl_est = nonlinear_vqr_est._fitted_solution._B
A_nl_est = nonlinear_vqr_est._fitted_solution._A
Y_nl_est = zeros([n, d])
for i in range(n):
    if isinstance(nonlinear_vqr_est.solver, NonlinearRVQRDualLSESolver):
        Y_hat = (
            B_nl_est
            @ nonlinear_vqr_est.solver._net(
                tensor(X[i, :][:, None].numpy(), dtype=float32).T
            )
            .detach()
            .numpy()
            .T
            + A_nl_est
        )
    else:
        Y_hat = (
            B_nl_est
            @ tensor(X[i, :][:, None].numpy(), dtype=float32).T.detach().numpy().T
            + A_nl_est
        )
    Q1, Q2 = decode_quantile_values(T, d=2, Y_hat=Y_hat)
    u1, u2 = randint(0, T), randint(0, T)
    Y_nl_est[i, :] = array((Q1[u1, u2], Q2[u1, u2]))

w2_Y = w2(Y.numpy() if isinstance(Y, torch.Tensor) else Y, Y_nl_est)

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[1].scatter(Y_nl_est[:, 0], Y_nl_est[:, 1])
# ax[1].set_xlim(-4, 4)
# ax[1].set_ylim(-1.5, 3)
# ax[1].set_title(f"{w2_Y}")
#
# ax[0].scatter(Y[:, 0], Y[:, 1])
# ax[0].set_xlim(-4, 4)
# ax[0].set_ylim(-1.5, 3)
# plt.show()

Y_nl_est_given_X15 = zeros([n, d])
Y_nl_est_given_X20 = zeros([n, d])
Y_nl_est_given_X25 = zeros([n, d])

for i in range(n):
    if isinstance(nonlinear_vqr_est.solver, NonlinearRVQRDualLSESolver):
        Y_hat25 = (
            B_nl_est
            @ nonlinear_vqr_est.solver._net(
                tensor(array([2.5]), dtype=float32)[None, :]
            )
            .detach()
            .numpy()
            .T
            + A_nl_est
        )
        Y_hat15 = (
            B_nl_est
            @ nonlinear_vqr_est.solver._net(
                tensor(array([1.5]), dtype=float32)[None, :]
            )
            .detach()
            .numpy()
            .T
            + A_nl_est
        )
        Y_hat20 = (
            B_nl_est
            @ nonlinear_vqr_est.solver._net(
                tensor(array([2.0]), dtype=float32)[None, :]
            )
            .detach()
            .numpy()
            .T
            + A_nl_est
        )
    else:
        Y_hat25 = (
            B_nl_est @ tensor(array([2.5]), dtype=float32)[None, :].detach().numpy().T
            + A_nl_est
        )
        Y_hat15 = (
            B_nl_est @ tensor(array([1.5]), dtype=float32)[None, :].detach().numpy().T
            + A_nl_est
        )
        Y_hat20 = (
            B_nl_est @ tensor(array([2.0]), dtype=float32)[None, :].detach().numpy().T
            + A_nl_est
        )
    u1, u2 = randint(0, T), randint(0, T)

    Q1_15, Q2_15 = decode_quantile_values(T, d=2, Y_hat=Y_hat15)
    Q1_20, Q2_20 = decode_quantile_values(T, d=2, Y_hat=Y_hat20)
    Q1_25, Q2_25 = decode_quantile_values(T, d=2, Y_hat=Y_hat25)

    Y_nl_est_given_X15[i, :] = array((Q1_15[u1, u2], Q2_15[u1, u2]))
    Y_nl_est_given_X20[i, :] = array((Q1_20[u1, u2], Q2_20[u1, u2]))
    Y_nl_est_given_X25[i, :] = array((Q1_25[u1, u2], Q2_25[u1, u2]))

w2_15 = w2(torch.stack([Y_15_1, Y_15_2]).numpy().T, Y_nl_est_given_X15)
w2_20 = w2(torch.stack([Y_20_1, Y_20_2]).numpy().T, Y_nl_est_given_X20)
w2_25 = w2(torch.stack([Y_25_1, Y_25_2]).numpy().T, Y_nl_est_given_X25)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].scatter(Y[:, 0], Y[:, 1], c="k", label="Y")
axes[0].scatter(Y_15_1, Y_15_2, c="r", label="Y|X=1.5")
axes[0].scatter(Y_20_1, Y_20_2, c="g", label="Y|X=2.0")
axes[0].scatter(Y_25_1, Y_25_2, c="b", label="Y|X=2.5")
axes[0].set_xlim(-4, 4)
axes[0].set_ylim(-0.5, 2.5)
axes[0].legend()

axes[1].scatter(Y_nl_est[:, 0], Y_nl_est[:, 1], c="k", label=f"Y({w2_Y:.4f})")
axes[1].scatter(
    Y_nl_est_given_X15[:, 0],
    Y_nl_est_given_X15[:, 1],
    c="r",
    label=f"Y|X=1.5 ({w2_15:.4f})",
)
axes[1].scatter(
    Y_nl_est_given_X20[:, 0],
    Y_nl_est_given_X20[:, 1],
    c="g",
    label=f"Y|X=2.0 ({w2_20:.4f})",
)
axes[1].scatter(
    Y_nl_est_given_X25[:, 0],
    Y_nl_est_given_X25[:, 1],
    c="b",
    label=f"Y|X=2.5 ({w2_25:.4f})",
)
axes[1].set_xlim(-4, 4)
axes[1].set_ylim(-0.5, 2.5)
axes[1].legend()

plt.show()
# plt.title(f"{w2(Y.numpy() if isinstance(Y, torch.Tensor) else Y, Y_nl_est)}")
