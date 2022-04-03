import ot
import torch
from numpy import array, zeros
from torch import tensor
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
from matplotlib import pyplot as plt
from numpy.random import randint

from vqr.api import VectorQuantileRegressor
from experiments.data.nonlinear_data import get_k_dim_banana
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)

GPU_DEVICE_NUM = 7


def w2(Y_gt_, Y_est_, emd: bool = False):
    if emd:
        return ot.emd2(
            a=[], b=[], M=ot.dist(Y_gt_, Y_est_), numItermax=200_000, numThreads=32
        )
    else:
        device = "cpu" if GPU_DEVICE_NUM is None else f"cuda:{GPU_DEVICE_NUM}"
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)(
            tensor(Y_gt_, dtype=torch.float32, device=torch.device(device)),
            tensor(Y_est_, dtype=torch.float32, device=torch.device(device)),
        )
        return loss


n = 1000000
d = 2
k = 1
T = 50
num_epochs = 10000
linear = True

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

epsilon = 1e-9
if linear:
    solver = RegularizedDualVQRSolver(
        verbose=True,
        num_epochs=num_epochs,
        epsilon=epsilon,
        learning_rate=2.9,
        gpu=True,
        full_precision=False,
        device_num=GPU_DEVICE_NUM,
        batchsize_y=50000,
        batchsize_u=2000,
        inference_batch_size=100,
    )
else:
    solver = MLPRegularizedDualVQRSolver(
        verbose=True,
        num_epochs=num_epochs,
        epsilon=epsilon,
        learning_rate=2.9,
        gpu=True,
        skip=True,
        batchnorm=True,
        hidden_layers=(1000, 1000, 1000),
        device_num=GPU_DEVICE_NUM,
        batchsize_y=50000,
        batchsize_u=2000,
        inference_batch_size=100,
    )
vqr_est = VectorQuantileRegressor(n_levels=T, solver=solver)

vqr_est.fit(X, Y)
Y_nl_est = zeros([n, d])
Y_nl_est_given_X15 = zeros([n, d])
Y_nl_est_given_X20 = zeros([n, d])
Y_nl_est_given_X25 = zeros([n, d])

u1 = randint(0 + 1, T - 1, size=(n,))
u2 = randint(0 + 1, T - 1, size=(n,))

# Q1, Q2 = nonlinear_vqr_est.vector_quantiles(X)[0]
Q1_15, Q2_15 = vqr_est.vector_quantiles(array([1.5]))[0]
Q1_20, Q2_20 = vqr_est.vector_quantiles(array([2.0]))[0]
Q1_25, Q2_25 = vqr_est.vector_quantiles(array([2.5]))[0]

Y_nl_est_given_X15[:, 0] = Q1_15[u1, u2]
Y_nl_est_given_X15[:, 1] = Q2_15[u1, u2]

Y_nl_est_given_X20[:, 0] = Q1_20[u1, u2]
Y_nl_est_given_X20[:, 1] = Q2_20[u1, u2]

Y_nl_est_given_X25[:, 0] = Q1_25[u1, u2]
Y_nl_est_given_X25[:, 1] = Q2_25[u1, u2]


# Creates the Y for all X used in training

# for i in range(n):
#     if i % 1000 == 0:
#         print(i)
#     u1, u2 = randint(0 + 1, T - 1), randint(0 + 1, T - 1)
#
#     Q1, Q2 = nonlinear_vqr_est.vector_quantiles(X[i, :][:, None].numpy())[0]
#
#     Y_nl_est[i, :] = array((Q1[u1, u2], Q2[u1, u2]))
#     Y_nl_est_given_X15[i, :] = array((Q1_15[u1, u2], Q2_15[u1, u2]))
#     Y_nl_est_given_X20[i, :] = array((Q1_20[u1, u2], Q2_20[u1, u2]))
#     Y_nl_est_given_X25[i, :] = array((Q1_25[u1, u2], Q2_25[u1, u2]))
# w2_Y = w2(Y.numpy() if isinstance(Y, torch.Tensor) else Y, Y_nl_est)


w2_15 = w2(torch.stack([Y_15_1, Y_15_2]).numpy().T, Y_nl_est_given_X15)
w2_20 = w2(torch.stack([Y_20_1, Y_20_2]).numpy().T, Y_nl_est_given_X20)
w2_25 = w2(torch.stack([Y_25_1, Y_25_2]).numpy().T, Y_nl_est_given_X25)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# axes[0].scatter(Y[:, 0], Y[:, 1], c="k", label="Y", alpha=0.5)
axes[0].scatter(Y_15_1, Y_15_2, c="r", label="Y|X=1.5", alpha=0.5)
axes[0].scatter(Y_20_1, Y_20_2, c="g", label="Y|X=2.0", alpha=0.5)
axes[0].scatter(Y_25_1, Y_25_2, c="b", label="Y|X=2.5", alpha=0.5)
axes[0].set_xlim(-4, 4)
axes[0].set_ylim(-0.5, 2.5)
axes[0].legend()

# axes[1].scatter(
#     Y_nl_est[:, 0], Y_nl_est[:, 1], c="k", label=f"Y ({w2_Y:.4f})", alpha=0.5
# )
axes[1].scatter(
    Y_nl_est_given_X15[:, 0],
    Y_nl_est_given_X15[:, 1],
    c="r",
    label=f"Y|X=1.5 ({w2_15:.4f})",
    alpha=0.5,
)
axes[1].scatter(
    Y_nl_est_given_X20[:, 0],
    Y_nl_est_given_X20[:, 1],
    c="g",
    label=f"Y|X=2.0 ({w2_20:.4f})",
    alpha=0.5,
)
axes[1].scatter(
    Y_nl_est_given_X25[:, 0],
    Y_nl_est_given_X25[:, 1],
    c="b",
    label=f"Y|X=2.5 ({w2_25:.4f})",
    alpha=0.5,
)
axes[1].set_xlim(-4, 4)
axes[1].set_ylim(-0.5, 2.5)
axes[1].legend()
axes[1].set_title(
    f"Nonlinear VQR ({epsilon})"
    if isinstance(vqr_est.solver, MLPRegularizedDualVQRSolver)
    else f"Linear VQR ({epsilon})"
)

plt.savefig(f"cond_banana_{n=}_{T=}_{k=}_{d=}_{linear=}_{num_epochs=}.png")
