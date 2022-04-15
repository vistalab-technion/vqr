import torch
from numpy import array, zeros
from torch import Tensor, tensor
from matplotlib import cm
from matplotlib import pyplot as plt

from vqr.api import VectorQuantileRegressor
from experiments.utils.metrics import kde, kde_l1, w2_pot, w2_keops
from experiments.data.cond_banana import ConditionalBananaDataProvider
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)


def plot_kde(kde_map_1, kde_map_2, l1_distance: float, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(
        kde_map_1,
        interpolation="bilinear",
        origin="lower",
        cmap=cm.RdPu,
        # extent=(0, 1, 0, 1),
    )
    axes[1].imshow(
        kde_map_2,
        interpolation="bilinear",
        origin="lower",
        cmap=cm.RdPu,
        extent=(0, 1, 0, 1),
    )
    plt.title(f"L1 distance: {l1_distance}")
    plt.show()  # f"{filename}.png")


n = 10000
d = 2
k = 1
T = 50
num_epochs = 200
linear = True
sigma = 0.1
GPU_DEVICE_NUM = 4
device = f"cuda:{GPU_DEVICE_NUM}" if GPU_DEVICE_NUM is not None else "cpu"
dtype = torch.float32
epsilon = 1e-9

data_provider = ConditionalBananaDataProvider(k=k, d=d, nonlinear=True)
X, Y = data_provider.sample(n=n)

if linear:
    solver = RegularizedDualVQRSolver(
        verbose=True,
        num_epochs=num_epochs,
        epsilon=epsilon,
        lr=2.9,
        gpu=True,
        full_precision=False,
        device_num=GPU_DEVICE_NUM,
        batchsize_y=100000,
        batchsize_u=2500,
        inference_batch_size=100,
        lr_factor=0.9,
        lr_patience=300,
        lr_threshold=0.5 * 0.01,
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
        batchsize_y=100000,
        batchsize_u=2500,
        inference_batch_size=100,
        lr_factor=0.9,
        lr_patience=300,
        lr_threshold=0.5 * 0.01,
    )

vqr_est = VectorQuantileRegressor(n_levels=T, solver=solver)

vqr_est.fit(X, Y)

# Generate conditional distributions for the below X's
Xs = [
    tensor(array([[1.5] * k]), dtype=dtype),
    tensor(array([[2.0] * k]), dtype=dtype),
    tensor(array([[2.5] * k]), dtype=dtype),
]


for cond_X in Xs:
    _, cond_Y_gt = data_provider.sample(n=n, X=cond_X.numpy())
    cond_Y_gt = tensor(cond_Y_gt, dtype=dtype)

    cond_Y_est = vqr_est.sample(n=n, x=cond_X.numpy())
    cond_Y_est = tensor(cond_Y_est, dtype=dtype)

    # w2 distance
    w2_dist = w2_keops(cond_Y_gt, cond_Y_est, gpu_device=GPU_DEVICE_NUM)

    # Estimate KDEs
    kde_orig = kde(
        cond_Y_gt,
        grid_resolution=T,
        device=device,
        sigma=sigma,
    )

    kde_est = kde(
        cond_Y_est,
        grid_resolution=T,
        device=device,
        sigma=sigma,
    )

    # Calculate KDE-L1 distance
    kde_l1_dist = kde_l1(
        cond_Y_gt, cond_Y_est, grid_resolution=T, device=device, sigma=sigma
    )

    # Plot KDEs
    plot_kde(kde_orig.T, kde_est.T, kde_l1_dist, f"{cond_X=}_{linear=}")
