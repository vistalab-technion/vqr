import pickle

import numpy as np
import torch
from numpy import array, zeros
from torch import Tensor, tensor
from matplotlib import cm
from matplotlib import pyplot as plt

from vqr.api import VectorQuantileRegressor
from experiments.utils.metrics import kde, kde_l1, w2_pot, w2_keops
from experiments.datasets.cond_banana import ConditionalBananaDataProvider
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
        extent=(0, 1, 0, 1),
    )
    axes[1].imshow(
        kde_map_2,
        interpolation="bilinear",
        origin="lower",
        cmap=cm.RdPu,
        extent=(0, 1, 0, 1),
    )
    plt.title(f"L1 distance: {l1_distance}")
    plt.savefig(f"{filename}.png")


n = 20000
d = 2
k = 1
T = 50
num_epochs = 20000
linear = False
sigma = 0.1
GPU_DEVICE_NUM = 1
device = f"cuda:{GPU_DEVICE_NUM}" if GPU_DEVICE_NUM is not None else "cpu"
dtype = torch.float32
epsilon = 5e-3

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
        batchsize_y=None,
        batchsize_u=None,
        inference_batch_size=100,
        lr_factor=0.9,
        lr_patience=500,
        lr_threshold=0.5 * 0.01,
    )
else:
    solver = MLPRegularizedDualVQRSolver(
        verbose=True,
        num_epochs=num_epochs,
        epsilon=epsilon,
        lr=0.4,
        gpu=True,
        skip=False,
        batchnorm=False,
        hidden_layers=(2, 10, 20),
        device_num=GPU_DEVICE_NUM,
        batchsize_y=None,
        batchsize_u=None,
        inference_batch_size=100,
        lr_factor=0.9,
        lr_patience=300,
        lr_threshold=0.5 * 0.01,
    )

vqr_est = VectorQuantileRegressor(n_levels=T, solver=solver)

vqr_est.fit(X, Y)

# Generate conditional distributions for the below X's
Xs = [tensor(array([[x] * k]), dtype=dtype) for x in np.linspace(1.0, 3.0, 20)]
kde_l1_dists = []

for cond_X in Xs:
    _, cond_Y_gt = data_provider.sample(n=n, x=cond_X.numpy())
    cond_Y_gt = tensor(cond_Y_gt, dtype=dtype)

    cond_Y_est = vqr_est.sample(n=n, x=cond_X.numpy())
    cond_Y_est = tensor(cond_Y_est, dtype=dtype)

    # w2 distance
    w2_dist = w2_keops(cond_Y_gt, cond_Y_est, device=device)

    # Estimate KDEs
    kde_orig = kde(
        cond_Y_gt,
        grid_resolution=T * 2,
        device=device,
        sigma=sigma,
    )

    kde_est = kde(
        cond_Y_est,
        grid_resolution=T * 2,
        device=device,
        sigma=sigma,
    )

    # Calculate KDE-L1 distance
    kde_l1_dist = kde_l1(
        cond_Y_gt, cond_Y_est, grid_resolution=T * 2, device=device, sigma=sigma
    )
    kde_l1_dists.append(kde_l1_dist)

    # Plot KDEs
    plot_kde(
        kde_orig.T,
        kde_est.T,
        kde_l1_dist,
        f"Y_given_X={cond_X.squeeze().item():.1f}_{linear=}",
    )


with open(f"./kde-l1-dists-{linear=}.pkl", "wb") as f:
    pickle.dump(kde_l1_dists, f)
    print(np.mean(kde_l1_dists))
    f.close()
