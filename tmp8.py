import os
import pickle

import numpy as np
import torch
from numpy import exp, array, zeros, arange, histogram
from torch import Tensor, tensor
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.special import xlogy

from vqr.api import VectorQuantileEstimator, VectorQuantileRegressor
from experiments.optimization import _compare_conditional_quantiles
from experiments.utils.metrics import kde, ranks, kde_l1, w2_pot, w2_keops
from experiments.data.cond_banana import ConditionalBananaDataProvider
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)
from vqr.solvers.dual.alt_regularized_lse import (
    AlternativeRegularizedDualVQRSolver,
    MLPAlternativeRegularizedDualVQRSolver,
    ImplicitAlternativeRegularizedDualVQRSolver,
    ImplicitMLPAlternativeRegularizedDualVQRSolver,
)


def entropy(w):
    return (exp(-xlogy(w, w).sum()) - 1) / (len(w) - 1)


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
num_epochs = 10000
linear = False
sigma = 0.1
GPU_DEVICE_NUM = 1
device = f"cuda:{GPU_DEVICE_NUM}" if GPU_DEVICE_NUM is not None else "cpu"
dtype = torch.float32
epsilon = 0.2 * 1e-1

os.makedirs(f"{linear=}", exist_ok=True)

data_provider = ConditionalBananaDataProvider(k=k, d=d, nonlinear=True)
X, Y = data_provider.sample(n=n)

if linear:
    # solver = AlternativeRegularizedDualVQRSolver(
    solver = ImplicitAlternativeRegularizedDualVQRSolver(
        verbose=True,
        num_epochs=num_epochs,
        epsilon=epsilon,
        lr=0.001,
        # lr=2.9,
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
    # solver = MLPAlternativeRegularizedDualVQRSolver(
    solver = ImplicitMLPAlternativeRegularizedDualVQRSolver(
        verbose=True,
        num_epochs=num_epochs,
        epsilon=epsilon,
        lr=0.001,
        gpu=True,
        skip=False,
        batchnorm=False,
        hidden_layers=(2, 10, 20),
        device_num=GPU_DEVICE_NUM,
        batchsize_y=None,
        batchsize_u=None,
        inference_batch_size=100,
        lr_factor=0.9,
        lr_patience=500,
        lr_threshold=0.5 * 0.01,
        lr_max_steps=100,
    )

vqr_est = VectorQuantileRegressor(n_levels=T, solver=solver)

vqr_est.fit(X, Y)

# Generate conditional distributions for the below X's
Xs = [tensor(array([[x] * k]), dtype=dtype) for x in np.linspace(1.0, 3.0, 20)]
kde_l1_dists = []
entropies_is = []
entropies_oos = []
q_minus_q_stars = []
samples_gt = {}
samples_est = {}
kdes_gt = {}
kdes_est = {}

n_test = 4000

for cond_X in Xs:
    _, cond_Y_gt = data_provider.sample(n=n_test, x=cond_X.numpy())

    samples_gt[f"{round(cond_X.item(), 1)}"] = cond_Y_gt

    cond_Y_gt = tensor(cond_Y_gt, dtype=dtype)
    vqe = VectorQuantileEstimator(
        n_levels=T, solver="vqe_pot", solver_opts={"numItermax": 2e6}
    )
    vqe.fit(cond_Y_gt.numpy())
    cond_vq_gt = vqe.vector_quantiles(refine=True)

    cond_Y_est = vqr_est.sample(n=n_test, x=cond_X.numpy())
    samples_est[f"{round(cond_X.item(), 1)}"] = cond_Y_est

    cond_Y_est = tensor(cond_Y_est, dtype=dtype)

    # w2 distance
    # w2_dist = w2_keops(cond_Y_gt, cond_Y_est, device=device)

    # Estimate KDEs
    kde_orig = kde(
        cond_Y_gt,
        grid_resolution=100,
        device=device,
        sigma=sigma,
    )
    kdes_gt[f"{round(cond_X.item(), 1)}"] = kde_orig

    kde_est = kde(
        cond_Y_est,
        grid_resolution=100,
        device=device,
        sigma=sigma,
    )
    kdes_est[f"{round(cond_X.item(), 1)}"] = kde_est

    # Calculate KDE-L1 distance
    kde_l1_dist = kde_l1(
        cond_Y_gt, cond_Y_est, grid_resolution=100, device=device, sigma=sigma
    )
    kde_l1_dists.append(kde_l1_dist)

    # Plot KDEs
    plot_kde(
        kde_orig.T,
        kde_est.T,
        kde_l1_dist,
        f"{linear=}/kde_Y_given_X={cond_X.squeeze().item():.1f}",
    )

    # get quantiles
    cond_vq_est = vqr_est.vector_quantiles(X=cond_X.numpy(), refine=True)[0]
    quantiles = cond_vq_est.values.reshape(2, -1).T

    # Q - Q*
    q_minus_q_star = _compare_conditional_quantiles(
        cond_vq_gt, cond_vq_est, t_factor=1, ignore_X=True
    )
    q_minus_q_stars.append(q_minus_q_star)
    print(f"{round(cond_X.item(), 1)}", q_minus_q_star)

    # Uniformity of ranks
    ranks_is = ranks(quantiles, cond_Y_est.numpy()).squeeze()
    ranks_oos = ranks(quantiles, cond_Y_gt.numpy()).squeeze()
    hist_is, _ = histogram(ranks_is, arange(ranks_is.min(), ranks_is.max()))
    entropy_is = entropy(hist_is / hist_is.sum())
    hist_oos, _ = histogram(ranks_oos, arange(ranks_oos.min(), ranks_oos.max()))
    entropy_oos = entropy(hist_oos / hist_oos.sum())
    entropies_is.append(entropy_is)
    entropies_oos.append(entropy_oos)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].hist(ranks_is, bins=100)
    ax[0].set_title(f"IS - entropy: {entropy_is:.3f}")
    ax[1].hist(ranks_oos, bins=100)
    ax[1].set_title(f"OOS - entropy: {entropy_oos:.3f}")
    fig.suptitle(f"(X={cond_X.item():.1f})")
    plt.savefig(
        f"{linear=}/Entropy_Y_given_X={cond_X.squeeze().item():.1f}_{linear=}.png"
    )


with open(f"./cond-banana-{linear=}.pkl", "wb") as f:
    pickle.dump(
        {
            "kde_dists": kde_l1_dists,
            "entropy_is": entropies_is,
            "entropy_oos": entropies_oos,
            "q_minus_q_stars": q_minus_q_stars,
            "cond_Y_samples_gt": samples_gt,
            "cond_Y_samples_est": samples_est,
            "kdes_gt": kdes_gt,
            "kdes_est": kdes_est,
        },
        f,
    )
    print("KDE:", np.mean(kde_l1_dists))
    print("Entropy IS:", np.mean(entropies_is))
    print("Entropy OOS:", np.mean(entropies_oos))
    print("||Q - Q*||:", np.mean(q_minus_q_stars))
    f.close()
