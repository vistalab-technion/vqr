import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from torch import Tensor
from matplotlib import cm

from experiments.utils.metrics import kde, kde_l1

# CVAE baseline:
# Produced using the open-source implementation by Feldman et al. 2021.
# Available at: https://github.com/shai128/mqr
# Download results from
# https://drive.google.com/drive/folders/1mkrOraclAiPv-YrYibQSBhY_J4MEis9Z?usp=sharing


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


cvae_results_path = "../cvae_results3/"
GPU_DEVICE_NUM = 0
device = f"cuda:{GPU_DEVICE_NUM}" if GPU_DEVICE_NUM is not None else "cpu"
sigma = 0.1

kde_l1_dists = []
samples_gt = {}
samples_est = {}
kdes_gt = {}
kdes_est = {}

for folder_name in os.listdir(cvae_results_path):
    X = folder_name[-3:]
    with open(f"{cvae_results_path}/x={X}/y_recon_y.pkl", "rb") as f:
        file_contents = pickle.load(f)
        f.close()
    y = Tensor(array(file_contents["y"]))
    y_reconstructed = Tensor(array(file_contents["y_reconstructed"]))
    samples_gt[str(X)] = y.numpy()
    samples_est[str(X)] = y_reconstructed.numpy()

    # Estimate KDEs
    kde_orig = kde(
        y,
        grid_resolution=100,
        device=device,
        sigma=sigma,
    )
    kdes_gt[str(X)] = kde_orig

    kde_est = kde(
        y_reconstructed,
        grid_resolution=100,
        device=device,
        sigma=sigma,
    )
    kdes_est[str(X)] = kde_est

    # Calculate KDE-L1 distance
    kde_l1_dist = kde_l1(
        y, y_reconstructed, grid_resolution=100, device=device, sigma=sigma
    )

    kde_l1_dists.append(kde_l1_dist)

    # Plot KDEs
    plot_kde(
        kde_orig.T,
        kde_est.T,
        kde_l1_dist,
        f"Y_given_X={float(X):.1f}_cvae",
    )

with open(f"./cond-banana-cvae.pkl", "wb") as f:
    pickle.dump(
        {
            "cond_Y_samples_gt": samples_gt,
            "cond_Y_samples_est": samples_est,
            "kde_dists": kde_l1_dists,
            "entropy_is": None,
            "entropy_oos": None,
            "q_minus_q_stars": None,
            "kdes_gt": kdes_gt,
            "kdes_est": kdes_est,
        },
        f,
    )
    print(np.mean(kde_l1_dists))
    f.close()
