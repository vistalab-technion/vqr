import numpy as np
import torch
from torch import tensor
from matplotlib import cm
from matplotlib import pyplot as plt

from vqr import VectorQuantileRegressor
from experiments.utils.metrics import kde
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)
from experiments.data.synthetic_glasses import SyntheticGlassesDataProvider

n = 20000
d = 1
k = 1
T = 1000
num_epochs = 50000
linear = False
sigma = 0.1
GPU_DEVICE_NUM = 1
device = f"cuda:{GPU_DEVICE_NUM}" if GPU_DEVICE_NUM is not None else "cpu"
dtype = torch.float32
epsilon = 1e-9
X, Y = SyntheticGlassesDataProvider().sample(n)


if linear:
    solver = RegularizedDualVQRSolver(
        verbose=True,
        num_epochs=num_epochs,
        epsilon=epsilon,
        lr=2.9,
        gpu=True,
        full_precision=False,
        device_num=GPU_DEVICE_NUM,
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
        skip=True,
        batchnorm=True,
        hidden_layers=(1000, 1000, 1000),
        device_num=GPU_DEVICE_NUM,
        lr_factor=0.9,
        lr_patience=300,
        lr_threshold=0.5 * 0.01,
    )
vqr_est = VectorQuantileRegressor(n_levels=T, solver=solver)

vqr_est.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
Y_est = np.concatenate(
    [vqr_est.sample(n=1, x=np.array([X[i]])[:, None])[0] for i in range(n)]
)
kde_1 = kde(
    tensor(np.stack([X, Y_est]), dtype=torch.float32).T,
    grid_resolution=T,
    device=device,
    sigma=sigma,
)
plt.figure()
plt.imshow(
    kde_1.T,
    interpolation="bilinear",
    origin="lower",
    cmap=cm.RdPu,
    extent=(0, 1, 0, 1),
)
plt.show()

kde_gt = kde(
    tensor(np.stack([X, Y]), dtype=torch.float32).T,
    grid_resolution=T,
    device=device,
    sigma=sigma,
)
plt.figure()
plt.imshow(
    kde_gt.T,
    interpolation="bilinear",
    origin="lower",
    cmap=cm.RdPu,
    extent=(0, 1, 0, 1),
)
plt.show()

plt.figure()
plt.plot(X, Y_est, ".")
plt.show()
