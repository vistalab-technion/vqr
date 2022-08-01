import pickle

import ot
import numpy as np
import torch
from torch import tensor
from matplotlib import cm
from matplotlib import pyplot as plt

from vqr import VectorQuantileRegressor
from vqr.solvers.primal.cvx import SIMILARITY_FN_INNER_PROD
from experiments.utils.metrics import kde
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)
from experiments.data.synthetic_glasses import SyntheticGlassesDataProvider

n = 10000
d = 1
k = 1
T = 100
num_epochs = 40000
linear = False
refine = False
sigma = 0.1
GPU_DEVICE_NUM = 2
device = f"cuda:{GPU_DEVICE_NUM}" if GPU_DEVICE_NUM is not None else "cpu"
dtype = torch.float32
epsilon = 1e-3
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
quantiles = []

for i in range(n):
    X_i = np.array([X[i]])[:, None]
    quantile_est = vqr_est.vector_quantiles(X=X_i, refine=refine)[0].values.squeeze()
    quantiles.append(quantile_est)

quantiles = np.stack(quantiles, axis=0)
Y_est = np.concatenate(
    [vqr_est.sample(n=1, x=np.array([X[i]])[:, None])[0] for i in range(n)]
)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(X, Y, ".", label="GT Y|X")
ax[0].set_ylabel("y")
ax[0].set_xlabel("x")
ax[0].set_title("GT")
ax[0].legend()
ax[1].plot(X, Y_est, ".", label="samples of Y|X")
for quantile_level in np.linspace(start=0.0, stop=1.0, num=20):
    if quantile_level == 0.0:
        ax[1].plot(
            X,
            quantiles[:, int(quantile_level * T) + 1],
            label=f"{quantile_level+1:.2f}",
        )
    elif quantile_level == 1.0:
        ax[1].plot(
            X,
            quantiles[:, int(quantile_level * T) - 1],
            label=f"{quantile_level-1:.2f}",
        )
    else:
        ax[1].plot(
            X,
            quantiles[:, int(quantile_level * T)],
            label=f"{quantile_level:.2f}",
        )
ax[1].legend()

ax[1].set_ylabel("y")
ax[1].set_xlabel("x")
ax[1].set_title("Non-linear QR" if not linear else "Linear QR")
plt.suptitle("Simultaneous scalar QR", fontsize="x-large")
plt.tight_layout()
plt.savefig(f"sqr_{linear=}_{refine=}.png")
# plt.show()

with open(f"sqr_{linear=}_{refine=}.pkl", "wb") as f:
    pickle.dump(
        {"quantiles": quantiles, "Y_gt": Y, "Y_est": Y_est},
        f,
    )
