import torch

from vqr import VectorQuantileRegressor
from experiments.data.real_data import DataFolder, CASPDataProvider
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)

data = DataFolder(train_ratio=0.6, val_ratio=0.1, test_ratio=0.3).generate_folds(
    CASPDataProvider(d=2)
)
train_X = data["train"][0]
train_Y = data["train"][1]

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
        hidden_layers=(50, 50, 50),
        device_num=GPU_DEVICE_NUM,
        batchsize_y=None,
        batchsize_u=None,
        inference_batch_size=100,
        lr_factor=0.9,
        lr_patience=300,
        lr_threshold=0.5 * 0.01,
    )

vqr_est = VectorQuantileRegressor(n_levels=T, solver=solver)

vqr_est.fit(train_X, train_Y)
