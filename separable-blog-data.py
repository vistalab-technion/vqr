import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from vqr import VectorQuantileRegressor
from experiments.utils.tensors import ensure_numpy
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)

dataset = "blog_data"
DATA_FILE_NAME = f"{dataset}.pkl"
DATA_FOLDER_NAME = "./data/"
num_trials = 10

with open(f"{DATA_FOLDER_NAME}{DATA_FILE_NAME}", "rb") as f:
    all_data = pickle.load(f)
    f.close()

train_size = all_data["x_train"].shape[0]
valid_size = all_data["x_test"].shape[0]
all_X = np.concatenate(
    [all_data["x_train"].cpu().numpy(), all_data["x_test"].cpu().numpy()], axis=0
)
all_Y = np.concatenate(
    [all_data["y_train"].cpu().numpy(), all_data["y_test"].cpu().numpy()], axis=0
)
global_coverages = []
global_widths = []

for trial_num in range(num_trials):
    permuted_indices = np.random.permutation(np.arange(0, train_size + valid_size))
    train_X, train_Y = (
        all_X[permuted_indices[:train_size], :],
        all_Y[permuted_indices[:train_size], :],
    )
    test_X, test_Y = (
        all_X[permuted_indices[train_size:], :],
        all_Y[permuted_indices[train_size:], :],
    )

    # Scalers
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(train_X)
    scaler_y.fit(train_Y)

    # Scaled train data
    train_X = scaler_x.transform(train_X)
    train_Y = scaler_y.transform(train_Y)

    T = 100
    num_epochs = 30000
    GPU_DEVICE_NUM = 0
    device = f"cuda:{GPU_DEVICE_NUM}" if GPU_DEVICE_NUM is not None else "cpu"
    epsilon = 1e-2
    solver_opts = {
        "hidden_layers": (50, 30, 10),
        "lr": 0.1,
        "skip": False,
        "batchnorm": False,
        "lr_factor": 0.9,
        "lr_patience": 300,
    }

    solver_1 = MLPRegularizedDualVQRSolver(
        verbose=True,
        num_epochs=num_epochs,
        epsilon=epsilon,
        gpu=True,
        activation="relu",
        device_num=GPU_DEVICE_NUM,
        batchsize_y=None,
        batchsize_u=None,
        inference_batch_size=100,
        **solver_opts,
    )

    vqr_est_1 = VectorQuantileRegressor(n_levels=T, solver=solver_1)
    vqr_est_1.fit(train_X, train_Y[:, [0]])

    solver_2 = MLPRegularizedDualVQRSolver(
        verbose=True,
        num_epochs=num_epochs,
        epsilon=epsilon,
        gpu=True,
        activation="relu",
        device_num=GPU_DEVICE_NUM,
        batchsize_y=None,
        batchsize_u=None,
        inference_batch_size=100,
        **solver_opts,
    )

    vqr_est_2 = VectorQuantileRegressor(n_levels=T, solver=solver_2)
    vqr_est_2.fit(train_X, train_Y[:, [1]])

    test_X, test_Y = scaler_x.transform(test_X), scaler_y.transform(test_Y)
    coverages = []
    widths = []
    alpha = 0.02

    for X_test_i, Y_test_i in zip(test_X, test_Y):
        coverage_1_i = vqr_est_1.coverage(
            Y=Y_test_i[None, [0]], x=X_test_i[None, :], alpha=alpha, refine=True
        )
        coverage_2_i = vqr_est_2.coverage(
            Y=Y_test_i[None, [1]], x=X_test_i[None, :], alpha=alpha, refine=True
        )

        width_1_i = vqr_est_1.width(x=X_test_i[None, :], alpha=alpha, refine=True)
        width_2_i = vqr_est_2.width(x=X_test_i[None, :], alpha=alpha, refine=True)

        coverage_i = coverage_1_i * coverage_2_i
        width_i = width_1_i * width_2_i

        coverages.append(coverage_i)
        widths.append(width_i)

    print(f"Trial {trial_num}, Coverage: {np.round(np.mean(coverages), 3)}")
    print(f"Trial {trial_num}, Area: {np.round(np.mean(widths), 3)}")

    global_coverages.append(np.mean(coverages))
    global_widths.append(np.mean(widths))

print(global_coverages, global_widths)
