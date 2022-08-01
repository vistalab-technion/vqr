import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from vqr import VectorQuantileRegressor
from experiments.utils.tensors import ensure_numpy
from vqr.solvers.dual.regularized_lse import MLPRegularizedDualVQRSolver

dataset = "bio"  # blog_data, bio
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

    test_X, test_Y = scaler_x.transform(test_X), scaler_y.transform(test_Y)

    # Scaled train data
    train_X = scaler_x.transform(train_X)
    train_Y = scaler_y.transform(train_Y)

    T = 50
    num_epochs = 20000
    linear = False
    GPU_DEVICE_NUM = 0
    device = f"cuda:{GPU_DEVICE_NUM}" if GPU_DEVICE_NUM is not None else "cpu"
    epsilon = 1e-2

    solver = MLPRegularizedDualVQRSolver(
        verbose=True,
        num_epochs=num_epochs,
        epsilon=epsilon,
        lr=0.8,
        gpu=True,
        skip=True,
        batchnorm=False,
        hidden_layers=(100, 100, 100),
        activation="relu",
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

    X_test, Y_test = all_data["x_test"], all_data["y_test"]
    X_test, Y_test = scaler_x.transform(ensure_numpy(X_test.cpu())), scaler_y.transform(
        ensure_numpy(Y_test.cpu())
    )
    coverages = []
    widths = []

    for X_test_i, Y_test_i in zip(test_X, test_Y):
        coverage_i = vqr_est.coverage(
            Y=Y_test_i[None, :], x=X_test_i[None, :], alpha=0.05
        )
        width_i = vqr_est.width(x=X_test_i[None, :], alpha=0.05)
        coverages.append(coverage_i)
        widths.append(width_i)

    print(f"Trial {trial_num}, Coverage: {np.round(np.mean(coverages), 3)}")
    print(f"Trial {trial_num}, Area: {np.round(np.mean(widths), 3)}")

    global_coverages.append(np.mean(coverages))
    global_widths.append(np.mean(widths))

print(global_coverages, global_widths)
