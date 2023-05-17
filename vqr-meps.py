import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from vqr import VectorQuantileRegressor
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)

dataset = "meps_20"  # blog_data, bio
DATA_FILE_NAME = f"{dataset}.pkl"
DATA_FOLDER_NAME = "./data/"
num_trials = 1


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
    # permuted_indices = np.random.permutation(np.arange(0, train_size + valid_size))
    permuted_indices = np.arange(0, train_size + valid_size)
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
    num_epochs = 10000
    linear = True
    GPU_DEVICE_NUM = 1
    device = f"cuda:{GPU_DEVICE_NUM}" if GPU_DEVICE_NUM is not None else "cpu"
    epsilon = 2e-2

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

    vqr_est = VectorQuantileRegressor(n_levels=T, solver=solver)
    vqr_est.fit(train_X, train_Y)

    coverages = []
    widths = []
    contours = []
    alpha = 0.06

    for X_test_i, Y_test_i in zip(test_X, test_Y):
        contour_i = vqr_est.quantile_contour(x=X_test_i, alpha=alpha, refine=False)[0]
        coverage_i = vqr_est.coverage(
            Y=Y_test_i[None, :], x=X_test_i[None, :], alpha=alpha
        )
        width_i = vqr_est.width(x=X_test_i[None, :], alpha=alpha)
        coverages.append(coverage_i)
        widths.append(width_i)
        contours.append(contour_i)

    print(f"Trial {trial_num}, Coverage: {np.round(np.mean(coverages), 3)}")
    print(f"Trial {trial_num}, Area: {np.round(np.mean(widths), 3)}")

    with open(f"vqr-{dataset}-contours.pkl", "wb") as f:
        pickle.dump(
            {
                "test_set": [test_X, test_Y],
                "coverages": coverages,
                "widths": widths,
                "contours": contours,
                "vqr_est": vqr_est,
            },
            f,
        )

    global_coverages.append(np.mean(coverages))
    global_widths.append(np.mean(widths))

print(global_coverages, global_widths)
