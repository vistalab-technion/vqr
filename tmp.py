from numpy import ceil, mean, array, floor

from vqr import VectorQuantileRegressor
from vqr.coverage import measure_width, measure_coverage
from experiments.data.real_data import DataFolder, CASPDataProvider
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)

data = DataFolder(train_ratio=0.6, val_ratio=0.1, test_ratio=0.05).generate_folds(
    CASPDataProvider(d=2)
)

train_X = data["train"][0]
train_Y = data["train"][1]

valid_X = data["calib"][0]
valid_Y = data["calib"][1]

test_X = data["test"][0]
test_Y = data["test"][1]

T = 50
num_epochs = 10000
linear = False
GPU_DEVICE_NUM = 1
device = f"cuda:{GPU_DEVICE_NUM}" if GPU_DEVICE_NUM is not None else "cpu"
epsilon = 1e-2


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
        lr=0.1,
        gpu=True,
        skip=True,
        batchnorm=False,
        hidden_layers=(1000, 1000, 1000),
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

alpha = 0.05

# Measure coverage and quantile width
i_lo = int(floor(T * alpha))
i_hi = int(ceil(T * (1 - alpha)))

coverages = []
widths = []

for X_, Y_ in zip(test_X, test_Y):
    Q1, Q2 = vqr_est.vector_quantiles(X_, refine=False)[0].values
    Q_surface = array(  # (N, 2)
        [
            [
                *Q1[i_lo:i_hi, i_hi],
                *Q1[i_lo, i_lo:i_hi],
                *Q1[i_lo:i_hi, i_lo],
                *Q1[i_hi, i_lo:i_hi],
            ],
            [
                *Q2[i_lo:i_hi, i_hi],
                *Q2[i_lo, i_lo:i_hi],
                *Q2[i_lo:i_hi, i_lo],
                *Q2[i_hi, i_lo:i_hi],
            ],
        ]
    ).T
    coverage = measure_coverage(Q_surface, Y_[None, :])
    width = measure_width(Q_surface)
    print(f"{coverage:.2f}, {width:.2f}")
    coverages.append(coverage)
    widths.append(width)

print(mean(coverages), mean(widths))
