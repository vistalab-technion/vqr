import logging
from time import time
from typing import Optional, Sequence
from itertools import product

import numpy as np
import torch.cuda
from numpy import ndarray

from vqr.api import VectorQuantileRegressor
from vqr.data import generate_linear_x_y_mvn_data
from experiments.utils import run_parallel_exp
from experiments.logging import setup_logging

_LOG = logging.getLogger(__name__)


def _measure_paired_coverage(
    X: ndarray,
    Y: ndarray,
    vqr: VectorQuantileRegressor,
    n_subsample: Optional[int] = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)

    N, d = Y.shape
    assert len(X) == N

    if n_subsample:
        n_coverage = min(n_subsample, N)
        idx_cov = rng.permutation(n_coverage)
    else:
        idx_cov = np.arange(N)

    cov = np.mean(
        [
            # Coverage for each single data point is just 0 or 1
            vqr.coverage(y.reshape(1, d), alpha=alpha, x=x)
            for (x, y) in zip(X[idx_cov, :], Y[idx_cov, :])
        ]
    )
    return cov.item()


def single_scale_exp(
    N: int,
    T: int,
    d: int,
    k: int,
    solver_type: str,
    solver_opts: dict,
    cov_n: int = 1000,
    cov_alpha: float = 0.05,
    validation_proportion: float = 0.25,
    seed: int = 42,
):
    # Generate data
    X, Y = generate_linear_x_y_mvn_data(n=N, d=d, k=k, seed=seed)
    N_valid = int(N * validation_proportion)
    X_valid, Y_valid = generate_linear_x_y_mvn_data(n=N_valid, d=d, k=k, seed=seed + 1)

    # Solve
    vqr = VectorQuantileRegressor(
        n_levels=T, solver=solver_type, solver_opts=solver_opts
    )

    start_time = time()
    vqr.fit(X, Y)
    vqr_fit_time = time() - start_time

    # Measure coverage on n_coverage samples
    cov_train = _measure_paired_coverage(X, Y, vqr, cov_n, cov_alpha, seed)
    cov_valid = _measure_paired_coverage(X_valid, Y_valid, vqr, cov_n, cov_alpha, seed)

    return {
        "N": N,
        "T": T,
        "d": d,
        "k": k,
        "solver_type": solver_type,
        "solver": solver_opts,  # note: non-consistent key name on purpose
        "train_coverage": cov_train,
        "valid_coverage": cov_valid,
        "w2": None,  # TODO: for this we need to generate data given X=x
        "time": vqr_fit_time,
    }


def run_scale_exps(
    N: Sequence[int],
    T: Sequence[int],
    d: Sequence[int],
    k: Sequence[int],
    bs_y: Sequence[Optional[int]],
    bs_u: Sequence[Optional[int]],
    eps: Sequence[float],
    gpu_device: int = 0,
    processes: int = 1,
):
    exp_configs = [
        dict(
            N=N_,
            T=T_,
            d=d_,
            k=k_,
            solver_type="regularized_dual",
            solver_opts=dict(
                verbose=False,
                epsilon=eps_,
                learning_rate=0.5,
                batchsize_y=bs_y_,
                batchsize_u=bs_u_,
                gpu=torch.cuda.is_available(),
                device_num=gpu_device,
            ),
        )
        for (N_, T_, d_, k_, bs_y_, bs_u_, eps_) in product(N, T, d, k, bs_y, bs_u, eps)
    ]

    results_df = run_parallel_exp(
        exp_name="scale",
        exp_fn=single_scale_exp,
        exp_configs=exp_configs,
        max_workers=processes,
    )

    print(results_df)


if __name__ == "__main__":
    setup_logging()

    run_scale_exps(
        N=[1000],
        T=[20],
        d=[2],
        k=[3, 5],
        bs_y=[None],
        bs_u=[None],
        eps=[1e-6],
        processes=2,
    )
