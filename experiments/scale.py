import logging
from typing import Optional

import click
import numpy as np
import torch
from numpy import ndarray

from vqr.api import VectorQuantileRegressor
from experiments.base import (
    parse_gpu_options,
    parse_vqr_options,
    parse_output_options,
    click_common_vqr_solver_options,
)
from experiments.data.mvn import LinearMVNDataProvider
from experiments.utils.helpers import experiment_id
from experiments.utils.metrics import kde_l1, w2_keops
from experiments.utils.parallel import run_parallel_exp

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
    solver_name: str,
    solver_opts: dict,
    cov_n: int = 1000,
    cov_alpha: float = 0.05,
    validation_proportion: float = 0.25,
    seed: int = 42,
):
    # Data provider
    data_provider = LinearMVNDataProvider(d=d, k=k, seed=seed)

    # Generate data
    X, Y = data_provider.sample(
        n=int(N * (1 + validation_proportion)),
    )
    X_valid, Y_valid = X[N:, :], Y[N:, :]
    X, Y = X[:N, :], Y[:N, :]

    # Solve
    vqr = VectorQuantileRegressor(
        n_levels=T, solver=solver_name, solver_opts=solver_opts
    )
    vqr.fit(X, Y)

    # Measure coverage on n_coverage samples
    cov_train = _measure_paired_coverage(X, Y, vqr, cov_n, cov_alpha, seed)
    cov_valid = _measure_paired_coverage(X_valid, Y_valid, vqr, cov_n, cov_alpha, seed)

    # Estimate d distribution and compare it with the gt cond distribution
    w2_dists = []
    kde_l1_dists = []
    for i in range(min(cov_n, len(Y_valid))):
        _, Y_gt = data_provider.sample(n=T**d, x=X_valid[[i], :])
        Y_est = vqr.sample(n=T**d, x=X_valid[[i], :])
        w2_dists.append(w2_keops(Y_gt, Y_est))
        kde_l1_dist = kde_l1(
            Y_gt,
            Y_est,
            grid_resolution=T,
            sigma=0.1,
            dtype=torch.float32,
            device="cuda" if solver_opts["gpu"] else "cpu",
        )
        kde_l1_dists.append(kde_l1_dist)

    # W2 metric
    w2_avg = np.mean(w2_dists)
    kde_l1_avg = np.mean(kde_l1_dists)

    return {
        "N": N,
        "T": T,
        "d": d,
        "k": k,
        "solver_type": solver_name,
        "solver": solver_opts,  # note: non-consistent key name on purpose
        "train_coverage": cov_train,
        "valid_coverage": cov_valid,
        "w2": w2_avg,
        "kde_l1": kde_l1_avg,
        **vqr.solution_metrics,
    }


@click.command(name="scale-exp")
@click.pass_context
@click_common_vqr_solver_options
def scale_exp(
    ctx: click.Context,
    **kw,
):
    output_opts = parse_output_options(ctx)
    gpu_opts = parse_gpu_options(ctx)
    exp_id = experiment_id(name="scale", tag=output_opts.out_tag)

    exp_configs = parse_vqr_options(ctx)

    results_df = run_parallel_exp(
        exp_name=exp_id,
        exp_fn=single_scale_exp,
        exp_configs=tuple(e.to_dict() for e in exp_configs),
        max_workers=gpu_opts.num_processes,
        gpu_enabled=gpu_opts.gpu_enabled,
        gpu_devices=gpu_opts.gpu_devices,
        workers_per_device=gpu_opts.ppd,
    )

    out_file_path = output_opts.out_dir.joinpath(f"{exp_id}.csv")
    results_df.to_csv(out_file_path, index=False)
    _LOG.info(f"Wrote output file: {out_file_path.absolute()!s}")
    return results_df
