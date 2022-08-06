import logging
from typing import Optional

import click
import numpy as np
import torch
from numpy import ndarray

from vqr.api import VectorQuantileRegressor
from experiments.base import VQROptions, run_exp_context
from experiments.datasets.mvn import LinearMVNDataProvider
from experiments.utils.metrics import kde_l1, w2_keops

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
    validation_proportion: float = 0.25,
    cov_n: int = 1000,
    cov_alpha: float = 0.05,
    dist_n: int = 1000,
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

    # Estimate cond distribution and compare it with the gt cond distribution
    w2_dists = []
    kde_l1_dists = []
    for i in range(min(dist_n, len(Y_valid))):
        x = X_valid[[i], :]
        _, Y_gt = data_provider.sample(n=T**d, x=x)
        Y_est = vqr.sample(n=T**d, x=x)
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
@VQROptions.cli
@click.option(
    "--validation-proportion",
    type=float,
    default=0.25,
    help="Proportion of between validation set and training set.",
)
@click.option(
    "--cov-n",
    type=int,
    default=1000,
    help="Number of validation-set samples to use for coverage calculation",
)
@click.option(
    "--cov-alpha",
    type=float,
    default=0.05,
    help="Quantile level for coverage calculation.",
)
@click.option(
    "--dist-n",
    type=int,
    default=1000,
    help=(
        "Number of validation-set samples to use for conditional distribution "
        "estimation"
    ),
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for data generation",
)
def scale_exp(
    ctx: click.Context,
    validation_proportion: float,
    cov_n: int,
    cov_alpha: float,
    dist_n: int,
    seed: int,
    **kw,
):

    # Generate experiment configs from CLI
    vqr_options = VQROptions.parse_multiple(ctx)
    exp_configs = {
        vqr_option.key(): dict(
            cov_n=cov_n,
            cov_alpha=cov_alpha,
            dist_n=dist_n,
            validation_proportion=validation_proportion,
            seed=seed,
            **vqr_option.to_dict(),
        )
        for vqr_option in vqr_options
    }

    return run_exp_context(
        ctx, exp_fn=single_scale_exp, exp_configs=exp_configs, write_csv=True
    )
