import logging
from typing import Optional, Sequence
from pathlib import Path
from itertools import product

import click
import numpy as np
from numpy import ndarray

from vqr.api import VectorQuantileRegressor
from experiments.data.mvn import LinearMVNDataProvider
from experiments.utils.helpers import experiment_id
from experiments.utils.metrics import w2_keops
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
    solver_type: str,
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
        n_levels=T, solver=solver_type, solver_opts=solver_opts
    )
    vqr.fit(X, Y)

    # Measure coverage on n_coverage samples
    cov_train = _measure_paired_coverage(X, Y, vqr, cov_n, cov_alpha, seed)
    cov_valid = _measure_paired_coverage(X_valid, Y_valid, vqr, cov_n, cov_alpha, seed)

    # Estimate d distribution and compare it with the gt cond distribution
    w2 = 0.0
    for i in range(5):
        _LOG.info(f"{X_valid[i, :].shape=}")
        _, Y_gt = data_provider.sample(n=1000, X=X_valid[i, :][None, :])
        Y_est = vqr.sample(n=1000, x=X_valid[i, :][None, :])
        w2 += w2_keops(Y_gt, Y_est).detach().cpu().item()

    # W2 metric
    w2 /= 5.0

    return {
        "N": N,
        "T": T,
        "d": d,
        "k": k,
        "solver_type": solver_type,
        "solver": solver_opts,  # note: non-consistent key name on purpose
        "train_coverage": cov_train,
        "valid_coverage": cov_valid,
        "w2": w2,
        **vqr.solution_metrics,
    }


@click.command(name="scale-exp")
@click.pass_context
@click.option("-N", "N", type=int, multiple=True, default=[1000])
@click.option("-T", "T", type=int, multiple=True, default=[20])
@click.option("-d", type=int, multiple=True, default=[2])
@click.option("-k", type=int, multiple=True, default=[3])
@click.option("--bs-y", type=int, multiple=True, default=[-1])
@click.option("--bs-u", type=int, multiple=True, default=[-1])
@click.option("--epochs", type=int, default=1000)
@click.option("--epsilon", type=float, default=1e-6)
@click.option("--lr", "learning_rate", type=float, default=0.5)
@click.option("--out-tag", type=str, default="")
def run_scale_exps(
    ctx: click.Context,
    N: Sequence[int],
    T: Sequence[int],
    d: Sequence[int],
    k: Sequence[int],
    bs_y: Sequence[Optional[int]],
    bs_u: Sequence[Optional[int]],
    epochs: int,
    epsilon: float,
    learning_rate: float,
    out_tag: str = None,
):
    # Get global options
    gpu_enabled: bool = ctx.parent.params["gpu"]
    gpu_devices: Optional[str] = ctx.parent.params["devices"]
    num_processes: int = ctx.parent.params["processes"]
    ppd: int = ctx.parent.params["ppd"]
    out_dir: Path = ctx.parent.params["out_dir"]

    exp_id = experiment_id(name="scale", tag=out_tag)

    exp_configs = [
        dict(
            N=N_,
            T=T_,
            d=d_,
            k=k_,
            solver_type="regularized_dual",
            solver_opts=dict(
                verbose=False,
                num_epochs=epochs,
                epsilon=epsilon,
                learning_rate=learning_rate,
                batchsize_y=bs_y_ if bs_y_ > 0 else None,
                batchsize_u=bs_u_ if bs_u_ > 0 else None,
                gpu=gpu_enabled,
            ),
        )
        for (N_, T_, d_, k_, bs_y_, bs_u_) in product(N, T, d, k, bs_y, bs_u)
    ]

    results_df = run_parallel_exp(
        exp_name=exp_id,
        exp_fn=single_scale_exp,
        exp_configs=exp_configs,
        max_workers=num_processes,
        gpu_enabled=gpu_enabled,
        gpu_devices=gpu_devices,
        workers_per_device=ppd,
    )

    out_file_path = out_dir.joinpath(f"{exp_id}.csv")
    results_df.to_csv(out_file_path, index=False)
    _LOG.info(f"Wrote output file: {out_file_path.absolute()!s}")
    return results_df
