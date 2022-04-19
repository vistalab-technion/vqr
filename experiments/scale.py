import logging
from typing import Optional, Sequence
from pathlib import Path
from itertools import product

import click
import numpy as np
import torch
from numpy import ndarray
from torch import tensor

from vqr.api import VectorQuantileRegressor
from experiments.data.mvn import LinearMVNDataProvider
from experiments.utils.helpers import experiment_id
from experiments.utils.metrics import kde_l1, w2_keops
from experiments.utils.parallel import run_parallel_exp
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)

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
@click.option("-N", "ns", type=int, multiple=True, default=[1000], help="Samples")
@click.option("-T", "ts", type=int, multiple=True, default=[20], help="Quantile levels")
@click.option("-d", "ds", type=int, multiple=True, default=[2], help="Y dimension")
@click.option("-k", "ks", type=int, multiple=True, default=[3], help="X dimension")
@click.option("-E", "epsilons", type=float, multiple=True, default=[1e-6], help="eps")
@click.option("--bs-y", "bys", type=int, multiple=True, default=[-1], help="Batch Y")
@click.option("--bs-u", "bus", type=int, multiple=True, default=[-1], help="Batch U")
@click.option("--epochs", type=int, default=1000, help="epochs")
@click.option("--lr", type=float, default=0.5, help="Learning rate")
@click.option("--lr-max-steps", type=int, default=10, help="LR sched. steps")
@click.option("--lr-factor", type=float, default=0.9, help="LR sched. factor")
@click.option("--lr-patience", type=int, default=500, help="LR sched. patience")
@click.option("--lr-threshold", type=float, default=0.01 * 5, help="LR sched. thresh.")
@click.option("--mlp/--no-mlp", type=bool, default=False, help="NL-VQR with MLP")
@click.option("--mlp-layers", type=str, default="32,32", help="comma-separated ints")
@click.option("--mlp-skip/--no-mlp-skip", type=bool, default=False, help="MLP residual")
@click.option("--mlp-activation", type=str, default="relu", help="MLP activation")
@click.option("--mlp-batchnorm/--no-mlp-batchnorm", type=bool, default=False)
@click.option("--mlp-dropout", type=float, default=0.0)
@click.option("--out-tag", type=str, default="")
def scale_exp(
    ctx: click.Context,
    ns: Sequence[int],
    ts: Sequence[int],
    ds: Sequence[int],
    ks: Sequence[int],
    epsilons: Sequence[float],
    bys: Sequence[Optional[int]],
    bus: Sequence[Optional[int]],
    epochs: int,
    lr: float,
    lr_max_steps: int,
    lr_factor: float,
    lr_patience: int,
    lr_threshold: float,
    mlp: bool,
    mlp_layers: Optional[str],
    mlp_skip: Optional[bool],
    mlp_activation: Optional[str],
    mlp_batchnorm: Optional[bool],
    mlp_dropout: Optional[float],
    out_tag: str = None,
):
    exp_id = experiment_id(name="scale", tag=out_tag)

    # Get global options
    gpu_enabled: bool = ctx.parent.params["gpu"]
    gpu_devices: Optional[str] = ctx.parent.params["devices"]
    num_processes: int = ctx.parent.params["processes"]
    ppd: int = ctx.parent.params["ppd"]
    out_dir: Path = ctx.parent.params["out_dir"]

    # parse mlp options
    mlp_opts = dict(
        hidden_layers=mlp_layers,
        activation=mlp_activation,
        skip=mlp_skip,
        batchnorm=mlp_batchnorm,
        dropout=mlp_dropout,
    )
    # Filter out defaults and remove everything if mlp==False.
    mlp_opts = {k: v for k, v in mlp_opts.items() if v is not None and mlp}

    solver_name = (
        MLPRegularizedDualVQRSolver.solver_name()
        if mlp
        else RegularizedDualVQRSolver.solver_name()
    )

    exp_configs = [
        dict(
            N=N_,
            T=T_,
            d=d_,
            k=k_,
            solver_name=solver_name,
            solver_opts=dict(
                verbose=False,
                num_epochs=epochs,
                epsilon=eps_,
                lr=lr,
                lr_max_steps=lr_max_steps,
                lr_factor=lr_factor,
                lr_patience=lr_patience,
                lr_threshold=lr_threshold,
                batchsize_y=bs_y_ if bs_y_ > 0 else None,
                batchsize_u=bs_u_ if bs_u_ > 0 else None,
                gpu=gpu_enabled,
                **mlp_opts,
            ),
        )
        for (N_, T_, d_, k_, eps_, bs_y_, bs_u_) in product(
            ns, ts, ds, ks, epsilons, bys, bus
        )
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
