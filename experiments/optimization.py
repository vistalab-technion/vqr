import logging
from typing import Any, Dict, Optional, Sequence

import click
import numpy as np

from vqr import QuantileFunction, VectorQuantileRegressor
from experiments.base import VQROptions, run_exp_context
from experiments.data.mvn import LinearMVNDataProvider
from experiments.data.quantile import QuantileFunctionDataProviderWrapper

_LOG = logging.getLogger(__name__)


def _compare_conditional_quantiles(
    vqf_gt: QuantileFunction, vqf_est: QuantileFunction, t_factor: int
) -> float:

    # Make sure shapes are consistent and that both quantile functions are
    # conditional on the same X.
    assert vqf_gt.d == vqf_est.d
    assert t_factor == vqf_gt.T // vqf_est.T
    assert np.allclose(vqf_gt.X, vqf_est.X)

    T = vqf_est.T
    d = vqf_est.d

    q_surfaces_gt = vqf_gt.values  # (d, T, T, ..., T)

    # We support having more levels in the gt function, in which case we index the
    # levels corresponding to the estimated levels.
    idx = (slice(None), *[slice(0, None, t_factor)] * d)
    q_surfaces_gt_subsampled = q_surfaces_gt[idx]

    q_surfaces_est = vqf_est.values  # (d, T, T, ..., T)
    assert q_surfaces_gt_subsampled.shape == q_surfaces_est.shape

    return np.linalg.norm(q_surfaces_gt_subsampled - q_surfaces_est).item()


def single_optim_exp(
    N: int,
    T: int,
    d: int,
    k: int,
    solver_name: str,
    solver_opts: dict,
    dp_vqr_n: int,
    dp_vqr_t_factor: int,
    dp_vqr_solver_opts: dict,
    n_eval_x: int,
    seed: int = 42,
) -> Dict[str, Any]:

    # Data provider
    wrapped_provider = LinearMVNDataProvider(d=d, k=k, seed=seed)
    data_provider = QuantileFunctionDataProviderWrapper(
        wrapped_provider=wrapped_provider,
        vqr_n_levels=T * dp_vqr_t_factor,
        vqr_fit_n=dp_vqr_n,
        vqr_solver_opts=dp_vqr_solver_opts,
    )

    # Sample values of x on which we evaluate
    eval_x = wrapped_provider.sample_x(n=n_eval_x)

    # Obtain g.t. VQR quantile functions
    vqfs_gt = data_provider.vqr.fitted_solution.vector_quantiles(X=eval_x)

    # Generate data
    X, Y = data_provider.sample(n=N)

    optimization_dists = []

    # Define a callback that will be invoked each iteration (epoch or batch)
    def _post_iter_callback(
        solution, batch_loss, epoch_loss, epoch_idx, batch_idx, num_epochs, num_batches
    ):
        # Obtain quantile functions from current iteration, conditioned on the same X's
        vqfs_est = solution.vector_quantiles(X=eval_x)

        # Calculate distance from g.t.
        dists = [
            _compare_conditional_quantiles(vqf_gt, vqf_est, t_factor=dp_vqr_t_factor)
            for vqf_gt, vqf_est in zip(vqfs_gt, vqfs_est)
        ]
        optimization_dists.append(dists)

    # Add a solver callback to evaluate the distance from g.t. solution
    solver_opts["post_iter_callback"] = _post_iter_callback

    ###
    solver_opts["verbose"] = True

    # Solve
    vqr = VectorQuantileRegressor(
        n_levels=T,
        solver=solver_name,
        solver_opts=solver_opts,
    ).fit(X, Y)

    # Remove callback so that it doesn't get serialized
    solver_opts.pop("post_iter_callback")

    return dict(
        N=N,
        T=T,
        d=d,
        k=k,
        solver_type=solver_name,
        solver=solver_opts,  # note: non-consistent key name on purpose
        optimization_dists=np.round(optimization_dists, decimals=5).tolist(),
    )


@click.command(name="optim-exp")
@click.pass_context
@VQROptions.cli
@click.option(
    "--dp-vqr-n",
    type=int,
    default=10000,
    help="Number of samples to use for fitting the data provider's VQR model",
)
@click.option(
    "--dp-t-factor",
    type=int,
    default=100,
    help=(
        "Factor between data provider's number of quantile levels and the fitted "
        "models' number of levels. Data provider will have T*t_factor levels."
    ),
)
@click.option(
    "--n-eval-x",
    type=int,
    default=10,
    help=("Number of X's to use when comparing g.t. to estimated quantile function"),
)
def optim_exp(
    ctx: click.Context,
    dp_vqr_n: int,
    dp_t_factor: int,
    n_eval_x: int,
    **kw,
):

    # Generate experiment configs from CLI
    vqr_options = VQROptions.parse_multiple(ctx)
    exp_configs = {
        vqr_option.key(): dict(
            dp_vqr_n=dp_vqr_n,
            dp_vqr_t_factor=dp_t_factor,
            n_eval_x=n_eval_x,
            dp_vqr_solver_opts=dict(epsilon=5e-3),
            **vqr_option.to_dict(),
        )
        for vqr_option in vqr_options
    }

    return run_exp_context(
        ctx, exp_fn=single_optim_exp, exp_configs=exp_configs, write_csv=True
    )
