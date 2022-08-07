import logging
from typing import Any, Dict

import click
import numpy as np
from matplotlib import pyplot as plt

from vqr import QuantileFunction, VectorQuantileRegressor
from experiments.base import VQROptions, run_exp_context
from experiments.datasets.mvn import LinearMVNDataProvider
from experiments.datasets.quantile import QuantileFunctionDataProviderWrapper

_LOG = logging.getLogger(__name__)


def _compare_conditional_quantiles(
    vqf_gt: QuantileFunction,
    vqf_est: QuantileFunction,
    t_factor: int,
) -> float:

    # Make sure shapes are consistent and that both quantile functions are
    # conditional on the same X.
    assert vqf_gt.d == vqf_est.d
    assert t_factor == vqf_gt.T // vqf_est.T
    assert np.allclose(vqf_gt.X, vqf_est.X)

    T = vqf_est.T
    d = vqf_est.d

    # We support having more levels in the gt function, in which case we index the
    # levels corresponding to the estimated levels.
    idx = (slice(None), *[slice(t_factor - 1, None, t_factor)] * d)

    q_surfaces_gt = vqf_gt.values[idx]  # (d, T, T, ..., T)
    q_surfaces_gt = q_surfaces_gt.reshape(d, -1)
    q_surfaces_est = vqf_est.values  # (d, T, T, ..., T)
    q_surfaces_est = q_surfaces_est.reshape(d, -1)
    assert q_surfaces_gt.shape == q_surfaces_est.shape

    # Make sure we're comparing the same quantile levels
    q_levels_gt = vqf_gt.levels[idx].reshape(d, -1)
    q_levels_est = vqf_est.levels.reshape(d, -1)
    assert q_levels_gt.shape == q_levels_est.shape
    assert np.allclose(q_levels_gt, q_levels_est)

    return (
        np.linalg.norm(q_surfaces_gt - q_surfaces_est) / np.linalg.norm(q_surfaces_gt)
    ).item()


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

    dp_vqr_solver_opts["verbose"] = True
    # Data provider
    wrapped_provider = LinearMVNDataProvider(d=d, k=k, seed=seed)
    data_provider = QuantileFunctionDataProviderWrapper(
        wrapped_provider=wrapped_provider,
        vqr_n_levels=T * dp_vqr_t_factor,
        vqr_fit_n=dp_vqr_n,
        vqr_solver_opts=dp_vqr_solver_opts,
        seed=seed,
    )

    # Sample values of x on which we evaluate
    eval_x = wrapped_provider.sample_x(n=n_eval_x)

    # Obtain g.t. VQR quantile functions
    vqfs_gt = data_provider.vqr.vector_quantiles(X=eval_x, refine=True)

    # Generate data
    X, Y = data_provider.sample(n=N)

    optimization_dists = []

    # Define a callback that will be invoked each iteration (epoch or batch)
    def _post_iter_callback(
        solution,
        batch_loss,
        epoch_loss,
        epoch_idx,
        batch_idx,
        num_epochs,
        num_batches,
        xy_slice,
        u_slice,
    ):
        # Obtain quantile functions from current iteration, conditioned on the same X's
        eval_x_scaled = data_provider.vqr._scaler.transform(eval_x)
        vqfs_est = solution.vector_quantiles(X=eval_x_scaled, refine=True)

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

    if plot := False:
        x = np.arange(len(optimization_dists))
        y = np.mean(optimization_dists, axis=1)
        yerr_min = np.min(optimization_dists, axis=1)
        yerr_max = np.max(optimization_dists, axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(x, y, "k-")
        ax.grid()
        ax.set_ylim([0, 2])
        ax.fill_between(x, y - yerr_min, y + yerr_max, alpha=0.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$\frac{||Q^{*}(u)-\hat{Q}(u)||_{2}}{||Q^{*}(u)||_{2}}$")
        plt.savefig("optim-exp.png", tight_layout=True)

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
@VQROptions.cli(prefix="dp")
@click.option(
    "--t-factor",
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
    t_factor: int,
    n_eval_x: int,
    **kw,
):

    # We don't support multiple DP options, only first is used
    vqr_options_dp = VQROptions.parse(ctx, prefix="dp")

    # Generate experiment configs from CLI
    vqr_options = VQROptions.parse_multiple(ctx)
    exp_configs = {
        vqr_option.key(): dict(
            n_eval_x=n_eval_x,
            dp_vqr_n=vqr_options_dp.N,
            dp_vqr_t_factor=t_factor,
            dp_vqr_solver_opts=vqr_options_dp.solver_opts,
            **vqr_option.to_dict(),
        )
        for vqr_option in vqr_options
    }

    return run_exp_context(
        ctx, exp_fn=single_optim_exp, exp_configs=exp_configs, write_csv=True
    )
