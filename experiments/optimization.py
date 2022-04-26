from typing import Any, Dict, Optional, Sequence

import click
import numpy as np

from vqr import VectorQuantileRegressor
from experiments.base import VQROptions, run_exp_context
from experiments.data.mvn import LinearMVNDataProvider
from experiments.data.quantile import QuantileFunctionDataProviderWrapper


def single_optim_exp(
    N: int,
    T: int,
    d: int,
    k: int,
    solver_name: str,
    solver_opts: dict,
    seed: int = 42,
) -> Dict[str, Any]:

    # Data provider
    t_factor = 100
    wrapped_provider = LinearMVNDataProvider(d=d, k=k, seed=seed)
    data_provider = QuantileFunctionDataProviderWrapper(
        wrapped_provider=wrapped_provider,
        vqr_n_levels=T * t_factor,
        vqr_fit_n=10000,
    )

    # Obtain g.t. VQR quantile function for x=1
    x_eval = np.ones((2, k))
    Qs_opt = data_provider.vqr.fitted_solution.vector_quantiles(X=x_eval)[0]
    Q_opt = np.stack(Qs_opt, axis=-1)

    idx = tuple([*[slice(0, None, t_factor)] * d, slice(None)])
    Q_opt_subsampled = Q_opt[idx]
    assert Q_opt_subsampled.shape == tuple([*[T] * d, 1])

    # Generate data
    X, Y = data_provider.sample(n=N)

    optimization_dists = []

    # Define a callback that will be invoked each iteration (epoch or batch)
    def _post_iter_callback(
        solution, batch_loss, epoch_loss, epoch_idx, batch_idx, num_epochs, num_batches
    ):
        Qs_est = solution.vector_quantiles(X=x_eval)[0]
        Q_est = np.stack(Qs_est, axis=-1)

        # Calculate distance from g.t.
        dist = np.linalg.norm(Q_est - Q_opt_subsampled)
        optimization_dists.append(dist.item())

        if epoch_idx % 100 == 0:
            import matplotlib.pyplot as plt

            plt.plot(Q_opt_subsampled, label="opt")
            plt.plot(Q_est, label="est")
            plt.title(f"{epoch_idx=}")
            plt.legend()
            plt.show()

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
def optim_exp(
    ctx: click.Context,
    **kw,
):

    # Generate experiment configs from CLI
    vqr_options = VQROptions.parse(ctx)
    exp_configs = {
        vqr_option.key(): dict(
            **vqr_option.to_dict(),
        )
        for vqr_option in vqr_options
    }

    return run_exp_context(
        ctx, exp_fn=single_optim_exp, exp_configs=exp_configs, write_csv=True
    )
