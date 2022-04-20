from __future__ import annotations

import logging
import functools
from abc import ABC, abstractmethod
from time import strftime
from typing import Any, Dict, List, Callable, Optional, Sequence
from pathlib import Path
from itertools import product

import click
import pandas as pd
from _socket import gethostname

from experiments import EXPERIMENTS_OUT_DIR
from experiments.utils.helpers import stable_hash
from experiments.utils.parallel import run_parallel_exp
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)

_LOG = logging.getLogger(__name__)


def _click_decorator(fn: Callable, options: Sequence[click.Option]):
    """
    Generates a decorator which applies a sequence of click.Options to function.
    :param fn: The function to decorate.
    :param options: The click options to apply to the function.
    :return: The decorated function.
    """
    # Reduce applies each option to f repeatedly, i.e. opt1(opt2(...optN(f) ...))
    wrapper = functools.reduce(lambda fn_, opt: opt(fn_), reversed(options), fn)
    return functools.wraps(fn)(wrapper)


class _CLIOptions(ABC):
    """
    Helper class for grouping multiple click CLI options together and then parsing them
    from the context.
    """

    @classmethod
    @abstractmethod
    def _cli_options(cls) -> Sequence[click.Option]:
        """
        :return: A list of click.Option decorators which can be applied to a
        click.Command.
        """
        pass

    @classmethod
    def cli(cls, fn: Callable) -> Callable:
        """
        A decorator which can be applied to a click.Command. Adds the options defined in
        the _cli_options method.
        :param fn:
        :return:
        """
        return _click_decorator(fn, cls._cli_options())

    @classmethod
    @abstractmethod
    def parse(cls, ctx: click.Context) -> _CLIOptions:
        """
        Parses a click.Context into a _CLIOptions instance.
        :param ctx: The click.Context.
        :return: An instance of this class, initialized based on parsing the context.
        """
        pass

    def key(self) -> str:
        """
        :return: A string which uniquely represents the current instance's options.
        """
        return stable_hash(self.__dict__)

    def __hash___(self):
        return hash(self.key())


class GPUOptions(_CLIOptions):
    """
    Encapsulates GPU options for an experiment.
    """

    def __init__(
        self,
        gpu_enabled: bool,
        gpu_devices: Optional[str],
        num_processes: int,
        ppd: int,
    ):
        self.gpu_enabled = gpu_enabled
        self.gpu_devices = gpu_devices
        self.num_processes = num_processes
        self.ppd = ppd

    @classmethod
    def _cli_options(cls) -> Sequence[click.Option]:
        return [  # type:ignore
            click.option(
                "-g", "--gpu/--no-gpu", default=False, help="Enable GPU support"
            ),
            click.option(
                "--devices",
                default=None,
                type=str,
                help="GPU devices (comma separated)",
            ),
            click.option(
                "-p",
                "--processes",
                type=int,
                default=-1,
                help="Number of processes",
            ),
            click.option("--ppd", type=int, default=1, help="Processes per GPU device"),
        ]

    @classmethod
    def parse(cls, ctx: click.Context) -> GPUOptions:
        """
        Parses GPU options from context.
        :param ctx: Click context.
        :return: A GPUOptions instance.
        """
        while ctx and "gpu" not in ctx.params:
            ctx = ctx.parent
        if not ctx:
            raise ValueError("GPU options not found in context")

        return GPUOptions(
            gpu_enabled=ctx.params["gpu"],
            gpu_devices=ctx.params["devices"],
            num_processes=ctx.params["processes"],
            ppd=ctx.params["ppd"],
        )


class OutputOptions(_CLIOptions):
    """
    Encapsulates output options for an experiment.
    """

    def __init__(self, out_dir: Path, out_tag: str):
        self.out_dir = out_dir
        self.out_tag = out_tag

    @classmethod
    def _cli_options(cls) -> Sequence[click.Option]:
        return [  # type:ignore
            click.option(
                "-o",
                "--out-dir",
                type=Path,
                default=EXPERIMENTS_OUT_DIR,
                help="Output directory",
            ),
            click.option("--out-tag", type=str, default="", help="Output tag"),
        ]

    @classmethod
    def parse(cls, ctx: click.Context) -> OutputOptions:
        """
        Parses output options from context.
        :param ctx: Click context.
        :return: An OutputOptions instance.
        """
        while ctx and "out_dir" not in ctx.params:
            ctx = ctx.parent
        if not ctx:
            raise ValueError("Output options not found in context")

        return OutputOptions(
            out_dir=ctx.params["out_dir"],
            out_tag=ctx.params["out_tag"],
        )


class LoggingOptions(_CLIOptions):
    """
    Encapsulates output options for an experiment.
    """

    def __init__(self, debug: bool):
        self.debug = debug

    @classmethod
    def _cli_options(cls) -> Sequence[click.Option]:
        return [  # type:ignore
            click.option("--debug/--no-debug", default=False, help="Use DEBUG logging"),
        ]

    @classmethod
    def parse(cls, ctx: click.Context) -> LoggingOptions:
        while ctx and "debug" not in ctx.params:
            ctx = ctx.parent
        if not ctx:
            raise ValueError("Logging options not found in context")
        return LoggingOptions(debug=ctx.params["debug"])


class VQROptions(_CLIOptions):
    """
    Encapsulates VQR options for an experiment.
    """

    def __init__(
        self,
        N: int,
        T: int,
        d: int,
        k: int,
        solver_name: str,
        solver_opts: Dict[str, Any],
    ):
        self.N = N
        self.T = T
        self.d = d
        self.k = k
        self.solver_name = solver_name
        self.solver_opts = solver_opts

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def _cli_options(cls) -> Sequence[click.Option]:
        return [  # type:ignore
            click.option(
                "-N", "ns", type=int, multiple=True, default=[1000], help="Samples"
            ),
            click.option(
                "-T",
                "ts",
                type=int,
                multiple=True,
                default=[20],
                help="Quantile levels",
            ),
            click.option(
                "-d", "ds", type=int, multiple=True, default=[2], help="Y dimension"
            ),
            click.option(
                "-k", "ks", type=int, multiple=True, default=[3], help="X dimension"
            ),
            click.option(
                "-E",
                "epsilons",
                type=float,
                multiple=True,
                default=[1e-6],
                help="epsilon",
            ),
            click.option(
                "--bs-y",
                "bys",
                type=int,
                multiple=True,
                default=[-1],
                help="Batch size of Y (samples), -1 disables batching",
            ),
            click.option(
                "--bs-u",
                "bus",
                type=int,
                multiple=True,
                default=[-1],
                help="Batch size of U (quantile levels), -1 disables batching",
            ),
            click.option("--epochs", type=int, default=1000, help="epochs"),
            click.option("--lr", type=float, default=0.5, help="Learning rate"),
            click.option(
                "--lr-max-steps", type=int, default=10, help="LR sched. steps"
            ),
            click.option(
                "--lr-factor", type=float, default=0.9, help="LR sched. factor"
            ),
            click.option(
                "--lr-patience", type=int, default=500, help="LR sched. patience"
            ),
            click.option(
                "--lr-threshold", type=float, default=0.01 * 5, help="LR sched. thresh."
            ),
            click.option(
                "--mlp/--no-mlp", type=bool, default=False, help="NL-VQR with MLP"
            ),
            click.option(
                "--mlp-layers", type=str, default="32,32", help="comma-separated ints"
            ),
            click.option(
                "--mlp-skip/--no-mlp-skip",
                type=bool,
                default=False,
                help="MLP residual",
            ),
            click.option(
                "--mlp-activation", type=str, default="relu", help="MLP activation"
            ),
            click.option(
                "--mlp-batchnorm/--no-mlp-batchnorm", type=bool, default=False
            ),
            click.option("--mlp-dropout", type=float, default=0.0),
        ]

    @classmethod
    def parse(cls, ctx: click.Context) -> Sequence[VQROptions]:

        while ctx and "ns" not in ctx.params:
            ctx = ctx.parent
        if not ctx:
            raise ValueError("VQR options not found in context")

        gpu_options = GPUOptions.parse(ctx)
        p = ctx.params

        # parse mlp options
        mlp_opts = dict(
            hidden_layers=p["mlp_layers"],
            activation=p["mlp_activation"],
            skip=p["mlp_skip"],
            batchnorm=p["mlp_batchnorm"],
            dropout=p["mlp_dropout"],
        )
        # Filter out defaults and remove everything if mlp==False.
        mlp_opts = {k: v for k, v in mlp_opts.items() if v is not None and p["mlp"]}

        solver_name = (
            MLPRegularizedDualVQRSolver.solver_name()
            if p["mlp"]
            else RegularizedDualVQRSolver.solver_name()
        )

        vqr_options = [
            VQROptions(
                N=N_,
                T=T_,
                d=d_,
                k=k_,
                solver_name=solver_name,
                solver_opts=dict(
                    verbose=False,
                    num_epochs=p["epochs"],
                    epsilon=eps_,
                    lr=p["lr"],
                    lr_max_steps=p["lr_max_steps"],
                    lr_factor=p["lr_factor"],
                    lr_patience=p["lr_patience"],
                    lr_threshold=p["lr_threshold"],
                    batchsize_y=bs_y_ if bs_y_ > 0 else None,
                    batchsize_u=bs_u_ if bs_u_ > 0 else None,
                    gpu=gpu_options.gpu_enabled,
                    **mlp_opts,
                ),
            )
            for (N_, T_, d_, k_, eps_, bs_y_, bs_u_) in product(
                p["ns"], p["ts"], p["ds"], p["ks"], p["epsilons"], p["bys"], p["bus"]
            )
        ]

        return vqr_options


def experiment_id(name: str, tag: str):
    """
    Creates a unique id for an experiment based on hostname, timestamp and a
    user-specified tag.
    :param name: An experiment name.
    :param tag: A user tag.
    :return: The experiment id.
    """
    hostname = gethostname()
    if hostname:
        hostname = hostname.split(".")[0].strip()
    else:
        hostname = "localhost"

    name = f"{name}-" if name else ""
    tag = f"-{tag}" if tag else ""
    timestamp = strftime(f"%Y%m%d_%H%M%S")
    exp_id = strftime(f"{name}{timestamp}-{hostname}{tag}")
    return exp_id


def run_exp_context(
    ctx: click.Context,
    exp_fn: Callable[[Any], Dict[str, Any]],
    exp_configs: Dict[str, dict],
    write_csv: bool = True,
) -> pd.DataFrame:
    """
    Runs multiple experiment configs based on common CLI arguments from the context.
    :param ctx: Click context.
    :param exp_fn: Callable which runs a single experiment. Should return a dict of
    results.
    :param exp_configs:  A mapping from an experiment name/identifier to a dict with
    configuration which will be passed to the exp_fn.
    :return: A pandas DataFrame with the collected results. Each row is a result,
    columns correspond to dict keys in the output of exp_fn.
    """
    output_opts = OutputOptions.parse(ctx)
    gpu_opts = GPUOptions.parse(ctx)
    exp_id = experiment_id(name=ctx.command.name, tag=output_opts.out_tag)

    results: Sequence[Dict[str, Any]] = run_parallel_exp(
        exp_name=exp_id,
        exp_fn=exp_fn,
        exp_configs=exp_configs,
        max_workers=gpu_opts.num_processes,
        gpu_enabled=gpu_opts.gpu_enabled,
        gpu_devices=gpu_opts.gpu_devices,
        workers_per_device=gpu_opts.ppd,
    )

    results_df = pd.json_normalize(list(results))

    if write_csv:
        out_file_path = output_opts.out_dir.joinpath(f"{exp_id}.csv")
        results_df.to_csv(out_file_path, index=False)
        _LOG.info(f"Wrote output file: {out_file_path.absolute()!s}")

    return results_df
