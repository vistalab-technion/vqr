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
    def _cli_options(cls, prefix: str = "") -> Sequence[click.Option]:
        """
        :param prefix: A prefix to add to each argument name.
        :return: A list of click.Option decorators which can be applied to a
        click.Command.
        """
        pass

    @classmethod
    def cli(cls, fn: Callable = None, *, prefix: str = None) -> Callable:
        """
        A decorator which can be applied to a click.Command. Adds the options defined in
        the _cli_options method.
        :param fn: The function to decorate, should be a click command or group.
        :param prefix: A prefix which will be applied to all CLI options.
        :return: A decorator which adds click options to the given command/group.
        """

        prefix = prefix or ""

        # This is to support the usage as a parametrized decorator, i.e. @cli(k=v)
        if fn is None:
            return functools.partial(cls.cli, prefix=prefix)

        return _click_decorator(fn, cls._cli_options(prefix=prefix))

    @classmethod
    def parse(cls, ctx: click.Context, prefix: str = "") -> _CLIOptions:
        """
        Parses a click.Context into a _CLIOptions instance.
        If there are multiple option values, returns the first one.
        :param ctx: The click.Context.
        :param prefix: The prefix which was applied to all CLI options.
        :return: An instance of this class, initialized based on parsing the context.
        """
        multi_opts = cls.parse_multiple(ctx, prefix=prefix)
        n_opts = len(multi_opts)
        if n_opts != 1:
            raise ValueError(f"Expected single value of parsed options, got {n_opts}")

        return multi_opts[0]

    @classmethod
    @abstractmethod
    def parse_multiple(
        cls,
        ctx: click.Context,
        prefix: str = "",
    ) -> Sequence[_CLIOptions]:
        """
        Parses a click.Context, which may contain some options with multiple
        values, into a sequence of _CLIOptions instance, generated as a product of
        the multiple values.
        :param ctx: The click.Context.
        :param prefix: The prefix which was applied to all CLI options.
        :return: An sequence of instances of this class, initialized based on parsing
        the context.
        """
        pass

    @classmethod
    def _prefix_option(cls, option: str, prefix: str):
        if not prefix:
            return option

        if option.startswith("--"):
            return option.replace("--", f"--{prefix}-")
        elif option.startswith("-"):
            return option.replace("-", f"-{prefix}_")
        else:
            raise ValueError(
                f"Expected that options start with '-' or '--', got {option}"
            )

    @classmethod
    def _prefix_name(cls, name: str, prefix: str):
        if not prefix:
            return name

        if name.startswith("-"):
            raise ValueError(
                f"Expected that option names don't start with '-', got {name}"
            )

        return f"{prefix}_{name}"

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
    def _cli_options(cls, prefix: str = "") -> Sequence[click.Option]:
        _p = functools.partial(cls._prefix_option, prefix=prefix)
        return [  # type:ignore
            click.option(
                _p("-g"), _p("--gpu/--no-gpu"), default=False, help="Enable GPU support"
            ),
            click.option(
                _p("--devices"),
                default=None,
                type=str,
                help="GPU devices (comma separated)",
            ),
            click.option(
                _p("-p"),
                _p("--processes"),
                type=int,
                default=-1,
                help="Number of processes",
            ),
            click.option(
                _p("--ppd"), type=int, default=1, help="Processes per GPU device"
            ),
        ]

    @classmethod
    def parse_multiple(
        cls,
        ctx: click.Context,
        prefix: str = "",
    ) -> Sequence[GPUOptions]:
        _n = functools.partial(cls._prefix_name, prefix=prefix)

        while ctx and _n("gpu") not in ctx.params:
            ctx = ctx.parent
        if not ctx:
            raise ValueError(f"GPU options not found in context ({prefix=})")

        return (
            GPUOptions(
                gpu_enabled=ctx.params[_n("gpu")],
                gpu_devices=ctx.params[_n("devices")],
                num_processes=ctx.params[_n("processes")],
                ppd=ctx.params[_n("ppd")],
            ),
        )


class OutputOptions(_CLIOptions):
    """
    Encapsulates output options for an experiment.
    """

    def __init__(self, out_dir: Path, out_tag: str):
        self.out_dir = out_dir
        self.out_tag = out_tag

    @classmethod
    def _cli_options(cls, prefix: str = "") -> Sequence[click.Option]:
        _p = functools.partial(cls._prefix_option, prefix=prefix)
        return [  # type:ignore
            click.option(
                _p("-o"),
                _p("--out-dir"),
                type=Path,
                default=EXPERIMENTS_OUT_DIR,
                help="Output directory",
            ),
            click.option(_p("--out-tag"), type=str, default="", help="Output tag"),
        ]

    @classmethod
    def parse_multiple(
        cls,
        ctx: click.Context,
        prefix: str = "",
    ) -> Sequence[OutputOptions]:
        _n = functools.partial(cls._prefix_name, prefix=prefix)

        while ctx and _n("out_dir") not in ctx.params:
            ctx = ctx.parent
        if not ctx:
            raise ValueError(f"Output options not found in context {prefix=}")

        return (
            OutputOptions(
                out_dir=ctx.params[_n("out_dir")],
                out_tag=ctx.params[_n("out_tag")],
            ),
        )


class LoggingOptions(_CLIOptions):
    """
    Encapsulates output options for an experiment.
    """

    def __init__(self, debug: bool):
        self.debug = debug

    @classmethod
    def _cli_options(cls, prefix: str = "") -> Sequence[click.Option]:
        _p = functools.partial(cls._prefix_option, prefix=prefix)
        return [  # type:ignore
            click.option(
                _p("--debug/--no-debug"), default=False, help="Use DEBUG logging"
            ),
        ]

    @classmethod
    def parse_multiple(
        cls,
        ctx: click.Context,
        prefix: str = "",
    ) -> Sequence[LoggingOptions]:
        _n = functools.partial(cls._prefix_name, prefix=prefix)

        while ctx and _n("debug") not in ctx.params:
            ctx = ctx.parent
        if not ctx:
            raise ValueError(f"Logging options not found in context {prefix=}")

        return (LoggingOptions(debug=ctx.params[_n("debug")]),)


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
    def _cli_options(cls, prefix: str = "") -> Sequence[click.Option]:
        _p = functools.partial(cls._prefix_option, prefix=prefix)
        _n = functools.partial(cls._prefix_name, prefix=prefix)

        return [  # type:ignore
            click.option(
                _p("-N"),
                _n("ns"),
                type=int,
                multiple=True,
                default=[1000],
                help="Samples",
            ),
            click.option(
                _p("-T"),
                _n("ts"),
                type=int,
                multiple=True,
                default=[20],
                help="Quantile levels",
            ),
            click.option(
                _p("-d"),
                _n("ds"),
                type=int,
                multiple=True,
                default=[2],
                help="Y dimension",
            ),
            click.option(
                _p("-k"),
                _n("ks"),
                type=int,
                multiple=True,
                default=[3],
                help="X dimension",
            ),
            click.option(
                _p("-E"),
                _n("epsilons"),
                type=float,
                multiple=True,
                default=[1e-6],
                help="epsilon",
            ),
            click.option(
                _p("--bs-y"),
                _n("bys"),
                type=int,
                multiple=True,
                default=[-1],
                help="Batch size of Y (samples), -1 disables batching",
            ),
            click.option(
                _p("--bs-u"),
                _n("bus"),
                type=int,
                multiple=True,
                default=[-1],
                help="Batch size of U (quantile levels), -1 disables batching",
            ),
            click.option(_p("--epochs"), type=int, default=1000, help="epochs"),
            click.option(_p("--lr"), type=float, default=0.5, help="Learning rate"),
            click.option(
                _p("--lr-max-steps"), type=int, default=10, help="LR sched. steps"
            ),
            click.option(
                _p("--lr-factor"), type=float, default=0.9, help="LR sched. factor"
            ),
            click.option(
                _p("--lr-patience"), type=int, default=500, help="LR sched. patience"
            ),
            click.option(
                _p("--lr-threshold"),
                type=float,
                default=0.01 * 5,
                help="LR sched. " "thresh.",
            ),
            click.option(
                _p("--mlp/--no-mlp"), type=bool, default=False, help="NL-VQR with MLP"
            ),
            click.option(
                _p("--mlp-layers"),
                type=str,
                default="32,32",
                help="comma-separated ints",
            ),
            click.option(
                _p("--mlp-skip/--no-mlp-skip"),
                type=bool,
                default=False,
                help="MLP residual",
            ),
            click.option(
                _p("--mlp-activation"), type=str, default="relu", help="MLP activation"
            ),
            click.option(
                _p("--mlp-batchnorm/--no-mlp-batchnorm"), type=bool, default=False
            ),
            click.option(_p("--mlp-dropout"), type=float, default=0.0),
        ]

    @classmethod
    def parse_multiple(
        cls,
        ctx: click.Context,
        prefix: str = "",
    ) -> Sequence[VQROptions]:
        _n = functools.partial(cls._prefix_name, prefix=prefix)

        while ctx and _n("ns") not in ctx.params:
            ctx = ctx.parent
        if not ctx:
            raise ValueError(f"VQR options not found in context {prefix=}")

        gpu_options = GPUOptions.parse(ctx)
        p = ctx.params

        # parse mlp options
        mlp_opts = dict(
            hidden_layers=p[_n("mlp_layers")],
            activation=p[_n("mlp_activation")],
            skip=p[_n("mlp_skip")],
            batchnorm=p[_n("mlp_batchnorm")],
            dropout=p[_n("mlp_dropout")],
        )
        # Filter out defaults and remove everything if mlp==False.
        mlp_opts = {k: v for k, v in mlp_opts.items() if v is not None and p[_n("mlp")]}

        solver_name = (
            MLPRegularizedDualVQRSolver.solver_name()
            if p[_n("mlp")]
            else RegularizedDualVQRSolver.solver_name()
        )

        vqr_options = tuple(
            VQROptions(
                N=N_,
                T=T_,
                d=d_,
                k=k_,
                solver_name=solver_name,
                solver_opts=dict(
                    verbose=False,
                    num_epochs=p[_n("epochs")],
                    epsilon=eps_,
                    lr=p[_n("lr")],
                    lr_max_steps=p[_n("lr_max_steps")],
                    lr_factor=p[_n("lr_factor")],
                    lr_patience=p[_n("lr_patience")],
                    lr_threshold=p[_n("lr_threshold")],
                    batchsize_y=bs_y_ if bs_y_ > 0 else None,
                    batchsize_u=bs_u_ if bs_u_ > 0 else None,
                    gpu=gpu_options.gpu_enabled,
                    **mlp_opts,
                ),
            )
            for (N_, T_, d_, k_, eps_, bs_y_, bs_u_) in product(
                p[_n("ns")],
                p[_n("ts")],
                p[_n("ds")],
                p[_n("ks")],
                p[_n("epsilons")],
                p[_n("bys")],
                p[_n("bus")],
            )
        )

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
    :param write_csv: Whether to write the results to a CSV.
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
