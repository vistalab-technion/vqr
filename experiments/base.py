import functools
from typing import Any, Dict, Callable, Optional, Sequence
from pathlib import Path
from itertools import product
from dataclasses import dataclass

import click

from experiments import EXPERIMENTS_OUT_DIR
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)


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


def click_common_gpu_options(exp_fn: Callable) -> Callable:
    options = [
        click.option("-g", "--gpu/--no-gpu", default=False, help="Enable GPU support"),
        click.option(
            "--devices",
            default=None,
            type=str,
            help="GPU devices (comma separated)",
        ),
        click.option(
            "-p", "--processes", type=int, default=-1, help="Number of processes"
        ),
        click.option("--ppd", type=int, default=1, help="Processes per GPU device"),
    ]
    return _click_decorator(
        exp_fn,
        options=options,
    )


def click_common_logging_options(exp_fn: Callable) -> Callable:
    options = [
        click.option("--debug/--no-debug", default=False, help="Use DEBUG logging"),
    ]
    return _click_decorator(
        exp_fn,
        options=options,
    )


def click_common_output_options(exp_fn: Callable) -> Callable:
    options = [
        click.option(
            "-o",
            "--out-dir",
            type=Path,
            default=EXPERIMENTS_OUT_DIR,
            help="Output directory",
        ),
        click.option("--out-tag", type=str, default="", help="Output tag"),
    ]
    return _click_decorator(
        exp_fn,
        options=options,
    )


def click_common_vqr_solver_options(exp_fn: Callable) -> Callable:
    """
    A decorator that adds common VQR solver CLI options for experiments.

    :param exp_fn: The decorated experiment function.
    :return: A new experiment function with the CLI options added.
    """
    options = [
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
            "-E", "epsilons", type=float, multiple=True, default=[1e-6], help="epsilon"
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
        click.option("--lr-max-steps", type=int, default=10, help="LR sched. steps"),
        click.option("--lr-factor", type=float, default=0.9, help="LR sched. factor"),
        click.option("--lr-patience", type=int, default=500, help="LR sched. patience"),
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
        click.option("--mlp-batchnorm/--no-mlp-batchnorm", type=bool, default=False),
        click.option("--mlp-dropout", type=float, default=0.0),
    ]
    return _click_decorator(exp_fn, options)


@dataclass
class GPUOptions:
    """
    Encapsulates GPU options for an experiment.
    """

    gpu_enabled: bool
    gpu_devices: Optional[str]
    num_processes: int
    ppd: int


@dataclass
class OutputOptions:
    """
    Encapsulates output options for an experiment.
    """

    out_dir: Path
    out_tag: str


@dataclass
class VQROptions:
    """
    Encapsulates VQR options for an experiment.
    """

    N: int
    T: int
    d: int
    k: int
    solver_name: str
    solver_opts: Dict[str, Any]

    def to_dict(self):
        return self.__dict__.copy()


def parse_output_options(ctx: click.Context) -> OutputOptions:
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


def parse_gpu_options(ctx: click.Context) -> GPUOptions:
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


def parse_vqr_options(
    ctx: click.Context,
) -> Sequence[VQROptions]:

    while ctx and "ns" not in ctx.params:
        ctx = ctx.parent
    if not ctx:
        raise ValueError("VQR options not found in context")

    gpu_options = parse_gpu_options(ctx)
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
