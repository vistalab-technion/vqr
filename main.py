import logging
from pathlib import Path
from functools import wraps, partial

import click

from experiments import EXPERIMENTS_OUT_DIR
from experiments.scale import XExperiment, scale_exp
from experiments.logging import setup_logging

_LOG = logging.getLogger(__name__)


@click.group(context_settings=dict(show_default=True))
@click.pass_context
@click.option("--debug/--no-debug", default=False, help="Use DEBUG logging")
@click.option("-g", "--gpu/--no-gpu", default=False, help="Enable GPU support")
@click.option("--devices", default=None, type=str, help="GPU devices (comma separated)")
@click.option("-p", "--processes", type=int, default=-1, help="Number of processes")
@click.option("--ppd", type=int, default=1, help="Processes per GPU device")
@click.option("-o", "--out-dir", type=Path, default=EXPERIMENTS_OUT_DIR, help="Out dir")
def main(
    ctx: click.Context,
    debug: bool,
    gpu: bool,
    devices: str,
    processes: int,
    ppd: int,
    out_dir: Path,
):
    setup_logging(level=logging.DEBUG if debug else logging.INFO)


if __name__ == "__main__":
    main.add_command(scale_exp)
    main()
