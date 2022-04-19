import logging
from pathlib import Path

import click

from experiments.base import (
    click_common_gpu_options,
    click_common_output_options,
    click_common_logging_options,
)
from experiments.scale import scale_exp
from experiments.logging import setup_logging

_LOG = logging.getLogger(__name__)


@click.group(context_settings=dict(show_default=True))
@click.pass_context
@click_common_logging_options
@click_common_gpu_options
@click_common_output_options
def main(ctx: click.Context, debug: bool, **kw):
    setup_logging(level=logging.DEBUG if debug else logging.INFO)


if __name__ == "__main__":
    main.add_command(scale_exp)
    main()
