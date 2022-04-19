import logging

import click

from experiments.base import GPUOptions, OutputOptions, LoggingOptions
from experiments.scale import scale_exp
from experiments.logging import setup_logging

_LOG = logging.getLogger(__name__)


@click.group(context_settings=dict(show_default=True))
@click.pass_context
@LoggingOptions.cli
@GPUOptions.cli
@OutputOptions.cli
def main(ctx: click.Context, **kw):
    logging_opts = LoggingOptions.parse(ctx)
    setup_logging(level=logging.DEBUG if logging_opts.debug else logging.INFO)


if __name__ == "__main__":
    main.add_command(scale_exp)
    main()
