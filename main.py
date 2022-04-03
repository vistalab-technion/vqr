import logging

import click

from experiments.scale import run_scale_exps
from experiments.logging import setup_logging

_LOG = logging.getLogger(__name__)


@click.group(context_settings=dict(show_default=True))
@click.option("--debug/--no-debug", default=False, help="Use DEBUG logging")
def main(debug):
    setup_logging(level=logging.DEBUG if debug else logging.INFO)


if __name__ == "__main__":
    main.add_command(run_scale_exps)
    main()
