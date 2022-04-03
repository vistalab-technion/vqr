import logging

import click

from experiments.logging import setup_logging

_LOG = logging.getLogger()


if __name__ == "__main__":
    setup_logging()
    _LOG.info("hello world1")
