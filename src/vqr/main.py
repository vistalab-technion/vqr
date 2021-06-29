import sys
import logging
import argparse
import logging.config
from typing import Sequence
from importlib.metadata import PackageNotFoundError, version

import yaml

from vqr.cfg import LOGGING_CONFIG

try:
    __version__ = version("vqr")
except PackageNotFoundError:
    # package is not installed
    __version__ = "<unknown>"

_LOG = logging.getLogger(__name__)


def parse_cli(args: Sequence[str]):
    hf = argparse.ArgumentDefaultsHelpFormatter
    p = argparse.ArgumentParser(
        description="vqr command-line interface", formatter_class=hf
    )

    p.set_defaults(handler=None)

    # Top-level configuration parameters
    p.add_argument(
        "--version",
        "-v",
        action="version",
        version=__version__,
        help="Show version and die.",
    )

    return p.parse_args(args)


def setup_logging():
    with open(LOGGING_CONFIG, "r") as f:
        logging_config = yaml.safe_load(f)

    logging.config.dictConfig(logging_config)


def main():
    setup_logging()

    args = sys.argv[1:]
    parsed_args = parse_cli(args)

    # Convert to a dict
    parsed_args = vars(parsed_args)

    try:
        # Get the function to invoke
        handler_fn = parsed_args.pop("handler")

        # Invoke it with the remaining arguments
        if handler_fn:
            handler_fn(**parsed_args)
        else:
            _LOG.warning(f"No action, stopping.")
    except KeyboardInterrupt as e:
        _LOG.warning(f"Interrupted by user, stopping.")
    except Exception as e:
        _LOG.error(f"{e.__class__.__name__}: {e}", exc_info=e)


if __name__ == "__main__":
    main()
