import logging
import logging.config
from pathlib import Path

import yaml

LOGGING_CONFIG = Path(__file__).parent.joinpath("logging.yaml")


def setup_logging():
    with open(LOGGING_CONFIG, "r") as f:
        logging_config = yaml.safe_load(f)

    logging.config.dictConfig(logging_config)
