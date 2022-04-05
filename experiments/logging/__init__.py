import logging
import logging.config
from typing import Optional
from pathlib import Path

import yaml

LOGGING_CONFIG = Path(__file__).parent.joinpath("logging.yaml")


def setup_logging(level: Optional[int] = None):
    with open(LOGGING_CONFIG, "r") as f:
        logging_config = yaml.safe_load(f)

    if level is not None:
        for logger_name, logger_settings in logging_config["loggers"].items():
            logger_settings["level"] = logging.getLevelName(level)

    logging.config.dictConfig(logging_config)
