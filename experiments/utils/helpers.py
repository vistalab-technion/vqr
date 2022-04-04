import logging
from time import strftime
from typing import NamedTuple

from _socket import gethostname

_LOG = logging.getLogger(__name__)


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


def sec_to_time(sec: float):
    """
    Converts a time duration in seconds to days, hours, minutes, seconds and
    milliseconds.
    :param sec: Time in seconds.
    :return: An object with d, h, m, s and ms fields representing the above,
    respectively.
    """

    class __T(NamedTuple):
        d: int
        h: int
        m: int
        s: int
        ms: int

        def __repr__(self):
            return (
                f'{"" if self.d == 0 else f"{self.d}+"}'
                f"{self.h:02d}:{self.m:02d}:{self.s:02d}.{self.ms:03d}"
            )

    if sec < 0:
        raise ValueError("Invalid argument value")

    d = int(sec // (3600 * 24))
    h = int((sec // 3600) % 24)
    m = int((sec // 60) % 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return __T(d, h, m, s, ms)
