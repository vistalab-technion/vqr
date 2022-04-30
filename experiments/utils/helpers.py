import pickle
import logging
from typing import Any, NamedTuple
from hashlib import blake2b

_LOG = logging.getLogger(__name__)


def stable_hash(obj: Any, hash_len: int = 8) -> str:
    """
    :param obj: An object to hash.
    :param hash_len: Desired length of output string.
    :return: A unique and repeatable hash of the given object.
    """
    return blake2b(pickle.dumps(obj), digest_size=hash_len // 2).hexdigest()


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
