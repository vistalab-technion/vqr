import logging
import multiprocessing as mp
import concurrent.futures
from time import time, strftime
from socket import gethostname
from typing import (
    Any,
    Dict,
    Tuple,
    Union,
    TypeVar,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    NamedTuple,
    cast,
)
from concurrent.futures import Future, TimeoutError, CancelledError, ProcessPoolExecutor

import ot
import torch
import pandas as pd
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

_T = TypeVar("_T")

_LOG = logging.getLogger(__name__)


def w2_keops(Y_gt, Y_est, dtype=torch.float32, gpu_device: Optional[int] = None):
    device = torch.device("cpu" if gpu_device is None else f"cuda:{gpu_device}")
    return SamplesLoss(loss="sinkhorn", p=2, blur=0.05)(
        torch.tensor(Y_gt, dtype=dtype, device=device),
        torch.tensor(Y_est, dtype=dtype, device=device),
    )


def w2_pot(Y_gt, Y_est, num_iter_max=200_000, num_threads=32):
    return ot.emd2(
        a=[],
        b=[],
        M=ot.dist(Y_gt, Y_est),
        numItermax=num_iter_max,
        numThreads=num_threads,
    )


def _exp_fn_wrapper(_exp_fn, _exp_idx: int, **kwargs):
    exp_result = _exp_fn(**kwargs)
    if not exp_result or not isinstance(exp_result, dict):
        raise ValueError(
            f"Got no return value from experiment index {_exp_idx}, config={kwargs}"
        )
    return exp_result


def run_parallel_exp(
    exp_name: str,
    exp_fn: Callable[[Any], Dict[str, Any]],
    exp_configs: Iterable[dict],
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Runs multiple experiment configurations with parallel workers.
    :param exp_name: Name of experiment (for logging only).
    :param exp_fn: Callable that runs a single experiment configuration. It must
    return a dict with string keys.
    :param exp_configs: Iterable which iterates over experiment configurations
    represented as dicts. Each such config dict will be passed as **kwargs to exp_fn.
    :param max_workers: Maximal number of parallel worker processes to use. None
    means use number of physical cores.
    :return: A Dataframe with the results. Rows correspond to experiment
    configurations and columns correspond to keys in the results dicts (can be
    nested, in which case the columns will be e.g. key1.key2.key3 etc.)
    """
    exp_configs = tuple(exp_configs)
    n_exps = len(exp_configs)

    _LOG.info(f"Starting experiment {exp_name} with {n_exps} configurations...")
    start_time = time()

    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp.get_context("spawn")
    ) as executor:

        futures = [
            executor.submit(_exp_fn_wrapper, exp_fn, i, **exp_config)
            for i, exp_config in enumerate(exp_configs)
        ]

        results = []
        for i, (_, result) in enumerate(
            yield_future_results(
                futures,
                wait_time_sec=1.0,
                re_raise=False,
            )
        ):
            _LOG.info(f"Collected result {i+1}/{n_exps} ({100*(i+1)/n_exps:.0f}%)")
            results.append(result)

    elapsed_time = sec_to_time(time() - start_time)
    _LOG.info(f"Completed {len(results)}/{n_exps}, elapsed={elapsed_time}")

    df = pd.json_normalize(results)
    return df


def yield_future_results(
    futures: Union[Dict[_T, Future], Sequence[Future]],
    wait_time_sec=0.1,
    max_retries=None,
    re_raise=True,
    raise_on_max_retries=True,
) -> Iterator[Tuple[_T, Any]]:
    """
    Waits for futures to be ready, and yields their results. This function waits for
    each result for a fixed time, and moves to the next result if it's not ready.
    Therefore, the order of yielded results is not guaranteed to be the same as the
    order of the Future objects.

    :param futures: Either a dict mapping from some name to an
        Future to wait for, or a list of Future (in which case a name
        will be generated for each one based on it's index).
    :param wait_time_sec: Time to wait for each Future before moving to
        the next one if the current one is not ready.
    :param max_retries: Maximal number of times to wait for the same
        Future, before giving up on it. None means never give up. If
        max_retries is exceeded, an error will be logged.
    :param re_raise: Whether to re-raise an exception thrown in on of the
        tasks and stop handling. If False, exception will be logged instead and
        handling will continue.
    :param raise_on_max_retries:  Whether to raise an exception or only log an error
        in case max_retries is reached for a specific result. The type of exception
        raised will be :class:`concurrent.futures.TimeoutError`.
    :return: A generator, where each element is a tuple. The first element
        in the tuple is the name of the result, and the second element is the
        actual result. In case the task raised an exception, the second element
        will be None.
    """

    futures_d: Dict[_T, Future]
    if isinstance(futures, (list, tuple)):
        # If it's a sequence, map from index to future
        futures_d = {cast(_T, i): r for i, r in enumerate(futures)}
    elif isinstance(futures, dict):
        futures_d = futures
    else:
        raise ValueError("Expected sequence or dict of futures")

    if len(futures_d) == 0:
        raise ValueError("No futures to wait for")

    # Map result to number of retries
    retry_counts = {res_name: 0 for res_name in futures_d.keys()}

    while len(retry_counts) > 0:
        retry_counts_next = {}

        for res_name, retry_count in retry_counts.items():
            future: Future = futures_d[res_name]
            result = None

            try:
                result = future.result(timeout=wait_time_sec)

            except concurrent.futures.CancelledError:
                _LOG.warning(f"Result {res_name} was cancelled")

            except concurrent.futures.TimeoutError:
                retries = retry_counts[res_name] + 1
                if max_retries is not None and retries > max_retries:
                    msg = f"Result {res_name} timed out with {max_retries=}"
                    if raise_on_max_retries:
                        raise concurrent.futures.TimeoutError(msg)
                    _LOG.error(msg)
                else:
                    retry_counts_next[res_name] = retries

                continue

            except Exception as e:
                if re_raise:
                    raise e
                _LOG.error(f"Result {res_name} raised {type(e)}: " f"{e}", exc_info=e)

            yield res_name, result

        retry_counts = retry_counts_next


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
