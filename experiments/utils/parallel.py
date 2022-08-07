import os
import queue
import logging
import multiprocessing as mp
import concurrent.futures
from time import time
from queue import Queue
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
    cast,
)
from concurrent.futures import Future, ProcessPoolExecutor

import torch
import pandas as pd

from experiments.log import setup_logging
from experiments.utils.helpers import sec_to_time

_LOG = logging.getLogger(__name__)


_T = TypeVar("_T")


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
    exp_configs: Dict[str, dict],
    max_workers: Optional[int] = None,
    gpu_enabled: bool = False,
    gpu_devices: Optional[Union[Sequence[int], str]] = None,
    workers_per_device: int = 1,
) -> Sequence[Dict[str, Any]]:
    """
    Runs multiple experiment configurations with parallel workers.
    :param exp_name: Name of experiment (for logging only).
    :param exp_fn: Callable that runs a single experiment configuration. It must
    return a dict with string keys.
    :param exp_configs: Mapping from a string, representing a single experiments
    id/name, to a dict of kwargs containing the configuration of that experiment.
    Each such config dict will be passed as **kwargs to exp_fn.
    :param max_workers: Maximal number of parallel worker processes to use. None
    means use number of physical cores.
    :param gpu_enabled: Whether to enable GPU support.
    :param gpu_devices: Either a list of device IDs (ints) or a comma-separated string.
    :param workers_per_device: Number of worker processes that may share a GPU.
    :return: A sequence with the results returned by each invocation of the exp_fn.
    """
    n_exps = len(exp_configs)

    if not max_workers or max_workers < 0:
        max_workers = os.cpu_count()

    if gpu_devices is None:
        # Allow all devices
        gpu_devices = tuple(range(torch.cuda.device_count()))
    elif isinstance(gpu_devices, str):
        # parse comma separated
        gpu_devices = tuple(int(d) for d in gpu_devices.split(","))

    if gpu_enabled:
        if not torch.cuda.is_available():
            _LOG.warning(f"Requested GPU, but CUDA is not available")
            gpu_enabled = False
        else:
            assert len(gpu_devices) > 0
            _LOG.info(f"GPU enabled, {gpu_devices=}")

            # Limit number of workers based on effective number of devices
            max_workers = min(max_workers, len(gpu_devices) * workers_per_device)

    _LOG.info(
        f"Starting experiment {exp_name} with {n_exps} configurations and "
        f"{max_workers} workers..."
    )
    start_time = time()

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
        initargs=cuda_worker_init_args(
            enable_cuda=gpu_enabled,
            cuda_devices=gpu_devices,
            workers_per_device=workers_per_device,
        ),
        initializer=cuda_worker_init_fn,
    ) as executor:

        futures = {
            exp_name: executor.submit(_exp_fn_wrapper, exp_fn, i, **exp_config)
            for i, (exp_name, exp_config) in enumerate(exp_configs.items())
        }

        results = []
        for i, (exp_name, exp_result) in enumerate(
            yield_future_results(
                futures,
                wait_time_sec=1.0,
                re_raise=False,
            )
        ):
            _LOG.info(
                f"Collected result {exp_name} {i+1}/{n_exps} "
                f"({100*(i+1)/n_exps:.0f}%)"
            )
            results.append(exp_result)

    elapsed_time = sec_to_time(time() - start_time)
    _LOG.info(f"Completed {len(results)}/{n_exps}, elapsed={elapsed_time}")

    return tuple(results)


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


def cuda_worker_init_args(
    enable_cuda: bool = None,
    cuda_devices: Optional[Sequence[int]] = None,
    workers_per_device: int = 1,
) -> Tuple[Any, ...]:
    """
    Creates the init args which should be passed to :obj:`cuda_worker_init_fn` when it
    runs on a worker process.

    :param enable_cuda: Whether to enable CUDA on the workers. If None, the global
        CudaState will be interrogated for this value.
    :param cuda_devices: List of cuda device numbers to use.
    :param workers_per_device: Number of workers which are allowed to share a device.
    :return: A tuple containing the init args.
    """

    gpu_queue: Optional[Queue] = None

    if enable_cuda:

        if not cuda_devices:
            n_devices = torch.cuda.device_count() or 0

            # If we're using GPUs, create a queue of GPU device ids.
            # Each device id is repeated n_workers_per_device times.
            device_list = [*range(n_devices)] * workers_per_device
        else:
            device_list = [*cuda_devices] * workers_per_device

        if device_list:
            manager = mp.Manager()
            gpu_queue = manager.Queue()

            for device_id in device_list:
                gpu_queue.put(device_id)

    return (gpu_queue,)


def cuda_worker_init_fn(*args: Any) -> None:
    """
    An initializer function to be run on workers. Sets the GPU state for a worker
    process to use a worker-specific CUDA device.

    :param args: Arguments tuple which must be created by calling
        :obj:`cuda_worker_init_args`.
    """

    setup_logging()

    gpu_device_queue: Optional[Queue]
    gpu_device_queue, *_ = args

    pid = os.getpid()

    if gpu_device_queue is None:
        return

    try:
        # pull a unique device id from the queue - guarantees that different workers
        # will get different device ids (as long a unique ids were placed into the
        # queue).
        device = gpu_device_queue.get(block=False)
        torch.cuda.set_device(device)
        _LOG.info(f"Set GPU {device=} for worker process {pid}")
    except queue.Empty:
        msg = f"Can't obtain a GPU device number for worker process {pid}."
        raise ValueError(msg)
