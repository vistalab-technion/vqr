import logging
from typing import Optional

import ot
import torch
from geomloss import SamplesLoss

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
