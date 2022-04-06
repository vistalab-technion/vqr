import logging
from typing import Optional, Sequence

import ot
import numpy as np
import torch
from torch import Tensor, ones_like, from_numpy
from geomloss import SamplesLoss
from pykeops.torch import Pm, Vi, Vj, LazyTensor

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


def kde2d_keops(
    x1: Tensor,
    xticks: Sequence[float],
    yticks: Sequence[float],
    device: str,
    sigma: float,
):
    dtype = x1.dtype
    x1 = x1.contiguous().to(device)
    X, Y = np.meshgrid(xticks, yticks)
    x2 = (
        from_numpy(np.vstack((X.ravel(), Y.ravel())).T)
        .contiguous()
        .type(dtype)
        .to(device)
    )
    sigma = Tensor([sigma]).type(dtype).to(device).contiguous()
    gamma = 1.0 / sigma**2
    b = ones_like(x1[:, 0]).type(dtype).to(device)
    b = b.view(-1, 1).contiguous()
    heatmap = (-Vi(x2).weightedsqdist(Vj(x1), Pm(gamma))).exp() @ b
    heatmap = heatmap.view(len(xticks), len(yticks)).cpu().numpy()
    heatmap /= heatmap.sum()
    return heatmap
