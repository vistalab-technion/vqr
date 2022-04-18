import logging
from typing import Optional, Sequence

import ot
import numpy as np
import torch
from numpy import array
from torch import Tensor, ones_like, from_numpy
from geomloss import SamplesLoss
from pykeops.torch import Pm, Vi, Vj, LazyTensor

_LOG = logging.getLogger(__name__)

# TODO: add tests for metrics


def w2_keops(
    Y_gt,
    Y_est,
    dtype=torch.float32,
    gpu_device: Optional[int] = None,
):
    device = torch.device("cpu" if gpu_device is None else f"cuda:{gpu_device}")
    if isinstance(Y_gt, Tensor):
        Y_gt = Y_gt.clone().detach().numpy()
    if isinstance(Y_est, Tensor):
        Y_est = Y_est.clone().detach().numpy()
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


def get_grid_points(Y: Tensor, grid_resolution: int) -> Tensor:
    """
    Get grid points in R^d. The dimension is determined by the dimension of Y.
    Grid borders are determined by the max and min values of Y along each dimension.

    :param Y: Samples in R^d.
    :param grid_resolution: Number of bins to tessellate each dimension.
    :return: Co-ordinates of points in each dimension. Shape=[grid_resolution^d, d]
    """
    Y_max = Y.max(dim=0).values  # torch.tensor([3.0, 2.3]): hack for cond banana plots
    Y_min = Y.min(dim=0).values  # torch.tensor([-3.0, 0.0]): hack for cond banana plots
    axes = [
        torch.linspace(y_i_min.item(), y_i_max.item(), grid_resolution)
        for y_i_min, y_i_max in zip(Y_min, Y_max)
    ]
    grids = torch.meshgrid(axes, indexing="ij")
    grid_points = torch.stack([g.ravel() for g in grids], dim=1)
    return grid_points


def kde_l1(
    Y_gt: Tensor, Y_est: Tensor, grid_resolution: int, device: str, sigma: float
) -> float:
    """
    Measures L1 distance between the KDEs of two sample distributions.

    :param Y_gt: Samples from the ground-truth distribution
    :param Y_est: Samples from the estimated distribution
    :param grid_resolution: Number of grid points per dimension
    :param device: Device on which to execute the metric
    :param sigma: the bandwidth for the kernel in KDE
    :return: KDE L1 distance per grid point.
    """
    assert Y_gt.shape[1] == Y_est.shape[1]  # same dimensions
    grid_shape = [grid_resolution for _ in range(Y_gt.shape[1])]
    grid_points = get_grid_points(Y_gt, grid_resolution)
    kde_gt = kde_keops(Y_gt, grid_points, grid_shape, device, sigma)
    kde_est = kde_keops(Y_est, grid_points, grid_shape, device, sigma)
    return torch.mean(torch.abs(kde_gt - kde_est)).item()


def kde(Y: Tensor, grid_resolution: int, device: str, sigma: float):
    """
    Estimates KDE of the empirical distribution Y.

    :param Y: Samples (empirical distribution).
    :param grid_resolution: Number of grid points per dimension
    :param device: Device on which to execute the metric
    :param sigma: the bandwidth for the kernel in KDE
    :return: KDE L1 distance per grid point.
    """
    return kde_keops(
        Y,
        get_grid_points(Y, grid_resolution),
        [grid_resolution for _ in range(Y.shape[1])],
        device,
        sigma,
    )


def kde_keops(
    Y: Tensor,
    grid: Tensor,
    grid_shape: Sequence[int],
    device: str,
    sigma: float,
):
    """
    Performs kernel density estimation of Y on the grid.

    :param Y: Samples whose KDE needs to be estimated.
    :param grid: Grid points on which the density needs to be estimated.
    :param grid_shape: Shape of the grid.
    :param device: Device to execute on. If gpu needs to be used provide "cuda:{
    device_num}" else, cpu()
    :param sigma: Bandwidth of KDE.
    :return: KDE of the data over the grid.
    """
    assert grid.shape[0] == np.prod(grid_shape)
    assert grid.shape[1] == len(grid_shape)
    d = len(grid_shape)

    dtype = Y.dtype
    Y = Y.contiguous().to(device)
    grid = grid.contiguous().type(dtype).to(device)

    # The distance scales proportional to d, so we need to scale sigma by sqrt(d) to
    # make KDE values computed on different dimensional comparable.
    sigma = Tensor([sigma * np.sqrt(d)]).type(dtype).to(device).contiguous()
    gamma = 1.0 / sigma**2
    b = ones_like(Y[:, 0]).type(dtype).to(device)
    b = b.view(-1, 1).contiguous()
    heatmap = (-Vi(grid).weightedsqdist(Vj(Y), Pm(gamma))).exp() @ b
    heatmap = heatmap.view(*grid_shape).cpu()
    heatmap /= heatmap.sum()
    return heatmap
