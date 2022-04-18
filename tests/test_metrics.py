import torch
import pytest
from numpy import array
from torch import tensor

from experiments.utils.metrics import kde_l1


class TestKDE:
    @pytest.mark.parametrize("resolution", [10, 12])
    @pytest.mark.parametrize("N", [1000, 2000, 3000])
    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    def test_kde_l1(self, N, d, resolution):
        dtype = torch.float32
        Y1 = torch.rand(size=(N, d), dtype=dtype)
        Y2 = torch.rand(size=(N, d), dtype=dtype)
        kde_same_dist = kde_l1(
            Y1, Y2, device="cpu", grid_resolution=resolution, sigma=0.05
        )
        Y3 = torch.rand(size=(N, d), dtype=dtype) + tensor(array([1.0]), dtype=dtype)
        kde_diff_dist = kde_l1(
            Y1, Y3, device="cpu", grid_resolution=resolution, sigma=0.05
        )
        print(f"{N=}, {d=}, {resolution=}, {kde_same_dist:.3e}, {kde_diff_dist:.3e}")
        assert kde_same_dist < kde_diff_dist
