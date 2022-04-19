import numpy as np
import pytest

from experiments.utils.metrics import kde_l1, w2_pot, w2_keops


class TestMetrics:
    def _generate_data(self, N: int, d: int):
        dtype = np.float32

        # Y1 & Y2 are from the same dist, Y3 is not
        Y1 = np.random.randn(N, d).astype(dtype)
        Y2 = np.random.randn(N, d).astype(dtype)
        Y3 = np.random.randn(N, d).astype(dtype) + 1.0 + (d - 1) * 0.1

        return Y1, Y2, Y3

    @pytest.mark.parametrize("resolution", [10, 12])
    @pytest.mark.parametrize("N", [1000, 2000, 3000])
    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    def test_kde_l1(self, N, d, resolution):
        Y1, Y2, Y3 = self._generate_data(N, d)
        kde_same_dist = kde_l1(
            Y1, Y2, device="cpu", grid_resolution=resolution, sigma=0.05
        )
        kde_diff_dist = kde_l1(
            Y1, Y3, device="cpu", grid_resolution=resolution, sigma=0.05
        )
        print(f"{N=}, {d=}, {resolution=}, {kde_same_dist=:.3e}, {kde_diff_dist=:.3e}")
        assert kde_same_dist < kde_diff_dist

    @pytest.mark.parametrize("N", [1000, 2000, 3000])
    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    def test_w2_keops(self, N, d):
        Y1, Y2, Y3 = self._generate_data(N, d)
        w2_same_dist = w2_keops(Y1, Y2, device="cpu")
        w2_diff_dist = w2_keops(Y1, Y3, device="cpu")
        print(f"{N=}, {d=}, {w2_same_dist=:.3e}, {w2_diff_dist=:.3e}")
        assert w2_same_dist < w2_diff_dist

    @pytest.mark.parametrize("N", [1000, 2000, 3000])
    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    def test_w2_pot(self, N, d):

        Y1, Y2, Y3 = self._generate_data(N, d)
        w2_same_dist = w2_pot(Y1, Y2)
        w2_diff_dist = w2_pot(Y1, Y3)
        print(f"{N=}, {d=}, {w2_same_dist=:.3e}, {w2_diff_dist=:.3e}")
        assert w2_same_dist < w2_diff_dist
