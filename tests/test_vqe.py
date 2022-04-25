import numpy as np
import pytest

from tests.conftest import _test_monotonicity
from vqr.solvers.primal.cvx import CVXVQRSolver
from vqr.solvers.primal.pot import POTVQESolver


class TestPOTVQE:
    # Important that N is odd
    @pytest.mark.parametrize("N", [101, 201, 301, 1001])
    @pytest.mark.parametrize("T", [10, 20])
    def test_pot_quantile_vs_sorted_Y_1d(self, N, T):
        Y = np.random.randn(N, 1)
        est_quantiles = POTVQESolver().solve_vqr(T=T, Y=Y).vector_quantiles()[0][0]
        sorted_Y = np.sort(Y.squeeze())[:: int(N / T)]
        assert np.allclose(est_quantiles[1:], sorted_Y[1:-1])

    @pytest.mark.parametrize("N", [101, 201, 301, 501])
    @pytest.mark.parametrize("T", [10, 20])
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_pot_quantile_vs_sorted_Y_2d(self, N, T, d):
        Y = np.random.randn(N, d)
        est_quantiles = POTVQESolver().solve_vqr(T=T, Y=Y).vector_quantiles()[0]
        est_quantiles_cvx = CVXVQRSolver().solve_vqr(T=T, Y=Y).vector_quantiles()[0]
        for Q, Q_cvx in zip(est_quantiles, est_quantiles_cvx):
            assert Q.shape == (T,) * d
            assert Q_cvx.shape == (T,) * d
            assert np.allclose(Q, Q_cvx)

    @pytest.mark.parametrize("T", [10, 20, 50, 100])
    @pytest.mark.parametrize("N", [101, 201, 301, 1001])
    def test_vqe_vs_cvx(self, T, N):
        Y = np.random.randn(N, 1)
        est_quantiles = POTVQESolver().solve_vqr(T=T, Y=Y).vector_quantiles()[0][0]
        est_quantiles_cvx = CVXVQRSolver().solve_vqr(T=T, Y=Y).vector_quantiles()[0][0]
        assert np.allclose(est_quantiles, est_quantiles_cvx)

    def test_no_X(self):
        Y = np.random.randn(1000, 1)
        with pytest.raises(AssertionError, match="POTVQESolver can't work with*"):
            POTVQESolver().solve_vqr(T=50, Y=Y, X=np.random.randn(1000, 2))

    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    def test_multiple_d(self, d):
        T = 10
        Y = np.random.randn(1000, d)
        quantiles = POTVQESolver().solve_vqr(T=T, Y=Y, X=None).vector_quantiles()[0]
        assert len(quantiles) == d

    @pytest.mark.parametrize("N", [1001, 2001, 3001])
    @pytest.mark.parametrize("T", [5, 10, 20, 25, 30])
    def test_comonotonicity(self, N, T):
        Y = np.random.randn(N, 2)
        vqe_soln = POTVQESolver().solve_vqr(T=T, Y=Y, X=None)
        quantiles = vqe_soln.vector_quantiles()[0]
        quantile_grids = vqe_soln.quantile_grid
        _test_monotonicity(Us=quantile_grids, Qs=quantiles, T=T)
