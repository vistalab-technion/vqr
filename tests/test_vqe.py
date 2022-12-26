import numpy as np
import pytest
from matplotlib import pyplot as plt
from sklearn.exceptions import NotFittedError

from vqr import DiscreteVQF, VectorQuantileEstimator
from tests.conftest import _test_monotonicity, monotonicity_offending_projections
from vqr.solvers.cvx import CVXVQRSolver
from vqr.solvers.pot import POTVQESolver
from experiments.datasets.mvn import IndependentDataProvider


class TestVectorQuantileEstimator(object):
    @pytest.fixture(
        scope="class",
        params=[
            (
                "regularized_dual",
                {"verbose": True, "lr": 0.1, "epsilon": 1e-6},
            ),
            ("cvx_primal", {"verbose": True}),
            ("vqe_pot", {}),
        ],
        ids=["rvqr_linear", "cvx_primal", "vqe_pot"],
    )
    def solver(self, request):
        return request.param

    @pytest.fixture(
        scope="class",
        params=[
            {"d": 1, "N": 200, "T": 20},
            {"d": 2, "N": 200, "T": 20},
            {"d": 3, "N": 200, "T": 10},
        ],
        ids=[
            "d=1",
            "d=2",
            "d=3",
        ],
    )
    def problem_size(self, request):
        param = request.param
        d, N, T = param["d"], param["N"], param["T"]
        return (d, N, T)

    @pytest.fixture(scope="class")
    def vqe_fitted(self, solver, problem_size):
        d, N, T = problem_size
        solver_name, solver_opts = solver
        solver_opts["T"] = T

        X, Y = IndependentDataProvider(d=d, k=1).sample(n=N)
        vqe = VectorQuantileEstimator(
            solver=solver_name,
            solver_opts=solver_opts,
        )
        vqe.fit(Y)

        return Y, vqe

    def test_shapes(self, vqe_fitted, problem_size):
        d, N, T = problem_size
        Y, vqe = vqe_fitted

        assert vqe.dim_y == d
        assert len(vqe.quantile_grid) == d
        assert all(q.shape == (T,) * d for q in vqe.quantile_grid)

    def test_vector_quantiles(self, vqe_fitted, problem_size):
        d, N, T = problem_size
        Y, vqe = vqe_fitted

        vqf: DiscreteVQF = vqe.vector_quantiles()
        assert len(vqf) == d
        assert all(q_surface.shape == (T,) * d for q_surface in vqf)
        assert vqf.values.shape == (d, *[T] * d)
        assert vqf.quantile_grid.shape == (d, *[T] * d)

    def test_sample(self, vqe_fitted, problem_size, test_out_dir):
        d, N, T = problem_size
        Y, vqe = vqe_fitted

        n = 1000
        Y_samp = vqe.sample(n)
        assert Y_samp.shape == (n, d)

        if d == 2:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.scatter(Y[:, 0], Y[:, 1], c="k", label="Y (GT)")
            ax.scatter(Y_samp[:, 0], Y_samp[:, 1], c="C0", label="Y")
            ax.legend()
            fig.savefig(test_out_dir.joinpath("vqe_sample.pdf"), bbox_inches="tight")

    @pytest.mark.flaky(reruns=1)
    def test_coverage(self, vqe_fitted, problem_size, test_out_dir):
        d, N, T = problem_size
        Y, vqe = vqe_fitted

        cov = vqe.coverage(Y, alpha=0.05)
        print(f"{cov=}")
        assert cov > (0.7 if d < 3 else 0.25)

    def test_not_fitted(self, solver):
        solver_name, solver_opts = solver
        vq = VectorQuantileEstimator(solver=solver_name, solver_opts=solver_opts)
        with pytest.raises(NotFittedError):
            _ = vq.quantile_grid
        with pytest.raises(NotFittedError):
            _ = vq.vector_quantiles()

    def test_monotonicity(self, vqe_fitted):
        Y, vqe = vqe_fitted

        _test_monotonicity(
            Us=vqe.quantile_grid,
            Qs=list(vqe.vector_quantiles(refine=True)),
        )

    def test_vmr_vqe(self, solver):
        solver_name, solver_opts = solver

        T = 15
        N = 1000
        d = 2
        solver_opts["T"] = T

        _, Y = IndependentDataProvider(d=d, k=1).sample(n=N)
        vqe = VectorQuantileEstimator(solver=solver_name, solver_opts=solver_opts)
        vqe.fit(Y)

        off_projs, _ = monotonicity_offending_projections(
            Us=vqe.quantile_grid,
            Qs=list(vqe.vector_quantiles(refine=False)),
            projection_tolerance=0.0,
        )

        off_projs_refined, _ = monotonicity_offending_projections(
            Us=vqe.quantile_grid,
            Qs=list(vqe.vector_quantiles(refine=True)),
            projection_tolerance=0.0,
        )

        print(len(off_projs), len(off_projs_refined))
        assert len(off_projs_refined) <= len(off_projs)


class TestPOTVQE:
    # Important that N is odd
    @pytest.mark.parametrize("N", [101, 201, 301, 1001])
    @pytest.mark.parametrize("T", [10, 20])
    def test_pot_quantile_vs_sorted_Y_1d(self, N, T):
        Y = np.random.randn(N, 1)
        est_quantiles = POTVQESolver(T=T).solve_vqe(Y=Y).values[0]
        sorted_Y = np.sort(Y.squeeze())[:: int(N / T)]
        assert np.allclose(est_quantiles[1:], sorted_Y[1:-1])

    @pytest.mark.parametrize("N", [101, 201, 301])
    @pytest.mark.parametrize("T", [5, 10])
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_pot_quantile_vs_sorted_Y_2d(self, N, T, d):
        Y = np.random.randn(N, d)
        est_quantiles = POTVQESolver(T=T).solve_vqe(Y=Y).values
        est_quantiles_cvx = CVXVQRSolver(T=T).solve_vqe(Y=Y).values
        for Q, Q_cvx in zip(est_quantiles, est_quantiles_cvx):
            assert Q.shape == (T,) * d
            assert Q_cvx.shape == (T,) * d
            assert np.allclose(Q, Q_cvx)

    @pytest.mark.parametrize("T", [10, 20, 50, 100])
    @pytest.mark.parametrize("N", [101, 201, 301, 1001])
    def test_vqe_vs_cvx(self, T, N):
        Y = np.random.randn(N, 1)
        est_quantiles = POTVQESolver(T=T).solve_vqe(Y=Y).values[0]
        est_quantiles_cvx = CVXVQRSolver(T=T).solve_vqe(Y=Y).values[0]
        assert np.allclose(est_quantiles, est_quantiles_cvx)

    @pytest.mark.parametrize("d", [1, 2, 3, 4])
    def test_multiple_d(self, d):
        T = 10
        Y = np.random.randn(1000, d)
        quantiles = POTVQESolver(T=T).solve_vqe(Y=Y).values
        assert len(quantiles) == d

    @pytest.mark.parametrize("N", [1001, 2001, 3001])
    @pytest.mark.parametrize("T", [5, 10, 20, 25, 30])
    def test_comonotonicity(self, N, T):
        Y = np.random.randn(N, 2)
        vqf = POTVQESolver(T=T).solve_vqe(Y=Y)
        _test_monotonicity(Us=vqf.quantile_grid, Qs=vqf.values)
