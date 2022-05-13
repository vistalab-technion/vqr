import random

import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError

from vqr import (
    VQRSolution,
    QuantileFunction,
    VectorQuantileEstimator,
    VectorQuantileRegressor,
)
from tests.conftest import _test_monotonicity, monotonicity_offending_projections
from experiments.data.mvn import LinearMVNDataProvider, IndependentDataProvider
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)


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
    def vqr_solver_opts(self, request):
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
    def vqe_fitted(self, request, vqr_solver_opts):
        params = request.param
        solver, solver_opts = vqr_solver_opts
        d, N, T = params["d"], params["N"], params["T"]
        X, Y = IndependentDataProvider(d=d, k=1).sample(n=N)
        vqe = VectorQuantileEstimator(
            n_levels=T,
            solver=solver,
            solver_opts=solver_opts,
        )
        vqe.fit(Y)
        return Y, vqe

    def test_shapes(self, vqe_fitted):
        Y, vqe = vqe_fitted
        N, d = Y.shape
        T = vqe.n_levels

        assert vqe.dim_y == d
        assert len(vqe.quantile_grid) == d
        assert all(q.shape == (T,) * d for q in vqe.quantile_grid)

    def test_vector_quantiles(self, vqe_fitted):
        Y, vqe = vqe_fitted
        N, d = Y.shape
        T = vqe.n_levels

        vqf: QuantileFunction = vqe.vector_quantiles()
        assert len(vqf) == d
        assert all(q_surface.shape == (T,) * d for q_surface in vqf)
        assert vqf.values.shape == (d, *[T] * d)

    def test_sample(self, vqe_fitted, test_out_dir):
        Y, vqe = vqe_fitted
        N, d = Y.shape[0], Y.shape[1]
        T = vqe.n_levels

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
    def test_coverage(self, vqe_fitted, test_out_dir):
        Y, vqe = vqe_fitted
        N, d = Y.shape
        T = vqe.n_levels

        cov = vqe.coverage(Y, alpha=0.05)
        print(f"{cov=}")
        assert cov > (0.7 if d < 3 else 0.25)

    def test_not_fitted(self, vqr_solver_opts):
        solver, solver_opts = vqr_solver_opts
        vq = VectorQuantileEstimator(
            n_levels=100, solver=solver, solver_opts=solver_opts
        )
        with pytest.raises(NotFittedError):
            _ = vq.quantile_grid
        with pytest.raises(NotFittedError):
            _ = vq.vector_quantiles()

    def test_monotonicity(self, vqe_fitted):
        Y, vqe = vqe_fitted
        N, d = Y.shape
        T = vqe.n_levels

        _test_monotonicity(
            Us=vqe.quantile_grid,
            Qs=list(vqe.vector_quantiles(refine=True)),
            T=vqe.n_levels,
        )

    def test_vmr_vqe(self, vqr_solver_opts):
        solver, solver_opts = vqr_solver_opts

        T = 15
        N = 1000
        d = 2

        _, Y = IndependentDataProvider(d=d, k=1).sample(n=N)
        vqe = VectorQuantileEstimator(
            n_levels=T, solver=solver, solver_opts=solver_opts
        )
        vqe.fit(Y)
        off_projs, _ = monotonicity_offending_projections(
            Us=vqe.quantile_grid,
            Qs=list(vqe.vector_quantiles(refine=False)),
            T=vqe.n_levels,
            projection_tolerance=0.0,
        )
        off_projs_refined, _ = monotonicity_offending_projections(
            Us=vqe.quantile_grid,
            Qs=list(vqe.vector_quantiles(refine=True)),
            T=vqe.n_levels,
            projection_tolerance=0.0,
        )
        print(len(off_projs), len(off_projs_refined))
        assert len(off_projs_refined) <= len(off_projs)


class TestVectorQuantileRegressor(object):
    @pytest.fixture(
        scope="class",
        params=[
            RegularizedDualVQRSolver(
                verbose=True,
                lr=0.5,
                epsilon=1e-2,
            ),
            RegularizedDualVQRSolver(
                verbose=True,
                lr=0.5,
                epsilon=1e-2,
                batchsize_y=1000,
                batchsize_u=100,
            ),
            MLPRegularizedDualVQRSolver(verbose=True, lr=0.5, epsilon=1e-2),
            MLPRegularizedDualVQRSolver(
                verbose=True,
                lr=0.5,
                epsilon=1e-2,
                hidden_layers=[2, 4],
                skip=False,  # No skip, so output will have different k
                num_epochs=1500,
            ),
        ],
        ids=[
            "rvqr_linear",
            "rvqr_linear_batches",
            "rvqr_mlp",
            "rvqr_mlp_change_k",
        ],
    )
    def vqr_solver(self, request):
        return request.param

    @pytest.fixture(
        scope="class",
        params=[
            {"d": 1, "k": 5, "N": 1000, "T": 20},
            {"d": 2, "k": 7, "N": 1000, "T": 20},
            {"d": 3, "k": 3, "N": 1000, "T": 10},
        ],
        ids=[
            "d=1,k=5",
            "d=2,k=7",
            "d=3,k=3",
        ],
    )
    def vqr_fitted(self, request, vqr_solver):
        params = request.param
        N, d, k, T = params["N"], params["d"], params["k"], params["T"]
        X, Y = LinearMVNDataProvider(d=d, k=k).sample(n=N)

        vqr = VectorQuantileRegressor(n_levels=T, solver=vqr_solver)
        vqr.fit(X, Y)

        return X, Y, vqr

    def test_shapes(self, vqr_fitted):
        X, Y, vqr = vqr_fitted
        N, d = Y.shape
        N, k = X.shape
        T = vqr.n_levels

        assert vqr.dim_y == d
        assert vqr.dim_x == k
        assert len(vqr.quantile_grid) == d
        assert all(q.shape == (T,) * d for q in vqr.quantile_grid)
        assert vqr.solution_metrics is not None

        for X_ in [X[[0], :], X[0:, :]]:  # Single and multiple X should be valid
            N_ = len(X_)

            Y_hat = vqr.predict(X_)
            assert Y_hat.shape == (N_, d, *[T] * d)

    def test_vector_quantiles(self, vqr_fitted):
        X, Y, vqr = vqr_fitted
        N, d = Y.shape
        N, k = X.shape
        T = vqr.n_levels
        U_grid = vqr.quantile_grid

        # QuantileFunction per X
        vqfs = vqr.vector_quantiles(X=X)
        assert len(vqfs) == N

        vqf: QuantileFunction
        for vqf in vqfs:

            # Iterating over QuantileFunction returns its surfaces
            assert all(vq_surface.shape == (T,) * d for vq_surface in vqf)

            # All values of the quantile function
            assert vqf.values.shape == (d, *[T] * d)

            for j in range(10):
                # Obtain a random quantile level
                u_idx = np.random.randint(low=0, high=T, size=(d,), dtype=int)

                # Obtain a vector-quantile at level u
                vq = vqf(u_idx)

                # d-dimensional vector quantile value
                assert vq.shape == (d,)
                assert np.all(vq == vqf.values[(slice(None), *u_idx)])

    def test_sample(self, vqr_fitted, test_out_dir):
        X, Y, vqr = vqr_fitted
        N, d = Y.shape
        N, k = X.shape
        T = vqr.n_levels

        n = 1000
        xs = []
        YXs = []
        for i in range(5):
            x = X[i]
            YX_samp = vqr.sample(n, x=x)
            assert YX_samp.shape == (n, d)
            xs.append(x)
            YXs.append(YX_samp)

        if d == 2:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.scatter(Y[:, 0], Y[:, 1], c="k", label="Y (GT)")
            for i, (x, YX_samp) in enumerate(zip(xs, YXs)):
                ax.scatter(
                    YX_samp[:, 0],
                    YX_samp[:, 1],
                    c=f"C{i+1}",
                    label=f"$Y|X=x_{i}$",
                    alpha=0.2,
                )
            ax.legend()
            fig.savefig(test_out_dir.joinpath("vqr_sample.pdf"), bbox_inches="tight")

    @pytest.mark.flaky(reruns=1)
    def test_coverage(self, vqr_fitted, test_out_dir):
        X, Y, vqr = vqr_fitted
        N, d = Y.shape
        N, k = X.shape
        T = vqr.n_levels

        cov = np.mean(
            [
                # Coverage for each single data point is just 0 or 1
                vqr.coverage(y.reshape(1, d), x=x, alpha=0.05)
                for (x, y) in zip(X, Y)
            ]
        )

        print(f"{cov=}")
        assert cov > (0.6 if d < 3 else 0.2)

    def test_monotonicity(self, vqr_fitted):
        X, Y, vqr = vqr_fitted
        N = len(Y)

        # Test with a few random X's
        idxs = np.random.permutation(N)
        for i in idxs[:10]:
            _test_monotonicity(
                Us=vqr.quantile_grid,
                Qs=list(vqr.vector_quantiles(X=X[[i]], refine=True)[0]),
                T=vqr.n_levels,
            )

    @pytest.mark.flaky(reruns=1)
    def test_vector_monotone_rearrangement(self, vqr_solver):
        N = 1000
        d = 2
        k = 3
        T = 15
        X, Y = LinearMVNDataProvider(d=d, k=k, seed=42).sample(n=N)

        vqr = VectorQuantileRegressor(n_levels=T, solver=vqr_solver)
        vqr.fit(X, Y)

        all_off_projs_refined = []
        all_off_projs = []

        for _ in range(25):
            i = random.randrange(0, N)
            off_projs, all_projs = monotonicity_offending_projections(
                Us=vqr.quantile_grid,
                Qs=list(vqr.vector_quantiles(X=X[[i]], refine=False)[0]),
                T=vqr.n_levels,
                projection_tolerance=0.0,
            )

            off_projs_refined, all_projs_refined = monotonicity_offending_projections(
                Us=vqr.quantile_grid,
                Qs=list(vqr.vector_quantiles(X=X[[i]], refine=True)[0]),
                T=vqr.n_levels,
                projection_tolerance=0.0,
            )
            all_off_projs.append(len(off_projs) / len(all_projs))
            all_off_projs_refined.append(
                len(off_projs_refined) / len(all_projs_refined)
            )

        print(np.mean(all_off_projs_refined), np.mean(all_off_projs))
        print(np.median(all_off_projs_refined), np.median(all_off_projs))
        assert (np.median(all_off_projs_refined) <= np.median(all_off_projs)) or (
            (np.mean(all_off_projs_refined) <= np.mean(all_off_projs))
        )

    @pytest.mark.parametrize("with_batches", [False, True])
    def test_callback(self, with_batches):

        callback_kwargs = []

        def _callback(*args, **kwargs):
            callback_kwargs.append(kwargs)

        N = 1000
        num_epochs = 100
        batchsize_y = (N // 5) if with_batches else None
        num_batches = np.ceil(N / batchsize_y) if with_batches else 1

        X, Y = LinearMVNDataProvider(d=2, k=3).sample(n=N)
        solver = RegularizedDualVQRSolver(
            verbose=False,
            num_epochs=num_epochs,
            batchsize_y=batchsize_y,
            post_iter_callback=_callback,
        )
        vqr = VectorQuantileRegressor(n_levels=15, solver=solver)
        vqr.fit(X, Y)

        assert len(callback_kwargs) == num_epochs * num_batches
        for kw in callback_kwargs:
            assert kw.keys() == callback_kwargs[0].keys()
            solution = kw["solution"]
            assert isinstance(solution, VQRSolution)

    def test_not_fitted(self, vqr_solver):
        X, Y = LinearMVNDataProvider(d=2, k=3).sample(n=100)
        vq = VectorQuantileRegressor(
            n_levels=100,
            solver=vqr_solver,
        )
        with pytest.raises(NotFittedError):
            _ = vq.quantile_grid
        with pytest.raises(NotFittedError):
            _ = vq.vector_quantiles(X)

    def test_no_x(self, vqr_fitted):
        X, Y, vqr = vqr_fitted
        error_message = "Must provide covariates "
        with pytest.raises(ValueError, match=error_message):
            vqr.sample(n=100, x=None)
        with pytest.raises(ValueError, match=error_message):
            vqr.coverage(Y, x=None)
        with pytest.raises(ValueError, match=error_message):
            vqr.vector_quantiles(X=None)
