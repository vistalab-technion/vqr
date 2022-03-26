import itertools as it
from typing import Sequence

import numpy as np
import pytest
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.exceptions import NotFittedError

from vqr import VectorQuantileEstimator, VectorQuantileRegressor
from vqr.data import generate_mvn_data, generate_linear_x_y_mvn_data
from vqr.solvers.dual.regularized_lse import (
    RegularizedDualVQRSolver,
    MLPRegularizedDualVQRSolver,
)


class TestVectorQuantileEstimator(object):
    SOLVER = "regularized_dual"
    SOLVER_OPTS = {"verbose": True, "learning_rate": 0.1, "epsilon": 1e-6}

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
    def vqe_fitted(self, request):
        params = request.param
        d, N, T = params["d"], params["N"], params["T"]
        X, Y = generate_mvn_data(N, d, k=1)
        vqe = VectorQuantileEstimator(
            n_levels=T,
            solver=self.SOLVER,
            solver_opts=self.SOLVER_OPTS,
        )
        vqe.fit(Y)
        return Y, vqe

    def test_shapes(self, vqe_fitted):
        Y, vqe = vqe_fitted
        N, d = Y.shape[0], Y.shape[1]
        T = vqe.n_levels

        assert vqe.dim_y == d
        assert len(vqe.quantile_grid) == d
        assert len(vqe.vector_quantiles()) == d

        assert all(q.shape == (T,) * d for q in vqe.vector_quantiles())
        assert all(q.shape == (T,) * d for q in vqe.quantile_grid)

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

    def test_coverage(self, vqe_fitted, test_out_dir):
        Y, vqe = vqe_fitted
        N, d = Y.shape
        T = vqe.n_levels

        cov = vqe.coverage(Y, alpha=0.05)
        print(f"{cov=}")
        assert cov > (0.7 if d < 3 else 0.25)

    def test_not_fitted(self):
        vq = VectorQuantileEstimator(n_levels=100)
        with pytest.raises(NotFittedError):
            _ = vq.quantile_grid
        with pytest.raises(NotFittedError):
            _ = vq.vector_quantiles()

    @pytest.mark.repeat(5)
    def test_monotonicity(self):
        T = 15
        N = 1000
        d = 2

        _, Y = generate_mvn_data(n=N, d=d, k=1)
        vqe = VectorQuantileEstimator(
            n_levels=T, solver=self.SOLVER, solver_opts=self.SOLVER_OPTS
        )
        vqe.fit(Y)

        _test_monotonicity(
            Us=vqe.quantile_grid,
            Qs=vqe.vector_quantiles(),
            T=vqe.n_levels,
        )


class TestVectorQuantileRegressor(object):
    @pytest.fixture(
        scope="class",
        params=[
            RegularizedDualVQRSolver(verbose=True, learning_rate=0.5, epsilon=1e-5),
            MLPRegularizedDualVQRSolver(verbose=True, learning_rate=0.5, epsilon=1e-5),
            MLPRegularizedDualVQRSolver(
                verbose=True,
                learning_rate=0.5,
                epsilon=1e-5,
                hidden_layers=[2, 4],
                skip=False,  # No skip, so output will have different k
                num_epochs=1500,
            ),
        ],
        ids=[
            "rvqr_linear",
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
        X, Y = generate_linear_x_y_mvn_data(N, d=d, k=k)

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

        for X_ in [None, X]:
            N_ = N if X_ is not None else 1

            vq_samples = vqr.vector_quantiles(X=X_)

            assert len(vq_samples) == N_

            for vq_sample in vq_samples:
                assert all(vq.shape == (T,) * d for vq in vq_sample)

            Y_hat = vqr.predict(X_)
            assert Y_hat.shape == (N_, d, *[T] * d)

    def test_sample(self, vqr_fitted, test_out_dir):
        X, Y, vqr = vqr_fitted
        N, d = Y.shape
        N, k = X.shape
        T = vqr.n_levels

        n = 1000

        Y_samp = vqr.sample(n, x=None)
        assert Y_samp.shape == (n, d)

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
            ax.scatter(Y_samp[:, 0], Y_samp[:, 1], c="C0", label="Y")
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

    def test_coverage(self, vqr_fitted, test_out_dir):
        X, Y, vqr = vqr_fitted
        N, d = Y.shape
        N, k = X.shape
        T = vqr.n_levels

        cov = vqr.coverage(Y, alpha=0.05, x=np.median(X, axis=0))
        print(f"{cov=}")
        assert cov > (0.6 if d < 3 else 0.2)

    @pytest.mark.parametrize("i", range(5))
    def test_monotonicity(self, i, vqr_solver):
        N = 1000
        d = 2
        k = 3
        T = 15
        X, Y = generate_linear_x_y_mvn_data(n=N, d=d, k=k)

        vqr = VectorQuantileRegressor(n_levels=T, solver=vqr_solver)
        vqr.fit(X, Y)

        _test_monotonicity(
            Us=vqr.quantile_grid,
            Qs=vqr.vector_quantiles(X=X[[i]])[0],
            T=vqr.n_levels,
        )


def _test_monotonicity(
    Us: Sequence[np.ndarray],
    Qs: Sequence[np.ndarray],
    T: int,
    projection_tolerance: float = 0.0,
    offending_proportion_limit: float = 0.005,
):
    # Only supports 2d for now.
    U1, U2 = Us
    Q1, Q2 = Qs

    ii = jj = tuple(range(1, T))

    n, n_c = 0, 0
    projections = []
    offending_projections = []
    for i0, j0 in it.product(ii, jj):
        u0 = np.array([U1[i0, j0], U2[i0, j0]])
        q0 = np.array([Q1[i0, j0], Q2[i0, j0]])

        for i1, j1 in it.product(ii, jj):
            n += 1

            u1 = np.array([U1[i1, j1], U2[i1, j1]])
            q1 = np.array([Q1[i1, j1], Q2[i1, j1]])
            du = u1 - u0
            dq = q1 - q0
            projection = np.dot(dq, du)

            # normalize projection to [-1, 1]
            # but only if it has any length (to prevent 0/0 -> NaN)
            if np.abs(projection) > 0:
                projection = projection / norm(dq) / norm(du)

            assert not np.isnan(projection)
            if projection < -projection_tolerance:
                offending_projections.append(projection.item())
                n_c += 1

            projections.append(projection)

    offending_proportion = n_c / n
    if offending_projections:
        q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        print(f"err quantiles: {np.quantile(offending_projections, q=q)}")
        print(f"all quantiles: {np.quantile(projections, q=q)}")
        print(f"{n=}, {n_c=}, {n_c/n=}")

    assert offending_proportion < offending_proportion_limit
