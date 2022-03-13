import itertools as it

import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError

from vqr import VectorQuantileEstimator, VectorQuantileRegressor
from vqr.data import generate_mvn_data, generate_linear_x_y_mvn_data


class TestVectorQuantileEstimator(object):
    @pytest.fixture(
        scope="class",
        params=[
            {"d": 1, "N": 100, "T": 10},
            {"d": 2, "N": 100, "T": 10},
            {"d": 3, "N": 100, "T": 10},
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
        vqe = VectorQuantileEstimator(n_levels=T, solver_opts={"verbose": True})
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

    def test_not_fitted(self):
        vq = VectorQuantileEstimator(n_levels=100)
        with pytest.raises(NotFittedError):
            _ = vq.quantile_grid
        with pytest.raises(NotFittedError):
            _ = vq.vector_quantiles()

    @pytest.mark.repeat(5)
    def test_monotonicity(self):
        T = 25
        N = 100
        d = 2
        EPS = 0.06

        _, Y = generate_mvn_data(n=N, d=d, k=1)
        vqe = VectorQuantileEstimator(n_levels=T, solver_opts={"verbose": True})
        vqe.fit(Y)
        U1, U2 = vqe.quantile_grid
        Q1, Q2 = vqe.vector_quantiles()

        ii = jj = tuple(range(1, T))

        n, n_c = 0, 0
        offending_points = []
        offending_dists = []
        for i0, j0 in it.product(ii, jj):
            u0 = np.array([U1[i0, j0], U2[i0, j0]])
            q0 = np.array([Q1[i0, j0], Q2[i0, j0]])

            for i1, j1 in it.product(ii, jj):
                n += 1

                u1 = np.array([U1[i1, j1], U2[i1, j1]])
                q1 = np.array([Q1[i1, j1], Q2[i1, j1]])

                if np.dot(q1 - q0, u1 - u0) < -EPS:
                    offending = (
                        f"{(i0, j0)=}, {(i1, j1)=}, "
                        f"{q1-q0=}, "
                        f"{u1-u0=}, "
                        f"{np.dot(q1-q0, u1-u0)=}"
                    )
                    offending_points.append(offending)
                    offending_dists.append(np.dot(q1 - q0, u1 - u0).item())
                    n_c += 1
        print(offending_points, offending_dists)
        assert len(offending_points) == 0, f"{n=}, {n_c=}, {n_c/n=:.2f}"


class TestVectorQuantileRegressor(object):
    @pytest.fixture(
        scope="class",
        params=[
            {"d": 1, "k": 5, "N": 500, "T": 20},
            {"d": 2, "k": 7, "N": 500, "T": 20},
            {"d": 3, "k": 3, "N": 500, "T": 10},
        ],
        ids=[
            "d=1,k=5",
            "d=2,k=7",
            "d=3,k=3",
        ],
    )
    def vqr_fitted(self, request):
        params = request.param
        N, d, k, T = params["N"], params["d"], params["k"], params["T"]
        X, Y = generate_linear_x_y_mvn_data(N, d=d, k=k)

        vqr = VectorQuantileRegressor(n_levels=T, solver_opts={"verbose": True})
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
