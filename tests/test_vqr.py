import itertools as it

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from vqr import VectorQuantileEstimator, VectorQuantileRegressor
from vqr.data import generate_mvn_data


class TestVectorQuantileEstimator(object):
    @pytest.fixture(
        params=[
            {"d": 1},
            {"d": 2},
            {"d": 3},
        ],
        ids=[
            "d=1",
            "d=2",
            "d=3",
        ],
    )
    def dataset(self, request):
        params = request.param
        d = params["d"]
        N = 100
        X, Y = generate_mvn_data(N, d, k=1)
        return Y

    def test_shapes(self, dataset):
        Y = dataset
        N, d = Y.shape[0], Y.shape[1]
        T = 10

        vqe = VectorQuantileEstimator(n_levels=T, solver_opts={"verbose": True})
        vqe.fit(Y)

        assert vqe.dim_y == d
        assert len(vqe.quantile_grid) == d
        assert len(vqe.vector_quantiles()) == d

        assert all(q.shape == (T,) * d for q in vqe.vector_quantiles())
        assert all(q.shape == (T,) * d for q in vqe.quantile_grid)

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
        params=[
            {"d": 1, "k": 5},
            {"d": 2, "k": 7},
            {"d": 3, "k": 3},
        ],
        ids=[
            "d=1, k=5",
            "d=2, k=7",
            "d=3, k=3",
        ],
    )
    def dataset(self, request):
        params = request.param
        d, k = params["d"], params["k"]
        N = 100
        X, Y = generate_mvn_data(N, d=d, k=k)
        return X, Y

    def test_shapes(self, dataset):
        X, Y = dataset
        N, d = Y.shape
        N, k = X.shape
        T = 10

        vqr = VectorQuantileRegressor(n_levels=T, solver_opts={"verbose": True})
        vqr.fit(X, Y)

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
