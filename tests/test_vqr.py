import itertools as it

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from vqr import VectorQuantileRegressor
from vqr.data import generate_mvn_data


class TestVectorQuantileRegressor(object):
    @pytest.fixture(
        params=[
            #
            {"d": 1, "k": 2},
            {"d": 2, "k": 1},
            {"d": 3, "k": 3},
        ],
        ids=[
            #
            "d=1, k=2",
            "d=2, k=1",
            "d=3, k=3",
        ],
    )
    def dataset(self, request):
        params = request.param
        d, k = params["d"], params["k"]
        N = 100
        return generate_mvn_data(N, d, k)

    def test_shapes(self, dataset):
        X, Y = dataset
        N, d, k = Y.shape[0], Y.shape[1], X.shape[1]
        T = 10

        vqr = VectorQuantileRegressor(n_levels=T, solver_opts={"verbose": True})
        vqr.fit(X, Y)

        assert vqr.quantile_dimension == d
        assert len(vqr.quantile_values) == d
        assert all(q.shape == (T,) * d for q in vqr.quantile_values)
        assert len(vqr.quantile_grid) == d
        assert all(q.shape == (T,) * d for q in vqr.quantile_grid)

    def test_not_fitted(self):
        vqr = VectorQuantileRegressor(n_levels=100)
        with pytest.raises(NotFittedError):
            _ = vqr.quantile_grid
        with pytest.raises(NotFittedError):
            _ = vqr.quantile_values

    @pytest.mark.repeat(5)
    def test_monotonicity(self):
        T = 25
        N = 100
        d = 2
        EPS = 0.0001

        X, Y = generate_mvn_data(n=N, d=d, k=1)
        vqr = VectorQuantileRegressor(n_levels=T, solver_opts={"verbose": True})
        vqr.fit(X, Y)
        U1, U2 = vqr.quantile_grid
        Q1, Q2 = vqr.quantile_values

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

        assert len(offending_points) == 0, f"{n=}, {n_c=}, {n_c/n=:.2f}"
