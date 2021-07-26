import numpy as np
import pytest

from vqr import VectorQuantileRegressor


class TestVectorQuantileRegressor(object):
    @pytest.fixture(autouse=True)
    def setup_rng(self):
        self.rng = np.random.default_rng()

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

        # Generate random orthonormal matrix Q
        Q, R = np.linalg.qr(self.rng.normal(size=(d, d)))

        # Generate positive eigenvalues
        eigs = self.rng.uniform(size=(d,))

        # PSD Covariance matrix, zero mean
        S = Q.T @ np.diag(eigs) @ Q
        mu = np.zeros(d)

        # Generate correlated targets (Y) and uncorrelated features (X)
        Y = self.rng.multivariate_normal(mean=mu, cov=S, size=(N,))
        X = self.rng.normal(size=(N, k))

        return X, Y

    def test_shapes(self, dataset):
        X, Y = dataset
        N, d, k = Y.shape[0], Y.shape[1], X.shape[1]
        T = 10

        vqr = VectorQuantileRegressor(n_levels=T, solver_opts={"verbose": True})

        vqr.fit(X, Y)

        assert len(vqr.quantile_values) == d
        assert all(q.shape == (T,) * d for q in vqr.quantile_values)
        assert len(vqr.quantile_grid) == d
        assert all(q.shape == (T,) * d for q in vqr.quantile_grid)
