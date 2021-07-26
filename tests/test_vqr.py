import numpy as np
import pytest

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

        assert len(vqr.quantile_values) == d
        assert all(q.shape == (T,) * d for q in vqr.quantile_values)
        assert len(vqr.quantile_grid) == d
        assert all(q.shape == (T,) * d for q in vqr.quantile_grid)
