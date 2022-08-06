from vqr.api import ScalarQuantileEstimator
from experiments.datasets.mvn import IndependentDataProvider


class TestScalarQuantileEstimator:
    def test_init(self):
        n_levels = 50
        N = 100
        d = 2
        _, Y = IndependentDataProvider(d=d, k=1).sample(n=N)

        sq = ScalarQuantileEstimator(n_levels)
        fitted_sq = sq.fit(Y)
        assert fitted_sq.quantile_values.shape[0] == n_levels
        assert len(fitted_sq.quantile_levels) == n_levels
