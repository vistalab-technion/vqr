import pytest

from experiments.data.mvn import IndependentDataProvider


class TestMVNData(object):
    @pytest.mark.parametrize("k", [1, 20, 100])
    @pytest.mark.parametrize("d", [1, 2, 10])
    @pytest.mark.parametrize("n", [1000, 5000, 10000])
    def test_shapes(self, n, d, k):
        X, Y = IndependentDataProvider(k, d).sample(n=n)

        assert X.shape == (n, k)
        assert Y.shape == (n, d)
