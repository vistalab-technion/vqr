import pytest

from vqr.data import generate_mvn_data


class TestMVNData(object):
    @pytest.mark.parametrize("k", [1, 20, 100])
    @pytest.mark.parametrize("d", [1, 2, 10])
    @pytest.mark.parametrize("n", [1000, 5000, 10000])
    def test_shapes(self, n, d, k):
        X, Y = generate_mvn_data(n, d, k)

        assert X.shape == (n, k)
        assert Y.shape == (n, d)
