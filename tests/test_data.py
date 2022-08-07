import os

import pytest

from tests import TESTS_OUT_DIR
from experiments.datasets.mvn import LinearMVNDataProvider, IndependentDataProvider
from experiments.datasets.quantile import QuantileFunctionDataProviderWrapper
from experiments.datasets.cond_banana import ConditionalBananaDataProvider


class TestMVNData(object):
    @pytest.mark.parametrize("k", [1, 20, 100])
    @pytest.mark.parametrize("d", [1, 2, 10])
    @pytest.mark.parametrize("n", [1000, 5000, 10000])
    def test_shapes(self, n, d, k):
        X, Y = IndependentDataProvider(k, d, seed=1023).sample(n=n)

        assert X.shape == (n, k)
        assert Y.shape == (n, d)


class TestCondBanana(object):
    @pytest.mark.parametrize("k", [1, 20, 100])
    @pytest.mark.parametrize("d", [2, 3, 4])
    @pytest.mark.parametrize("n", [1000, 5000, 10000])
    @pytest.mark.parametrize("nonlinear", [True, False])
    def test_shapes(self, n, d, k, nonlinear):
        X, Y = ConditionalBananaDataProvider(
            k=k, d=d, nonlinear=nonlinear, seed=63
        ).sample(n)

        assert X.shape == (n, k)
        assert Y.shape == (n, d)


class TestQuantileFunctionDataProviderWrapper:
    @pytest.fixture(autouse=True)
    def setup(self, test_out_dir):
        self.wrapped_provider = LinearMVNDataProvider(k=3, d=1)
        self.cache_dir = test_out_dir

    @pytest.fixture(autouse=True, scope="class")
    def providers(self, test_out_dir_class_pid):
        wrapped_provider = LinearMVNDataProvider(k=3, d=1, seed=5040)
        cache_dir = test_out_dir_class_pid

        vqr_n_levels = 25
        vqr_fit_n = 10000

        dp1 = QuantileFunctionDataProviderWrapper(
            wrapped_provider=wrapped_provider,
            vqr_solver_opts=dict(verbose=True),
            vqr_n_levels=vqr_n_levels,
            vqr_fit_n=vqr_fit_n,
            cache_dir=cache_dir,
        )

        dp1_cached = QuantileFunctionDataProviderWrapper(
            wrapped_provider=wrapped_provider,
            vqr_solver_opts=dict(verbose=True),
            vqr_n_levels=vqr_n_levels,
            vqr_fit_n=vqr_fit_n,
            cache_dir=cache_dir,
        )

        dp2 = QuantileFunctionDataProviderWrapper(
            wrapped_provider=wrapped_provider,
            vqr_solver_opts=dict(verbose=True),
            vqr_n_levels=vqr_n_levels // 2,
            vqr_fit_n=vqr_fit_n // 2,
            cache_dir=cache_dir,
        )

        return wrapped_provider, dp1, dp1_cached, dp2

    def test_cache(self, providers):
        wrapped_provider, dp1, dp1_cached, dp2 = providers

        assert not dp1.cache_loaded
        assert dp1.cache_saved
        assert dp1_cached.cache_loaded
        assert not dp1_cached.cache_saved

        assert not dp2.cache_loaded
        assert dp2.cache_saved

        assert dp1.cached_path == dp1_cached.cached_path
        assert dp1.cached_path != dp2

    def test_wrapped_shapes(self, providers):
        wrapped_provider, dp1, dp1_cached, dp2 = providers

        for dp in [dp1, dp1_cached, dp2]:
            assert dp.k == wrapped_provider.k
            assert dp.d == wrapped_provider.d

            n = 100

            for x in [None, dp.sample_x(n=1)]:
                if x is not None:
                    assert x.shape == (1, dp.k)

                X, Y = dp1.sample(n=n, x=x)
                assert Y.shape == (n, dp.d)
                assert X.shape == (n, dp.k)
