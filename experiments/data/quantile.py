import os
import pickle
import logging
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
from filelock import FileLock

from vqr.api import DEFAULT_SOLVER_NAME, VectorQuantileRegressor
from experiments import EXPERIMENTS_OUT_DIR
from experiments.data.base import Array, DataProvider
from experiments.utils.helpers import stable_hash

_LOG = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = EXPERIMENTS_OUT_DIR.joinpath("cache")


class QuantileFunctionMVNDataProvider(DataProvider):
    """
    A data provider which models the conditional quantile function of an MVN variable Y|X.

    The Conditional quantile function is defined by

    Q_{Y|X=x}(u) = mu(x) + Sigma^{1/2}(x) u

    Where mu(x) is the mean, Sigma(x) is the covariance and u is a vector rank
    which must be sampled from N(0, I).

    Based on Carlier et. al 2016.
    """

    def __init__(self, k: int, d: int, seed: Optional[int] = 42):
        """
        :param d: Dimension of targets.
        :param k: Dimension of features.
        :param seed: Random seed to use for generation. None means don't set.
        """
        super().__init__(seed=seed)
        assert d > 0
        assert k >= 0
        self._d = d
        self._k = k
        # self._loc = self._rng.random((d,))
        # self._scale = self._rng.random((d, d))
        self._loc = np.zeros((d,))
        self._scale = np.eye(N=d)

    @property
    def k(self) -> int:
        return self._k

    @property
    def d(self) -> int:
        return self._d

    def _mu_sigma(self, x: Array) -> Tuple[Array, Array]:
        x2 = np.sum(x**1)  # will have chi-squared distribution with k-DOF

        mu = self._loc + x2  # (d,)
        sigma = self._scale * x2  # (d, d)
        return mu, sigma

    def sample_x(self, n: int) -> Array:
        return self._rng.normal(size=(n, self.k))

    def sample(self, n: int, x: Optional[Array] = None) -> Tuple[Array, Array]:
        if x is None:
            X = self.sample_x(n=n)
        else:
            x = np.reshape(x, (1, -1))
            assert x.shape[1] == self.k
            X = np.concatenate([x for _ in range(n)], axis=0)

        mu, sigma = zip(*[self._mu_sigma(x) for x in X])

        U = self._rng.normal(size=(n, self.d))  # (n, d)
        M = np.array(mu)  # (n,d)
        S = np.array(sigma)  # (n,d,d)
        SU = np.einsum("Nij, Nj -> Ni", S, U)  # (n,d,d) @ (n,d) -> (n,d)

        Y = M + SU

        return X, Y


class QuantileFunctionDataProviderWrapper(DataProvider):
    """
    A wrapper around a data provider which first estimates the conditional quantile
    function by solving VQR, and then samples from the quantile function to generate
    new data.

    Uses a local cache to avoid re-calculating the same VQR solution multiple times.
    """

    def __init__(
        self,
        wrapped_provider: DataProvider,
        vqr_n_levels: int = 50,
        vqr_fit_n: int = 10000,
        vqr_solver_name: str = DEFAULT_SOLVER_NAME,
        vqr_solver_opts: Optional[dict] = None,
        cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
        seed: Optional[int] = 42,
    ):
        """
        :param wrapped_provider: The data provider which will be used to draw samples
        on which a VQR model will be fitted.
        :param vqr_n_levels: Number of quantile levels to fit the VQR model with.
        :param vqr_fit_n: Number of samples from the wrapped provider to fit the VQR
        model with.
        :param vqr_solver_name: VQR solver name.
        :param vqr_solver_opts: VQR solver options.
        :param cache_dir: Override for the default cache directory. Set None for no
        caching.
        :param seed: Random seed.
        """
        super().__init__(seed=seed)

        self.wrapped_provider = wrapped_provider
        self._vqr_n_levels = vqr_n_levels
        self._vqr_fit_n = vqr_fit_n
        self._vqr_solver_name = vqr_solver_name
        self._vqr_solver_opts = vqr_solver_opts
        self._cache_dir = Path(cache_dir) if cache_dir else None

        if not self._cache_dir:
            self.vqr = self._fit_vqr()

        else:
            (
                self.vqr,
                self.cached_path,
                self.cache_loaded,
                self.cache_saved,
            ) = self._vqr_from_cache()

    def _fit_vqr(self) -> VectorQuantileRegressor:
        X, Y = self.wrapped_provider.sample(n=self._vqr_fit_n)
        return VectorQuantileRegressor(
            n_levels=self._vqr_n_levels,
            solver=self._vqr_solver_name,
            solver_opts=self._vqr_solver_opts,
        ).fit(X, Y)

    def _vqr_from_cache(self):
        os.makedirs(self._cache_dir, exist_ok=True)

        cache_args = [
            self.wrapped_provider,
            self._vqr_fit_n,
            self._vqr_n_levels,
            self._vqr_solver_name,
            self._vqr_solver_opts,
            self._seed,
        ]
        cache_key = stable_hash(cache_args, hash_len=8)

        cached_file_path = self._cache_dir.joinpath(f"{cache_key}.pkl")
        lock_file_path = self._cache_dir.joinpath(f"{cache_key}.lock")

        # Make sure only one process calculates the same cached VQR
        with FileLock(str(lock_file_path), timeout=-1):
            need_to_refit = True
            cache_loaded = False
            cache_saved = False

            # First we try to load from cache
            if cached_file_path.is_file():
                with open(cached_file_path, "rb") as f:
                    try:
                        fitted_vqr = pickle.load(f)
                        _LOG.info(f"Loaded cached VQR@{cache_key}")
                        need_to_refit = False
                        cache_loaded = True
                    except pickle.PickleError as e:
                        _LOG.warning(f"Failed to load cached VQR@{cache_key}: {e}")

            # If there is no cached result, or we were unsuccessful loading it,
            # fit VQR and save the cached result.
            if need_to_refit:
                fitted_vqr = self._fit_vqr()

                with open(cached_file_path, "wb") as f:
                    pickle.dump(fitted_vqr, f)
                    _LOG.info(f"Saved cached VQR@{cache_key}")
                    cache_saved = True

        return fitted_vqr, cached_file_path, cache_loaded, cache_saved

    @property
    def k(self) -> int:
        return self.wrapped_provider.k

    @property
    def d(self) -> int:
        return self.wrapped_provider.d

    def sample_x(self, n: int) -> Array:
        return self.wrapped_provider.sample_x(n)

    def sample(self, n: int, x: Optional[Array] = None) -> Tuple[Array, Array]:
        if x is None:
            X = self.sample_x(n=n)
            Y = np.concatenate([self.vqr.sample(n=1, x=x) for x in X], axis=0)
        else:
            x = np.reshape(x, (1, -1))
            assert x.shape[1] == self.k
            X = np.concatenate([x for _ in range(n)], axis=0)
            Y = self.vqr.sample(n=n, x=x)

        return X, Y
