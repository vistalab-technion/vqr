from typing import Tuple, Optional, Sequence

import numpy as np
from numpy import array
from numpy.random import Generator

from experiments.data.base import DataProvider


class LinearMVNDataProvider(DataProvider):
    """
    Generates a dataset of linearly dependent features and targets.
    Features are sampled uniformly [0, 1]. Targets are obtained via
        Y = A X + N
    where A is a random matrix whose entries are sampled i.i.d from
    standard normal distribution. N is a random vector drawn from a
    multivariate normal distribution with a random covariance matrix.

    """

    def __init__(
        self, k: int, d: int, A: Optional[array] = None, seed: Optional[int] = 42
    ):
        """
        :param d: Dimension of targets.
        :param k: Dimension of features.
        :param seed: Random seed to use for generation. None means don't set.
        """
        assert d > 0
        assert k >= 0
        self._d = d
        self._k = k
        self._rng = np.random.default_rng(seed)
        if A is not None:
            assert k == A.shape[0]
            assert d == A.shape[1]
        self._A = self._make_A() if A is None else A
        self._noise_generator = MVNNoiseGenerator(d=d, rng=self._rng, random_cov=True)

    @property
    def k(self) -> int:
        return self._k

    @property
    def d(self) -> int:
        return self._d

    def sample(self, n: int, X: Optional[array] = None) -> Sequence[array]:
        """
        :param n: Number of samples.
        :param X: Features whose conditional distribution needs to be sampled.
        :return: A tuple (X, Y), where X is of shape (n, k) and contains the features
        and Y is of shape (n, d) and contains the responses.
        """

        if X is None:
            X = self._rng.uniform(size=(n, self.k))
            X -= np.mean(X, axis=0)
        else:
            assert len(X.shape) == 2
            assert X.shape[1] == self._k
            assert X.shape[0] == 1
            X = np.concatenate([X for _ in range(n)], axis=0)

        N = self._noise_generator.sample(n)
        Y = X @ self._A + N
        return X, Y

    def _make_A(self) -> array:
        return self._rng.random(size=(self.k, self.d))


class MVNNoiseGenerator:
    """
    Generates multivariate normal distribution.
    """

    def __init__(
        self,
        d: int,
        rng: Optional[Generator] = None,
        seed: int = 42,
        random_cov: bool = False,
    ):
        """

        :param d: dimension of the distribution
        :param rng: a numpy random number generator
        :param seed: Random seed to use for generation. If a random number generator
        is provided, this seed will be ignored.
        :param random_cov: Whether to use a random covariance. If set to False, a
        specific construction of covariance for anti-correlated r.v.s will be used (
        only when d=2).
        """
        assert d > 0
        self._d = d
        self._rng = np.random.default_rng(seed) if rng is None else rng
        self._Sigma = self._make_Sigma(random_cov)
        self._mu = self._make_mu()

    def sample(self, n: int) -> array:
        return self._rng.multivariate_normal(mean=self._mu, cov=self._Sigma, size=(n,))

    def _make_mu(self) -> array:
        return np.zeros(self._d)

    def _make_Sigma(self, random_cov: bool = False) -> array:
        if (not random_cov) and self._d == 2:
            S = np.array([[1.0, -0.7], [-0.7, 1.0]])

        else:
            # Generate random orthonormal matrix Q
            Q, R = np.linalg.qr(self._rng.normal(size=(self._d, self._d)))

            # Generate positive eigenvalues
            eigs = self._rng.uniform(size=(self._d,))

            # PSD Covariance matrix, zero mean
            S = Q.T @ np.diag(eigs) @ Q
        return S


class IndependentDataProvider(LinearMVNDataProvider):
    """
    Generates a dataset of independent features and targets.
    Features are sampled i.i.d from a standard normal distribution. In case of >1
    dimensional features, each sample in the vector-valued feature is drawn i.i.d
    from standard normal distribution.
    Targets are sampled i.i.d from a multivariate normal distribution.
    """

    def __init__(self, k: int, d: int, seed: Optional[int] = 42):
        super().__init__(k=k, d=d, A=np.zeros(shape=(k, d)), seed=seed)