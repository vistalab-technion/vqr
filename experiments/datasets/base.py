from abc import ABC, abstractmethod
from typing import Tuple, Optional, Sequence

import numpy as np
from numpy import ndarray as Array


class SerializableRandomDataGenerator(ABC):
    """
    A wrapper for a numpy number generator which allows serialization and
    de-serialization without maintaining the state of the generator.
    This is useful when persistent hashes need to be generated from the provider's
    state.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = self._create_rng()

    def _create_rng(self) -> np.random.Generator:
        return np.random.default_rng(self._seed)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_rng"] = None
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        self._rng = self._create_rng()


class DataProvider(SerializableRandomDataGenerator):
    """
    Base class for DataProviders which represent a possibly conditional distribution
    of Y|X where Y is d-dimensional and X is k-dimensional. Models the distributions
    of  X, Y|X and the joint (X,Y).
    """

    @property
    @abstractmethod
    def k(self) -> int:
        """
        :return: Number of features (covariates) in X.
        """
        pass

    @property
    @abstractmethod
    def d(self) -> int:
        """
        :return: Number of dimensions in Y.
        """
        pass

    @abstractmethod
    def sample_x(self, n: int) -> Array:
        """
        Samples feature vectors X.
        :param n: Number of samples required.
        :return: An array of shape (n, k) with the samples.
        """
        pass

    @abstractmethod
    def sample(self, n: int, x: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Samples either (x,y) pairs from the joint distribution or multiple y values
        from Y|X=x.
        :param n: Number of samples to draw.
        :param x: The feature vector to condition on. Should be of shape (1, k).
        If not provided, will be sampled n times.
        :return: A tuple (X, Y) containing n samples from Y|X=x if x was provided (X
        will contain the same sample n times), otherwise n samples from Y where a
        different X was sampled each time, representing samples from the joint
        distribution of (X,Y).
        In all cases, X will be of shape (n, k) and Y will be of shape (n, d).
        """
        pass
