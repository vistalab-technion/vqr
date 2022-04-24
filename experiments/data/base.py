from abc import ABC, abstractmethod
from typing import Tuple, Optional, Sequence

from numpy.typing import ArrayLike as Array


class DataProvider(ABC):
    """
    Base class for DataProviders which represent a possibly conditional distribution
    of Y|X where Y is d-dimensional and X is k-dimensional.
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
    def sample(self, n: int, x: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        :param n: Number of samples to draw.
        :param x: The feature vector to condition on. If not provided, will be sampled.
        :return: A tuple (X, Y) containing n samples from Y|X=x if x was provided,
        otherwise n samples from Y where a different X was sampled each time.
        """
        pass
