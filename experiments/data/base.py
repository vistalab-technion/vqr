from abc import ABC, abstractmethod
from typing import Optional, Sequence

from numpy import array


class DataProvider(ABC):
    @property
    @abstractmethod
    def k(self) -> int:
        pass

    @property
    @abstractmethod
    def d(self) -> int:
        pass

    @abstractmethod
    def sample(self, n: int, x: Optional[array] = None) -> Sequence[array]:
        pass
