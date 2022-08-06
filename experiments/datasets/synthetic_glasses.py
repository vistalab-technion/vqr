from typing import Optional, Sequence

import numpy as np
from numpy import array
from numpy import ndarray as Array
from matplotlib import pyplot as plt

from experiments.datasets.base import DataProvider


class SyntheticGlassesDataProvider(DataProvider):
    @property
    def k(self) -> int:
        return 1

    @property
    def d(self) -> int:
        return 1

    def sample_x(self, n: int) -> Array:
        return np.linspace(0, 1, num=n)

    def sample(self, n: int, x: Optional[array] = None) -> Sequence[array]:
        Z = np.random.rand(n) > 0.5
        Y_i = (
            5 * np.sin(np.linspace(start=0.0, stop=3 * np.pi, num=n))
            + 2.5
            + 2 * np.random.beta(a=0.5, b=1, size=(n,))
        )
        Y_j = (
            5 * np.sin(np.linspace(start=np.pi, stop=4 * np.pi, num=n))
            + 2.5
            - 2 * np.random.beta(a=0.5, b=1, size=(n,))
        )
        return self.sample_x(n), Z * Y_i + (~Z) * Y_j


if __name__ == "__main__":
    dp = SyntheticGlassesDataProvider()
    X_, Y_ = dp.sample(n=3000)
    plt.plot(X_, Y_, ".")
    plt.show()
