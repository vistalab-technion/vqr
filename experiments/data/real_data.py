from typing import Dict, Tuple, Optional

import pandas as pd
from numpy import array, stack, arange
from numpy import ndarray as Array
from numpy.random import randint
from sklearn.preprocessing import StandardScaler

from experiments.data.base import RealDataProvider
from experiments.utils.split import split_train_calib_test

DATA_FOLDER_PATH = "/home/sanketh/vqr/data/"


class CASPDataProvider(RealDataProvider):
    DATASET_PATH = DATA_FOLDER_PATH + "CASP.csv"
    TOTAL_DIMS = 10

    def __init__(self, d: int, seed: Optional[int] = None):
        df = pd.read_csv(self.DATASET_PATH)
        if d == 1:
            self.y = array(df.iloc[:, 0].values)[:, None]
            self.x = array(df.iloc[:, 1:].values)
        elif d == 2:
            y_indices = [0, 6]
            self.y = stack([array(df.iloc[:, idx].values) for idx in y_indices], axis=1)
            self.x = stack(
                [
                    array(df.iloc[:, idx].values)
                    for idx in range(df.shape[1])
                    if idx not in y_indices
                ],
                axis=1,
            )
        else:
            raise ValueError("CASP dataset supports only upto d=2.")
        assert self.x.shape[0] == self.y.shape[0]
        self._k = self.x.shape[1]
        self._d = self.y.shape[1]
        self._n = self.x.shape[0]
        self._seed = seed

    @property
    def k(self) -> int:
        return self._k

    @property
    def d(self) -> int:
        return self._d

    @property
    def n(self) -> int:
        return self._n

    def sample(self, n: int, idx: Optional[Array] = None) -> Tuple[Array, Array]:
        if idx is not None:
            assert idx.shape[0] <= self.n
        else:
            idx = randint(low=0, high=self.x.shape[0], size=n)
        return self.x[idx, :], self.y[idx, :]


class DataFolder:
    def __init__(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        scale: bool = True,
    ):
        if train_ratio + val_ratio + test_ratio != 1.0:
            sum_ratios = train_ratio + val_ratio + test_ratio
            train_ratio = train_ratio / sum_ratios
            val_ratio = val_ratio / sum_ratios
            test_ratio = test_ratio / sum_ratios

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.scale = scale

    def generate_folds(
        self, data_provider: RealDataProvider
    ) -> Dict[str, Tuple[array, array]]:
        X, Y = data_provider.sample(n=data_provider.n, idx=arange(0, data_provider.n))
        data = split_train_calib_test(
            X, Y, split_ratios=(self.train_ratio, self.train_ratio + self.val_ratio)
        )
        if not self.scale:
            return data
        else:
            x_scaler = StandardScaler().fit(X=data["train"][0])
            y_scaler = StandardScaler().fit(X=data["train"][1])
            processed_data = {
                "train": (
                    x_scaler.transform(data["train"][0]),
                    y_scaler.transform(data["train"][1]),
                ),
                "calib": (
                    x_scaler.transform(data["calib"][0]),
                    y_scaler.transform(data["calib"][1]),
                ),
                "test": (
                    x_scaler.transform(data["test"][0]),
                    y_scaler.transform(data["test"][1]),
                ),
            }
            return processed_data


if __name__ == "__main__":
    d = 1
    total_dims = 10
    N = 45730
    dp = CASPDataProvider(d=d)
    assert dp.y.shape[1] == d
    assert dp.y.shape[0] == dp.x.shape[0] == N
    assert dp.x.shape[1] == total_dims - d
    print(dp.x.shape, dp.y.shape)
    x_sampled, y_sampled = dp.sample(n=10000)
    assert x_sampled.shape == (10000, total_dims - d)
    assert y_sampled.shape == (10000, d)

    folds = DataFolder(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2).generate_folds(
        data_provider=dp
    )
    assert folds["train"][0].shape[0] == folds["train"][1].shape[0]
    assert folds["calib"][0].shape[0] == folds["calib"][1].shape[0]
    assert folds["test"][0].shape[0] == folds["test"][1].shape[0]
    assert folds["train"][0].shape[1] == folds["calib"][0].shape[1]
    assert folds["train"][1].shape[1] == folds["calib"][1].shape[1]
