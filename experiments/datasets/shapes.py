from typing import Tuple, Optional

import numpy as np
import pylab as pl
from numpy import ndarray
from numpy import ndarray as Array

from experiments import EXPERIMENTS_DATA_DIR
from experiments.datasets.base import DataProvider


def generate_heart() -> Tuple[ndarray, ndarray]:
    """
    Generates independent X and Y.
    X is sampled i.i.d from a uniform distribution [0, 1]
    Y is an r.v whose distribution is heart-shaped.
    :return: X, Y
    """
    # Load data from image
    image = 1 - pl.imread(EXPERIMENTS_DATA_DIR / "heart.png")[:, :, 2]
    # image = image[::2, ::2]
    image = image / np.sum(image)
    idces = image.nonzero()
    Y = np.zeros([len(idces[0]), 2])
    Y[:, 0] = idces[0] / idces[0].max()
    Y[:, 1] = idces[1] / idces[1].max()

    rng = np.random.default_rng(None)
    X = rng.uniform(size=(Y.shape[0], 1))
    return X, Y


def generate_star() -> Tuple[ndarray, ndarray]:
    """
    Generates independent X and Y.
    X is sampled i.i.d from a uniform distribution [0, 1]
    Y is an r.v whose distribution is star-shaped.
    :return: X, Y
    """
    image = 1 - pl.imread(EXPERIMENTS_DATA_DIR / "star.jpg") / 255
    # image = image[::7, ::7]
    image = image / np.sum(image)
    idces = image.nonzero()
    Y = np.zeros([len(idces[0]), 2])
    Y[:, 0] = idces[0] / idces[0].max()
    Y[:, 1] = idces[1] / idces[1].max()
    rng = np.random.default_rng(None)
    X = rng.uniform(size=(Y.shape[0], 1))
    return X, Y


def _rotation_matrix(theta: float) -> ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class ImageDataProvider(DataProvider):
    """
    Generates (X, Y) where Y is sampled from an images, rotated by X angles.
    """

    def __init__(
        self,
        image_data: ndarray,
        initial_rotation_deg=0,
        noise_std=0.0,
        x_max_deg=90.0,
        x_discrete=False,
        seed=42,
    ):
        Y = image_data
        # Zero mean
        Y -= np.mean(Y, axis=0)

        # Rotate
        rot_theta = _rotation_matrix(np.deg2rad(initial_rotation_deg))
        Y = Y @ rot_theta

        self._Y_gt = Y
        self._x_high = x_max_deg
        self._x_discrete = x_discrete
        self._noise_std = noise_std
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def k(self) -> int:
        return 1

    @property
    def d(self) -> int:
        return 2

    def sample_x(self, n: int) -> Array:
        if self._x_discrete:
            return self._rng.integers(0, self._x_high + 1, size=(n,)).reshape(n, -1)
        else:
            return self._rng.uniform(0, self._x_high, size=(n,)).reshape(n, -1)

    def sample(self, n: int, x: Optional[Array] = None) -> Tuple[Array, Array]:
        if x is None:
            X = self.sample_x(n)
            R = np.stack([_rotation_matrix(np.deg2rad(-x).item()) for x in X], axis=0)
        else:
            x = np.reshape(x, (1, -1))
            assert x.shape[1] == self.k
            X = np.concatenate([x for _ in range(n)], axis=0)
            R = np.stack([_rotation_matrix(np.deg2rad(-x).item())] * n, axis=0)

        N = len(self._Y_gt)
        sample_idx = self._rng.integers(0, N, size=(n,))

        Y_sampled = self._Y_gt[sample_idx]

        # Rotate samples by the corresponding x degrees
        Y_sampled = np.einsum("bji,bi -> bj", R, Y_sampled)

        # Additive noise
        eta = self._rng.normal(size=Y_sampled.shape) * self._noise_std
        Y_sampled += eta

        return X, Y_sampled


class HeartDataProvider(ImageDataProvider):
    def __init__(
        self,
        initial_rotation_deg=0,
        noise_std=0.0,
        x_max_deg=10.0,
        x_discrete=False,
        seed=42,
    ):
        Y = generate_heart()
        super().__init__(
            image_data=generate_heart()[1],
            initial_rotation_deg=initial_rotation_deg,
            noise_std=noise_std,
            x_max_deg=x_max_deg,
            x_discrete=x_discrete,
            seed=seed,
        )


class StarDataProvider(ImageDataProvider):
    def __init__(
        self,
        initial_rotation_deg=0,
        noise_std=0.0,
        x_max_deg=10.0,
        x_discrete=False,
        seed=42,
    ):
        Y = generate_heart()
        super().__init__(
            image_data=generate_star()[1],
            initial_rotation_deg=initial_rotation_deg,
            noise_std=noise_std,
            x_max_deg=x_max_deg,
            x_discrete=x_discrete,
            seed=seed,
        )
