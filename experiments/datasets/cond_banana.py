from typing import Tuple, Optional, Sequence

import numpy as np
import torch

from experiments.datasets.base import Array, DataProvider


class ConditionalBananaDataProvider(DataProvider):
    def __init__(
        self, k: int, d: int, seed: Optional[int] = 42, nonlinear: bool = False
    ):
        assert d in [2, 3, 4]
        assert k > 0
        super().__init__(seed=seed)
        self._d = d
        self._k = k
        self._beta = self._make_beta()
        self._nonlinear = nonlinear

    @property
    def k(self) -> int:
        return self._k

    @property
    def d(self) -> int:
        return self._d

    def sample_x(self, n: int) -> Array:
        X = self._rng.uniform(low=0.8, high=3.2, size=(n, self.k))
        return X

    def sample(self, n: int, x: Optional[Array] = None) -> Tuple[Array, Array]:
        if x is None:
            X = self.sample_x(n=n)
        else:
            assert len(x.shape) == 2
            assert x.shape[0] == 1
            X = np.concatenate([x for _ in range(n)], axis=0)

        Z = (self._rng.uniform(size=(n,)) - 0.5) * 2
        one_dim_X = self._beta @ X.T
        Z = Z * np.pi / one_dim_X

        phi = (self._rng.uniform(size=(n,))) * (2 * np.pi)
        R = 0.1 * (self._rng.uniform(size=(n,)) - 0.5) * 2
        Y1 = Z + R * np.cos(phi)
        Y2 = (-np.cos(one_dim_X * Z) + 1) / 2 + R * np.sin(phi)

        if self._nonlinear:
            Y2 += np.sin(X.mean(axis=1))

        if self._d == 2:
            Y = np.stack([Y1, Y2], axis=1)
        elif self._d == 3:
            Y3 = np.sin(Z)
            Y = np.stack([Y1, Y2, Y3], axis=1)
        elif self._d == 4:
            Y3 = np.sin(Z)
            Y4 = np.cos(np.sin(Z) + R * np.sin(phi) * np.cos(phi))
            Y = np.stack([Y1, Y2, Y3, Y4], axis=1)
        else:
            raise NotImplementedError("Cond banana is implemented only for d=2,3,4.")

        return X, Y

    def _make_beta(self) -> Array:
        beta = self._rng.uniform(low=0, high=1, size=(self.k,))
        beta /= np.linalg.norm(beta, ord=1)
        return beta


def generate_x(dataset_name, n, k):
    if dataset_name == "cond_banana" or dataset_name == "cond_quad_banana":
        X = torch.FloatTensor(n, k).uniform_(0.8, 3.2)
    else:
        assert False
    return X


def get_k_dim_banana(n, k=10, X=None, is_nonlinear=False, d=2):
    assert d in [2, 3, 4]

    beta = torch.rand(k)
    beta /= beta.norm(p=1)
    banana_beta = beta

    if X is None:
        X = generate_x("cond_banana", n, k)
    else:
        assert len(X.shape) == 2
        X = X.clone().repeat(n, 1)

    X_to_output = X

    Z = (torch.rand(n) - 0.5) * 2
    one_dim_x = banana_beta @ X.T
    Z = Z * np.pi / one_dim_x

    phi = torch.rand(n) * (2 * np.pi)
    R = 0.1 * (torch.rand(n) - 0.5) * 2
    Y1 = Z + R * torch.cos(phi)
    Y2 = (-torch.cos(one_dim_x * Z) + 1) / 2 + R * torch.sin(phi)

    if is_nonlinear:
        tmp_X = X.clone()
        if len(X.shape) == 1:
            tmp_X = tmp_X.reshape(1, len(tmp_X))
        Y2 += torch.sin(tmp_X.mean(dim=1))

    decomposed = torch.cat([R.unsqueeze(-1), phi.unsqueeze(-1), Z.unsqueeze(-1)], dim=1)
    if d == 2:
        return Y1, Y2, X_to_output, decomposed
    elif d == 3:
        Y3 = torch.sin(Z)
        return Y1, Y2, Y3, X_to_output, decomposed
    elif d == 4:
        Y4 = torch.sin(Z)
        Y3 = torch.cos(torch.sin(Z)) + R * torch.sin(phi) * torch.cos(phi)
        return Y1, Y2, Y3, Y4, X_to_output, decomposed


def get_syn_data(
    dataset_name, get_decomposed=False, k=1, is_nonlinear=False, X=None, n=None
):
    if n is None:
        if is_nonlinear:
            n = 20000
        elif k < 5:
            n = 20000
        elif k < 25:
            n = 20000
        elif k <= 60:
            n = 80000
        else:
            n = 100000

    if dataset_name == "banana":
        T = (torch.rand(n) - 0.5) * 2
        phi = torch.rand(n) * (2 * np.pi)
        Z = torch.rand(n)
        R = 0.2 * Z * (1 + (1 - T.abs()) / 2)
        Y1 = T + R * torch.cos(phi)
        Y2 = T**2 + R * torch.sin(phi)
        decomposed = torch.cat(
            [T.unsqueeze(-1), phi.unsqueeze(-1), R.unsqueeze(-1)], dim=1
        )
        Ys = [Y1, Y2]
    elif dataset_name == "cond_banana" or dataset_name == "sin_banana":
        Y1, Y2, X, decomposed = get_k_dim_banana(n, k=k, is_nonlinear=is_nonlinear, X=X)
        Ys = [Y1, Y2]
    elif dataset_name == "cond_triple_banana":
        Y1, Y2, Y3, X, decomposed = get_k_dim_banana(
            n, k=k, is_nonlinear=is_nonlinear, d=3, X=X
        )
        Ys = [Y1, Y2, Y3]
    elif dataset_name == "cond_quad_banana":
        Y1, Y2, Y3, Y4, X, decomposed = get_k_dim_banana(
            n, k=k, is_nonlinear=is_nonlinear, d=4, X=X
        )
        Ys = [Y1, Y2, Y3, Y4]
    else:
        assert False

    Y = torch.cat([y.reshape(len(y), 1) for y in Ys], dim=1)

    if get_decomposed:
        return Y, X, decomposed

    return Y, X
