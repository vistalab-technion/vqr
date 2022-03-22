from __future__ import annotations

from time import time
from typing import Union, Callable, Optional, Sequence
from functools import partial

import numpy as np
import torch
from torch import tensor
from numpy.typing import ArrayLike as Array
from torch.optim.lr_scheduler import ReduceLROnPlateau

from vqr import VQRSolver, VectorQuantiles
from vqr.vqr import quantile_levels
from vqr.models import MLP


class NonlinearRVQRDualLSESolver(VQRSolver):
    """
    Solves the Regularized Dual formulation of Vector Quantile Regression using
    torch with SGD as a solver backend.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        num_epochs: int = 1000,
        learning_rate: float = 0.9,
        verbose: bool = False,
        nn_hidden_layers: Optional[Sequence[int]] = None,
        nn_activation: Optional[Union[str, torch.nn.Module]] = None,
        nn_init: Optional[Callable[[int], torch.nn.Module]] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        **solver_opts,
    ):
        """

        :param epsilon:
        :param num_epochs:
        :param learning_rate:
        :param verbose:
        :param nn_hidden_layers: List of hidden layer dimensions. Will create an MLP
        with these hidden dims.
        :param nn_activation:
        :param nn_init:
        :param dtype:
        :param device:
        :param solver_opts:
        """
        super().__init__(verbose=verbose, **solver_opts)

        self._epsilon = epsilon
        self._num_epochs = num_epochs
        self._lr = learning_rate
        self._dtype = dtype
        self._device = device

        if nn_hidden_layers is not None and nn_init is not None:
            raise ValueError(
                f"Can specify either nn_hidden_layers or nn_init, not both"
            )
        if nn_hidden_layers is not None:
            self._nn_init = partial(
                MLP,
                hidden_dims=nn_hidden_layers,
                skip=True,
                bn=False,
                nl=nn_activation or "relu",
            )

        else:
            self._nn_init = nn_init

    def solve_vqr(self, T: int, Y: Array, X: Optional[Array] = None) -> VectorQuantiles:
        N = len(Y)
        Y = np.reshape(Y, (N, -1))

        if X is not None:
            X = np.reshape(X, (N, -1))

        d: int = Y.shape[1]  # number or target dimensions
        k: int = X.shape[1] if X is not None else 0  # Number of features (can be zero)

        # All quantile levels
        Td: int = T ** d

        # Quantile levels grid: list of grid coordinate matrices, one per dimension
        # d arrays of shape (T,..., T)
        U_grids: Sequence[Array] = np.meshgrid(*([quantile_levels(T)] * d))
        # Stack all nd-grid coordinates into one long matrix, of shape (T**d, d)
        U: Array = np.stack([U_grid.reshape(-1) for U_grid in U_grids], axis=1)
        assert U.shape == (Td, d)

        #####
        dtd = dict(dtype=self._dtype, device=self._device)
        epsilon = self._epsilon
        num_epochs = self._num_epochs

        one_N = torch.ones(N, 1)
        one_T = torch.ones(Td, 1)
        Y_th = tensor(Y, **dtd)
        U_th = tensor(U, **dtd)
        mu = tensor(one_T / Td, **dtd)
        nu = tensor(one_N / N, **dtd)

        if k > 0:
            X_th = tensor(X, **dtd)

            # Construct network and add batchnorm at the end
            if self._nn_init is not None:
                inner_net = self._nn_init(k)
            else:
                inner_net = torch.nn.Identity()
            inner_net.to(**dtd)
            out_dim = inner_net(torch.zeros(1, k, **dtd)).shape[1]
            net = torch.nn.Sequential(
                inner_net,
                torch.nn.BatchNorm1d(
                    num_features=out_dim, affine=False, track_running_stats=True, **dtd
                ),
            )
            b = torch.zeros(*(Td, out_dim), requires_grad=True, **dtd)
        else:
            X_th = None
            b = None
            net = None

        psi = torch.full(size=(N, 1), fill_value=0.1, requires_grad=True, **dtd)

        UY = U_th @ Y_th.T  # (T^d, N)

        optimizer = torch.optim.SGD(
            [
                dict(params=[*net.parameters()] if k > 0 else []),
                dict(
                    params=[b, psi] if k > 0 else [psi],
                ),
            ],
            lr=self._lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0,
        )

        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=100,
            threshold=5 * 0.01,  # loss needs to decrease by x% every patience epochs
            threshold_mode="rel",
            min_lr=self._lr * 0.5 ** 10,
            verbose=self._verbose,
        )

        def _evaluate_phi():
            bX = b @ net(X_th).T if k > 0 else 0
            max_arg = UY - bX - psi.reshape(1, -1)  # (T^d,N)-(T^d,N)-(1,N) = (T^d, N)
            max_val = torch.max(max_arg, dim=1, keepdim=True)[0]  # (T^d, 1)
            phi_ = (
                epsilon
                * torch.log(torch.sum(torch.exp((max_arg - max_val) / epsilon), dim=1))
                + max_val[:, 0]
            )
            return phi_.reshape(-1, 1)  # (T^d, 1)

        total_time, last_print_time = 0, time()
        for epoch_idx in range(num_epochs):
            epoch_start_time = time()

            optimizer.zero_grad()
            phi = _evaluate_phi()
            constraint_loss = phi.T @ mu
            objective = psi.T @ nu + constraint_loss
            objective.backward()
            optimizer.step()
            scheduler.step(objective)
            total_loss = objective.item()
            constraint_loss = constraint_loss.item()

            epoch_elapsed_time = time() - epoch_start_time
            total_time += epoch_elapsed_time
            if self._verbose and (epoch_idx % 100 == 0 or epoch_idx == num_epochs - 1):
                elapsed = time() - last_print_time
                print(
                    f"{epoch_idx=}, {total_loss=:.6f} {constraint_loss=:.6f}, "
                    f"{elapsed=:.2f}s"
                )
                last_print_time = time()

        # Finalize phi
        phi = _evaluate_phi()

        A = phi.detach().cpu().numpy()
        if k > 0:
            B = b.detach().cpu().numpy()

            # Finalize network
            net = net.cpu().to(dtype=self._dtype)
            net.eval()
            x_transform_fn = partial(
                self._features_transform, net=net, dtype=self._dtype
            )
        else:
            B = None
            x_transform_fn = None

        if self._verbose:
            print(f"{total_time=:.2f}s")

        return VectorQuantiles(T, d, U, A, B, X_transform=x_transform_fn)

    @staticmethod
    def _features_transform(X: Array, net: torch.nn.Module, dtype: torch.dtype):
        # Assumes net is on cpu and in eval mode.
        X_th = torch.from_numpy(X).to(dtype=dtype)
        return net(X_th).detach().numpy()
