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


class RegularizedDualVQRSolver(VQRSolver):
    """
    Solves the Regularized Dual formulation of Vector Quantile Regression using
    pytorch with gradient-based optimization.

    Can solve a non-linear VQR problem, given an arbitrary neural network that acts
    as a learnable non-linear transformation of the input features.
    """

    def __init__(
        self,
        epsilon: float = 1e-3,
        num_epochs: int = 1000,
        learning_rate: float = 0.9,
        verbose: bool = False,
        nn_init: Optional[Callable[[int], torch.nn.Module]] = None,
        full_precision: bool = False,
        gpu: bool = False,
        **solver_opts,
    ):
        """
        :param epsilon: Regularization. The lower, the more exact the solution.
        :param num_epochs: Number of epochs (full iterations over all data) to
        optimize for.
        :param learning_rate: Optimizer learning rate.
        :param verbose: Whether to print verbose output.
        :param nn_init: Function that initializes a neural net given the number of
        input features. Must be a callable that accepts a single int (the number of
        input features, k) and returns a torch.nn.Module which for an input of
        shape (N, k) produces an output of shape (N, k').
        :param full_precision: Whether to use full precision (float64) or only
        double precision (float32).
        :param gpu: Whether to perform optimization on GPU. Outputs will in any case
        be numpy arrays on CPU.
        :param solver_opts: Other opts for the base class.
        """
        super().__init__(verbose=verbose, **solver_opts)

        self._epsilon = epsilon
        self._num_epochs = num_epochs
        self._lr = learning_rate
        self._dtype = torch.float64 if full_precision else torch.float32
        self._device = torch.device("cuda") if gpu else torch.device("cpu")
        if nn_init is None:
            self._nn_init = lambda k: torch.nn.Identity()
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

        UY = U_th @ Y_th.T  # (T^d, N)
        psi = torch.full(size=(N, 1), fill_value=0.1, requires_grad=True, **dtd)
        X_th = None
        b = None
        net = None
        if k > 0:
            X_th = tensor(X, **dtd)

            # Instantiate custom neural network
            inner_net = self._nn_init(k)
            inner_net.to(**dtd)

            # Make sure the network produces the right output shape, and use it as k
            with torch.no_grad():
                inner_net.train(False)
                net_out_shape = inner_net(torch.zeros(1, k, **dtd)).shape
            if len(net_out_shape) != 2 or net_out_shape[0] != 1:
                raise ValueError(
                    "Invalid output shape from custom neural net: "
                    "expected (N, k_in) input to produce (N, k_out) output"
                )
            k_out = net_out_shape[1]

            # Add a non-trainable BatchNorm at the end
            net = torch.nn.Sequential(
                inner_net,
                torch.nn.BatchNorm1d(
                    num_features=k_out, affine=False, track_running_stats=True, **dtd
                ),
            )
            net.train(True)  # note: also applied to inner_net
            b = torch.zeros(*(Td, k_out), requires_grad=True, **dtd)

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

        # Finalize phi and calculate VQR coefficients A and B
        phi = _evaluate_phi()
        A = phi.detach().cpu().numpy()
        B = None
        x_transform_fn = None
        if k > 0:
            B = b.detach().cpu().numpy()

            # Finalize network: move to CPU, set eval mode, wrap with callable
            net = net.cpu().to(dtype=self._dtype)
            net.train(False)
            x_transform_fn = partial(
                self._features_transform, net=net, dtype=self._dtype
            )

        if self._verbose:
            print(f"{total_time=:.2f}s")

        return VectorQuantiles(T, d, U, A, B, X_transform=x_transform_fn)

    @staticmethod
    def _features_transform(X: Array, net: torch.nn.Module, dtype: torch.dtype):
        # Assumes net is on cpu and in eval mode.
        X_th = torch.from_numpy(X).to(dtype=dtype)
        return net(X_th).detach().numpy()
