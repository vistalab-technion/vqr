from __future__ import annotations

import sys
import logging
from copy import deepcopy
from time import time
from typing import Any, Union, Callable, Optional, Sequence
from functools import partial

import numpy as np
import torch
from numpy import ndarray as Array
from torch import Tensor, tensor
from tqdm.auto import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from vqr import VQRSolver, VQRSolution
from vqr.vqr import vector_quantile_levels
from vqr.models import MLP

_LOG = logging.getLogger(__name__)


class RegularizedDualVQRSolver(VQRSolver):
    """
    Solves the Regularized Dual formulation of Vector Quantile Regression using
    pytorch with gradient-based optimization.

    Can solve a non-linear VQR problem, given an arbitrary neural network that acts
    as a learnable non-linear transformation of the input features.
    """

    @classmethod
    def solver_name(cls) -> str:
        return "regularized_dual"

    def __init__(
        self,
        epsilon: float = 1e-3,
        num_epochs: int = 1000,
        lr: float = 0.9,
        lr_max_steps: int = 10,
        lr_factor: float = 0.5,
        lr_patience: int = 100,
        lr_threshold: float = 5 * 0.01,
        verbose: bool = False,
        nn_init: Optional[Callable[[int], torch.nn.Module]] = None,
        batchsize_y: Optional[int] = None,
        batchsize_u: Optional[int] = None,
        inference_batch_size: int = 1,
        full_precision: bool = False,
        gpu: bool = False,
        device_num: Optional[int] = None,
        post_iter_callback: Optional[Callable[[Any], None]] = None,
    ):
        """
        :param epsilon: Regularization. The lower, the more exact the solution.
        :param num_epochs: Number of epochs (full iterations over all data) to
        optimize for.
        :param lr: Optimizer learning rate.
        :param lr_max_steps: Maximum number of times lr will be adjusted using
        :class:`ReduceLROnPlateau` scheduler. Value of 0 (default) disables scheduling.
        :param lr_factor: Factor for adjusting lr using :class:`ReduceLROnPlateau`
        scheduler. Must be in (0, 1).
        :param lr_patience: patience of :class:`ReduceLROnPlateau`.
        :param lr_threshold: threshold of :class:`ReduceLROnPlateau` (using relative
        mode).
        :param verbose: Whether to print verbose output.
        :param nn_init: Function that initializes a neural net given the number of
        input features. Must be a callable that accepts a single int (the number of
        input features, k) and returns a torch.nn.Module which for an input of
        shape (N, k) produces an output of shape (N, k').
        :param batchsize_u: The batch size of quantile levels during training.
        If set to None, full batch will be used.
        :param batchsize_y: Batch size of samples during training. If set to None,
        full batch will be used.
        :param inference_batch_size: Batch size to be used for inference. Default is 1.
        :param full_precision: Whether to use full precision (float64) or only
        double precision (float32).
        :param gpu: Whether to perform optimization on GPU. Outputs will in any case
        be numpy arrays on CPU.
        :param device_num: the GPU number on which to run, used if gpu=True. If None,
        then no device will be specified and torch will choose automatically.
        :param post_iter_callback: Optional Callback to invoke after each
        optimization iteration (after each epoch if not using batches,
        or otherwise after each batch). It should accept kwargs.
        It will be passed the following kwargs:
        (solution, batch_loss, epoch_loss, epoch_idx, batch_idx, num_epochs,
        num_batches). The solution is a VQRSolution object containing the
        intermediate solution for the iteration on which the callback is invoked.
        """
        super().__init__()

        if not 0 < epsilon < 1:
            raise ValueError(f"invalid {epsilon=}, must be in (0, 1)")
        if not num_epochs > 0:
            raise ValueError(f"invalid {num_epochs=}, must be > 0")

        if not lr > 0:
            raise ValueError(f"invalid {lr=}, must be > 0")
        if not 0 < lr_factor < 1.0:
            raise ValueError(f"invalid {lr_factor=}, must be in (0, 1)")
        if not lr_patience > 0:
            raise ValueError(f"invalid {lr_patience=}, must be > 0")
        if not 0 <= lr_threshold <= 1:
            raise ValueError(f"invalid {lr_threshold=}, must be in [0,1]")
        if not lr_max_steps >= 0:
            raise ValueError(f"invalid {lr_max_steps=}, must be >= 0")

        if (batchsize_y and not batchsize_y > 0) or (
            batchsize_u and not batchsize_u > 0
        ):
            raise ValueError(f"invalid {batchsize_y=} or {batchsize_u}, must be > 0")

        self._verbose = verbose
        self._epsilon = epsilon
        self._num_epochs = num_epochs
        self._lr = lr
        self._lr_factor = lr_factor
        self._lr_patience = lr_patience
        self._lr_threshold = lr_threshold
        self._lr_max_steps = lr_max_steps
        self._dtype = torch.float64 if full_precision else torch.float32
        self._device = (
            torch.device("cuda" if device_num is None else f"cuda:{device_num}")
            if gpu and torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._dtd = dict(dtype=self._dtype, device=self._device)

        if nn_init is None:
            self._nn_init = self._default_nn_init
        else:
            self._nn_init = nn_init

        self._batchsize_y = batchsize_y
        self._batchsize_u = batchsize_u
        self._inference_batch_size = inference_batch_size
        self._callback: Optional[Callable] = post_iter_callback

    def solve_vqr(self, T: int, Y: Array, X: Optional[Array] = None) -> VQRSolution:
        start_time = time()
        log_level = logging.INFO if self._verbose else logging.NOTSET

        N = len(Y)
        Y = np.reshape(Y, (N, -1))

        if X is not None:
            X = np.reshape(X, (N, -1))

        d: int = Y.shape[1]  # number or target dimensions
        k: int = X.shape[1] if X is not None else 0  # Number of features (can be zero)

        # All quantile levels
        Td: int = T**d
        U: Array = vector_quantile_levels(T, d)
        assert U.shape == (Td, d)

        #####
        dtd = self._dtd
        epsilon = self._epsilon
        num_epochs = self._num_epochs

        Y_th = tensor(Y, **dtd)
        U_th = tensor(U, **dtd)

        psi = torch.full(size=(N, 1), fill_value=0.1, requires_grad=True, **dtd)
        X_th = None
        b = None
        net = None
        k_out = k
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
            # drop reference to inner_net to prevent memory being held after we move
            # net to RAM
            del inner_net
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
            factor=self._lr_factor,
            # loss needs to decrease by factor of 1-threshold every patience epochs
            # in min+rel mode: dynamic_threshold = best_loss * ( 1 - threshold )
            patience=self._lr_patience,
            threshold=self._lr_threshold,
            threshold_mode="rel",
            # when lr_max_steps==0, min_lr==lr so it will not be reduced.
            min_lr=self._lr * self._lr_factor**self._lr_max_steps,
            verbose=False,
        )

        _LOG.log(
            log_level,
            f"{self}: Solving with {N=}, {T=}, {d=}, {k=}, {k_out=}, "
            f"batchsize_y={self._batchsize_y}, batchsize_u={self._batchsize_u}",
        )

        final_loss = None
        with tqdm(
            total=num_epochs,
            file=sys.stdout,
            unit="epochs",
            disable=not self._verbose,
            mininterval=0.2,
            position=0,
            leave=True,
            ncols=100,
        ) as pbar:
            if self._batchsize_y or self._batchsize_u:
                final_loss = self._train_minibatch(
                    T, Y_th, U_th, psi, X_th, b, net, optimizer, scheduler, pbar
                )

            else:
                final_loss = self._train_fullbatch(
                    T, Y_th, U_th, psi, X_th, b, net, optimizer, scheduler, pbar
                )

        train_time = time() - start_time

        # Move data back to main memory, and free GPU memory for inference
        Y_th = Y_th.cpu()
        U_th = U_th.cpu()
        psi = psi.detach_().cpu()
        if k > 0:
            X_th = X_th.cpu()
            b = b.detach_().cpu()
            net = net.cpu()
            net.train(False)  # Finalize network
        torch.cuda.empty_cache()

        # Finalize phi and calculate VQR coefficients A and B
        phi = self._evaluate_phi_inference(Y_th, U_th, psi, epsilon, X_th, b, net)

        total_time = time() - start_time
        inference_time = total_time - train_time
        _LOG.log(log_level, f"{self}: total_time={total_time:.2f}s")

        return self._create_solution(
            T=T,
            d=d,
            k=k,
            U=U,
            phi=phi,
            b=b,
            net=net,
            solution_metrics=dict(
                train_time=train_time,
                inference_time=inference_time,
                total_time=total_time,
                final_loss=final_loss,
            ),
        )

    def _create_solution(
        self, T, d, k, U, phi, b, net, solution_metrics: dict = None
    ) -> VQRSolution:

        A = phi.detach().cpu().numpy()
        B = None
        x_transform_fn = None
        if k > 0:
            B = b.detach().cpu().numpy()

            # Create a copy of the net on CPU, without moving the original net to CPU
            # Note: deepcopy seems to create a copy of all tensors on cpu, but calling
            # cpu() to make sure.
            net_copy: torch.nn.Module = deepcopy(net).cpu()
            x_transform_fn = partial(
                self._features_transform, net=net_copy, dtype=self._dtype
            )

        return VQRSolution(
            T,
            d,
            U,
            A,
            B,
            X_transform=x_transform_fn,
            k_in=k,
            solution_metrics=solution_metrics,
        )

    @staticmethod
    def _features_transform(
        X: Array, net: torch.nn.Module, dtype: torch.dtype
    ) -> Array:
        # Assumes net is on cpu and in eval mode.
        X_th = torch.from_numpy(X).to(dtype=dtype)
        with torch.no_grad():
            return net(X_th).numpy()

    def _train_fullbatch(
        self,
        T: int,
        Y: Tensor,
        U: Tensor,
        psi: Tensor,
        X: Tensor,
        b: Tensor,
        net: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        pbar: tqdm,
    ) -> float:
        N = len(Y)
        Td = len(U)
        d: int = Y.shape[1]
        k: int = X.shape[1] if X is not None else 0

        mu = torch.ones(Td, 1, **self._dtd) / Td
        nu = torch.ones(N, 1, **self._dtd) / N

        UY = U @ Y.T  # (T^d, N)

        epsilon = self._epsilon
        objective = tensor(float("nan"))
        for epoch_idx in range(self._num_epochs):

            # Optimize
            optimizer.zero_grad()
            phi = self._evaluate_phi(Y, U, psi, epsilon, X, b, net, UY)
            constraint_loss = phi.T @ mu
            objective = psi.T @ nu + constraint_loss
            objective.backward()
            optimizer.step()
            scheduler.step(objective)

            # Update progress and stats
            pbar.update(1)
            pbar.set_postfix(
                total_loss=objective.item(),
                constraint_loss=constraint_loss.item(),
                lr=optimizer.param_groups[0]["lr"],
                refresh=False,
            )

            # Invoke callback
            if self._callback:
                self._callback(
                    solution=self._create_solution(T, d, k, U, phi, b, net),
                    batch_loss=None,
                    epoch_loss=objective.item(),
                    epoch_idx=epoch_idx,
                    batch_idx=None,
                    num_epochs=self._num_epochs,
                    num_batches=None,
                    xy_slice=None,
                    u_slice=None,
                )

        return objective.item()

    def _train_minibatch(
        self,
        T: int,
        Y: Tensor,
        U: Tensor,
        psi: Tensor,
        X: Tensor,
        b: Tensor,
        net: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        pbar: tqdm,
    ) -> float:
        N = len(Y)
        Td = len(U)
        d: int = Y.shape[1]
        k: int = X.shape[1] if X is not None else 0

        epsilon = self._epsilon

        def _num_batches(num_samples, batch_size):
            if not batch_size:
                batch_size = num_samples
            return int(np.ceil(num_samples / batch_size))

        def _yield_batches(num_samples, batch_size):
            if not batch_size or batch_size > num_samples:
                batch_size = num_samples

            while True:
                idx = np.concatenate(
                    [
                        np.random.permutation(num_samples),
                        np.random.permutation(num_samples),
                    ],
                    axis=0,
                )

                for batch_idx in range(_num_batches(num_samples, batch_size)):
                    batch_slice = idx[
                        (batch_size * batch_idx) : (batch_size * (batch_idx + 1))
                    ]

                    if batch_size == num_samples:
                        batch_slice = sorted(batch_slice)

                    assert len(batch_slice) == batch_size
                    yield batch_slice

        num_batches_xy = _num_batches(N, self._batchsize_y)
        num_batches_u = _num_batches(Td, self._batchsize_u)
        num_batches_per_epoch = max(num_batches_xy, num_batches_u)

        total_objective = tensor(float("nan"))
        for epoch_idx in range(self._num_epochs):

            total_objective = tensor(0.0, **self._dtd)

            for batch_idx, xy_slice, u_slice in zip(
                range(num_batches_per_epoch),
                _yield_batches(N, self._batchsize_y),
                _yield_batches(Td, self._batchsize_u),
            ):
                Y_batch = Y[xy_slice]
                U_batch = U[u_slice]
                psi_batch = psi[xy_slice, :]
                mu_batch = torch.ones(len(U_batch), 1, **self._dtd) / len(U_batch)
                nu_batch = torch.ones(len(Y_batch), 1, **self._dtd) / len(Y_batch)
                X_batch, b_batch = None, None
                if X is not None:
                    X_batch = X[xy_slice]
                    b_batch = b[u_slice, :]

                # Optimize
                optimizer.zero_grad()
                phi_batch = self._evaluate_phi(
                    Y_batch, U_batch, psi_batch, epsilon, X_batch, b_batch, net, UY=None
                )
                constraint_loss = phi_batch.T @ mu_batch
                objective = psi_batch.T @ nu_batch + constraint_loss
                objective.backward()
                optimizer.step()

                total_objective = total_objective + objective

                # Invoke callback
                if self._callback:
                    if not self._batchsize_u:
                        phi_all_levels = phi_batch
                    else:
                        # In the case of U-batches, we need to compute phi with all
                        # T^d levels, otherwise we can't create a valid solution object
                        with torch.no_grad():
                            phi_all_levels = self._evaluate_phi(
                                Y_batch, U, psi_batch, epsilon, X_batch, b, net, UY=None
                            )
                    self._callback(
                        solution=self._create_solution(
                            T, d, k, U, phi_all_levels, b, net
                        ),
                        batch_loss=objective.item(),
                        epoch_loss=total_objective.item() / (batch_idx + 1),
                        epoch_idx=epoch_idx,
                        batch_idx=batch_idx + epoch_idx * num_batches_per_epoch,
                        num_epochs=self._num_epochs,
                        num_batches=num_batches_per_epoch * self._num_epochs,
                        xy_slice=xy_slice,
                        u_slice=u_slice,
                    )

            total_objective /= num_batches_per_epoch
            scheduler.step(total_objective)

            # Update progress and stats
            pbar.update(1)
            pbar.set_postfix(
                total_loss=total_objective.item(),
                lr=optimizer.param_groups[0]["lr"],
                batch=(
                    f"{batch_idx + epoch_idx * num_batches_per_epoch}/"
                    f"{num_batches_per_epoch * self._num_epochs}"
                ),
                refresh=False,
            )

        return total_objective.item()

    @staticmethod
    def _evaluate_phi(
        Y: Tensor,
        U: Tensor,
        psi: Tensor,
        epsilon: float,
        X: Tensor,
        b: Tensor,
        net: torch.nn.Module,
        UY: Optional[Tensor] = None,
    ):
        """
        Calculates the phi optimization variable of the VQR objective.

        Phi is defined for quantile level i as follows:
            phi_i = max_j { u_i y_j - b_i x_j - psi_j }

        This implementation uses a log-sum-exp reduction with
        epsilon-smoothing instead of taking max_j.
        """
        if UY is None:
            UY = U @ Y.T  # (T^d, N)

        bX = 0
        if X is not None:
            bX = b @ net(X).T

        max_arg = UY - bX - psi.reshape(1, -1)  # (T^d,N)-(T^d,N)-(1,N) = (T^d, N)
        phi = epsilon * torch.logsumexp(max_arg / epsilon, dim=1)

        return phi.reshape(-1, 1)  # (T^d, 1)

    def _evaluate_phi_inference(
        self,
        Y: Tensor,
        U: Tensor,
        psi: Tensor,
        epsilon: float,
        X: Tensor,
        b: Tensor,
        net: torch.nn.Module,
    ):
        Td, d = U.shape
        N, _ = Y.shape

        phi_full = torch.empty(Td, 1, dtype=self._dtype, device="cpu")
        num_batches_us = int(np.ceil(Td / self._inference_batch_size))
        idx = np.arange(Td)
        Y = Y.to(self._device)
        psi = psi.to(self._device)
        X = X.to(self._device) if X is not None else None
        net = net.to(self._device) if net is not None else None

        with torch.no_grad():
            for batch_idx in range(num_batches_us):
                batch_slice = idx[
                    self._inference_batch_size
                    * batch_idx : min(self._inference_batch_size * (batch_idx + 1), Td)
                ]
                U_batch = U[batch_slice].to(self._device)
                b_batch = b[batch_slice].to(self._device) if b is not None else None
                phi = self._evaluate_phi(
                    Y,
                    U_batch,
                    psi,
                    epsilon,
                    X,
                    b_batch,
                    net,
                    UY=None,
                )
                phi = phi.cpu().detach()
                phi_full[batch_slice] = phi

        return phi_full.reshape(-1, 1)

    def _default_nn_init(self, *args, **kwargs):
        """
        Default neural-net init function. For internal use.
        """
        return torch.nn.Identity()

    def __getstate__(self):
        # Prevent serialization of callback if the solver is serialized.
        state = self.__dict__.copy()
        state["_callback"] = None
        return state

    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self._epsilon:.0e})"


class MLPRegularizedDualVQRSolver(RegularizedDualVQRSolver):
    """
    Same as RegularizedDualVQRSolver, but with a general-purpose MLP as a
    learnable non-linear feature transformation.
    """

    @classmethod
    def solver_name(cls) -> str:
        return "regularized_dual_mlp"

    def __init__(
        self,
        hidden_layers: Union[str, Sequence[int]] = (32,),
        activation: Union[str, torch.nn.Module] = "relu",
        skip: bool = True,
        batchnorm: bool = False,
        dropout: float = 0,
        **solver_opts,
    ):
        """
        Supports init args of both :class:`MLP` and :class:`RegularizedDualVQRSolver`.
        """
        if "nn_init" in solver_opts:
            raise ValueError("Can't provide nn_init to this solver")

        super().__init__(
            nn_init=partial(
                MLP,
                hidden_dims=hidden_layers,
                nl=activation,
                skip=skip,
                batchnorm=batchnorm,
                dropout=dropout,
            ),
            **solver_opts,
        )
