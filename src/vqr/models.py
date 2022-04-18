from typing import Union, Sequence

import torch
from numpy import array
from torch import Tensor, nn, eye, diag
from torch import ones as ones_th


class QuadraticModel(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.C1 = nn.Parameter(eye(k, k, dtype=torch.float32, requires_grad=True))
        self.C2 = nn.Parameter(
            ones_th(k, dtype=torch.float32, requires_grad=True),
        )
        self.C3 = nn.Parameter(
            torch.tensor(array([0.0]), dtype=torch.float32, requires_grad=True)
        )
        self.batch_norm = nn.BatchNorm1d(
            num_features=k, affine=False, track_running_stats=True
        )

    def forward(self, X: Tensor):
        return self.batch_norm(diag(X @ self.C1 @ X.T)[:, None] + self.C2 * X + self.C3)


NLS = {
    "relu": torch.nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "silu": nn.SiLU,
    "lrelu": nn.LeakyReLU,
}


class MLP(nn.Module):
    """
    A Simple MLP.

    Structure is:
    FC -> [BN] -> ACT -> [DROPOUT] -> ... FC -> [BN] -> ACT -> [DROPOUT] -> FC

    Note that BatchNorm and Dropout are optional and the MLP always ends with an FC.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Union[str, Sequence[int]],
        nl: Union[str, nn.Module] = "silu",
        skip: bool = False,
        batchnorm: bool = False,
        dropout: float = 0,
    ):
        """
        :param in_dim: Input feature dimension.
        :param hidden_dims: Hidden dimensions. Will be converted to FC layers. Last
        entry is the output dimension. If a string is provided, it will be parsed as
        a comma-separated list of integer values, e.g. '12,34,46,7'.
        :param nl: Non-linearity to use between FC layers.
        :param skip: Whether to use a skip-connection (over all layers). This
        requires that in_dim==out_dim, thus if skip==True and hidden_dims[-1]!=in_dim
        then the last hidden layer will be changed to produce an output of size in_dim.
        :param batchnorm: Whether to use Batch Normalization
        (before each non-linearity).
        :param dropout: Whether to use dropout (after each non-linearity).
        Zero means no dropout, otherwise means dropout probability.
        """
        super().__init__()

        if isinstance(hidden_dims, str):
            try:
                hidden_dims = tuple(map(int, hidden_dims.strip(", ").split(",")))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "When hidden_dims is a string it must be a comma-separated "
                    "sequence of integers, e.g. '11,22,34,45'"
                ) from e

        if not hidden_dims:
            raise ValueError(f"got {hidden_dims=} but must have at least one")

        if isinstance(nl, nn.Module):
            non_linearity = nl
        else:
            if nl not in NLS:
                raise ValueError(f"got {nl=} but must be one of {[*NLS.keys()]}")
            non_linearity = NLS[nl]

        if not 0 <= dropout < 1:
            raise ValueError(f"got {dropout=} but must be in [0, 1)")

        # Split output dimension from the hidden dimensions
        *hidden_dims, out_dim = hidden_dims
        if skip and out_dim != in_dim:
            out_dim = in_dim

        layers = []
        fc_dims = [in_dim, *hidden_dims]
        for d1, d2 in zip(fc_dims[:-1], fc_dims[1:]):
            layers.append(nn.Linear(d1, d2, bias=True))
            if batchnorm:
                layers.append(nn.BatchNorm1d(num_features=d2))
            layers.append(non_linearity())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        # Always end with FC
        layers.append(nn.Linear(fc_dims[-1], out_dim, bias=True))

        self.fc_layers = nn.Sequential(*layers)
        self.skip = skip

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        z = self.fc_layers(x)

        if self.skip:
            z += x

        return z
