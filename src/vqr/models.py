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


class DeepNet(nn.Module):
    def __init__(self, hidden_width=2000, depth=1, k=2):
        super().__init__()
        self.nl = nn.ReLU()
        self.fc_first = nn.Linear(k, hidden_width)
        self.fc_last = nn.Linear(hidden_width, k)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_width, hidden_width) for _ in range(depth)]
        )
        self.bn_last = nn.BatchNorm1d(
            num_features=k, affine=False, track_running_stats=True
        )
        self.bn_hidden = nn.BatchNorm1d(
            num_features=hidden_width, affine=False, track_running_stats=True
        )

    def forward(self, X_in):
        X = self.nl(self.fc_first(X_in))

        for hidden in self.fc_hidden:
            X_hidden = self.nl(hidden(X))
            X = self.bn_hidden(X_hidden + X)

        X = self.bn_last(self.fc_last(X) + X_in)
        return X


NLS = {
    "relu": torch.nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
}


class MLP(nn.Module):
    """
    A Simple MLP.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        nl: Union[str, nn.Module] = "relu",
        skip: bool = False,
        bn: bool = False,
    ):
        """
        :param in_dim: Input feature dimension.
        :param hidden_dims: Hidden dimensions. Will be converted to FC layers.
        :param nl: Nonlinearity to use between FC layers.
        :param skip: Whether to use a skip-connection (over all layers).
        :param bn: Whether to use batchnorm (single one at the end, after the skip
        connection).
        """
        super().__init__()

        if skip and hidden_dims[-1] != in_dim:
            hidden_dims = [*hidden_dims, in_dim]

        all_dims = [in_dim, *hidden_dims]

        if isinstance(nl, nn.Module):
            non_linearity = nl
        else:
            non_linearity = NLS[nl]

        layers = []
        for d1, d2 in zip(all_dims[:-1], all_dims[1:]):
            layers += [
                nn.Linear(d1, d2, bias=True),
                non_linearity(),
            ]

        self.fc_layers = nn.Sequential(*layers[:-1])

        self.bn = None
        if bn:
            self.bn = nn.BatchNorm1d(
                num_features=all_dims[-1], affine=False, track_running_stats=True
            )

        self.skip = skip

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        z = self.fc_layers(x)

        if self.skip:
            z += x

        if self.bn is not None:
            z = self.bn(z)

        return z
