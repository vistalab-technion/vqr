from typing import Union, Optional, Sequence

import torch
import torch.nn.functional as F
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
    "celu": nn.CELU,
}


def get_nl(nl: Union[str, nn.Module]):
    if isinstance(nl, nn.Module):
        non_linearity = nl
    else:
        if nl not in NLS:
            raise ValueError(f"got {nl=} but must be one of {[*NLS.keys()]}")
        non_linearity = NLS[nl]
    return non_linearity


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
        last_nl: Optional[Union[str, nn.Module]] = None,
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

        non_linearity = get_nl(nl)

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

        if last_nl is not None:
            layers.append(get_nl(last_nl)())

        self.fc_layers = nn.Sequential(*layers)
        self.skip = skip

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        z = self.fc_layers(x)

        if self.skip:
            z += x

        return z

    def init_weights(self):
        def _init_weights(mod: nn.Module):
            if isinstance(mod, nn.Linear):
                nn.init.constant_(mod.weight, val=0.01)
                nn.init.constant_(mod.bias, val=0.0)

        self.apply(_init_weights)


class DenseICNN(nn.Module):
    """Fully connected ICNN with input-quadratic skip connections"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (32, 32, 32),
        nl: str = "celu",
        dropout: float = 0.0,
        rank: int = 1,
        strong_convexity: float = 1e-6,
        quadratic: bool = False,
    ):
        super(DenseICNN, self).__init__()
        self.strong_convexity = strong_convexity
        self.activation = get_nl(nl)()
        self.unconstrained_layers = nn.ModuleList(
            [
                nn.Sequential(
                    ConvexQuadratic(in_dim, out_features, rank=rank, bias=True)
                    if quadratic
                    else nn.Linear(in_dim, out_features, bias=True),
                    nn.Dropout(dropout),
                )
                for out_features in hidden_dims
            ]
        )

        sizes = zip(hidden_dims[:-1], hidden_dims[1:])
        self.nonnegative_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, out_features, bias=False),
                    nn.Dropout(dropout),
                )
                for (in_features, out_features) in sizes
            ]
        )
        self.final_layer = nn.Linear(hidden_dims[-1], out_dim, bias=False)

    def forward(self, input_tensor: Tensor):
        self.convexify()

        output = self.unconstrained_layers[0](input_tensor)
        for quadratic_layer, convex_layer in zip(
            self.unconstrained_layers[1:], self.nonnegative_layers
        ):
            output = convex_layer(output) + quadratic_layer(input_tensor)
            output = self.activation(output)

        return self.final_layer(output) + 0.5 * self.strong_convexity * (
            input_tensor**2
        ).sum(dim=1).reshape(-1, 1)

    def convexify(self):
        layer: nn.Sequential
        for layer in self.nonnegative_layers:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    sublayer.weight.data.clamp_(0)
        self.final_layer.weight.data.clamp_(0)


class ConvexQuadratic(nn.Module):
    """Convex Quadratic Layer"""

    __constants__ = [
        "in_features",
        "out_features",
        "quadratic_decomposed",
        "weight",
        "bias",
    ]

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, rank: int = 1
    ):
        super(ConvexQuadratic, self).__init__()

        self.quadratic_decomposed = nn.Parameter(
            torch.Tensor(torch.randn(in_features, rank, out_features))
        )
        self.weight = nn.Parameter(torch.Tensor(torch.randn(out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input_tensor: Tensor):
        quad = (
            (
                input_tensor.matmul(
                    self.quadratic_decomposed.transpose(1, 0)
                ).transpose(1, 0)
            )
            ** 2
        ).sum(dim=1)
        linear = F.linear(input_tensor, self.weight, self.bias)
        return quad + linear
