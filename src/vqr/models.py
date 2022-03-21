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
            # X = self.bn_hidden(diag(X_hidden @ X_hidden.T)[:, None] + X_hidden + X)
        X = self.bn_last(self.fc_last(X) + X_in)
        return X
