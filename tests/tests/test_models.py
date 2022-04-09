import torch
import pytest

from vqr.models import MLP


class TestMLP(object):
    @pytest.fixture(autouse=True)
    def setup(self):
        pass

    @pytest.mark.parametrize(
        "in_dim, hidden_dims, nl, skip, batchnorm, dropout",
        [
            [3, [5], "relu", False, True, 0.1],
            [3, [5, 10], "relu", False, True, 0.1],
            [3, [5, 10, 3], "relu", True, True, 0.1],
            [3, [5, 10, 5], "relu", True, True, 0.1],
            [3, "5, 10,5", "relu", True, True, 0.1],
            #
            [3, [5], "relu", False, True, 0],
            [3, [5, 10], "relu", False, True, 0],
            [3, [5, 10, 3], "relu", True, True, 0],
            [3, [5, 10, 5], "relu", True, True, 0],
            [3, ",5,10, 5 ", "relu", True, True, 0],
            #
            [3, [5], "tanh", False, False, 0.1],
            [3, [5, 10], "tanh", False, False, 0.1],
            [3, [5, 10, 3], "tanh", True, False, 0.2],
            [3, [5, 10, 5], "tanh", True, False, 0.2],
            #
            [3, [5], "tanh", False, False, 0],
            [3, [5, 10], "tanh", False, False, 0],
            [3, [5, 10, 3], "tanh", True, False, 0],
            [3, [5, 10, 5], "tanh", True, False, 0],
            [3, "5,10,5", "tanh", True, False, 0],
        ],
    )
    def test_mlp(self, in_dim, hidden_dims, nl, skip, batchnorm, dropout):
        mlp = MLP(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            nl=nl,
            skip=skip,
            batchnorm=batchnorm,
            dropout=dropout,
        )

        N = 100
        x = torch.randn(N, in_dim)
        y = mlp(x)

        print(mlp)
        assert isinstance(mlp.fc_layers[-1], torch.nn.Linear)
        if skip:
            assert y.shape == (N, in_dim)
        else:
            assert y.shape == (N, hidden_dims[-1])

    @pytest.mark.parametrize(
        "in_dim, hidden_dims, nl, skip, batchnorm, dropout, error_msg",
        [
            # no hidden dims
            [3, [], "relu", False, True, 0.1, "hidden_dims="],
            [3, None, "relu", False, True, 0.1, "hidden_dims="],
            # invalid hidden dims
            [3, "", "relu", False, True, 0.1, "hidden_dims is a string"],
            [3, ",", "relu", False, True, 0.1, "hidden_dims is a string"],
            [3, "1,2,a,3", "relu", False, True, 0.1, "hidden_dims is a string"],
            # unknown nl
            [3, [5], "wtf", False, True, 0.1, "nl="],
            # invalid dropout
            [3, [5], "relu", False, True, -0.1, "dropout="],
            [3, [5], "relu", False, True, 1.0, "dropout="],
            [3, [5], "relu", False, True, 1.1, "dropout="],
        ],
    )
    def test_init_exceptions(
        self, in_dim, hidden_dims, nl, skip, batchnorm, dropout, error_msg
    ):
        with pytest.raises(ValueError, match=error_msg):
            MLP(
                in_dim=in_dim,
                hidden_dims=hidden_dims,
                nl=nl,
                skip=skip,
                batchnorm=batchnorm,
                dropout=dropout,
            )
