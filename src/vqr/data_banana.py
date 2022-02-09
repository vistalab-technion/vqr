import numpy as np
import torch

banana_beta = None


def generate_x(dataset_name, n, k):
    if dataset_name == "cond_banana" or dataset_name == "cond_quad_banana":
        X = torch.FloatTensor(n, k).uniform_(1 - 0.2, 3 + 0.2)
    else:
        assert False
    return X


def get_k_dim_banana(n, k=10, X=None, is_nonlinear=False, d=2):
    assert d in [2, 3, 4]
    global banana_beta
    if banana_beta is None:
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
        Y2 = T ** 2 + R * torch.sin(phi)
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
        return X.numpy(), Y.numpy(), decomposed

    return X.numpy(), Y.numpy()
