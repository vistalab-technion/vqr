from typing import Union, Optional

import torch
from numpy import ndarray
from torch import Tensor


def ensure_numpy(x: Union[ndarray, Tensor]) -> ndarray:
    """
    Ensures that the input tensor is a numpy array, and converts it if not.
    :param x: Either a numpy array or a torch tensor.
    :return: The input as a numpy array.
    """
    if isinstance(x, Tensor):
        return x.clone().detach().numpy()
    elif isinstance(x, ndarray):
        return x
    else:
        raise ValueError(f"Unexpected type, {type(x)=}")


def ensure_torch(
    x: Union[ndarray, Tensor],
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
) -> Tensor:
    """
    Ensures that the input tensor is a torch tensor of a specified type on a
    specified device, and converts it if not.
    :param x: Either a numpy array or a torch tensor.
    :param dtype: The datatype of the tensor to return. Will be converted if necessary.
    :param device: The device on which the result should be. Will be copied if
    necessary.
    :return: The input as a torch tensor with the specified type and device.
    """
    if isinstance(x, Tensor):
        return x.to(dtype=dtype, device=device)
    elif isinstance(x, ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    else:
        raise ValueError(f"Unexpected type, {type(x)=}")
