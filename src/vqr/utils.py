import inspect
from typing import Sequence


def get_kwargs(ignore: Sequence[str] = ()) -> dict:
    """
    Assembles the arguments and kwargs of the caller.

    :param ignore: Keys to ignore.
    :return: The args/kwargs of the calling function as a dict.
    """
    frame = inspect.currentframe().f_back

    ignore_set = {"self", *ignore}
    arg_names, _, _, arg_values = inspect.getargvalues(frame)
    kwargs = {
        arg_name: arg_values[arg_name]
        for arg_name in arg_names
        if arg_name not in ignore_set
    }

    return kwargs
