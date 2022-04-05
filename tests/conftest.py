import os
import shutil
from pathlib import Path

import pytest
from _pytest.fixtures import FixtureRequest

from tests import TESTS_OUT_DIR
from experiments.logging import setup_logging

setup_logging()


@pytest.fixture(scope="function")
def test_out_dir(request: FixtureRequest) -> Path:
    """
    Creates a unique output directory in which a test can write some results.

    The directory will live under :obj:`TESTS_OUT_DIR` and be named according to the
    test class and test method.
    :return: A :class:`Path` representing the directory.
    """
    return _test_out_dir(request)


@pytest.fixture(scope="class")
def test_out_dir_class(request: FixtureRequest) -> Path:
    """
    Same as tests_out_dir but for class-level fixtures.
    """
    # For class-level dirs, we can't clear because multiple processes running tests
    # in the same class will remove each-others folders.
    return _test_out_dir(request, clear=False)


def _test_out_dir(request: FixtureRequest, clear: bool = True):
    filename, *classname_testname = str.split(request.node.nodeid, "::")
    out_dir = TESTS_OUT_DIR.joinpath(*classname_testname)

    # Clear any previous outputs to prevent any possibility of previous results to
    # affect the current run.
    if clear and out_dir.is_dir():
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    return out_dir
