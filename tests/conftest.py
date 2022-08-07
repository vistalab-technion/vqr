import os
import shutil
import itertools as it
from typing import Tuple, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.linalg import norm
from _pytest.fixtures import FixtureRequest

from tests import TESTS_OUT_DIR
from vqr.vqr import check_comonotonicity, decode_quantile_grid, vector_quantile_levels
from experiments.log import setup_logging

setup_logging()


@pytest.fixture(scope="function")
def test_out_dir(request: FixtureRequest) -> Path:
    """
    Creates a unique output directory in which a test can write some results.
    It will be cleared if it existed before.

    The directory will live under :obj:`TESTS_OUT_DIR` and be named according to the
    test class and test method.
    :return: A :class:`Path` representing the directory.
    """
    return _test_out_dir(request)


@pytest.fixture(scope="class")
def test_out_dir_class(request: FixtureRequest) -> Path:
    """
    Same as tests_out_dir but for class-level fixtures.

    Note: the returned dir is not cleared!
    For class-level dirs, we can't clear because multiple processes running tests
    in the same class will remove each-others folders.
    """
    return _test_out_dir(request, clear=False)


@pytest.fixture(scope="class")
def test_out_dir_class_pid(request: FixtureRequest) -> Path:
    """
    Same as tests_out_dir but for class-level fixtures.
    A separate out dir will be created for every test process.
    It will be cleared if it existed before.
    """
    return _test_out_dir(request, clear=True, with_pid=True)


def _test_out_dir(request: FixtureRequest, clear: bool = True, with_pid: bool = False):
    filename, *classname_testname = str.split(request.node.nodeid, "::")
    out_dir = TESTS_OUT_DIR.joinpath(*classname_testname)
    if with_pid:
        out_dir = out_dir.joinpath(f"pid_{os.getpid()}")

    # Clear any previous outputs to prevent any possibility of previous results to
    # affect the current run.
    if clear and out_dir.is_dir():
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _test_monotonicity(
    Us: Sequence[np.ndarray],
    Qs: Sequence[np.ndarray],
    T: int,
    projection_tolerance: float = 0.0,
    offending_proportion_limit: float = 0.005,
):
    offending_projections, projections = monotonicity_offending_projections(
        Qs, Us, T, projection_tolerance
    )
    n_c, n = len(offending_projections), len(projections)

    offending_proportion = n_c / n
    if offending_projections:
        q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        print(f"err quantiles: {np.quantile(offending_projections, q=q)}")
        print(f"all quantiles: {np.quantile(projections, q=q)}")
        print(f"{n=}, {n_c=}, {n_c/n=}")

    assert offending_proportion < offending_proportion_limit


def monotonicity_offending_projections(
    Qs: Sequence[np.ndarray],
    Us: Sequence[np.ndarray],
    T: int,
    projection_tolerance: float,
) -> Tuple[Sequence[float], Sequence[float]]:
    assert len(Qs) == len(Us)

    pairwise_comonotonicity_mat = check_comonotonicity(T=T, d=len(Qs), Qs=Qs, Us=Us)
    offending_projections = pairwise_comonotonicity_mat[
        np.where(np.triu(pairwise_comonotonicity_mat) < projection_tolerance)
    ].tolist()
    projections = np.triu(pairwise_comonotonicity_mat).ravel().tolist()

    return offending_projections, projections
