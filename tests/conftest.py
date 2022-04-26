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
    # Only supports 2d for now.
    U1, U2 = Us
    Q1, Q2 = Qs
    ii = jj = tuple(range(1, T))
    projections = []
    offending_projections = []

    for i0, j0 in it.product(ii, jj):
        u0 = np.array([U1[i0, j0], U2[i0, j0]])
        q0 = np.array([Q1[i0, j0], Q2[i0, j0]])

        for i1, j1 in it.product(ii, jj):
            u1 = np.array([U1[i1, j1], U2[i1, j1]])
            q1 = np.array([Q1[i1, j1], Q2[i1, j1]])
            du = u1 - u0
            dq = q1 - q0
            projection = np.dot(dq, du)

            # normalize projection to [-1, 1]
            # but only if it has any length (to prevent 0/0 -> NaN)
            if np.abs(projection) > 0:
                projection = projection / norm(dq) / norm(du)

            assert not np.isnan(projection)
            if projection < -projection_tolerance:
                offending_projections.append(projection.item())

            projections.append(projection)

    return offending_projections, projections
