from __future__ import division

import os
import re
import sys
from math import ceil
from typing import Any, Dict, Tuple, Union, Iterator, Optional, Sequence
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pyhrv
import pandas as pd
import matplotlib.pyplot as plt
from pyhrv.utils import sec_to_time, standardize_rri_trr
from numpy.typing import ArrayLike as Array
from pyhrv.wfdb.rri import ecgrr
from matplotlib.axes import Axes
from pyhrv.rri.processing import filtrr
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import train_test_split

import vqr
from vqr import VectorQuantileEstimator, VectorQuantileRegressor
from vqr.plot import plot_coverage_2d

DATASET_BASE_PATH = Path("/Users/avivr/dev/_datasets/physionet")
assert DATASET_BASE_PATH.is_dir()


import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
_LOG = logging.getLogger(__name__)


@dataclass
class WFDBRecord:
    """
    Represents a single WFDB record.
    """

    path: Path
    dataset_name: str
    ann_ext: str = None

    @property
    def name(self) -> str:
        return f"{self.dataset_name}/{self.path.name}"


@dataclass
class WFDBDataset:
    """
    Represents a WFDB dataset.
    """

    name: str
    path: Path = None
    ann_ext: str = None

    def __post_init__(self):
        if not self.path:
            self.path = DATASET_BASE_PATH.joinpath(self.name)

        self._record_paths = tuple(
            self.path.joinpath(dat_path.stem) for dat_path in self.path.glob("*.dat")
        )

    @property
    def records(self) -> Sequence[WFDBRecord]:
        return tuple(
            WFDBRecord(dataset_name=self.name, ann_ext=self.ann_ext, path=path)
            for path in self._record_paths
        )

    def __len__(self):
        return len(self._record_paths)

    def __iter__(self) -> Iterator[WFDBRecord]:
        return iter(self.records)


DATASETS = {
    ds.name: ds
    for ds in [
        WFDBDataset(name="nsrdb", ann_ext="atr"),
        WFDBDataset(name="chfdb", ann_ext="ecg"),
        WFDBDataset(name="afdb", ann_ext="qrs"),
    ]
}

NSRDB = DATASETS["nsrdb"]
CHFDB = DATASETS["chfdb"]
AFDB = DATASETS["afdb"]


def plot_rec(
    rec: WFDBRecord,
    from_time: str = None,
    to_time: str = None,
    ax: Axes = None,
):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), squeeze=True)
    else:
        fig = ax.figure

    trr, xrr = ecgrr(
        rec_path=str(rec.path),
        ann_ext=rec.ann_ext,
        from_time=from_time,
        to_time=to_time,
    )
    trr, xrr = filtrr(trr, xrr, enable_range=True, enable_moving_average=False)
    ax.plot(trr, xrr, label=rec.name)
    ax.set_ylim([0.2, 2.0])
    ax.legend()
    ax.grid("on")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("RR [s]")
    return fig, ax


def chunk_data(
    data: np.ndarray,
    window_size: int,
    overlap_size: int = 0,
    flatten_inside_window: bool = True,
):
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (
        num_windows * window_size - (num_windows - 1) * overlap_size
    )

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros(
            (
                num_windows * window_size - (num_windows - 1) * overlap_size,
                data.shape[1],
            )
        )
        newdata[: data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = as_strided(
        data,
        shape=(num_windows, window_size * data.shape[1]),
        strides=((window_size - overlap_size) * data.shape[1] * sz, sz),
    )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows, -1, data.shape[1]))


def split_wfdb(
    rec: WFDBRecord,
    d: int = 2,
    k: Union[int, float] = 20.0,
    window_overlap: float = 0.9,
    gap_delta: Union[int, float] = 0,
    dilation_x: Union[int, float] = 1,
    dilation_y: Union[int, float] = 1,
    validation_proportion: float = 0.0,
) -> Tuple[Array, Array, Optional[Array], Optional[Array]]:
    """
    Loads a WFDB record and splits it into pairs of covariate and target windows (X,Y).

    :param rec: WFDB record.
    :param d: Dimension of the response variable (Y).
    :param k: Dimension or duration of the covariates (X). If int, means number of
    samples and then will exactly correspond to dimension. If float, means duration
    in seconds, and will be converted to number of samples based on mean interval
    duration. Note that if dilation_x>1 then they will represent a longer temporal
    duration by a factor of dilation_x.
    :param window_overlap: Percent overlap between windows used to create XY samples.
    :param gap_delta: Gap between covariates and targets. If int will be interpreted
    as samples; otherwise as number of seconds. Zero means targets start immediately
    after covariates, i.e. no gap.
    :param dilation_x: Dilation of the covariates in time. E.g. if k=25 and
    dilation_x=2, then k=25 points will be taken from each window of 50=25*5 points,
    such that there's a gap of one point between each two points taken.
    :param dilation_y: Same as above but for the targets.
    :param validation_proportion: Proportion of data windows to designate for
    validation. Zero means that no validation set will be created.
    :return: A tuple (X_train, Y_train, X_valid, Y_valid) where the X's are of shape
    (N, k) and the Y's are of shape (N, d). If validation_proportion is set to zero,
    then X_valid and Y_valid will be None.
    """

    if not (d > 0 and k > 0):
        raise ValueError(f"Invalid dimensions {d=}, {k=}, both must be > 0")
    if not 0 < window_overlap <= 1.0:
        raise ValueError(f"Invalid {window_overlap=}, must be in (0,1]")
    if not gap_delta >= 0:
        raise ValueError(f"Invalid {gap_delta=}, must be >= 0")
    if not (dilation_x >= 1 and dilation_y >= 1):
        raise ValueError(f"Invalid {dilation_x=}, {dilation_y=}, both must be >= 1")
    if not 0 <= validation_proportion < 1.0:
        raise ValueError(f"Invalid {validation_proportion=}, must be in [0, 1)")

    # Load record data
    trr, xrr = ecgrr(rec.path, rec.ann_ext, from_time=None, to_time=None)
    xrr, trr = standardize_rri_trr(rri=xrr, trr=trr)

    # Pre-process, but only to remove non-physiological intervals
    trr, xrr = filtrr(trr, xrr, enable_range=True, enable_moving_average=False)
    assert len(trr) == len(xrr)

    n = len(xrr)
    rec_duration = sec_to_time(trr[-1])

    def _sec_to_samples(s: Union[int, float]):
        if isinstance(s, int):
            # Assuming it's in samples
            return s

        # Assuming it's in seconds: convert to samples by dividing by mean RR
        # interval duration in the training data.
        mean_rr = np.mean(xrr[: int((1 - validation_proportion) * n)])  # sec / sample
        return int(s / mean_rr) + 1

    k = _sec_to_samples(k)
    gap_delta = _sec_to_samples(gap_delta)
    dilation_x = _sec_to_samples(dilation_x)
    dilation_y = _sec_to_samples(dilation_y)

    window_size = d * dilation_y + k * dilation_x + gap_delta
    overlap_size = int((window_size - 1) * window_overlap)
    XY = chunk_data(xrr, window_size=window_size, overlap_size=overlap_size)

    X, Y = (
        XY[:, : k * dilation_x : dilation_x],
        XY[:, k * dilation_x + gap_delta :: dilation_y],
    )
    _LOG.info(
        f"{rec.path!s}: Loaded {len(xrr)} intervals ({rec_duration}), {len(X)} windows "
        f"[{d=}, {k=}, {gap_delta=}, dx={dilation_x}, dy={dilation_y}]"
    )
    if validation_proportion > 0:
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X, Y, test_size=validation_proportion, shuffle=False
        )
        _LOG.info(
            f"{rec.path!s}: Split into {len(X_train)} train, {len(X_valid)} "
            f"validation windows"
        )
        return X_train, Y_train, X_valid, Y_valid
    else:
        return X, Y, None, None


def _single_record_exp(
    rec: WFDBRecord,
    T: int = 25,
    split_opts: Dict[str, Any] = {},
    vqr_solver_opts: Dict[str, Any] = {},
):
    """

    :param rec: WFDBRecord to load.
    :param T: Number of quanltile levels for VQR.
    :param split_opts: Kwargs for split_wfdb.
    :param vqr_solver_opts: Kwargs for the VQR solver.
    """

    # Load and split the data
    X_train, Y_train, X_valid, Y_valid = split_wfdb(rec, **split_opts)

    vqr = VectorQuantileRegressor(n_levels=T, solver_opts=vqr_solver_opts)
    vqr.fit(X_train, Y_train)
    vqr_samples = vqr.vector_quantiles(X_valid)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for t in range(0, 5):
        Q1, Q2 = vqr_samples[t]
        plot_coverage_2d(
            Q1,
            Q2,
            Y_train=Y_train,
            Y_valid=Y_valid[[t], :],
            alpha=0.05,
            title=rec.name,
            ax=ax,
            xylim=[0.4, 1.1],
            xlabel="$R_i$",
            ylabel="$R_{i+1}$",
            contour_color=f"C{t}",
            contour_label=f"VQR {t=}",
        )

    plt.show()


def run_exp():
    exp_args = dict(
        T=25,
        split_opts=dict(
            d=2,
            k=20.0,
            window_overlap=0.90,
            gap_delta=1.0,
            dilation_x=1,
            dilation_y=1,
            validation_proportion=0.5,
        ),
        vqr_solver_opts={
            "verbose": True,
            "epsilon": 1e-5,
            "num_epochs": 500,
            "learning_rate": 0.5,
        },
    )
    _single_record_exp(NSRDB.records[-1], **exp_args)
    # _single_record_exp(CHFDB.records[-1],**exp_args)
    # _single_record_exp(AFDB.records[0], **exp_args)


if __name__ == "__main__":
    run_exp()
