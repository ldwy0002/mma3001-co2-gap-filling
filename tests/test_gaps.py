"""Tests for gaps.py using a small synthetic series."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gaps import make_gaps, select_series  # noqa: E402


def _series(n=500):
    times = pd.date_range("2024-03-01", periods=n, freq="10min", tz="Australia/Melbourne")
    return pd.Series(np.arange(n, dtype=float), index=times)


def _runs(mask):
    """Return the length of each block of consecutive True values."""
    m = np.concatenate([[0], mask.astype(int), [0]])
    starts = np.flatnonzero(np.diff(m) == 1)
    ends = np.flatnonzero(np.diff(m) == -1)
    return ends - starts


def test_select_series_drops_missing():
    times = pd.date_range("2024-03-01", periods=3, freq="10min")
    idx = pd.MultiIndex.from_product([["s1"], times], names=["sensorid", "timestamp"])
    wide = pd.DataFrame({"co2_ppm": [400.0, np.nan, 420.0]}, index=idx)
    out = select_series(wide, "s1")
    assert list(out) == [400.0, 420.0]


def test_right_number_and_length_of_gaps():
    _, mask = make_gaps(_series(), gap_len=6, n_gaps=10, seed=0)
    runs = _runs(mask.values)
    assert len(runs) == 10
    assert (runs == 6).all()


def test_gaps_never_touch_the_ends():
    for seed in range(20):
        _, mask = make_gaps(_series(), gap_len=18, n_gaps=10, seed=seed)
        assert not mask.iloc[0] and not mask.iloc[-1]


def test_gappy_is_nan_only_inside_gaps():
    s = _series()
    gappy, mask = make_gaps(s, gap_len=3, n_gaps=20, seed=1)
    assert gappy[mask].isna().all()
    assert (gappy[~mask] == s[~mask]).all()


def test_same_seed_gives_same_gaps():
    s = _series()
    _, m1 = make_gaps(s, gap_len=6, n_gaps=10, seed=42)
    _, m2 = make_gaps(s, gap_len=6, n_gaps=10, seed=42)
    _, m3 = make_gaps(s, gap_len=6, n_gaps=10, seed=43)
    assert (m1 == m2).all()
    assert not (m1 == m3).all()


def test_impossible_request_raises():
    with pytest.raises(ValueError):
        make_gaps(_series(50), gap_len=10, n_gaps=10, seed=0)