"""Tests for experiment.py helpers using small synthetic data."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiment import longest_clean_stretch, rmse, run_one  # noqa: E402

TZ = "Australia/Melbourne"


def test_longest_clean_stretch_picks_the_longest_block():
    a = pd.date_range("2024-03-01 00:00", periods=5, freq="10min", tz=TZ)
    b = pd.date_range("2024-03-02 00:00", periods=8, freq="10min", tz=TZ)
    c = pd.date_range("2024-03-03 00:00", periods=3, freq="10min", tz=TZ)
    s = pd.Series(1.0, index=a.append(b).append(c))
    out = longest_clean_stretch(s)
    assert len(out) == 8
    assert out.index[0] == b[0] and out.index[-1] == b[-1]


def test_rmse_only_scores_deleted_points():
    truth = pd.Series([1.0, 2.0, 3.0, 4.0])
    filled = pd.Series([100.0, 2.0, 5.0, 4.0])
    mask = pd.Series([False, False, True, False])
    # Index 0 is wrong but not deleted, so only the error of 2 at index 2 counts.
    assert rmse(filled, truth, mask) == 2.0


def test_run_one_scores_all_three_methods():
    idx = pd.date_range("2024-03-01", periods=300, freq="10min", tz=TZ)
    co2 = pd.Series(400 + 50 * np.sin(np.arange(300) / 20), index=idx)
    X = pd.DataFrame({"hour": idx.hour + idx.minute / 60}, index=idx)
    rows = run_one(co2, X, gap_len=3, seed=0)
    assert [r["method"] for r in rows] == ["linear", "spline", "forest"]
    assert all(r["rmse_ppm"] >= 0 and r["seconds"] >= 0 for r in rows)