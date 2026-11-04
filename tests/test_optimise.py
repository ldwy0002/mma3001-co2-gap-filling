"""Tests for optimise.py using small synthetic data."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fill_methods import fill_forest  # noqa: E402
from optimise import TREE_COUNTS, operation_estimates, profile_forest, tree_sweep  # noqa: E402

TZ = "Australia/Melbourne"


def _data(n=600):
    idx = pd.date_range("2024-03-01", periods=n, freq="10min", tz=TZ)
    co2 = pd.Series(400 + 50 * np.sin(np.arange(n) / 20), index=idx)
    X = pd.DataFrame({"hour": idx.hour + idx.minute / 60}, index=idx)
    return co2, X


def test_fill_forest_uses_n_trees():
    co2, X = _data()
    gappy = co2.copy()
    gappy.iloc[100:110] = np.nan
    a = fill_forest(gappy, X, n_trees=5)
    b = fill_forest(gappy, X, n_trees=50)
    assert not np.allclose(a, b)


def test_tree_sweep_has_every_tree_count_and_seed():
    co2, X = _data()
    sweep = tree_sweep(co2, X, gap_len=3)
    assert sorted(sweep["n_trees"].unique()) == TREE_COUNTS
    assert len(sweep) == len(TREE_COUNTS) * sweep["seed"].nunique()
    assert (sweep["rmse_ppm"] >= 0).all()


def test_profile_mentions_fit():
    co2, X = _data()
    report = profile_forest(co2, X, gap_len=3)
    assert "fit" in report


def test_forest_estimate_far_above_linear():
    ops = operation_estimates(13800, 4, 100)
    assert ops["linear"] < ops["spline"] < ops["forest"]
    assert ops["forest"] / ops["linear"] > 1000