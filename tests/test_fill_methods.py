"""Tests for fill_methods.py using synthetic series with known answers."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fill_methods import fill_forest, fill_linear, fill_spline, make_inputs  # noqa: E402

TZ = "Australia/Melbourne"


def _times(n):
    return pd.date_range("2024-03-01", periods=n, freq="10min", tz=TZ)


def _with_gap(s, start=40, length=6):
    gappy = s.copy()
    gappy.iloc[start:start + length] = np.nan
    return gappy


def test_linear_is_exact_on_a_straight_line():
    s = pd.Series(400 + 2.0 * np.arange(100), index=_times(100))
    filled = fill_linear(_with_gap(s))
    assert np.allclose(filled, s)


def test_linear_uses_real_time_spacing():
    times = _times(3).delete(1).insert(1, pd.Timestamp("2024-03-01 00:15", tz=TZ))
    s = pd.Series([0.0, np.nan, 20.0], index=times)
    # 15 min into a 20 min span should give 3/4 of the way: 15.
    assert fill_linear(s).iloc[1] == 15.0


def test_spline_is_exact_on_a_cubic():
    t = np.arange(100, dtype=float)
    s = pd.Series(400 + 0.5 * t - 0.02 * t**2 + 0.0003 * t**3, index=_times(100))
    filled = fill_spline(_with_gap(s))
    assert np.allclose(filled, s)


def test_known_values_never_change():
    rng = np.random.default_rng(0)
    s = pd.Series(400 + rng.normal(0, 20, 200), index=_times(200))
    gappy = _with_gap(s)
    X = pd.DataFrame({"a": rng.normal(size=200)}, index=s.index)
    known = gappy.notna()
    for filled in [fill_linear(gappy), fill_spline(gappy), fill_forest(gappy, X)]:
        assert filled.notna().all()
        assert (filled[known] == s[known]).all()


def test_forest_learns_from_inputs_and_repeats_with_seed():
    rng = np.random.default_rng(1)
    temp = pd.Series(20 + rng.uniform(0, 5, 500), index=_times(500))
    s = 300 + 20 * temp
    gappy = _with_gap(s, start=200, length=36)
    X = temp.to_frame("temp_C")
    f1 = fill_forest(gappy, X, seed=0)
    f2 = fill_forest(gappy, X, seed=0)
    gap = gappy.isna()
    assert np.sqrt(np.mean((f1[gap] - s[gap]) ** 2)) < 5
    assert (f1 == f2).all()


def test_make_inputs_columns_and_no_missing():
    idx = _times(6)
    mi = pd.MultiIndex.from_product([["s1"], idx], names=["sensorid", "timestamp"])
    wide = pd.DataFrame(
        {"temp_C": [20, np.nan, 22, 23, 24, 25.0], "rh_pct": [50.0] * 6,
         "co2_ppm": [400.0] * 6},
        index=mi,
    )
    X = make_inputs(wide, "s1", idx)
    assert list(X.columns) == ["temp_C", "rh_pct", "hour", "weekday"]
    assert X.notna().all().all()
    assert X["temp_C"].iloc[1] == 21.0