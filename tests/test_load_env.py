"""Tests for load_env using a small synthetic export."""

import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from load_env import audit, explode_json, read_raw, to_wide  # noqa: E402

# 2024-06-01 00:00 UTC
T0 = 1717200000


def _msg(t, co2, temp, hcho_err=False):
    hcho = {"time": t, "value": 17}
    if hcho_err:
        hcho["errorCode"] = 8
    return json.dumps([
        {"values": [{"time": t, "value": co2}],
         "variable": {"name": "Carbon dioxide", "unit": "ppm"}},
        {"values": [{"time": t, "value": temp}],
         "variable": {"name": "Temperature", "unit": "°C"}},
        {"values": [hcho], "variable": {"name": "Formaldehyde", "unit": "µg/m3"}},
    ])


def _write_sample(path):
    rows = [
        # id, sensorid, json, status
        (1, "6012000000001", _msg(T0, 500, 22.0), "Processed"),
        (2, "6012000000001", _msg(T0, 500, 22.0), "Processed"),  # duplicate
        (3, "6012000000001", _msg(T0 + 600, 520, 22.5, hcho_err=True), "Processed"),
        (4, "6012000000002", _msg(T0, 450, 21.0), "Processed"),
        (5, "6012000000002", "not json", "Processed"),
        (6, "6012000000002", _msg(T0 + 600, 999, 99.0), "Failed"),
        (7, "6012000000001", _msg(1683326565, 425, 18.8), "Processed"),  # 2023 backlog
    ]
    df = pd.DataFrame(rows, columns=["id", "sensorid", "jsondata", "status"])
    df["devicetype"] = "env"
    df.to_csv(path, index=False)


def test_sensor_ids_stay_exact(tmp_path):
    p = tmp_path / "s.csv"
    _write_sample(p)
    raw = read_raw(p)
    assert set(raw["sensorid"]) == {"6012000000001", "6012000000002"}


def test_skips_bad_rows_and_flags_errors(tmp_path):
    p = tmp_path / "s.csv"
    _write_sample(p)
    long = explode_json(read_raw(p))
    # Bad JSON row and non-Processed row are skipped: 5 good messages x 3 vars.
    assert len(long) == 15
    flagged = long[long["error_code"].notna()]
    assert len(flagged) == 1 and math.isnan(flagged["value"].iloc[0])


def test_dedup_and_pivot(tmp_path):
    p = tmp_path / "s.csv"
    _write_sample(p)
    wide = to_wide(explode_json(read_raw(p)))
    # Sensor 1: 3 unique times (the duplicate is removed), sorted 2023 first.
    s1 = wide.loc["6012000000001"]
    assert len(s1) == 3
    assert list(s1["co2_ppm"]) == [425, 500, 520]
    assert math.isnan(s1["hcho_ugm3"].iloc[2])
    assert len(wide.loc["6012000000002"]) == 1


def test_audit_reports_interval(tmp_path):
    p = tmp_path / "s.csv"
    _write_sample(p)
    wide = to_wide(explode_json(read_raw(p)))
    wide = wide[wide.index.get_level_values("timestamp").year == 2024]
    rep = audit(wide)
    assert rep.loc["6012000000001", "median_interval_min"] == 10.0