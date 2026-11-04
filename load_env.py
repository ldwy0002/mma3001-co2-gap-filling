"""Load, clean and audit the MSI environmental sensor export.

The raw export stores one server message per row. Each message holds a JSON
list of 11 channels (CO2, temperature, humidity, ...), and each reading
carries its own Unix timestamp in UTC. This module turns that into a clean
table with one row per sensor per timestamp and one column per channel.

Typical use::

    from load_env import load_env, audit
    wide = load_env("5EnvSensor_MayToDec2024_180kRows.zip")
    print(audit(wide))

Run directly (``python load_env.py``) to print the full data audit.
"""

import json

import pandas as pd

# Raw channel name -> short column name with its unit
SHORT_NAMES = {
    "Battery": "battery_V",
    "Carbon dioxide": "co2_ppm",
    "Humidity": "rh_pct",
    "Formaldehyde": "hcho_ugm3",
    "Pressure": "pressure_mb",
    "Temperature": "temp_C",
    "Light Volatile Organic Compounds": "lvoc_ppb",
    "Particulate matter 1": "pm1_ugm3",
    "Particulate matter 2.5": "pm25_ugm3",
    "Particulate matter 4": "pm4_ugm3",
    "Particulate matter 10": "pm10_ugm3",
}

LOCAL_TZ = "Australia/Melbourne"


def read_raw(path):
    """Read the raw export file.

    Args:
        path: Path to the export, either a CSV or a zip holding one CSV.

    Returns:
        pandas.DataFrame with the original columns. ``sensorid`` stays as
        text so the 13-digit IDs are not rounded.
    """
    # Keep IDs as text; Excel-style rounding gives 6.012E+12
    return pd.read_csv(path, dtype={"sensorid": str})


def explode_json(raw):
    """Unpack the ``jsondata`` column into one row per reading.

    Rows whose status is not ``Processed``, or whose JSON does not parse,
    are skipped. Readings with an ``errorCode`` keep their row but get a
    NaN value, so they count as missing data rather than real values.

    Args:
        raw: DataFrame from :func:`read_raw`.

    Returns:
        pandas.DataFrame in long format with columns ``sensorid``,
        ``time`` (Unix seconds, UTC), ``variable``, ``value`` and
        ``error_code``.
    """
    records = []
    for sid, js, status in zip(raw["sensorid"], raw["jsondata"], raw["status"]):
        if status != "Processed" or not isinstance(js, str):
            continue
        try:
            items = json.loads(js)
        except json.JSONDecodeError:
            continue  # skip corrupted rows
        for item in items:
            name = SHORT_NAMES.get(item["variable"]["name"], item["variable"]["name"])
            for v in item["values"]:
                err = v.get("errorCode")
                # Flagged readings become missing
                val = v.get("value") if err is None else float("nan")
                records.append((sid, v["time"], name, val, err))
    return pd.DataFrame(
        records, columns=["sensorid", "time", "variable", "value", "error_code"]
    )


def to_wide(long):
    """Remove duplicate readings and pivot to one column per channel.

    Args:
        long: DataFrame from :func:`explode_json`.

    Returns:
        pandas.DataFrame indexed by (``sensorid``, ``timestamp``), with
        timestamps in Melbourne time, sorted by time within each sensor.
    """
    # Server re-sends: same sensor, time and channel
    long = long.drop_duplicates(subset=["sensorid", "time", "variable"])
    long = long.assign(
        timestamp=pd.to_datetime(long["time"], unit="s", utc=True).dt.tz_convert(LOCAL_TZ)
    )
    # groupby/unstack keeps only real (sensor, time) pairs
    wide = (
        long.groupby(["sensorid", "timestamp", "variable"])["value"]
        .first()
        .unstack("variable")
    )
    wide.columns.name = None
    return wide.sort_index()


def load_env(path):
    """Read, unpack, deduplicate and pivot the export in one call.

    Args:
        path: Path to the raw export.

    Returns:
        pandas.DataFrame: the clean wide table (see :func:`to_wide`).
    """
    return to_wide(explode_json(read_raw(path)))


def audit(wide, gap_factor=3.0):
    """Summarise coverage and real gaps for each sensor.

    A real gap is any step between consecutive readings longer than
    ``gap_factor`` times that sensor's median sampling interval.

    Args:
        wide: DataFrame from :func:`to_wide`.
        gap_factor: Multiple of the median interval that counts as a gap.

    Returns:
        pandas.DataFrame with one row per sensor: number of readings,
        first and last timestamp, median interval (min), number of gaps
        and longest gap (h).
    """
    rows = []
    for sid, grp in wide.groupby(level="sensorid"):
        t = grp.index.get_level_values("timestamp").to_series()
        dt = t.diff().dropna()
        med = dt.median()  # median ignores long outages
        gaps = dt[dt > gap_factor * med]
        rows.append({
            "sensorid": sid,
            "n_readings": len(grp),
            "first": t.min(),
            "last": t.max(),
            "median_interval_min": med.total_seconds() / 60,
            "n_gaps": len(gaps),
            "longest_gap_h": gaps.max().total_seconds() / 3600 if len(gaps) else 0.0,
        })
    return pd.DataFrame(rows).set_index("sensorid")


if __name__ == "__main__":
    from pathlib import Path

    # Data sits one folder above the code
    data_dir = Path(__file__).resolve().parent.parent
    raw = read_raw(data_dir / "5EnvSensor_MayToDec2024_180kRows.zip")

    long = explode_json(raw)
    n_raw = len(long)
    n_unique = len(long.drop_duplicates(subset=["sensorid", "time", "variable"]))
    print(f"Rows in file: {len(raw)}")
    print(f"Readings: {n_raw} raw, {n_unique} unique "
          f"({100 * (1 - n_unique / n_raw):.1f}% duplicates removed)")
    print(f"Error-coded readings: {long['error_code'].notna().sum()}")

    years = pd.to_datetime(long["time"], unit="s").dt.year.value_counts().sort_index()
    print("Readings per measurement year:")
    print(years.to_string())

    wide = to_wide(long)
    print(audit(wide).to_string())