import json
import pandas as pd

SHORT_NAMES = { # left side is the key, right side is the value 
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

def read_raw(path): # reads only header row, pandas 
    return pd.read_csv(path, dtype={"sensorid": str})

def explode_json(raw): # main function, each item of json has a values and a variable section, pulls every reading from json into its own row, got real time measurement, skipped invalid rows, turned invalid values into NaN, so they count as missing data
    records = []
    for sid, js, status in zip(raw["sensorid"], raw["jsondata"], raw["status"]):
        if status != "Processed" or not isinstance(js, str):
            continue
        try:
            items = json.loads(js)
        except json.JSONDecodeError:
            continue
        for item in items:
            name = SHORT_NAMES.get(item["variable"]["name"], item["variable"]["name"])
            for v in item["values"]:
                err = v.get("errorCode")
                val = v.get("value") if err is None else float("nan")
                records.append((sid, v["time"], name, val, err))

    return pd.DataFrame(
        records, columns=["sensorid", "time", "variable", "value", "error_code"]
    )

def to_wide(long): # long is the table, that explode json produced 
    long = long.drop_duplicates(subset=["sensorid", "time", "variable"]) # removes duplicates
    long = long.assign(
        timestamp=pd.to_datetime(long["time"], unit="s", utc=True).dt.tz_convert(LOCAL_TZ)
    ) # converting time
    wide = (
        long.groupby(["sensorid", "timestamp", "variable"])["value"]
        .first()
        .unstack("variable") 
    ) # adjusting layout 
    wide.columns.name = None
    return wide.sort_index()

def load_env(path):
    return to_wide(explode_json(read_raw(path)))

def audit(wide, gap_factor=3.0): # not a cleaning funciton, just describes the clean data, for each sensor, when it starts and stops, how often it reports, where it drops out, 
    rows = []
    for sid, grp in wide.groupby(level='sensorid'):
        t = grp.index.get_level_values("timestamp").to_series()
        dt = t.diff().dropna()
        med = dt.median()
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

if __name__ == "__main__": # generates report for the data audit section
    from pathlib import Path

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