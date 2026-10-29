import numpy as np
import pandas as pd

def select_series(wide, sensorid, variable="co2_ppm"):
    return wide.loc[sensorid, variable].dropna()

def make_gaps(series, gap_len, n_gaps, seed=0):

    n = len(series)
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)

    placed = 0
    attempts = 0 
    while placed < n_gaps:
        attempts += 1
        if attempts > 100 * n_gaps:
            raise ValueError(
                f"Could not fit {n_gaps} gaps of {gap_len} readings into "
                f"{n} readings. Use fewer or shorter gaps."
            )
        start = rng.integers(1, n - gap_len)
        if mask[start - 1 : start + gap_len + 1].any():
            continue
        mask[start : start + gap_len] = True
        placed += 1

    gappy = series.copy()
    gappy[mask] = np.nan

    return gappy, pd.Series(mask, index = series.index)

if __name__ == "__main__":
    from pathlib import Path

    from load_env import load_env

    data_dir = Path(__file__).resolve().parent.parent
    wide = load_env(data_dir / "5EnvSensor_MayToDec2024_180kRows.zip")
    co2 = select_series(wide, "6012002000869")
    print(f"Sensor 869 CO2: {len(co2)} readings, "
          f"{co2.index.min()} to {co2.index.max()}")

    for gap_len in [1, 3, 6, 18, 36]:
        gappy, mask = make_gaps(co2, gap_len=gap_len, n_gaps=50, seed=0)
        print(f"gap_len={gap_len:2d}: deleted {mask.sum()} readings "
              f"({100 * mask.mean():.1f}% of the record)")

    gappy, mask = make_gaps(co2, gap_len=6, n_gaps=50, seed=0)
    first = mask.to_numpy().argmax()
    view = pd.DataFrame({"co2": co2, "gappy": gappy, "mask": mask})
    print(view.iloc[first - 3 : first + 9])
    view.to_csv(data_dir / "gappy_check.csv")
