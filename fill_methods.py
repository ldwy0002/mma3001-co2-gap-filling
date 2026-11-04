"""Three ways to fill gaps in a sensor time series.

Each ``fill_*`` function takes the gappy series from :func:`gaps.make_gaps`
and returns a copy with every NaN replaced. Known readings never change.

- :func:`fill_linear`: straight line between the readings either side
- :func:`fill_spline`: cubic spline through all known readings
- :func:`fill_forest`: random forest that predicts CO2 from temperature,
  humidity and time

Typical use::

    filled = fill_linear(gappy)
    X = make_inputs(wide, "6012002000869", gappy.index)
    filled = fill_forest(gappy, X)

Run directly (``python fill_methods.py``) to score all three on 1 h gaps.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.ensemble import RandomForestRegressor


def fill_linear(gappy):
    """Fill gaps with a straight line between the neighbouring readings.

    Uses the real timestamps, so uneven spacing is handled correctly.

    Args:
        gappy: pandas.Series with a DatetimeIndex and NaN in the gaps.

    Returns:
        pandas.Series with the gaps filled.
    """
    # "time" weights by real time, not by row count
    return gappy.interpolate(method="time")


def fill_spline(gappy):
    """Fill gaps with a cubic spline through every known reading.

    Args:
        gappy: pandas.Series with a DatetimeIndex and NaN in the gaps.

    Returns:
        pandas.Series with the gaps filled.
    """
    # SciPy needs plain numbers, so convert times to seconds
    t = (gappy.index - gappy.index[0]).total_seconds().to_numpy()
    known = gappy.notna().to_numpy()
    spline = CubicSpline(t[known], gappy.to_numpy()[known])
    filled = gappy.copy()
    filled[~known] = spline(t[~known])
    return filled


def make_inputs(wide, sensorid, index):
    """Build the random forest inputs for one sensor.

    Inputs are temperature and humidity from the same sensor, plus hour of
    day and day of week. Short holes in temperature or humidity are filled
    linearly so every row has a complete set of inputs.

    Args:
        wide: DataFrame from :func:`load_env.load_env`.
        sensorid: Sensor ID as text.
        index: Timestamps to build inputs for (use ``gappy.index``).

    Returns:
        pandas.DataFrame indexed by ``index`` with columns ``temp_C``,
        ``rh_pct``, ``hour`` and ``weekday``.
    """
    X = wide.loc[sensorid].loc[index, ["temp_C", "rh_pct"]].copy()
    X = X.interpolate(method="time").bfill()  # forest cannot take NaN
    X["hour"] = index.hour + index.minute / 60  # e.g. 9:30 -> 9.5
    X["weekday"] = index.dayofweek  # 0 = Monday
    return X


def fill_forest(gappy, X, seed=0, n_trees=100):
    """Fill gaps with a random forest trained on the known readings.

    Args:
        gappy: pandas.Series with NaN in the gaps.
        X: Inputs from :func:`make_inputs`, same index as ``gappy``.
        seed: Random seed for the forest, so results repeat exactly.
        n_trees: Number of trees in the forest.

    Returns:
        pandas.Series with the gaps filled.
    """
    # Train only on known readings. The gaps are NaN in ``gappy``, so the
    # model never sees the values it has to predict (see gaps.make_gaps).
    known = gappy.notna()
    model = RandomForestRegressor(n_estimators=n_trees, random_state=seed)
    model.fit(X[known], gappy[known])
    filled = gappy.copy()
    filled[~known] = model.predict(X[~known])
    return filled


if __name__ == "__main__":
    import time
    from pathlib import Path

    from gaps import make_gaps, select_series
    from load_env import load_env

    data_dir = Path(__file__).resolve().parent.parent
    wide = load_env(data_dir / "5EnvSensor_MayToDec2024_180kRows.zip")
    sid = "6012002000869"
    co2 = select_series(wide, sid)
    gappy, mask = make_gaps(co2, gap_len=6, n_gaps=50, seed=0)  # 1 h gaps
    X = make_inputs(wide, sid, co2.index)

    # lambda lets each method be called the same way
    for name, fill in [
        ("linear", lambda: fill_linear(gappy)),
        ("spline", lambda: fill_spline(gappy)),
        ("forest", lambda: fill_forest(gappy, X)),
    ]:
        t0 = time.perf_counter()
        filled = fill()
        seconds = time.perf_counter() - t0
        rmse = np.sqrt(np.mean((filled[mask] - co2[mask]) ** 2))
        print(f"{name:6s}: RMSE {rmse:6.1f} ppm, {seconds:.3f} s")