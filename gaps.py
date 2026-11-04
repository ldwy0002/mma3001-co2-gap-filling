"""Create artificial gaps in a clean sensor record.

I delete readings from a record with no real gaps, so I know the true value
of every point I remove. The filling methods only see the gappy series, and
I score them against the values I kept.

Typical use::

    from load_env import load_env
    from gaps import select_series, make_gaps

    wide = load_env("5EnvSensor_MayToDec2024_180kRows.zip")
    co2 = select_series(wide, "6012002000869")
    gappy, mask = make_gaps(co2, gap_len=6, n_gaps=50, seed=0)

Run directly (``python gaps.py``) to see how much each gap length deletes.
"""

import numpy as np
import pandas as pd


def select_series(wide, sensorid, variable="co2_ppm"):
    """Pull one channel for one sensor out of the wide table.

    Args:
        wide: DataFrame from :func:`load_env.load_env`.
        sensorid: Sensor ID as text, e.g. ``"6012002000869"``.
        variable: Column name, e.g. ``"co2_ppm"``.

    Returns:
        pandas.Series indexed by timestamp, with missing values removed.
    """
    return wide.loc[sensorid, variable].dropna()


def make_gaps(series, gap_len, n_gaps, seed=0):
    """Delete ``n_gaps`` blocks of ``gap_len`` consecutive readings.

    Gaps go at random positions, but never touch the first or last reading
    and never touch each other. Every gap then has a known reading on both
    sides, so filling it is interpolation, not extrapolation.

    Args:
        series: Clean pandas.Series from :func:`select_series`.
        gap_len: Readings per gap (1 reading is about 10 min).
        n_gaps: Number of gaps to create.
        seed: Random seed. The same seed always gives the same gaps.

    Returns:
        Tuple ``(gappy, mask)``. ``gappy`` is a copy of ``series`` with NaN
        in the gaps. ``mask`` is a boolean Series, True where a reading was
        deleted.

    Raises:
        ValueError: If the gaps cannot fit without touching.
    """
    n = len(series)
    rng = np.random.default_rng(seed)  # fixed seed = repeatable gaps
    mask = np.zeros(n, dtype=bool)

    placed = 0
    attempts = 0
    while placed < n_gaps:
        attempts += 1
        if attempts > 100 * n_gaps:  # stop instead of looping forever
            raise ValueError(
                f"Could not fit {n_gaps} gaps of {gap_len} readings into "
                f"{n} readings. Use fewer or shorter gaps."
            )
        start = rng.integers(1, n - gap_len)  # never at either end
        # Reject if it overlaps or touches an earlier gap
        if mask[start - 1 : start + gap_len + 1].any():
            continue
        mask[start : start + gap_len] = True
        placed += 1

    gappy = series.copy()  # keep the true values untouched
    gappy[mask] = np.nan

    # The mask marks every reading I deleted. Any method that learns from data
    # (the random forest) must only train on rows where the mask is False.
    # Training on masked rows would let the model see the true values before
    # it predicts them (data leakage), which makes the error look far smaller
    # than it would be on a real sensor dropout.
    return gappy, pd.Series(mask, index=series.index)


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