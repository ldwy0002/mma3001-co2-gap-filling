"""Run the full gap-filling experiment and save the results and figures.

For each sensor, gap length and random seed, I delete CO2 readings with
:func:`gaps.make_gaps`, fill them with all three methods from
:mod:`fill_methods`, and record the RMSE and runtime of each method.

Outputs (written to ``results/``):

- ``results.csv``: one row per sensor, gap length, seed and method
- ``summary.csv``: mean and standard deviation over the seeds
- ``rmse_vs_gap.png``: RMSE against gap length for each method
- ``example_gap.png``: one 3 h gap with the true and filled CO2

Run from the ``code`` folder::

    python experiment.py
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fill_methods import fill_forest, fill_linear, fill_spline, make_inputs
from gaps import make_gaps, select_series

# Experiment settings, all in one place
SENSORS = ["6012002000869", "6012002000326"]
GAP_LENGTHS = [1, 3, 6, 18, 36]  # readings: 10 min to 6 h
SEEDS = [0, 1, 2, 3, 4]
N_GAPS = 50
MAX_STEP = pd.Timedelta("30min")  # longer step = real gap
METHODS = ["linear", "spline", "forest"]


def longest_clean_stretch(series, max_step=MAX_STEP):
    """Return the longest part of a series with no real gaps.

    A real gap is any step between readings longer than ``max_step``.

    Args:
        series: pandas.Series with a sorted DatetimeIndex.
        max_step: Longest allowed time between consecutive readings.

    Returns:
        pandas.Series: the longest unbroken stretch of ``series``.
    """
    breaks = series.index.to_series().diff() > max_step
    block = breaks.cumsum().to_numpy()  # new block number after each break
    biggest = np.bincount(block).argmax()
    return series[block == biggest]


def rmse(filled, truth, mask):
    """Root mean square error at the deleted readings only.

    Args:
        filled: Series returned by a fill method.
        truth: The original series before gaps were made.
        mask: Boolean Series, True where readings were deleted.

    Returns:
        float: RMSE in the units of ``truth`` (ppm for CO2).
    """
    return float(np.sqrt(np.mean((filled[mask] - truth[mask]) ** 2)))


def run_one(co2, X, gap_len, seed):
    """Make one set of gaps and score every method on it.

    Args:
        co2: Clean CO2 series (the truth).
        X: Random forest inputs from :func:`fill_methods.make_inputs`.
        gap_len: Readings per gap.
        seed: Random seed for gap positions and the forest.

    Returns:
        list of dict: one result per method with ``method``, ``rmse_ppm``
        and ``seconds``.
    """
    # Same gaps for all three methods, so the comparison is fair
    gappy, mask = make_gaps(co2, gap_len=gap_len, n_gaps=N_GAPS, seed=seed)
    methods = {
        "linear": lambda: fill_linear(gappy),
        "spline": lambda: fill_spline(gappy),
        "forest": lambda: fill_forest(gappy, X, seed=seed),
    }
    rows = []
    for name, fill in methods.items():
        t0 = time.perf_counter()
        filled = fill()
        seconds = time.perf_counter() - t0
        rows.append({"method": name, "rmse_ppm": rmse(filled, co2, mask),
                     "seconds": seconds})
    return rows


def run_all(wide):
    """Run every sensor, gap length and seed.

    Args:
        wide: DataFrame from :func:`load_env.load_env`.

    Returns:
        pandas.DataFrame with columns ``sensor``, ``gap_len``, ``gap_min``,
        ``seed``, ``method``, ``rmse_ppm`` and ``seconds`` (150 rows).
    """
    rows = []
    for sid in SENSORS:
        co2 = longest_clean_stretch(select_series(wide, sid))
        X = make_inputs(wide, sid, co2.index)
        print(f"Sensor {sid}: {len(co2)} readings, "
              f"{co2.index[0]:%d %b %Y} to {co2.index[-1]:%d %b %Y}")
        for gap_len in GAP_LENGTHS:
            for seed in SEEDS:
                for r in run_one(co2, X, gap_len, seed):
                    rows.append({"sensor": sid[-3:], "gap_len": gap_len,
                                 "gap_min": gap_len * 10, "seed": seed, **r})
            print(f"  gap {gap_len * 10:3d} min done")
    return pd.DataFrame(rows)


def summarise(results):
    """Average the results over the seeds.

    Args:
        results: DataFrame from :func:`run_all`.

    Returns:
        pandas.DataFrame indexed by sensor, gap length and method, with the
        mean and standard deviation of RMSE and runtime.
    """
    return (
        results.groupby(["sensor", "gap_min", "method"])[["rmse_ppm", "seconds"]]
        .agg(["mean", "std"])
        .round(3)
    )


def plot_rmse(results, path):
    """Plot mean RMSE against gap length, one panel per sensor.

    Error bars show one standard deviation over the seeds.

    Args:
        results: DataFrame from :func:`run_all`.
        path: Where to save the PNG.
    """
    sensors = results["sensor"].unique()
    fig, axes = plt.subplots(1, len(sensors), figsize=(5 * len(sensors), 4),
                             sharey=True, squeeze=False)
    for ax, sensor in zip(axes[0], sensors):
        sub = results[results["sensor"] == sensor]
        for method in METHODS:
            stats = sub[sub["method"] == method].groupby("gap_min")["rmse_ppm"]
            ax.errorbar(stats.mean().index, stats.mean(), yerr=stats.std(),
                        marker="o", capsize=3, label=method)
        ax.set_xscale("log")  # gaps span 10 to 360 min
        ax.set_yscale("log")  # spline errors are about 10x the others
        ax.set_xticks([10, 30, 60, 180, 360], ["10", "30", "60", "180", "360"])
        ax.set_xlabel("Gap length (min)")
        ax.set_title(f"Sensor {sensor}")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("CO2 RMSE (ppm)")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example(co2, X, path, gap_len=18, seed=0):
    """Plot one gap with the true CO2 and all three filled versions.

    Args:
        co2: Clean CO2 series.
        X: Random forest inputs for ``co2``.
        path: Where to save the PNG.
        gap_len: Readings per gap (18 is 3 h).
        seed: Random seed for the gaps.
    """
    gappy, mask = make_gaps(co2, gap_len=gap_len, n_gaps=N_GAPS, seed=seed)
    filled = {
        "linear": fill_linear(gappy),
        "spline": fill_spline(gappy),
        "forest": fill_forest(gappy, X, seed=seed),
    }
    start = mask.to_numpy().argmax()  # first gap
    window = slice(max(start - 36, 0), start + gap_len + 36)  # 6 h either side
    t = co2.index[window]
    in_gap = pd.Series(False, index=co2.index)  # this one gap only
    in_gap.iloc[start:start + gap_len] = True
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, co2.iloc[window], color="black", label="true")
    ax.axvspan(co2.index[start], co2.index[start + gap_len - 1],
               color="grey", alpha=0.2, label="deleted")
    for name, series in filled.items():
        # Draw each method only inside the gap
        ax.plot(t, series.iloc[window].where(in_gap.iloc[window]), label=name)
    ax.set_ylabel("CO2 (ppm)")
    ax.set_title(f"Example {gap_len * 10} min gap")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from load_env import load_env

    data_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    wide = load_env(data_dir / "5EnvSensor_MayToDec2024_180kRows.zip")
    results = run_all(wide)  # several minutes, mostly the forest
    results.to_csv(out_dir / "results.csv", index=False)

    summary = summarise(results)
    summary.to_csv(out_dir / "summary.csv")
    print(summary.to_string())

    plot_rmse(results, out_dir / "rmse_vs_gap.png")
    co2 = select_series(wide, SENSORS[0])
    plot_example(co2, make_inputs(wide, SENSORS[0], co2.index),
                 out_dir / "example_gap.png")
    print(f"Saved results and figures to {out_dir}")