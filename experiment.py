import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fill_methods import fill_forest, fill_linear, fill_spline, make_inputs
from gaps import make_gaps, select_series

SENSORS = ["6012002000869", "6012002000326"]
GAP_LENGTHS  = [1, 3, 6, 18, 36]
SEEDS = [0, 1, 2, 3, 4]
N_GAPS = 50
MAX_STEP = pd.Timedelta("30min")
METHODS = ["linear", "spline", "forest"]

def longest_clean_stretch(series, max_step=MAX_STEP):
    breaks = series.index.to_series().diff() > max_step
    block = breaks.cumsum().to_numpy()
    biggest = np.bincount(block).argmax()
    return series[block == biggest]

def rmse(filled, truth, mask):
    return float(np.sqrt(np.mean((filled[mask] - truth[mask]) ** 2)))

def run_one(co2, X, gap_len, seed):
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
    rows = [] 
    for sid in SENSORS:
        co2 = longest_clean_stretch(select_series(wide, sid))
        X = make_inputs(wide, sid, co2.index)
        print(f"Sensor {sid}: {len(co2)} readings, "
            f"{co2.index[0]:%d %b %Y} to {co2.index[-1]:%d %b %Y}")
        for gap_len in GAP_LENGTHS: # 10 min, 30 min, 1 hr, 3hr, 6hr
            for seed in SEEDS: # 5 different sets of gap positions
                for r in run_one(co2, X, gap_len, seed):  # one trial gives 3 results, one per method
                    rows.append({"sensor": sid[-3:], "gap_len": gap_len, 
                                "gap_min": gap_len * 10, "seed": seed, **r})
            print(f"  gap {gap_len * 10:3d} min done") # message on progress
    return pd.DataFrame(rows)  # should be 150 rows, 2 sensors x 5 gaps x 5 seeds x 3 methods

def summarise(results):
    return (
        results.groupby(["sensor", "gap_min", "method"])[["rmse_ppm", "seconds"]]
        .agg(["mean", "std"])  # average
        .round(3) # rounding to 3dp
    )

def plot_rmse(results, path):
    sensors = results["sensor"].unique()  # 869 and 326, ending numbers of sensor
    fig, axes = plt.subplots(1, len(sensors), figsize=(5 * len(sensors), 4),
                            sharey=True, squeeze=False) 
    for ax, sensor in zip(axes[0], sensors):  
        sub = results[results["sensor"] == sensor] 
        for method in METHODS:  
            stats = sub[sub["method"] == method].groupby("gap_min")["rmse_ppm"] 
            ax.errorbar(stats.mean().index, stats.mean(), yerr=stats.std(),
                        marker="o", capsize=3, label=method) 
        ax.set_xscale("log") 
        ax.set_yscale("log")        
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
    gappy, mask = make_gaps(co2, gap_len=gap_len, n_gaps=N_GAPS, seed=seed)  # 3 h gaps
    filled = { 
        "linear": fill_linear(gappy),
        "spline": fill_spline(gappy),
        "forest": fill_forest(gappy, X, seed=seed),
    }
    start = mask.to_numpy().argmax() 
    window = slice(max(start - 36, 0), start + gap_len + 36)
    t = co2.index[window]  
    in_gap = pd.Series(False, index=co2.index)  
    in_gap.iloc[start:start + gap_len] = True  
    fig, ax = plt.subplots(figsize=(8, 4))  # single plot
    ax.plot(t, co2.iloc[window], color="black", label="true")  # real CO2
    ax.axvspan(co2.index[start], co2.index[start + gap_len - 1],
            color="grey", alpha=0.2, label="deleted")  
    for name, series in filled.items(): 
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
    results = run_all(wide)  
    results.to_csv(out_dir / "results.csv", index=False) 

    summary = summarise(results) 
    summary.to_csv(out_dir / "summary.csv")  
    print(summary.to_string()) 

    plot_rmse(results, out_dir / "rmse_vs_gap.png") 
    co2 = select_series(wide, SENSORS[0])
    plot_example(co2, make_inputs(wide, SENSORS[0], co2.index),
                out_dir / "example_gap.png") 
    print(f"Saved results and figures to {out_dir}")