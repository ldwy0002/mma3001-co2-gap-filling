import cProfile
import io
import math
import pstats
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiment import N_GAPS, SEEDS, SENSORS, longest_clean_stretch, rmse
from fill_methods import fill_forest, make_inputs
from gaps import make_gaps, select_series

TREE_COUNTS = [10, 25, 50, 100, 200]
SWEEP_GAP_LEN = 36


def profile_forest(co2, X, gap_len=SWEEP_GAP_LEN, n_lines=10):
    gappy, _ = make_gaps(co2, gap_len=gap_len, n_gaps=N_GAPS, seed=0)
    profiler = cProfile.Profile()
    profiler.enable()
    fill_forest(gappy, X)
    profiler.disable()
    out = io.StringIO()
    pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats(n_lines)
    return out.getvalue()


def tree_sweep(co2, X, gap_len=SWEEP_GAP_LEN):
    rows = []
    for seed in SEEDS:
        gappy, mask = make_gaps(co2, gap_len=gap_len, n_gaps=N_GAPS, seed=seed)
        for n_trees in TREE_COUNTS:
            t0 = time.perf_counter()
            filled = fill_forest(gappy, X, seed=seed, n_trees=n_trees)
            seconds = time.perf_counter() - t0
            rows.append({"n_trees": n_trees, "seed": seed,
                         "rmse_ppm": rmse(filled, co2, mask), "seconds": seconds})
    return pd.DataFrame(rows)


def operation_estimates(n, n_features, n_trees):
    log_n = math.log2(n)
    return {
        "linear": 10 * n,
        "spline": 20 * n,
        "forest": n_trees * n_features * n * log_n**2,
    }


def plot_sweep(sweep, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for sensor, sub in sweep.groupby("sensor"):
        stats = sub.groupby("n_trees")
        ax1.errorbar(stats["rmse_ppm"].mean().index, stats["rmse_ppm"].mean(),
                     yerr=stats["rmse_ppm"].std(), marker="o", capsize=3,
                     label=f"Sensor {sensor}")
        ax2.plot(stats["seconds"].mean().index, stats["seconds"].mean(),
                 marker="o", label=f"Sensor {sensor}")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks(TREE_COUNTS, [str(t) for t in TREE_COUNTS])
        ax.set_xlabel("Number of trees")
        ax.grid(alpha=0.3)
        ax.legend()
    ax1.set_ylabel("CO2 RMSE (ppm)")
    ax1.set_title(f"Accuracy, {SWEEP_GAP_LEN * 10} min gaps")
    ax2.set_yscale("log")
    ax2.set_ylabel("Runtime (s)")
    ax2.set_title("Cost")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from load_env import load_env

    data_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    wide = load_env(data_dir / "5EnvSensor_MayToDec2024_180kRows.zip")

    sweeps = []
    for sid in SENSORS:
        co2 = longest_clean_stretch(select_series(wide, sid))
        X = make_inputs(wide, sid, co2.index)

        if sid == SENSORS[0]:
            report = profile_forest(co2, X)
            (out_dir / "profile.txt").write_text(report)
            print(report)
            ops = operation_estimates(len(co2), X.shape[1], 100)
            for name, count in ops.items():
                print(f"{name:6s}: about {count:.1e} operations")

        sweep = tree_sweep(co2, X)
        sweep.insert(0, "sensor", sid[-3:])
        sweeps.append(sweep)
        print(f"Sensor {sid} tree sweep done")

    sweep = pd.concat(sweeps, ignore_index=True)
    sweep.to_csv(out_dir / "tree_sweep.csv", index=False)
    summary = sweep.groupby(["sensor", "n_trees"])[["rmse_ppm", "seconds"]].agg(["mean", "std"])
    print(summary.round(3).to_string())
    plot_sweep(sweep, out_dir / "tree_sweep.png")
    print(f"Saved profile, sweep and figure to {out_dir}")