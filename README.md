# Filling CO2 sensor gaps: linear vs spline vs random forest

MMA3001 individual project, Levi Dwyer, Monash University, 2026.

When a building's CO2 sensor drops out, how accurately can the missing readings be rebuilt, and which method should an engineer use for which gap length? This project deletes known CO2 readings from Monash Smart Infrastructure (MSI) sensor data in gaps of 10 min to 6 h, refills them with three methods, and scores each one against the true values.

**Result:** linear interpolation gave the lowest error at almost every gap length (about 9 to 25 ppm RMSE) and runs in about 1 ms. The cubic spline overshoots badly on long gaps, and the random forest only matched linear on 6 h gaps on one sensor, at about 2,000 times the runtime.

## Repository structure

```
code/
├── load_env.py        # read the raw export, unpack JSON, remove duplicates, audit sensors
├── gaps.py            # delete known readings at random, repeatable positions
├── fill_methods.py    # linear interpolation, cubic spline, random forest
├── experiment.py      # run every sensor, gap length and seed; save results and figures
├── optimise.py        # profile the forest and tune its number of trees
├── tests/             # pytest tests for every module
├── results/           # results tables, figures, profile and test report
├── docs/              # HTML documentation generated with pdoc
├── requirements.txt
├── LICENSE
└── README.md
```

## Data

The dataset is Dataset 2 from the MMA3001 project datasets (Monash Smart Infrastructure environmental sensor export). It belongs to Monash, so it is **not** included in this repository.

1. Download `5EnvSensor_MayToDec2024_180kRows.zip` from the unit's project datasets folder.
2. Place it **one folder above** this repository, next to the `code` folder:

```
mma3001assignment/
├── 5EnvSensor_MayToDec2024_180kRows.zip
└── code/              # this repository
```

Leave the file zipped; pandas reads it directly.

## Environment and installation

Tested with Python 3.12 on Windows.

```
pip install -r requirements.txt
```

## Usage

Run each step from the `code` folder. Each module also runs on its own as a quick demo.

| Command | What it does | Time |
|---|---|---|
| `python load_env.py` | Prints the data audit: duplicates, error codes, readings per year, per-sensor coverage and gaps | under 1 min |
| `python gaps.py` | Shows how many readings each gap length deletes | under 1 min |
| `python fill_methods.py` | Scores the three methods on 1 h gaps on sensor 869 | under 1 min |
| `python experiment.py` | Full experiment: 2 sensors x 5 gap lengths x 5 seeds x 3 methods | 5 to 10 min |
| `python optimise.py` | Profiles the forest and sweeps 10 to 200 trees | 5 to 10 min |

Outputs go to `results/`:

- `results.csv`: RMSE (ppm) and runtime (s) for every sensor, gap length, seed and method
- `summary.csv`: mean and standard deviation over the 5 seeds
- `rmse_vs_gap.png`: RMSE against gap length (main figure)
- `example_gap.png`: one 3 h gap with the true and filled CO2
- `tree_sweep.csv`, `tree_sweep.png`: forest accuracy and runtime against number of trees
- `profile.txt`: cProfile output for one forest fill

Every random choice uses a fixed seed, so a rerun gives the same gaps and the same RMSE values. Runtimes vary between machines and runs.

## Tests

```
python -m pytest tests -v
```

The tests check the core functions on small synthetic data with known answers, for example that linear interpolation rebuilds a straight line exactly and that the same seed gives the same gaps. The latest test report is in `results/test_report.txt`.

## Documentation

HTML documentation for every module, generated from the docstrings with pdoc, is in `docs/`. Open `docs/index.html` in a browser. To rebuild it:

```
pdoc load_env.py gaps.py fill_methods.py experiment.py optimise.py -d google -o docs
```

On Windows, `make_docs.bat` rebuilds the docs and the test report in one step.

## AI use

I used Claude (Anthropic) to help plan the project, draft and explain the code and tests, and debug. Details, including errors the AI made and how I checked its output, are in Section 7 of the written report.

## Licence

The code is released under the MIT Licence (see `LICENSE`). The dataset is not covered by this licence and is not redistributed here.
