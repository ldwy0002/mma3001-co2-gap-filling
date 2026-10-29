import numpy as np 
from scipy.interpolate import CubicSpline
from sklearn.ensemble import RandomForestRegressor

def fill_linear(gappy):
    return gappy.interpolate(method="time")

def fill_spline(gappy):
    t = (gappy.index - gappy.index[0]).total_seconds().to_numpy()
    known = gappy.notna().to_numpy()
    spline = CubicSpline(t[known], gappy.to_numpy()[known])
    filled = gappy.copy()
    filled[~known] = spline(t[~known])
    return filled

def make_inputs(wide, sensorid, index): # X is the machine learnign convention. X is the inputs, y is the target
    X = wide.loc[sensorid].loc[index, ["temp_C", "rh_pct"]].copy()
    X = X.interpolate(method="time").bfill()
    X["hour"] = index.hour + index.minute / 60
    X["weekday"] = index.dayofweek
    return X

def fill_forest(gappy, X, seed = 0, n_trees = 100):
    known = gappy.notna() # true for real readings, gaps are NaN so the model never sees them, avoiding leakage
    model = RandomForestRegressor(n_estimators=n_trees, random_state=seed) # 100 trees, fixed seed for repeatable results
    model.fit(X[known],gappy[known])
    filled = gappy.copy()
    filled[~known] = model.predict(X[~known])
    return filled

if __name__ == "__main__":
    import time  # for runtime
    from pathlib import Path

    from gaps import make_gaps, select_series 
    from load_env import load_env 

    data_dir = Path(__file__).resolve().parent.parent 
    wide = load_env(data_dir / "5EnvSensor_MayToDec2024_180kRows.zip")  # clean table
    sid = "6012002000869"  # main sensor
    co2 = select_series(wide, sid)  # true CO2 values
    gappy, mask = make_gaps(co2, gap_len=6, n_gaps=50, seed=0)  # 50 one-hour gaps
    X = make_inputs(wide, sid, co2.index)  # forest inputs

    # Each entry: a name, and a function that runs that method when called
    for name, fill in [
        ("linear", lambda: fill_linear(gappy)),
        ("spline", lambda: fill_spline(gappy)),
        ("forest", lambda: fill_forest(gappy, X)),
    ]:
        t0 = time.perf_counter()  # start the stopwatch
        filled = fill()  # run the method
        seconds = time.perf_counter() - t0  # stop the stopwatch
        rmse = np.sqrt(np.mean((filled[mask] - co2[mask]) ** 2))  # error at the deleted points only
        print(f"{name:6s}: RMSE {rmse:6.1f} ppm, {seconds:.3f} s")  # one line per method


