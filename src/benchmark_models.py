
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data files" / "processed" / "occupancy_hourly.csv"
OUTPUT_DIR = ROOT / "results" / "optimisation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "headcount_time_weighted"
REPEATS = 5

print("Loading data...", flush=True)

df = pd.read_csv(INPUT_FILE)
df = df[df["coverage_fraction"] >= 0.8].copy()
df = df.dropna(subset=[TARGET])

df["date"] = pd.to_datetime(df["date"])
df["day_number"] = df["day_number"].astype(int)
df["hour_of_day"] = df["hour_of_day"].astype(int)
df["month"] = df["month"].astype(int)

# Exclude Space 3, which has no observations
# in the later evaluation periods.
df = df[
    df["floorspaceid"] != "3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8"
].copy()

space_ids = sorted(df["floorspaceid"].unique())
mapping = {space_id: i for i, space_id in enumerate(space_ids)}
df["space_number"] = df["floorspaceid"].map(mapping)

dates = np.array(sorted(df["date"].unique()))
train_end = int(len(dates) * 0.70)
val_end = int(len(dates) * 0.85)

train = df[df["date"].isin(dates[:train_end])].copy()
validation = df[df["date"].isin(dates[train_end:val_end])].copy()
test = df[df["date"].isin(dates[val_end:])].copy()

FEATURES = ["space_number", "day_number", "hour_of_day", "month"]

def encode(data, columns=None):
    X = pd.get_dummies(
        data[FEATURES],
        columns=["space_number"],
        dtype=int
    )
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X

X_train = encode(train)
X_val = encode(validation, X_train.columns)
X_test = encode(test, X_train.columns)

def accuracy(actual, predicted):
    predicted = np.maximum(predicted, 0)
    return (
        mean_absolute_error(actual, predicted),
        np.sqrt(mean_squared_error(actual, predicted))
    )

def fit_weekly():
    weekly = train.groupby(
        ["floorspaceid", "day_number", "hour_of_day"]
    )[TARGET].mean()

    space_hour = train.groupby(
        ["floorspaceid", "hour_of_day"]
    )[TARGET].mean()

    space_mean = train.groupby("floorspaceid")[TARGET].mean()
    global_mean = train[TARGET].mean()

    return weekly, space_hour, space_mean, global_mean

def predict_weekly(data, fitted):
    weekly, space_hour, space_mean, global_mean = fitted

    keys = pd.MultiIndex.from_frame(
        data[["floorspaceid", "day_number", "hour_of_day"]]
    )

    predictions = weekly.reindex(keys).to_numpy(copy=True)

    missing = np.isnan(predictions)
    if missing.any():
        fallback = pd.MultiIndex.from_frame(
            data.loc[missing, ["floorspaceid", "hour_of_day"]]
        )
        predictions[missing] = space_hour.reindex(fallback).to_numpy()

    missing = np.isnan(predictions)
    if missing.any():
        predictions[missing] = (
            data.loc[missing, "floorspaceid"]
            .map(space_mean)
            .fillna(global_mean)
            .to_numpy()
        )

    return predictions

CONFIGURATIONS = [
    ("Weekly Pattern", 0, 0),
    ("Small RF", 30, 6),
    ("Medium RF", 75, 10),
    ("Original RF", 150, 12)
]

records = []

print(f"\nRunning {REPEATS} benchmark repetitions...", flush=True)

for repeat in range(1, REPEATS + 1):
    for name, trees, depth in CONFIGURATIONS:
        gc.collect()

        start = time.perf_counter()

        if name == "Weekly Pattern":
            model = fit_weekly()
        else:
            model = RandomForestRegressor(
                n_estimators=trees,
                max_depth=depth,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, train[TARGET])

        training_seconds = time.perf_counter() - start

        # Time predictions separately on each set.
        start = time.perf_counter()

        if name == "Weekly Pattern":
            val_pred = predict_weekly(validation, model)
        else:
            val_pred = model.predict(X_val)

        validation_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()

        if name == "Weekly Pattern":
            test_pred = predict_weekly(test, model)
        else:
            test_pred = model.predict(X_test)

        test_ms = (time.perf_counter() - start) * 1000

        val_mae, val_rmse = accuracy(validation[TARGET], val_pred)
        test_mae, test_rmse = accuracy(test[TARGET], test_pred)

        records.append({
            "Repeat": repeat,
            "Model": name,
            "Trees": trees,
            "Max Depth": depth,
            "Training Time (s)": training_seconds,
            "Validation Prediction (ms)": validation_ms,
            "Test Prediction (ms)": test_ms,
            "Validation MAE": val_mae,
            "Validation RMSE": val_rmse,
            "Test MAE": test_mae,
            "Test RMSE": test_rmse
        })

        print(
            f"Run {repeat}/{REPEATS} | {name} | "
            f"Train {training_seconds:.3f}s | "
            f"Val MAE {val_mae:.4f}",
            flush=True
        )

raw = pd.DataFrame(records)
raw.to_csv(OUTPUT_DIR / "benchmark_repeated_runs.csv", index=False)

summary = raw.groupby("Model", sort=False).agg(
    train_mean=("Training Time (s)", "mean"),
    train_std=("Training Time (s)", "std"),
    validation_prediction_mean_ms=("Validation Prediction (ms)", "mean"),
    validation_prediction_std_ms=("Validation Prediction (ms)", "std"),
    test_prediction_mean_ms=("Test Prediction (ms)", "mean"),
    test_prediction_std_ms=("Test Prediction (ms)", "std"),
    validation_mae=("Validation MAE", "mean"),
    test_mae=("Test MAE", "mean")
).reset_index()

summary.to_csv(OUTPUT_DIR / "benchmark_summary.csv", index=False)

print("\n===================================")
print("FIVE-RUN BENCHMARK SUMMARY")
print("===================================")
print(summary.to_string(index=False))

plt.figure(figsize=(10, 5))
plt.bar(
    summary["Model"],
    summary["train_mean"],
    yerr=summary["train_std"],
    capsize=5
)
plt.ylabel("Training time (seconds)")
plt.title("Mean Training Time Across Five Runs")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "repeated_training_time.png",
    dpi=200
)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(
    summary["Model"],
    summary["test_prediction_mean_ms"],
    yerr=summary["test_prediction_std_ms"],
    capsize=5
)
plt.ylabel("Prediction time (milliseconds)")
plt.title("Mean Test Prediction Time Across Five Runs")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "repeated_prediction_time.png",
    dpi=200
)
plt.close()

print("\nBenchmarking complete.")
