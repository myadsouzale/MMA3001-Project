
import pandas as pd
import numpy as np
import time
import tracemalloc
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. FILE PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data files" / "processed" /
    "occupancy_hourly.csv"
)

OUTPUT_DIR = ROOT / "results" / "optimisation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "headcount_time_weighted"

# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading data...", flush=True)

df = pd.read_csv(INPUT_FILE)

df = df[df["coverage_fraction"] >= 0.80].copy()
df = df.dropna(subset=[TARGET])

df["date"] = pd.to_datetime(df["date"])
df["day_number"] = df["day_number"].astype(int)
df["hour_of_day"] = df["hour_of_day"].astype(int)
df["month"] = df["month"].astype(int)

# Exclude Space 3 to match the previous
# four-space prediction experiment.
excluded_space = "3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8"

df = df[df["floorspaceid"] != excluded_space].copy()

space_ids = sorted(df["floorspaceid"].unique())

space_mapping = {
    space_id: i
    for i, space_id in enumerate(space_ids)
}

df["space_number"] = df["floorspaceid"].map(space_mapping)

# ==========================================
# 3. CHRONOLOGICAL SPLIT
# ==========================================

dates = np.array(sorted(df["date"].unique()))

train_end = int(len(dates) * 0.70)
val_end = int(len(dates) * 0.85)

train = df[df["date"].isin(dates[:train_end])].copy()

validation = df[
    df["date"].isin(dates[train_end:val_end])
].copy()

test = df[df["date"].isin(dates[val_end:])].copy()

print("Training hours:", len(train))
print("Validation hours:", len(validation))
print("Testing hours:", len(test))

# ==========================================
# 4. PREPARE FEATURES
# ==========================================

FEATURES = [
    "space_number",
    "day_number",
    "hour_of_day",
    "month"
]

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

y_train = train[TARGET]
y_val = validation[TARGET]
y_test = test[TARGET]

# ==========================================
# 5. EVALUATION FUNCTION
# ==========================================

def metrics(actual, predicted):
    predicted = np.maximum(predicted, 0)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    return mae, rmse

# ==========================================
# 6. WEEKLY PATTERN BASELINE
# ==========================================

print("\nBenchmarking weekly pattern model...")

start = time.perf_counter()

weekly_means = train.groupby(
    ["floorspaceid", "day_number", "hour_of_day"]
)[TARGET].mean()

space_hour_means = train.groupby(
    ["floorspaceid", "hour_of_day"]
)[TARGET].mean()

space_means = train.groupby("floorspaceid")[TARGET].mean()
global_mean = train[TARGET].mean()

weekly_training_time = time.perf_counter() - start

def predict_weekly(data):
    keys = pd.MultiIndex.from_frame(
        data[["floorspaceid", "day_number", "hour_of_day"]]
    )

    predicted = weekly_means.reindex(keys).to_numpy(copy=True)

    missing = np.isnan(predicted)

    if missing.any():
        fallback_keys = pd.MultiIndex.from_frame(
            data.loc[missing, ["floorspaceid", "hour_of_day"]]
        )

        predicted[missing] = (
            space_hour_means.reindex(fallback_keys).to_numpy()
        )

    missing = np.isnan(predicted)

    if missing.any():
        predicted[missing] = (
            data.loc[missing, "floorspaceid"]
            .map(space_means)
            .fillna(global_mean)
            .to_numpy()
        )

    return predicted

start = time.perf_counter()

weekly_val_predictions = predict_weekly(validation)
weekly_test_predictions = predict_weekly(test)

weekly_prediction_time = time.perf_counter() - start

weekly_val_mae, weekly_val_rmse = metrics(
    y_val, weekly_val_predictions
)

weekly_test_mae, weekly_test_rmse = metrics(
    y_test, weekly_test_predictions
)

results = [{
    "Model": "Weekly Pattern",
    "Trees": 0,
    "Max Depth": 0,
    "Training Time (s)": weekly_training_time,
    "Prediction Time (ms)": weekly_prediction_time * 1000,
    "Peak Python Memory (MB)": np.nan,
    "Validation MAE": weekly_val_mae,
    "Validation RMSE": weekly_val_rmse,
    "Test MAE": weekly_test_mae,
    "Test RMSE": weekly_test_rmse
}]

# ==========================================
# 7. RANDOM FOREST CONFIGURATIONS
# ==========================================

configurations = [
    ("Small RF", 30, 6),
    ("Medium RF", 75, 10),
    ("Original RF", 150, 12)
]

for name, trees, depth in configurations:

    print(f"\nBenchmarking {name}...", flush=True)

    model = RandomForestRegressor(
        n_estimators=trees,
        max_depth=depth,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    # Measure model training time and Python
    # allocation peak during the fit.
    tracemalloc.start()
    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    current_memory, peak_memory = (
        tracemalloc.get_traced_memory()
    )
    tracemalloc.stop()

    peak_memory_mb = peak_memory / (1024 ** 2)

    # Measure prediction time for both datasets
    start = time.perf_counter()

    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    prediction_time = (time.perf_counter() - start) * 1000

    val_mae, val_rmse = metrics(y_val, val_predictions)
    test_mae, test_rmse = metrics(y_test, test_predictions)

    results.append({
        "Model": name,
        "Trees": trees,
        "Max Depth": depth,
        "Training Time (s)": training_time,
        "Prediction Time (ms)": prediction_time,
        "Peak Python Memory (MB)": peak_memory_mb,
        "Validation MAE": val_mae,
        "Validation RMSE": val_rmse,
        "Test MAE": test_mae,
        "Test RMSE": test_rmse
    })

    print(
        f"Training: {training_time:.3f} s | "
        f"Validation MAE: {val_mae:.4f} | "
        f"Test MAE: {test_mae:.4f}"
    )

# ==========================================
# 8. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "optimisation_results.csv",
    index=False
)

print("\n===================================")
print("OPTIMISATION RESULTS")
print("===================================")

print(results_df.to_string(index=False))

# ==========================================
# 9. GRAPHS
# ==========================================

# Training time
plt.figure(figsize=(10, 5))
plt.bar(
    results_df["Model"],
    results_df["Training Time (s)"]
)
plt.ylabel("Training time (seconds)")
plt.title("Training Time by Model")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "training_time_comparison.png",
    dpi=200
)
plt.close()

# Validation and test MAE
x = np.arange(len(results_df))
width = 0.35

plt.figure(figsize=(11, 5))

plt.bar(
    x - width / 2,
    results_df["Validation MAE"],
    width,
    label="Validation"
)

plt.bar(
    x + width / 2,
    results_df["Test MAE"],
    width,
    label="Test"
)

plt.xticks(x, results_df["Model"])
plt.ylabel("MAE (people)")
plt.title("Prediction Accuracy vs Model Complexity")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "accuracy_comparison.png",
    dpi=200
)
plt.close()

# Runtime versus validation accuracy
plt.figure(figsize=(9, 6))

for _, row in results_df.iterrows():
    plt.scatter(
        row["Training Time (s)"],
        row["Validation MAE"],
        s=100
    )
    plt.annotate(
        row["Model"],
        (
            row["Training Time (s)"],
            row["Validation MAE"]
        ),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("Training time (seconds)")
plt.ylabel("Validation MAE (people)")
plt.title("Runtime vs Prediction Accuracy")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "runtime_accuracy_tradeoff.png",
    dpi=200
)
plt.close()

print("\nResults saved to:")
print(OUTPUT_DIR)
