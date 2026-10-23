
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. SETTINGS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data files" / "processed" / "occupancy_hourly.csv"
OUTPUT_DIR = ROOT / "results" / "sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "headcount_time_weighted"
THRESHOLDS = [0.50, 0.60, 0.80, 0.90]

# Use the same four spaces as the prediction study
EXCLUDED_SPACE = "3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8"

FEATURES = ["space_number", "day_number", "hour_of_day", "month"]

# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading hourly occupancy data...", flush=True)

df = pd.read_csv(INPUT_FILE)

df = df[df["floorspaceid"] != EXCLUDED_SPACE].copy()

df = df.dropna(
    subset=[TARGET, "coverage_fraction", "date"]
)

df["date"] = pd.to_datetime(df["date"])
df["day_number"] = df["day_number"].astype(int)
df["hour_of_day"] = df["hour_of_day"].astype(int)
df["month"] = df["month"].astype(int)

space_ids = sorted(df["floorspaceid"].unique())

mapping = {
    space_id: i for i, space_id in enumerate(space_ids)
}

df["space_number"] = df["floorspaceid"].map(mapping)

# ==========================================
# 3. FIXED CHRONOLOGICAL SPLIT
# ==========================================

# Determine split dates once, using the original
# 80% threshold, so every experiment uses
# exactly the same calendar boundaries.

reference = df[df["coverage_fraction"] >= 0.80]

dates = np.array(sorted(reference["date"].unique()))

train_end = int(len(dates) * 0.70)
val_end = int(len(dates) * 0.85)

train_cutoff = pd.Timestamp(dates[train_end])
test_cutoff = pd.Timestamp(dates[val_end])

print("\nFixed split boundaries:")
print("Validation begins:", train_cutoff.date())
print("Testing begins:", test_cutoff.date())

# ==========================================
# 4. FIXED HIGH-COVERAGE EVALUATION SETS
# ==========================================

# All models are evaluated on the same hours,
# regardless of their training threshold.

validation = df[
    (df["date"] >= train_cutoff) &
    (df["date"] < test_cutoff) &
    (df["coverage_fraction"] >= 0.90)
].copy()

test = df[
    (df["date"] >= test_cutoff) &
    (df["coverage_fraction"] >= 0.90)
].copy()

print("\nFixed evaluation observations:")
print("Validation:", len(validation))
print("Testing:", len(test))

# ==========================================
# 5. MODEL FUNCTIONS
# ==========================================

def encode(data, columns=None):
    X = pd.get_dummies(
        data[FEATURES],
        columns=["space_number"],
        dtype=int
    )

    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)

    return X


def calculate_metrics(actual, predicted):
    predicted = np.maximum(predicted, 0)

    return (
        mean_absolute_error(actual, predicted),
        np.sqrt(mean_squared_error(actual, predicted))
    )


def fit_baselines(train):
    space_means = train.groupby("floorspaceid")[TARGET].mean()

    weekly_means = train.groupby(
        ["floorspaceid", "day_number", "hour_of_day"]
    )[TARGET].mean()

    space_hour_means = train.groupby(
        ["floorspaceid", "hour_of_day"]
    )[TARGET].mean()

    global_mean = train[TARGET].mean()

    return (
        space_means,
        weekly_means,
        space_hour_means,
        global_mean
    )


def predict_space_mean(data, fitted):
    space_means, _, _, global_mean = fitted

    return (
        data["floorspaceid"]
        .map(space_means)
        .fillna(global_mean)
        .to_numpy()
    )


def predict_weekly(data, fitted):
    space_means, weekly_means, space_hour_means, global_mean = fitted

    keys = pd.MultiIndex.from_frame(
        data[["floorspaceid", "day_number", "hour_of_day"]]
    )

    predictions = weekly_means.reindex(keys).to_numpy(copy=True)

    missing = np.isnan(predictions)

    if missing.any():
        fallback_keys = pd.MultiIndex.from_frame(
            data.loc[missing, ["floorspaceid", "hour_of_day"]]
        )

        predictions[missing] = (
            space_hour_means.reindex(fallback_keys).to_numpy()
        )

    missing = np.isnan(predictions)

    if missing.any():
        predictions[missing] = (
            data.loc[missing, "floorspaceid"]
            .map(space_means)
            .fillna(global_mean)
            .to_numpy()
        )

    return predictions


# ==========================================
# 6. RUN SENSITIVITY EXPERIMENT
# ==========================================

results = []
availability = []

for threshold in THRESHOLDS:

    print(
        f"\nTesting coverage threshold: {threshold:.0%}",
        flush=True
    )

    filtered = df[df["coverage_fraction"] >= threshold]

    train = filtered[
        filtered["date"] < train_cutoff
    ].copy()

    availability.append({
        "Threshold": threshold,
        "Total Reliable Hours": len(filtered),
        "Training Hours": len(train),
        "Validation Hours Available": len(
            filtered[
                (filtered["date"] >= train_cutoff) &
                (filtered["date"] < test_cutoff)
            ]
        ),
        "Testing Hours Available": len(
            filtered[filtered["date"] >= test_cutoff]
        )
    })

    fitted = fit_baselines(train)

    X_train = encode(train)
    X_val = encode(validation, X_train.columns)
    X_test = encode(test, X_train.columns)

    models = {
        "Space Mean": (
            predict_space_mean(validation, fitted),
            predict_space_mean(test, fitted)
        ),
        "Weekly Pattern": (
            predict_weekly(validation, fitted),
            predict_weekly(test, fitted)
        )
    }

    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, train[TARGET])

    models["Random Forest"] = (
        rf.predict(X_val),
        rf.predict(X_test)
    )

    for name, (val_pred, test_pred) in models.items():

        val_mae, val_rmse = calculate_metrics(
            validation[TARGET], val_pred
        )

        test_mae, test_rmse = calculate_metrics(
            test[TARGET], test_pred
        )

        results.append({
            "Threshold": threshold,
            "Model": name,
            "Training Hours": len(train),
            "Validation MAE": val_mae,
            "Validation RMSE": val_rmse,
            "Test MAE": test_mae,
            "Test RMSE": test_rmse
        })

        print(
            f"{name}: "
            f"Val MAE={val_mae:.4f}, "
            f"Test MAE={test_mae:.4f}"
        )

# ==========================================
# 7. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)
availability_df = pd.DataFrame(availability)

results_df.to_csv(
    OUTPUT_DIR / "sensitivity_results.csv",
    index=False
)

availability_df.to_csv(
    OUTPUT_DIR / "coverage_availability.csv",
    index=False
)

print("\n===================================")
print("SENSITIVITY ANALYSIS RESULTS")
print("===================================")

print("\nCoverage availability:")
print(availability_df.to_string(index=False))

print("\nModel accuracy:")
print(results_df.to_string(index=False))

# ==========================================
# 8. CREATE GRAPHS
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    availability_df["Threshold"] * 100,
    availability_df["Training Hours"],
    marker="o"
)

plt.xlabel("Minimum hourly coverage (%)")
plt.ylabel("Number of training observations")
plt.title("Training Data Availability vs Coverage Threshold")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "coverage_vs_data_availability.png",
    dpi=200
)
plt.close()

plt.figure(figsize=(10, 6))

for model in results_df["Model"].unique():
    subset = results_df[results_df["Model"] == model]

    plt.plot(
        subset["Threshold"] * 100,
        subset["Validation MAE"],
        marker="o",
        label=model
    )

plt.xlabel("Minimum hourly coverage (%)")
plt.ylabel("Validation MAE (people)")
plt.title("Sensitivity of Prediction Accuracy to Coverage Threshold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "coverage_vs_validation_mae.png",
    dpi=200
)
plt.close()

plt.figure(figsize=(10, 6))

for model in results_df["Model"].unique():
    subset = results_df[results_df["Model"] == model]

    plt.plot(
        subset["Threshold"] * 100,
        subset["Test MAE"],
        marker="o",
        label=model
    )

plt.xlabel("Minimum hourly coverage (%)")
plt.ylabel("Test MAE (people)")
plt.title("Test Prediction Error vs Training Coverage Threshold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "coverage_vs_test_mae.png",
    dpi=200
)
plt.close()

print("\nSensitivity analysis complete.")
print("Results saved to:", OUTPUT_DIR)
