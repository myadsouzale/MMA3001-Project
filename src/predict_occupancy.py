
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. FILE PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data files" /
    "processed" / "occupancy_hourly.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD AND PREPARE DATA
# ==========================================

print("Loading hourly occupancy data...", flush=True)

df = pd.read_csv(INPUT_FILE)

df = df[df["coverage_fraction"] >= 0.80].copy()
df = df.dropna(subset=["headcount_time_weighted"])

df["date"] = pd.to_datetime(df["date"])
df["hour_of_day"] = df["hour_of_day"].astype(int)
df["day_number"] = df["day_number"].astype(int)
df["month"] = df["month"].astype(int)

# Give each floorspace a numeric code
space_ids = sorted(df["floorspaceid"].unique())

space_mapping = {
    space_id: i + 1
    for i, space_id in enumerate(space_ids)
}

df["space_number"] = df["floorspaceid"].map(space_mapping)
df["space"] = "Space " + df["space_number"].astype(str)

TARGET = "headcount_time_weighted"

# Sort chronologically
df = df.sort_values(["date", "hour_of_day", "space_number"])

print("Reliable hourly observations:", len(df))
print("Number of spaces:", len(space_ids))

# ==========================================
# 3. CHRONOLOGICAL TRAIN/VALIDATION/TEST
# ==========================================

# Use calendar dates to avoid splitting the same
# day between training and testing.
unique_dates = np.array(sorted(df["date"].unique()))

train_end = int(len(unique_dates) * 0.70)
val_end = int(len(unique_dates) * 0.85)

train_dates = unique_dates[:train_end]
val_dates = unique_dates[train_end:val_end]
test_dates = unique_dates[val_end:]

train = df[df["date"].isin(train_dates)].copy()
validation = df[df["date"].isin(val_dates)].copy()
test = df[df["date"].isin(test_dates)].copy()

print("\nData split:")
for name, subset in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:
    print(
        f"{name}: {len(subset)} rows, "
        f"{subset['date'].min().date()} to "
        f"{subset['date'].max().date()}"
    )

print("\n===================================")
print("DATA AVAILABILITY BY SPACE")
print("===================================")

for name, subset in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:
    print(f"\n{name}:")
    print(
        subset.groupby("space").agg(
            hours=(TARGET, "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            average_headcount=(TARGET, "mean")
        ).to_string()
    )

# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================

results = []

def evaluate(model_name, dataset_name, actual, predicted):
    predicted = np.maximum(np.asarray(predicted), 0)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    results.append({
        "Model": model_name,
        "Dataset": dataset_name,
        "MAE": mae,
        "RMSE": rmse
    })

    print(
        f"{model_name} ({dataset_name}): "
        f"MAE={mae:.4f}, RMSE={rmse:.4f}"
    )

    return predicted

# ==========================================
# 5. BASELINE 1: OVERALL SPACE MEAN
# ==========================================

print("\nTraining baseline models...")

space_means = train.groupby("floorspaceid")[TARGET].mean()
global_mean = train[TARGET].mean()

def predict_space_mean(data):
    return (
        data["floorspaceid"]
        .map(space_means)
        .fillna(global_mean)
        .to_numpy()
    )

# ==========================================
# 6. BASELINE 2: WEEKLY HISTORICAL PATTERN
# ==========================================

# Average headcount for each space, weekday, hour.
weekly_means = train.groupby(
    ["floorspaceid", "day_number", "hour_of_day"]
)[TARGET].mean()

# Fallback if a particular combination is absent
space_hour_means = train.groupby(
    ["floorspaceid", "hour_of_day"]
)[TARGET].mean()

def predict_weekly(data):
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
        predictions[missing] = predict_space_mean(
            data.loc[missing]
        )

    return predictions

# ==========================================
# 7. RANDOM FOREST MODEL
# ==========================================

print("Training Random Forest...")

FEATURES = [
    "space_number",
    "day_number",
    "hour_of_day",
    "month"
]

# One-hot encoding avoids treating space IDs
# as if they were naturally ordered.
X_train = pd.get_dummies(
    train[FEATURES],
    columns=["space_number"],
    dtype=int
)

X_val = pd.get_dummies(
    validation[FEATURES],
    columns=["space_number"],
    dtype=int
)

X_test = pd.get_dummies(
    test[FEATURES],
    columns=["space_number"],
    dtype=int
)

X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, train[TARGET])

# ==========================================
# 8. VALIDATION PERFORMANCE
# ==========================================

print("\nVALIDATION RESULTS")

evaluate(
    "Space Mean",
    "Validation",
    validation[TARGET],
    predict_space_mean(validation)
)

evaluate(
    "Weekly Pattern",
    "Validation",
    validation[TARGET],
    predict_weekly(validation)
)

evaluate(
    "Random Forest",
    "Validation",
    validation[TARGET],
    rf.predict(X_val)
)

# ==========================================
# 9. FINAL TEST PERFORMANCE
# ==========================================

print("\nTEST RESULTS")

test_predictions = {}

test_predictions["Space Mean"] = evaluate(
    "Space Mean",
    "Test",
    test[TARGET],
    predict_space_mean(test)
)

test_predictions["Weekly Pattern"] = evaluate(
    "Weekly Pattern",
    "Test",
    test[TARGET],
    predict_weekly(test)
)

test_predictions["Random Forest"] = evaluate(
    "Random Forest",
    "Test",
    test[TARGET],
    rf.predict(X_test)
)

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_DIR / "prediction_model_results.csv",
    index=False
)

# ==========================================
# 10. MODEL COMPARISON GRAPH
# ==========================================

print("\nCreating model comparison graphs...")

test_results = results_df[
    results_df["Dataset"] == "Test"
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(
    data=test_results,
    x="Model",
    y="MAE",
    ax=axes[0]
)
axes[0].set_title("Test MAE by Model")
axes[0].set_ylabel("Mean Absolute Error (people)")
axes[0].tick_params(axis="x", rotation=15)

sns.barplot(
    data=test_results,
    x="Model",
    y="RMSE",
    ax=axes[1]
)
axes[1].set_title("Test RMSE by Model")
axes[1].set_ylabel("Root Mean Squared Error (people)")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "model_comparison.png",
    dpi=200
)
plt.close()

# ==========================================
# 11. PREDICTED VS ACTUAL
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

actual = test[TARGET].to_numpy()

for ax, (model_name, predicted) in zip(
    axes, test_predictions.items()
):
    ax.scatter(
        actual,
        predicted,
        alpha=0.15,
        s=8
    )

    upper = max(
        np.percentile(actual, 99),
        np.percentile(predicted, 99),
        1
    )

    ax.plot([0, upper], [0, upper], "k--")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_title(model_name)
    ax.set_xlabel("Actual headcount")
    ax.set_ylabel("Predicted headcount")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "predicted_vs_actual.png",
    dpi=200
)
plt.close()

# ==========================================
# 12. ERROR BY SPACE
# ==========================================

space_results = []

for model_name, predictions in test_predictions.items():

    temp = test[["space", TARGET]].copy()
    temp["prediction"] = predictions

    for space, group in temp.groupby("space"):

        space_results.append({
            "Model": model_name,
            "Space": space,
            "MAE": mean_absolute_error(
                group[TARGET], group["prediction"]
            ),
            "RMSE": np.sqrt(mean_squared_error(
                group[TARGET], group["prediction"]
            )),
            "Test Hours": len(group)
        })

space_results_df = pd.DataFrame(space_results)

space_results_df.to_csv(
    RESULTS_DIR / "prediction_results_by_space.csv",
    index=False
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=space_results_df,
    x="Space",
    y="MAE",
    hue="Model"
)

plt.title("Prediction Error by Space")
plt.ylabel("MAE (people)")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "prediction_error_by_space.png",
    dpi=200
)
plt.close()

# ==========================================
# 13. SAVE TEST PREDICTIONS
# ==========================================

output = test[
    [
        "date", "space", "floorspaceid",
        "day_of_week", "hour_of_day",
        TARGET
    ]
].copy()

for model_name, predictions in test_predictions.items():
    column_name = (
        "prediction_" +
        model_name.lower().replace(" ", "_")
    )
    output[column_name] = predictions

output.to_csv(
    RESULTS_DIR / "test_predictions.csv",
    index=False
)

# ==========================================
# 14. FINAL SUMMARY
# ==========================================

print("\n===================================")
print("OCCUPANCY PREDICTION COMPLETE")
print("===================================")

print("\nTest model comparison:")
print(
    test_results[
        ["Model", "MAE", "RMSE"]
    ].to_string(index=False)
)

print("\nPrediction errors by space:")
print(space_results_df.to_string(index=False))

print("\nSpace ID mapping:")
for space_id, number in space_mapping.items():
    print(f"Space {number}: {space_id}")

print("\nResults saved in:")
print(RESULTS_DIR)
