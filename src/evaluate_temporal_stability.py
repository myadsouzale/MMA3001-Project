
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

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data files" / "processed" / "occupancy_hourly.csv"

OUTPUT_DIR = ROOT / "results" / "temporal_stability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "headcount_time_weighted"

# ==========================================
# 2. LOAD AND FILTER DATA
# ==========================================

print("Loading hourly occupancy data...", flush=True)

df = pd.read_csv(INPUT_FILE)

df = df[df["coverage_fraction"] >= 0.80].copy()
df = df.dropna(subset=[TARGET])

df["date"] = pd.to_datetime(df["date"])
df["day_number"] = df["day_number"].astype(int)
df["hour_of_day"] = df["hour_of_day"].astype(int)
df["month"] = df["month"].astype(int)

# Explicitly exclude Space 3
EXCLUDED_SPACE = "3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8"

df = df[df["floorspaceid"] != EXCLUDED_SPACE].copy()

# Preserve the original numbering from the five-space analysis
space_mapping = {
    "0f3f37f8-0335-41e0-ae87-a1d677dfc7d7": 1,
    "227f1068-9e41-469f-bef0-275f27a0bdba": 2,
    "941f0352-bb69-47d0-a9c5-ac29eb4f7319": 4,
    "cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b": 5
}

df["space_number"] = df["floorspaceid"].map(space_mapping)
df["space"] = "Space " + df["space_number"].astype(str)

df = df.sort_values(["date", "hour_of_day", "space_number"])

print("Reliable hourly observations:", len(df))
print("Spaces:", sorted(df["space"].unique()))

# ==========================================
# 3. CHRONOLOGICAL SPLIT
# ==========================================

dates = np.array(sorted(df["date"].unique()))

train_end = int(len(dates) * 0.70)
val_end = int(len(dates) * 0.85)

train_dates = dates[:train_end]
val_dates = dates[train_end:val_end]
test_dates = dates[val_end:]

train = df[df["date"].isin(train_dates)].copy()
validation = df[df["date"].isin(val_dates)].copy()
test = df[df["date"].isin(test_dates)].copy()

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

# ==========================================
# 4. MODEL DEFINITIONS
# ==========================================

space_means = train.groupby("floorspaceid")[TARGET].mean()
global_mean = train[TARGET].mean()

weekly_means = train.groupby(
    ["floorspaceid", "day_number", "hour_of_day"]
)[TARGET].mean()

space_hour_means = train.groupby(
    ["floorspaceid", "hour_of_day"]
)[TARGET].mean()

def predict_space_mean(data):
    return (
        data["floorspaceid"]
        .map(space_means)
        .fillna(global_mean)
        .to_numpy()
    )

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

print("\nTraining Random Forest...", flush=True)

rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, train[TARGET])

# ==========================================
# 5. GENERATE PREDICTIONS
# ==========================================

def make_predictions(data, X):
    return {
        "Space Mean": np.maximum(predict_space_mean(data), 0),
        "Weekly Pattern": np.maximum(predict_weekly(data), 0),
        "Random Forest": np.maximum(rf.predict(X), 0)
    }

val_predictions = make_predictions(validation, X_val)
test_predictions = make_predictions(test, X_test)

# ==========================================
# 6. EVALUATE MODELS
# ==========================================

print("\nMODEL COMPARISON")

results = []

for period, subset, predictions in [
    ("Validation", validation, val_predictions),
    ("Test", test, test_predictions)
]:
    for model, predicted in predictions.items():

        actual = subset[TARGET].to_numpy()

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        results.append({
            "Period": period,
            "Model": model,
            "MAE": mae,
            "RMSE": rmse,
            "Hours": len(subset)
        })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

# ==========================================
# 7. ERRORS BY MONTH AND SPACE
# ==========================================

detailed_results = []

for period, subset, predictions in [
    ("Validation", validation, val_predictions),
    ("Test", test, test_predictions)
]:
    for model, predicted in predictions.items():

        temp = subset[
            ["date", "space", TARGET]
        ].copy()

        temp["predicted"] = predicted
        temp["absolute_error"] = (
            temp[TARGET] - temp["predicted"]
        ).abs()

        temp["year_month"] = (
            temp["date"].dt.to_period("M").astype(str)
        )

        for (month, space), group in temp.groupby(
            ["year_month", "space"]
        ):
            detailed_results.append({
                "Period": period,
                "Month": month,
                "Space": space,
                "Model": model,
                "MAE": group["absolute_error"].mean(),
                "Hours": len(group)
            })

detail_df = pd.DataFrame(detailed_results)

detail_df.to_csv(
    OUTPUT_DIR / "monthly_space_errors.csv",
    index=False
)

# ==========================================
# 8. MONTHLY ERROR GRAPH
# ==========================================

monthly_errors = (
    detail_df.groupby(["Month", "Model"])
    .apply(
        lambda g: np.average(
            g["MAE"],
            weights=g["Hours"]
        ),
        include_groups=False
    )
    .reset_index(name="MAE")
)

plt.figure(figsize=(13, 6))

sns.lineplot(
    data=monthly_errors,
    x="Month",
    y="MAE",
    hue="Model",
    marker="o"
)

plt.title("Monthly Occupancy Prediction Error")
plt.xlabel("Month")
plt.ylabel("MAE (people)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_prediction_error.png",
    dpi=200
)
plt.close()

# ==========================================
# 9. ERROR BY SPACE
# ==========================================

plt.figure(figsize=(11, 6))

sns.barplot(
    data=detail_df,
    x="Space",
    y="MAE",
    hue="Model",
    errorbar=None
)

plt.title("Prediction Error by Space and Model")
plt.xlabel("Space")
plt.ylabel("Monthly mean MAE (people)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "prediction_error_by_space.png",
    dpi=200
)
plt.close()

# ==========================================
# 10. SAVE MONTHLY SUMMARY
# ==========================================

monthly_errors.to_csv(
    OUTPUT_DIR / "monthly_model_errors.csv",
    index=False
)

print("\n===================================")
print("TEMPORAL STABILITY ANALYSIS COMPLETE")
print("===================================")

print("\nMonthly prediction errors:")
print(monthly_errors.to_string(index=False))

print("\nFiles saved to:")
print(OUTPUT_DIR)
