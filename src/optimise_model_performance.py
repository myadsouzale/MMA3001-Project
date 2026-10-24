
"""
Optimisation of Occupancy Prediction Model Performance
======================================================

Compare the accuracy and computational performance of a
historical Weekly Pattern model and three Random Forest
regression configurations.

The experiment investigates whether reducing Random Forest
complexity can decrease training time and memory requirements
while maintaining acceptable prediction accuracy.

Models
------
1. Weekly Pattern baseline
2. Small RF: 30 trees, maximum depth 6
3. Medium RF: 75 trees, maximum depth 10
4. Original RF: 150 trees, maximum depth 12

All Random Forest models use a minimum leaf size of 10.

Methodology
-----------
Only hourly observations with at least 80% coverage are used.

Space 3 is excluded to match the previous four-space
prediction experiment.

The dataset is split chronologically by unique dates:

- 70% training
- 15% validation
- 15% testing

Each model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Training time
- Combined validation and test prediction time

Python-tracked peak memory allocations are also measured
during Random Forest training using tracemalloc.

Inputs
------
data files/processed/occupancy_hourly.csv
    Historical hourly occupancy observations.

Outputs
-------
results/optimisation/optimisation_results.csv
    Model accuracy and computational performance.

results/optimisation/training_time_comparison.png
    Training time for each model.

results/optimisation/accuracy_comparison.png
    Validation and testing MAE comparison.

results/optimisation/runtime_accuracy_tradeoff.png
    Training time versus validation accuracy.

Assumptions
-----------
- Hours with at least 80% coverage are sufficiently reliable.
- Time-weighted headcount approximates building occupancy.
- Chronological evaluation reflects prediction on later data.
- Calendar features provide useful occupancy information.
- Negative predicted headcounts are clipped to zero.

Limitations
-----------
Runtime measurements depend on the computer, software
environment and system load.

A single timed execution is not sufficient to establish
statistically stable runtime performance.

tracemalloc measures Python-tracked allocations and does
not represent total process memory consumption.

Test results are reported for comparison, but model
selection should primarily use validation performance
to avoid choosing models based on the test dataset.

Usage
-----
Run from the project root directory:

    python -u src/optimise_model_performance.py
"""

import pandas as pd
import numpy as np
import time
import tracemalloc
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 1. FILE PATHS AND SETTINGS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT
    / "data files"
    / "processed"
    / "occupancy_hourly.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "optimisation"
)

TARGET = "headcount_time_weighted"

MIN_COVERAGE = 0.80

EXCLUDED_SPACE = (
    "3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8"
)

FEATURES = [
    "space_number",
    "day_number",
    "hour_of_day",
    "month"
]

CONFIGURATIONS = [
    ("Small RF", 30, 6),
    ("Medium RF", 75, 10),
    ("Original RF", 150, 12)
]


# ==========================================
# 2. FEATURE ENCODING
# ==========================================

def encode(data, columns=None):
    """
    Encode calendar and building-space model features.

    Building-space identifiers are one-hot encoded.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing the model features.

    columns : pandas.Index or list, optional
        Feature columns from the training dataset.

    Returns
    -------
    pandas.DataFrame
        Encoded model features.

    Notes
    -----
    When training columns are provided, missing columns
    are filled with zero and additional columns removed.
    """

    X = pd.get_dummies(
        data[FEATURES],
        columns=["space_number"],
        dtype=int
    )

    if columns is not None:

        X = X.reindex(
            columns=columns,
            fill_value=0
        )

    return X


# ==========================================
# 3. EVALUATION METRICS
# ==========================================

def metrics(actual, predicted):
    """
    Calculate prediction accuracy using MAE and RMSE.

    Negative predicted headcounts are clipped to zero.

    Parameters
    ----------
    actual : array-like
        Observed occupancy headcounts.

    predicted : array-like
        Predicted occupancy headcounts.

    Returns
    -------
    tuple of float
        Mean Absolute Error and Root Mean Squared Error,
        both measured in people.
    """

    predicted = np.maximum(
        predicted,
        0
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    return mae, rmse


# ==========================================
# 4. WEEKLY PATTERN PREDICTION
# ==========================================

def predict_weekly(
    data,
    weekly_means,
    space_hour_means,
    space_means,
    global_mean
):
    """
    Predict occupancy using historical weekly averages.

    Predictions are based on mean training headcount
    for each space, weekday and hour combination.

    Missing combinations use broader historical averages.

    Fallback order
    --------------
    1. Space, weekday and hour average.
    2. Space and hour average.
    3. Overall space average.
    4. Global training average.

    Parameters
    ----------
    data : pandas.DataFrame
        Observations requiring predictions.

    weekly_means : pandas.Series
        Mean training headcount indexed by space,
        weekday and hour.

    space_hour_means : pandas.Series
        Mean training headcount indexed by space and hour.

    space_means : pandas.Series
        Mean training headcount indexed by space.

    global_mean : float
        Overall training mean headcount.

    Returns
    -------
    numpy.ndarray
        Predicted occupancy headcounts.
    """

    keys = pd.MultiIndex.from_frame(
        data[
            [
                "floorspaceid",
                "day_number",
                "hour_of_day"
            ]
        ]
    )

    predicted = (
        weekly_means
        .reindex(keys)
        .to_numpy(copy=True)
    )

    # Fallback to the same space and hour.

    missing = np.isnan(predicted)

    if missing.any():

        fallback_keys = pd.MultiIndex.from_frame(
            data.loc[
                missing,
                ["floorspaceid", "hour_of_day"]
            ]
        )

        predicted[missing] = (
            space_hour_means
            .reindex(fallback_keys)
            .to_numpy()
        )

    # Fallback to the overall space average,
    # followed by the global average.

    missing = np.isnan(predicted)

    if missing.any():

        predicted[missing] = (
            data.loc[
                missing,
                "floorspaceid"
            ]
            .map(space_means)
            .fillna(global_mean)
            .to_numpy()
        )

    return predicted


# ==========================================
# 5. MAIN OPTIMISATION FUNCTION
# ==========================================

def main():
    """
    Run the occupancy model optimisation experiment.

    Loads and filters hourly occupancy data, prepares
    chronological datasets and compares the historical
    Weekly Pattern model with three Random Forest models.

    Records accuracy, training time, prediction time and
    Python-tracked peak memory allocations.

    Returns
    -------
    None
        CSV results and PNG graphs are saved to OUTPUT_DIR.

    Raises
    ------
    FileNotFoundError
        If the hourly occupancy CSV does not exist.

    Notes
    -----
    Random Forest models use random_state=42 and
    min_samples_leaf=10.

    Existing result files with matching names are
    overwritten when the experiment is executed.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==========================================
    # 6. LOAD DATA
    # ==========================================

    print(
        "Loading data...",
        flush=True
    )

    df = pd.read_csv(INPUT_FILE)

    df = df[
        df["coverage_fraction"] >= MIN_COVERAGE
    ].copy()

    df = df.dropna(
        subset=[TARGET]
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["day_number"] = (
        df["day_number"].astype(int)
    )

    df["hour_of_day"] = (
        df["hour_of_day"].astype(int)
    )

    df["month"] = (
        df["month"].astype(int)
    )

    # Exclude Space 3 to match the previous
    # four-space prediction experiment.

    df = df[
        df["floorspaceid"] != EXCLUDED_SPACE
    ].copy()

    # Assign numeric space identifiers.

    space_ids = sorted(
        df["floorspaceid"].unique()
    )

    space_mapping = {
        space_id: i
        for i, space_id in enumerate(space_ids)
    }

    df["space_number"] = (
        df["floorspaceid"].map(space_mapping)
    )

    # ==========================================
    # 7. CHRONOLOGICAL SPLIT
    # ==========================================

    dates = np.array(
        sorted(df["date"].unique())
    )

    train_end = int(
        len(dates) * 0.70
    )

    val_end = int(
        len(dates) * 0.85
    )

    train = df[
        df["date"].isin(
            dates[:train_end]
        )
    ].copy()

    validation = df[
        df["date"].isin(
            dates[train_end:val_end]
        )
    ].copy()

    test = df[
        df["date"].isin(
            dates[val_end:]
        )
    ].copy()

    print(
        "Training hours:",
        len(train)
    )

    print(
        "Validation hours:",
        len(validation)
    )

    print(
        "Testing hours:",
        len(test)
    )

    # ==========================================
    # 8. PREPARE MODEL FEATURES
    # ==========================================

    X_train = encode(train)

    X_val = encode(
        validation,
        X_train.columns
    )

    X_test = encode(
        test,
        X_train.columns
    )

    y_train = train[TARGET]

    y_val = validation[TARGET]

    y_test = test[TARGET]

    # ==========================================
    # 9. BENCHMARK WEEKLY PATTERN MODEL
    # ==========================================

    print(
        "\nBenchmarking weekly pattern model..."
    )

    start = time.perf_counter()

    weekly_means = train.groupby(
        [
            "floorspaceid",
            "day_number",
            "hour_of_day"
        ]
    )[TARGET].mean()

    space_hour_means = train.groupby(
        [
            "floorspaceid",
            "hour_of_day"
        ]
    )[TARGET].mean()

    space_means = (
        train.groupby(
            "floorspaceid"
        )[TARGET]
        .mean()
    )

    global_mean = (
        train[TARGET].mean()
    )

    weekly_training_time = (
        time.perf_counter() - start
    )

    # Measure prediction time for both
    # validation and testing datasets.

    start = time.perf_counter()

    weekly_val_predictions = predict_weekly(
        validation,
        weekly_means,
        space_hour_means,
        space_means,
        global_mean
    )

    weekly_test_predictions = predict_weekly(
        test,
        weekly_means,
        space_hour_means,
        space_means,
        global_mean
    )

    weekly_prediction_time = (
        time.perf_counter() - start
    )

    # Calculate validation and testing errors.

    weekly_val_mae, weekly_val_rmse = metrics(
        y_val,
        weekly_val_predictions
    )

    weekly_test_mae, weekly_test_rmse = metrics(
        y_test,
        weekly_test_predictions
    )

    # Initialise the model comparison table.

    results = [
        {
            "Model": "Weekly Pattern",
            "Trees": 0,
            "Max Depth": 0,
            "Training Time (s)": (
                weekly_training_time
            ),
            "Prediction Time (ms)": (
                weekly_prediction_time * 1000
            ),
            "Peak Python Memory (MB)": np.nan,
            "Validation MAE": weekly_val_mae,
            "Validation RMSE": weekly_val_rmse,
            "Test MAE": weekly_test_mae,
            "Test RMSE": weekly_test_rmse
        }
    ]

    # ==========================================
    # 10. BENCHMARK RANDOM FOREST MODELS
    # ==========================================

    for name, trees, depth in CONFIGURATIONS:

        print(
            f"\nBenchmarking {name}...",
            flush=True
        )

        model = RandomForestRegressor(
            n_estimators=trees,
            max_depth=depth,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )

        # Measure Python-tracked peak memory
        # allocations during model training.

        tracemalloc.start()

        start = time.perf_counter()

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.perf_counter() - start
        )

        current_memory, peak_memory = (
            tracemalloc.get_traced_memory()
        )

        tracemalloc.stop()

        peak_memory_mb = (
            peak_memory / (1024 ** 2)
        )

        # Measure prediction time across
        # validation and testing datasets.

        start = time.perf_counter()

        val_predictions = model.predict(
            X_val
        )

        test_predictions = model.predict(
            X_test
        )

        prediction_time = (
            time.perf_counter() - start
        ) * 1000

        # Evaluate prediction accuracy.

        val_mae, val_rmse = metrics(
            y_val,
            val_predictions
        )

        test_mae, test_rmse = metrics(
            y_test,
            test_predictions
        )

        results.append(
            {
                "Model": name,
                "Trees": trees,
                "Max Depth": depth,
                "Training Time (s)": training_time,
                "Prediction Time (ms)": (
                    prediction_time
                ),
                "Peak Python Memory (MB)": (
                    peak_memory_mb
                ),
                "Validation MAE": val_mae,
                "Validation RMSE": val_rmse,
                "Test MAE": test_mae,
                "Test RMSE": test_rmse
            }
        )

        print(
            f"Training: {training_time:.3f} s | "
            f"Validation MAE: {val_mae:.4f} | "
            f"Test MAE: {test_mae:.4f}"
        )

    # ==========================================
    # 11. SAVE OPTIMISATION RESULTS
    # ==========================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_DIR / "optimisation_results.csv",
        index=False
    )

    print("\n===================================")
    print("OPTIMISATION RESULTS")
    print("===================================")

    print(
        results_df.to_string(
            index=False
        )
    )

    # ==========================================
    # 12. TRAINING TIME COMPARISON
    # ==========================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        results_df["Model"],
        results_df["Training Time (s)"]
    )

    plt.ylabel(
        "Training time (seconds)"
    )

    plt.title(
        "Training Time by Model"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_time_comparison.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 13. ACCURACY COMPARISON
    # ==========================================

    x = np.arange(
        len(results_df)
    )

    width = 0.35

    plt.figure(
        figsize=(11, 5)
    )

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

    plt.xticks(
        x,
        results_df["Model"]
    )

    plt.ylabel(
        "MAE (people)"
    )

    plt.title(
        "Prediction Accuracy vs Model Complexity"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "accuracy_comparison.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 14. RUNTIME VS ACCURACY
    # ==========================================

    plt.figure(
        figsize=(9, 6)
    )

    for _, row in (
        results_df.iterrows()
    ):

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

    plt.xlabel(
        "Training time (seconds)"
    )

    plt.ylabel(
        "Validation MAE (people)"
    )

    plt.title(
        "Runtime vs Prediction Accuracy"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "runtime_accuracy_tradeoff.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 15. FINAL SUMMARY
    # ==========================================

    print("\nResults saved to:")

    print(OUTPUT_DIR)


# ==========================================
# 16. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
