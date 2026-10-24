
"""
Repeated Runtime Benchmarking of Occupancy Prediction Models
============================================================

Benchmark the computational performance and prediction accuracy
of four occupancy prediction models over five repeated runs.

Models
------
1. Weekly Pattern baseline
2. Small Random Forest: 30 trees, maximum depth 6
3. Medium Random Forest: 75 trees, maximum depth 10
4. Original Random Forest: 150 trees, maximum depth 12

All Random Forest models use a minimum leaf size of 10
and a fixed random seed of 42.

Methodology
-----------
The analysis uses hourly observations with at least 80%
temporal coverage.

Space 3 is excluded to maintain consistency with the
four-space model comparison experiment.

Data is split chronologically by unique dates into:

- 70% training
- 15% validation
- 15% testing

Each model is trained and evaluated five times.

The experiment measures:

- Training time in seconds
- Validation prediction time in milliseconds
- Test prediction time in milliseconds
- Validation MAE and RMSE
- Test MAE and RMSE

Mean runtimes and sample standard deviations are calculated
across the repeated runs.

Inputs
------
data files/processed/occupancy_hourly.csv
    Historical hourly occupancy observations.

Outputs
-------
results/optimisation/benchmark_repeated_runs.csv
    Individual measurements from each benchmark run.

results/optimisation/benchmark_summary.csv
    Mean runtimes, standard deviations and average MAE.

results/optimisation/repeated_training_time.png
    Mean training times with standard deviation error bars.

results/optimisation/repeated_prediction_time.png
    Mean test prediction times with standard deviation bars.

Assumptions
-----------
- Hours with at least 80% coverage are sufficiently reliable.
- Time-weighted headcount estimates actual occupancy.
- Chronological splitting is suitable for future prediction.
- The same training and evaluation data can be used to
  compare computational performance fairly.
- Negative predicted headcounts are clipped to zero.

Limitations
-----------
Runtime measurements depend on hardware, operating system,
background processes and software versions.

Five repetitions provide an estimate of runtime variability,
but do not eliminate all measurement uncertainty.

The Random Forest uses a fixed random seed, so repeated
runs measure computational variability rather than
variation caused by different random model initialisations.

Garbage collection is requested before each model run,
but background system activity may still affect timings.

Usage
-----
Run from the project root directory:

    python -u src/benchmark_models.py
"""

import gc
import time

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

REPEATS = 5

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
    ("Weekly Pattern", 0, 0),
    ("Small RF", 30, 6),
    ("Medium RF", 75, 10),
    ("Original RF", 150, 12)
]


# ==========================================
# 2. FEATURE ENCODING
# ==========================================

def encode(data, columns=None):
    """
    Encode calendar and building-space features.

    Building-space identifiers are one-hot encoded
    for Random Forest regression.

    Parameters
    ----------
    data : pandas.DataFrame
        Observations containing the required model features.

    columns : pandas.Index or list, optional
        Training feature columns used to align
        validation and testing feature matrices.

    Returns
    -------
    pandas.DataFrame
        Encoded feature matrix.

    Notes
    -----
    When columns are supplied, missing feature columns
    are filled with zero and extra columns are removed.
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
# 3. ACCURACY METRICS
# ==========================================

def accuracy(actual, predicted):
    """
    Calculate Mean Absolute Error and RMSE.

    Negative predicted headcounts are clipped to zero.

    Parameters
    ----------
    actual : array-like
        Observed hourly headcounts.

    predicted : array-like
        Predicted hourly headcounts.

    Returns
    -------
    tuple of float
        MAE and RMSE, measured in people.
    """

    predicted = np.maximum(
        predicted,
        0
    )

    return (
        mean_absolute_error(
            actual,
            predicted
        ),
        np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )
    )


# ==========================================
# 4. TRAIN WEEKLY PATTERN MODEL
# ==========================================

def fit_weekly(train):
    """
    Train the historical Weekly Pattern model.

    The model calculates mean training headcounts for
    several levels of calendar and space grouping.

    Parameters
    ----------
    train : pandas.DataFrame
        Training observations containing floorspaceid,
        day_number, hour_of_day and the target headcount.

    Returns
    -------
    tuple
        Four historical averages:

        1. Space, weekday and hour means.
        2. Space and hour means.
        3. Overall space means.
        4. Global training mean.

    Notes
    -----
    The model uses historical averages rather than
    iterative machine-learning optimisation.
    """

    weekly = train.groupby(
        [
            "floorspaceid",
            "day_number",
            "hour_of_day"
        ]
    )[TARGET].mean()

    space_hour = train.groupby(
        [
            "floorspaceid",
            "hour_of_day"
        ]
    )[TARGET].mean()

    space_mean = (
        train.groupby(
            "floorspaceid"
        )[TARGET]
        .mean()
    )

    global_mean = (
        train[TARGET].mean()
    )

    return (
        weekly,
        space_hour,
        space_mean,
        global_mean
    )


# ==========================================
# 5. WEEKLY PATTERN PREDICTION
# ==========================================

def predict_weekly(data, fitted):
    """
    Predict occupancy using fitted weekly averages.

    Predictions are based on historical mean headcount
    for each building space, weekday and hour.

    Missing combinations use broader averages.

    Fallback order
    --------------
    1. Space, weekday and hour.
    2. Space and hour.
    3. Overall space average.
    4. Global training average.

    Parameters
    ----------
    data : pandas.DataFrame
        Observations requiring predictions.

    fitted : tuple
        Historical averages returned by fit_weekly().

    Returns
    -------
    numpy.ndarray
        Predicted hourly headcounts in people.
    """

    (
        weekly,
        space_hour,
        space_mean,
        global_mean
    ) = fitted

    keys = pd.MultiIndex.from_frame(
        data[
            [
                "floorspaceid",
                "day_number",
                "hour_of_day"
            ]
        ]
    )

    predictions = (
        weekly
        .reindex(keys)
        .to_numpy(copy=True)
    )

    # Fallback to space-hour averages.

    missing = np.isnan(
        predictions
    )

    if missing.any():

        fallback = pd.MultiIndex.from_frame(
            data.loc[
                missing,
                ["floorspaceid", "hour_of_day"]
            ]
        )

        predictions[missing] = (
            space_hour
            .reindex(fallback)
            .to_numpy()
        )

    # Fallback to space or global averages.

    missing = np.isnan(
        predictions
    )

    if missing.any():

        predictions[missing] = (
            data.loc[
                missing,
                "floorspaceid"
            ]
            .map(space_mean)
            .fillna(global_mean)
            .to_numpy()
        )

    return predictions


# ==========================================
# 6. MAIN BENCHMARK FUNCTION
# ==========================================

def main():
    """
    Run five repeated benchmarks for four prediction models.

    Loads and filters hourly occupancy data, creates
    chronological datasets and evaluates model training
    and prediction performance.

    The experiment records individual runtime measurements,
    calculates summary statistics and generates comparison
    graphs with standard deviation error bars.

    Returns
    -------
    None
        Benchmark CSV files and figures are saved
        to OUTPUT_DIR.

    Raises
    ------
    FileNotFoundError
        If the hourly occupancy dataset is missing.

    Notes
    -----
    The same chronological datasets are reused across
    all model configurations and benchmark repetitions.

    Existing benchmark outputs are overwritten.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==========================================
    # 7. LOAD AND FILTER DATA
    # ==========================================

    print(
        "Loading data...",
        flush=True
    )

    df = pd.read_csv(
        INPUT_FILE
    )

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

    # Exclude Space 3.

    df = df[
        df["floorspaceid"] != EXCLUDED_SPACE
    ].copy()

    # Assign numeric space identifiers.

    space_ids = sorted(
        df["floorspaceid"].unique()
    )

    mapping = {
        space_id: i
        for i, space_id in enumerate(space_ids)
    }

    df["space_number"] = (
        df["floorspaceid"].map(mapping)
    )

    # ==========================================
    # 8. CHRONOLOGICAL DATA SPLIT
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

    # ==========================================
    # 9. ENCODE FEATURES
    # ==========================================

    X_train = encode(
        train
    )

    X_val = encode(
        validation,
        X_train.columns
    )

    X_test = encode(
        test,
        X_train.columns
    )

    # ==========================================
    # 10. REPEATED MODEL BENCHMARKING
    # ==========================================

    records = []

    print(
        f"\nRunning {REPEATS} benchmark repetitions...",
        flush=True
    )

    for repeat in range(
        1,
        REPEATS + 1
    ):

        for name, trees, depth in (
            CONFIGURATIONS
        ):

            # Request garbage collection before
            # timing the next model.

            gc.collect()

            start = time.perf_counter()

            # ----------------------------------
            # Train the selected model
            # ----------------------------------

            if name == "Weekly Pattern":

                model = fit_weekly(
                    train
                )

            else:

                model = RandomForestRegressor(
                    n_estimators=trees,
                    max_depth=depth,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(
                    X_train,
                    train[TARGET]
                )

            training_seconds = (
                time.perf_counter() - start
            )

            # ----------------------------------
            # Measure validation prediction time
            # ----------------------------------

            start = time.perf_counter()

            if name == "Weekly Pattern":

                val_pred = predict_weekly(
                    validation,
                    model
                )

            else:

                val_pred = model.predict(
                    X_val
                )

            validation_ms = (
                time.perf_counter() - start
            ) * 1000

            # ----------------------------------
            # Measure test prediction time
            # ----------------------------------

            start = time.perf_counter()

            if name == "Weekly Pattern":

                test_pred = predict_weekly(
                    test,
                    model
                )

            else:

                test_pred = model.predict(
                    X_test
                )

            test_ms = (
                time.perf_counter() - start
            ) * 1000

            # ----------------------------------
            # Calculate prediction accuracy
            # ----------------------------------

            val_mae, val_rmse = accuracy(
                validation[TARGET],
                val_pred
            )

            test_mae, test_rmse = accuracy(
                test[TARGET],
                test_pred
            )

            # ----------------------------------
            # Store benchmark results
            # ----------------------------------

            records.append(
                {
                    "Repeat": repeat,
                    "Model": name,
                    "Trees": trees,
                    "Max Depth": depth,
                    "Training Time (s)": (
                        training_seconds
                    ),
                    "Validation Prediction (ms)": (
                        validation_ms
                    ),
                    "Test Prediction (ms)": (
                        test_ms
                    ),
                    "Validation MAE": val_mae,
                    "Validation RMSE": val_rmse,
                    "Test MAE": test_mae,
                    "Test RMSE": test_rmse
                }
            )

            print(
                f"Run {repeat}/{REPEATS} | {name} | "
                f"Train {training_seconds:.3f}s | "
                f"Val MAE {val_mae:.4f}",
                flush=True
            )

    # ==========================================
    # 11. SAVE INDIVIDUAL BENCHMARK RESULTS
    # ==========================================

    raw = pd.DataFrame(
        records
    )

    raw.to_csv(
        OUTPUT_DIR / "benchmark_repeated_runs.csv",
        index=False
    )

    # ==========================================
    # 12. CALCULATE SUMMARY STATISTICS
    # ==========================================

    summary = raw.groupby(
        "Model",
        sort=False
    ).agg(
        train_mean=(
            "Training Time (s)",
            "mean"
        ),
        train_std=(
            "Training Time (s)",
            "std"
        ),
        validation_prediction_mean_ms=(
            "Validation Prediction (ms)",
            "mean"
        ),
        validation_prediction_std_ms=(
            "Validation Prediction (ms)",
            "std"
        ),
        test_prediction_mean_ms=(
            "Test Prediction (ms)",
            "mean"
        ),
        test_prediction_std_ms=(
            "Test Prediction (ms)",
            "std"
        ),
        validation_mae=(
            "Validation MAE",
            "mean"
        ),
        test_mae=(
            "Test MAE",
            "mean"
        )
    ).reset_index()

    summary.to_csv(
        OUTPUT_DIR / "benchmark_summary.csv",
        index=False
    )

    # ==========================================
    # 13. PRINT BENCHMARK SUMMARY
    # ==========================================

    print("\n===================================")
    print("FIVE-RUN BENCHMARK SUMMARY")
    print("===================================")

    print(
        summary.to_string(
            index=False
        )
    )

    # ==========================================
    # 14. TRAINING TIME GRAPH
    # ==========================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        summary["Model"],
        summary["train_mean"],
        yerr=summary["train_std"],
        capsize=5
    )

    plt.ylabel(
        "Training time (seconds)"
    )

    plt.title(
        "Mean Training Time Across Five Runs"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "repeated_training_time.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 15. PREDICTION TIME GRAPH
    # ==========================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        summary["Model"],
        summary["test_prediction_mean_ms"],
        yerr=summary["test_prediction_std_ms"],
        capsize=5
    )

    plt.ylabel(
        "Prediction time (milliseconds)"
    )

    plt.title(
        "Mean Test Prediction Time Across Five Runs"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "repeated_prediction_time.png",
        dpi=200
    )

    plt.close()

    print(
        "\nBenchmarking complete."
    )


# ==========================================
# 16. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
