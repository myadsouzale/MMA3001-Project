
"""
Sensitivity Analysis of Occupancy Sensor Coverage Thresholds
============================================================

Investigate how the minimum hourly sensor coverage threshold
affects training data availability and occupancy prediction
accuracy across four monitored building spaces.

The experiment evaluates four training coverage thresholds:

- 50%
- 60%
- 80%
- 90%

Three prediction models are compared:

1. Space Mean baseline
2. Historical Weekly Pattern baseline
3. Random Forest regression

Methodology
-----------
The experiment uses fixed chronological boundaries derived
from the original dataset filtered at 80% coverage.

The dates are divided into:

- 70% training
- 15% validation
- 15% testing

For each threshold, the training dataset contains only
observations meeting that minimum coverage requirement.

The validation and testing datasets remain unchanged
throughout the experiment and require at least 90% coverage.

This allows changes in model accuracy to be investigated
without changing the evaluation observations.

Space 3 is excluded to maintain consistency with the
four-space prediction study.

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Inputs
------
data files/processed/occupancy_hourly.csv
    Historical hourly occupancy observations.

Outputs
-------
results/sensitivity/sensitivity_results.csv
    Validation and testing accuracy at each threshold.

results/sensitivity/coverage_availability.csv
    Number of observations available at each threshold.

results/sensitivity/coverage_vs_data_availability.png
    Training data availability versus coverage threshold.

results/sensitivity/coverage_vs_validation_mae.png
    Validation prediction error versus training threshold.

results/sensitivity/coverage_vs_test_mae.png
    Test prediction error versus training threshold.

Assumptions
-----------
- Time-weighted headcount approximates actual occupancy.
- Higher hourly coverage indicates more complete observations.
- The 90% coverage evaluation data provides a consistent
  reference for comparing different training thresholds.
- Calendar features may capture recurring occupancy patterns.
- Negative predicted headcounts are clipped to zero.

Limitations
-----------
A higher coverage threshold reduces the amount of
training data available.

The experiment evaluates sensitivity to the amount and
selection of training data, not sensor accuracy directly.

The fixed evaluation datasets represent high-coverage
observations and may not reflect performance during
periods of poor sensor availability.

Coverage filtering may also change the distribution
of times, dates and spaces represented in training.

Usage
-----
Run from the project root directory:

    python -u src/sensitivity_analysis.py
"""

import pandas as pd
import numpy as np
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
    / "sensitivity"
)

TARGET = "headcount_time_weighted"

THRESHOLDS = [
    0.50,
    0.60,
    0.80,
    0.90
]

REFERENCE_COVERAGE = 0.80

EVALUATION_COVERAGE = 0.90

EXCLUDED_SPACE = (
    "3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8"
)

FEATURES = [
    "space_number",
    "day_number",
    "hour_of_day",
    "month"
]


# ==========================================
# 2. FEATURE ENCODING
# ==========================================

def encode(data, columns=None):
    """
    Encode calendar and building-space features.

    Building-space identifiers are converted into
    one-hot encoded feature columns.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing the required model features.

    columns : pandas.Index or list, optional
        Feature columns from the training dataset.

    Returns
    -------
    pandas.DataFrame
        Encoded feature matrix.

    Notes
    -----
    Validation and testing features are aligned with
    the training feature columns when provided.
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

def calculate_metrics(actual, predicted):
    """
    Calculate Mean Absolute Error and RMSE.

    Negative predicted occupancy values are clipped
    to zero before evaluating model performance.

    Parameters
    ----------
    actual : array-like
        Observed hourly headcounts.

    predicted : array-like
        Predicted hourly headcounts.

    Returns
    -------
    tuple of float
        MAE and RMSE, both measured in people.
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
# 4. FIT BASELINE MODELS
# ==========================================

def fit_baselines(train):
    """
    Fit the historical occupancy baseline models.

    Calculates average training headcounts for
    different building-space and calendar groupings.

    Parameters
    ----------
    train : pandas.DataFrame
        Training observations containing space IDs,
        weekday numbers, hours and headcounts.

    Returns
    -------
    tuple
        Historical averages containing:

        1. Overall space means.
        2. Space, weekday and hour means.
        3. Space and hour means.
        4. Global training mean.
    """

    space_means = (
        train.groupby(
            "floorspaceid"
        )[TARGET]
        .mean()
    )

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

    global_mean = (
        train[TARGET].mean()
    )

    return (
        space_means,
        weekly_means,
        space_hour_means,
        global_mean
    )


# ==========================================
# 5. SPACE MEAN PREDICTION
# ==========================================

def predict_space_mean(data, fitted):
    """
    Predict occupancy using overall space averages.

    Each observation receives the historical mean
    headcount for its corresponding building space.

    Unknown spaces use the global training average.

    Parameters
    ----------
    data : pandas.DataFrame
        Observations requiring predictions.

    fitted : tuple
        Baseline model statistics returned by
        fit_baselines().

    Returns
    -------
    numpy.ndarray
        Predicted occupancy headcounts.
    """

    (
        space_means,
        _,
        _,
        global_mean
    ) = fitted

    return (
        data["floorspaceid"]
        .map(space_means)
        .fillna(global_mean)
        .to_numpy()
    )


# ==========================================
# 6. WEEKLY PATTERN PREDICTION
# ==========================================

def predict_weekly(data, fitted):
    """
    Predict occupancy using historical weekly patterns.

    Predictions are based on mean training headcount
    for each combination of space, weekday and hour.

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

    fitted : tuple
        Baseline statistics returned by fit_baselines().

    Returns
    -------
    numpy.ndarray
        Predicted hourly headcounts.
    """

    (
        space_means,
        weekly_means,
        space_hour_means,
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
        weekly_means
        .reindex(keys)
        .to_numpy(copy=True)
    )

    # Fallback to space-hour averages.

    missing = np.isnan(
        predictions
    )

    if missing.any():

        fallback_keys = pd.MultiIndex.from_frame(
            data.loc[
                missing,
                ["floorspaceid", "hour_of_day"]
            ]
        )

        predictions[missing] = (
            space_hour_means
            .reindex(fallback_keys)
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
            .map(space_means)
            .fillna(global_mean)
            .to_numpy()
        )

    return predictions


# ==========================================
# 7. MAIN SENSITIVITY ANALYSIS
# ==========================================

def main():
    """
    Run the coverage-threshold sensitivity experiment.

    Loads hourly occupancy data, determines fixed
    chronological boundaries and evaluates three
    prediction models at four training thresholds.

    The validation and testing datasets remain fixed
    at a minimum coverage requirement of 90%.

    Returns
    -------
    None
        CSV results and PNG graphs are saved to OUTPUT_DIR.

    Raises
    ------
    FileNotFoundError
        If the hourly occupancy dataset is missing.

    Notes
    -----
    Random Forest models use 150 trees, maximum depth 12,
    minimum leaf size 10 and random_state=42.

    Existing output files are overwritten.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==========================================
    # 8. LOAD DATA
    # ==========================================

    print(
        "Loading hourly occupancy data...",
        flush=True
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    # Exclude Space 3.

    df = df[
        df["floorspaceid"] != EXCLUDED_SPACE
    ].copy()

    # Remove observations missing required fields.

    df = df.dropna(
        subset=[
            TARGET,
            "coverage_fraction",
            "date"
        ]
    )

    # Prepare calendar features.

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
    # 9. FIXED CHRONOLOGICAL SPLIT
    # ==========================================

    # Determine split boundaries once using
    # the original 80% coverage threshold.

    reference = df[
        df["coverage_fraction"] >= REFERENCE_COVERAGE
    ]

    dates = np.array(
        sorted(reference["date"].unique())
    )

    train_end = int(
        len(dates) * 0.70
    )

    val_end = int(
        len(dates) * 0.85
    )

    train_cutoff = pd.Timestamp(
        dates[train_end]
    )

    test_cutoff = pd.Timestamp(
        dates[val_end]
    )

    print("\nFixed split boundaries:")

    print(
        "Validation begins:",
        train_cutoff.date()
    )

    print(
        "Testing begins:",
        test_cutoff.date()
    )

    # ==========================================
    # 10. FIXED HIGH-COVERAGE EVALUATION SETS
    # ==========================================

    # Every model is evaluated on the same
    # observations, regardless of its
    # training coverage threshold.

    validation = df[
        (df["date"] >= train_cutoff)
        & (df["date"] < test_cutoff)
        & (
            df["coverage_fraction"]
            >= EVALUATION_COVERAGE
        )
    ].copy()

    test = df[
        (df["date"] >= test_cutoff)
        & (
            df["coverage_fraction"]
            >= EVALUATION_COVERAGE
        )
    ].copy()

    print(
        "\nFixed evaluation observations:"
    )

    print(
        "Validation:",
        len(validation)
    )

    print(
        "Testing:",
        len(test)
    )

    # ==========================================
    # 11. RUN SENSITIVITY EXPERIMENT
    # ==========================================

    results = []

    availability = []

    for threshold in THRESHOLDS:

        print(
            f"\nTesting coverage threshold: "
            f"{threshold:.0%}",
            flush=True
        )

        # Select observations meeting the
        # current coverage threshold.

        filtered = df[
            df["coverage_fraction"] >= threshold
        ]

        # Training always ends before the
        # fixed validation start date.

        train = filtered[
            filtered["date"] < train_cutoff
        ].copy()

        # Record the number of observations
        # available at this threshold.

        availability.append(
            {
                "Threshold": threshold,
                "Total Reliable Hours": len(filtered),
                "Training Hours": len(train),
                "Validation Hours Available": len(
                    filtered[
                        (filtered["date"] >= train_cutoff)
                        & (filtered["date"] < test_cutoff)
                    ]
                ),
                "Testing Hours Available": len(
                    filtered[
                        filtered["date"] >= test_cutoff
                    ]
                )
            }
        )

        # ======================================
        # 12. TRAIN BASELINE MODELS
        # ======================================

        fitted = fit_baselines(
            train
        )

        # Prepare Random Forest features.

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

        # Generate baseline predictions.

        models = {
            "Space Mean": (
                predict_space_mean(
                    validation,
                    fitted
                ),
                predict_space_mean(
                    test,
                    fitted
                )
            ),

            "Weekly Pattern": (
                predict_weekly(
                    validation,
                    fitted
                ),
                predict_weekly(
                    test,
                    fitted
                )
            )
        }

        # ======================================
        # 13. TRAIN RANDOM FOREST
        # ======================================

        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(
            X_train,
            train[TARGET]
        )

        models["Random Forest"] = (
            rf.predict(X_val),
            rf.predict(X_test)
        )

        # ======================================
        # 14. EVALUATE MODEL ACCURACY
        # ======================================

        for name, (
            val_pred,
            test_pred
        ) in models.items():

            val_mae, val_rmse = (
                calculate_metrics(
                    validation[TARGET],
                    val_pred
                )
            )

            test_mae, test_rmse = (
                calculate_metrics(
                    test[TARGET],
                    test_pred
                )
            )

            results.append(
                {
                    "Threshold": threshold,
                    "Model": name,
                    "Training Hours": len(train),
                    "Validation MAE": val_mae,
                    "Validation RMSE": val_rmse,
                    "Test MAE": test_mae,
                    "Test RMSE": test_rmse
                }
            )

            print(
                f"{name}: "
                f"Val MAE={val_mae:.4f}, "
                f"Test MAE={test_mae:.4f}"
            )

    # ==========================================
    # 15. SAVE RESULTS
    # ==========================================

    results_df = pd.DataFrame(
        results
    )

    availability_df = pd.DataFrame(
        availability
    )

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

    print(
        availability_df.to_string(
            index=False
        )
    )

    print("\nModel accuracy:")

    print(
        results_df.to_string(
            index=False
        )
    )

    # ==========================================
    # 16. DATA AVAILABILITY GRAPH
    # ==========================================

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        availability_df["Threshold"] * 100,
        availability_df["Training Hours"],
        marker="o"
    )

    plt.xlabel(
        "Minimum hourly coverage (%)"
    )

    plt.ylabel(
        "Number of training observations"
    )

    plt.title(
        "Training Data Availability vs Coverage Threshold"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "coverage_vs_data_availability.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 17. VALIDATION ERROR GRAPH
    # ==========================================

    plt.figure(
        figsize=(10, 6)
    )

    for model in (
        results_df["Model"].unique()
    ):

        subset = results_df[
            results_df["Model"] == model
        ]

        plt.plot(
            subset["Threshold"] * 100,
            subset["Validation MAE"],
            marker="o",
            label=model
        )

    plt.xlabel(
        "Minimum hourly coverage (%)"
    )

    plt.ylabel(
        "Validation MAE (people)"
    )

    plt.title(
        "Sensitivity of Prediction Accuracy "
        "to Coverage Threshold"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "coverage_vs_validation_mae.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 18. TEST ERROR GRAPH
    # ==========================================

    plt.figure(
        figsize=(10, 6)
    )

    for model in (
        results_df["Model"].unique()
    ):

        subset = results_df[
            results_df["Model"] == model
        ]

        plt.plot(
            subset["Threshold"] * 100,
            subset["Test MAE"],
            marker="o",
            label=model
        )

    plt.xlabel(
        "Minimum hourly coverage (%)"
    )

    plt.ylabel(
        "Test MAE (people)"
    )

    plt.title(
        "Test Prediction Error vs "
        "Training Coverage Threshold"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "coverage_vs_test_mae.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 19. FINAL SUMMARY
    # ==========================================

    print(
        "\nSensitivity analysis complete."
    )

    print(
        "Results saved to:",
        OUTPUT_DIR
    )


# ==========================================
# 20. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
