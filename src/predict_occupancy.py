
"""
Building Occupancy Prediction and Model Evaluation
==================================================

Compare three methods for predicting hourly building occupancy:

1. Space Mean baseline
2. Historical Weekly Pattern baseline
3. Random Forest regression

The analysis uses time-weighted hourly headcount estimates
from building occupancy sensors.

Methodology
-----------
Hourly observations with at least 80% temporal coverage are
retained. Data is divided chronologically by unique calendar
dates into approximately:

- 70% training
- 15% validation
- 15% testing

Model accuracy is evaluated using Mean Absolute Error (MAE)
and Root Mean Squared Error (RMSE).

Inputs
------
data files/processed/occupancy_hourly.csv
    Hourly occupancy data containing space identifiers,
    time information, coverage fractions and headcounts.

Outputs
-------
results/prediction_model_results.csv
    Validation and test accuracy for each model.

results/prediction_results_by_space.csv
    Test prediction errors for each building space.

results/test_predictions.csv
    Actual and predicted hourly occupancy values.

results/model_comparison.png
    Comparison of test MAE and RMSE.

results/predicted_vs_actual.png
    Predicted versus actual occupancy plots.

results/prediction_error_by_space.png
    Comparison of prediction errors across spaces.

Assumptions
-----------
- Sensor headcounts estimate the number of people present.
- Hours with at least 80% coverage are suitable for analysis.
- Historical calendar patterns may help predict occupancy.
- Calendar-based features are available for future prediction.
- Missing observations are not equivalent to zero occupancy.

Limitations
-----------
The models do not explicitly include holidays, class
timetables, special events or other external influences.

Space identifiers are used without requiring physical
room names.

Historical prediction accuracy does not guarantee equivalent
performance under changing building operating conditions.

Usage
-----
From the project root directory, run:

    python -u src/predict_occupancy.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 1. FILE PATHS AND SETTINGS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data files"
    / "processed"
    / "occupancy_hourly.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

MIN_COVERAGE = 0.80
TARGET = "headcount_time_weighted"

FEATURES = [
    "space_number",
    "day_number",
    "hour_of_day",
    "month"
]


# ==========================================
# 2. EVALUATION FUNCTION
# ==========================================

def evaluate(model_name, dataset_name, actual, predicted, results):
    """
    Evaluate occupancy predictions using MAE and RMSE.

    Negative predictions are clipped to zero because
    physical occupancy cannot be negative.

    Parameters
    ----------
    model_name : str
        Name of the prediction model.

    dataset_name : str
        Name of the evaluation dataset, such as
        'Validation' or 'Test'.

    actual : array-like
        Observed hourly headcounts, measured in people.

    predicted : array-like
        Predicted hourly headcounts, measured in people.

    results : list of dict
        List used to store model evaluation results.

    Returns
    -------
    numpy.ndarray
        Non-negative predicted headcounts.

    Notes
    -----
    The function appends the calculated MAE and RMSE
    to the supplied results list.
    """

    predicted = np.maximum(np.asarray(predicted), 0)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

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
# 3. SPACE MEAN BASELINE
# ==========================================

def predict_space_mean(data, space_means, global_mean):
    """
    Predict occupancy using historical mean headcount.

    Each building space is assigned its average
    training headcount.

    For an unknown space, the global training
    average is used instead.

    Parameters
    ----------
    data : pandas.DataFrame
        Observations requiring predictions.
        Must contain 'floorspaceid'.

    space_means : pandas.Series
        Mean training headcount indexed by floorspaceid.

    global_mean : float
        Mean headcount across all training observations.

    Returns
    -------
    numpy.ndarray
        Predicted headcounts in people.
    """

    return (
        data["floorspaceid"]
        .map(space_means)
        .fillna(global_mean)
        .to_numpy()
    )


# ==========================================
# 4. WEEKLY HISTORICAL PATTERN BASELINE
# ==========================================

def predict_weekly(
    data,
    weekly_means,
    space_hour_means,
    space_means,
    global_mean
):
    """
    Predict occupancy using historical weekly patterns.

    Predictions use the average training headcount
    for each combination of:

    - Building space
    - Day of the week
    - Hour of the day

    Missing combinations use progressively broader
    fallback averages.

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
        Must contain floorspaceid, day_number
        and hour_of_day.

    weekly_means : pandas.Series
        Historical mean headcount indexed by
        space, weekday and hour.

    space_hour_means : pandas.Series
        Historical mean headcount indexed by
        space and hour.

    space_means : pandas.Series
        Historical mean headcount indexed by space.

    global_mean : float
        Overall training mean headcount.

    Returns
    -------
    numpy.ndarray
        Predicted headcounts in people.
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

    predictions = (
        weekly_means
        .reindex(keys)
        .to_numpy(copy=True)
    )

    # Fallback 1: historical average for the
    # same space and hour.

    missing = np.isnan(predictions)

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

    # Fallback 2: overall space average.
    # Unknown spaces use the global average.

    missing = np.isnan(predictions)

    if missing.any():

        predictions[missing] = predict_space_mean(
            data.loc[missing],
            space_means,
            global_mean
        )

    return predictions


# ==========================================
# 5. MAIN ANALYSIS
# ==========================================

def main():
    """
    Run the complete occupancy prediction workflow.

    The function loads hourly occupancy observations,
    creates chronological datasets, trains three
    prediction models and evaluates their performance.

    It also generates model comparison figures and
    saves prediction results to CSV files.

    Returns
    -------
    None
        Results are written to the results directory.

    Notes
    -----
    The Random Forest uses 150 trees, maximum depth
    of 12, minimum leaf size of 10 and random_state=42.

    Existing result files with matching names
    are overwritten.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==========================================
    # 6. LOAD AND PREPARE DATA
    # ==========================================

    print(
        "Loading hourly occupancy data...",
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

    df["hour_of_day"] = (
        df["hour_of_day"].astype(int)
    )

    df["day_number"] = (
        df["day_number"].astype(int)
    )

    df["month"] = (
        df["month"].astype(int)
    )

    # Assign a numeric label to each building space.

    space_ids = sorted(
        df["floorspaceid"].unique()
    )

    space_mapping = {
        space_id: i + 1
        for i, space_id in enumerate(space_ids)
    }

    df["space_number"] = (
        df["floorspaceid"].map(space_mapping)
    )

    df["space"] = (
        "Space "
        + df["space_number"].astype(str)
    )

    # Sort observations chronologically.

    df = df.sort_values(
        [
            "date",
            "hour_of_day",
            "space_number"
        ]
    )

    print(
        "Reliable hourly observations:",
        len(df)
    )

    print(
        "Number of spaces:",
        len(space_ids)
    )

    # ==========================================
    # 7. CHRONOLOGICAL DATA SPLIT
    # ==========================================

    # Split by calendar date to prevent the same
    # date appearing in multiple datasets.

    unique_dates = np.array(
        sorted(df["date"].unique())
    )

    train_end = int(
        len(unique_dates) * 0.70
    )

    val_end = int(
        len(unique_dates) * 0.85
    )

    train_dates = unique_dates[:train_end]

    val_dates = unique_dates[
        train_end:val_end
    ]

    test_dates = unique_dates[val_end:]

    train = df[
        df["date"].isin(train_dates)
    ].copy()

    validation = df[
        df["date"].isin(val_dates)
    ].copy()

    test = df[
        df["date"].isin(test_dates)
    ].copy()

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

    # ==========================================
    # 8. DATA AVAILABILITY BY SPACE
    # ==========================================

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
    # 9. TRAIN BASELINE MODELS
    # ==========================================

    print("\nTraining baseline models...")

    # Baseline 1: average headcount for each space.

    space_means = (
        train.groupby("floorspaceid")[TARGET]
        .mean()
    )

    global_mean = train[TARGET].mean()

    # Baseline 2: average headcount for each
    # space, weekday and hour.

    weekly_means = train.groupby(
        [
            "floorspaceid",
            "day_number",
            "hour_of_day"
        ]
    )[TARGET].mean()

    # Fallback: average for each space and hour.

    space_hour_means = train.groupby(
        [
            "floorspaceid",
            "hour_of_day"
        ]
    )[TARGET].mean()

    # ==========================================
    # 10. TRAIN RANDOM FOREST
    # ==========================================

    print("Training Random Forest...")

    # One-hot encoding prevents the model from
    # treating space identifiers as an ordered
    # numerical measurement.

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

    # Align validation and test features with
    # the columns used during model training.

    X_val = X_val.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

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

    # ==========================================
    # 11. VALIDATION PERFORMANCE
    # ==========================================

    results = []

    print("\nVALIDATION RESULTS")

    evaluate(
        "Space Mean",
        "Validation",
        validation[TARGET],
        predict_space_mean(
            validation,
            space_means,
            global_mean
        ),
        results
    )

    evaluate(
        "Weekly Pattern",
        "Validation",
        validation[TARGET],
        predict_weekly(
            validation,
            weekly_means,
            space_hour_means,
            space_means,
            global_mean
        ),
        results
    )

    evaluate(
        "Random Forest",
        "Validation",
        validation[TARGET],
        rf.predict(X_val),
        results
    )

    # ==========================================
    # 12. TEST PERFORMANCE
    # ==========================================

    print("\nTEST RESULTS")

    test_predictions = {}

    test_predictions["Space Mean"] = evaluate(
        "Space Mean",
        "Test",
        test[TARGET],
        predict_space_mean(
            test,
            space_means,
            global_mean
        ),
        results
    )

    test_predictions["Weekly Pattern"] = evaluate(
        "Weekly Pattern",
        "Test",
        test[TARGET],
        predict_weekly(
            test,
            weekly_means,
            space_hour_means,
            space_means,
            global_mean
        ),
        results
    )

    test_predictions["Random Forest"] = evaluate(
        "Random Forest",
        "Test",
        test[TARGET],
        rf.predict(X_test),
        results
    )

    # ==========================================
    # 13. SAVE MODEL COMPARISON RESULTS
    # ==========================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_DIR / "prediction_model_results.csv",
        index=False
    )

    # ==========================================
    # 14. MODEL COMPARISON GRAPH
    # ==========================================

    print(
        "\nCreating model comparison graphs..."
    )

    test_results = results_df[
        results_df["Dataset"] == "Test"
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    sns.barplot(
        data=test_results,
        x="Model",
        y="MAE",
        ax=axes[0]
    )

    axes[0].set_title(
        "Test MAE by Model"
    )

    axes[0].set_ylabel(
        "Mean Absolute Error (people)"
    )

    axes[0].tick_params(
        axis="x",
        rotation=15
    )

    sns.barplot(
        data=test_results,
        x="Model",
        y="RMSE",
        ax=axes[1]
    )

    axes[1].set_title(
        "Test RMSE by Model"
    )

    axes[1].set_ylabel(
        "Root Mean Squared Error (people)"
    )

    axes[1].tick_params(
        axis="x",
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "model_comparison.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 15. PREDICTED VS ACTUAL GRAPH
    # ==========================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )

    actual = test[TARGET].to_numpy()

    for ax, (model_name, predicted) in zip(
        axes,
        test_predictions.items()
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

        ax.plot(
            [0, upper],
            [0, upper],
            "k--"
        )

        ax.set_xlim(
            0,
            upper
        )

        ax.set_ylim(
            0,
            upper
        )

        ax.set_title(model_name)

        ax.set_xlabel(
            "Actual headcount"
        )

        ax.set_ylabel(
            "Predicted headcount"
        )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "predicted_vs_actual.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 16. ERROR BY SPACE
    # ==========================================

    space_results = []

    for model_name, predictions in (
        test_predictions.items()
    ):

        temp = test[
            ["space", TARGET]
        ].copy()

        temp["prediction"] = predictions

        for space, group in (
            temp.groupby("space")
        ):

            space_results.append({
                "Model": model_name,
                "Space": space,
                "MAE": mean_absolute_error(
                    group[TARGET],
                    group["prediction"]
                ),
                "RMSE": np.sqrt(
                    mean_squared_error(
                        group[TARGET],
                        group["prediction"]
                    )
                ),
                "Test Hours": len(group)
            })

    space_results_df = pd.DataFrame(
        space_results
    )

    space_results_df.to_csv(
        RESULTS_DIR
        / "prediction_results_by_space.csv",
        index=False
    )

    plt.figure(
        figsize=(11, 6)
    )

    sns.barplot(
        data=space_results_df,
        x="Space",
        y="MAE",
        hue="Model"
    )

    plt.title(
        "Prediction Error by Space"
    )

    plt.ylabel(
        "MAE (people)"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "prediction_error_by_space.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 17. SAVE TEST PREDICTIONS
    # ==========================================

    output = test[
        [
            "date",
            "space",
            "floorspaceid",
            "day_of_week",
            "hour_of_day",
            TARGET
        ]
    ].copy()

    for model_name, predictions in (
        test_predictions.items()
    ):

        column_name = (
            "prediction_"
            + model_name.lower().replace(" ", "_")
        )

        output[column_name] = predictions

    output.to_csv(
        RESULTS_DIR / "test_predictions.csv",
        index=False
    )

    # ==========================================
    # 18. FINAL SUMMARY
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

    print(
        "\nPrediction errors by space:"
    )

    print(
        space_results_df.to_string(
            index=False
        )
    )

    print("\nSpace ID mapping:")

    for space_id, number in (
        space_mapping.items()
    ):

        print(
            f"Space {number}: {space_id}"
        )

    print("\nResults saved in:")

    print(RESULTS_DIR)


# ==========================================
# 19. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
