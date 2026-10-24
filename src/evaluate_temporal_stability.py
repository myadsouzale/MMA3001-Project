
"""
Temporal Stability Analysis of Building Occupancy Predictions
=============================================================

Evaluate how occupancy prediction accuracy varies across
different months and monitored building spaces.

Three prediction methods are compared:

1. Space Mean baseline
2. Historical Weekly Pattern baseline
3. Random Forest regression

The analysis uses chronologically separated training,
validation and testing data.

Methodology
-----------
Only hourly observations with at least 80% temporal coverage
are included.

Space 3 is excluded because its available observations do
not extend into the later evaluation periods.

The remaining four spaces retain their original numbering
from the five-space occupancy analysis.

The dataset is split chronologically by unique dates:

- 70% training
- 15% validation
- 15% testing

Models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Prediction errors are also calculated for each month and
building space to investigate temporal stability.

Inputs
------
data files/processed/occupancy_hourly.csv
    Hourly occupancy observations containing space IDs,
    calendar features, coverage fractions and headcounts.

Outputs
-------
results/temporal_stability/model_comparison.csv
    Validation and testing accuracy for each model.

results/temporal_stability/monthly_space_errors.csv
    Monthly prediction errors for individual spaces.

results/temporal_stability/monthly_model_errors.csv
    Monthly prediction errors averaged across spaces,
    weighted by the number of observations.

results/temporal_stability/monthly_prediction_error.png
    Monthly prediction error trends.

results/temporal_stability/prediction_error_by_space.png
    Comparison of monthly prediction errors by space.

Assumptions
-----------
- Historical headcounts approximate actual occupancy.
- Hours with at least 80% coverage are sufficiently
  complete for the analysis.
- Calendar patterns may contain useful predictive information.
- Chronological splitting is appropriate for evaluating
  prediction on later observations.
- Negative predicted occupancy values are not physically
  meaningful and are clipped to zero.

Limitations
-----------
Monthly prediction errors may be influenced by changes in
occupancy behaviour, incomplete sensor coverage and the
number of observations available in each month.

The models do not explicitly account for public holidays,
class timetables, special events or other external drivers.

The analysis measures historical predictive performance
rather than proving that future model performance will
remain constant.

Usage
-----
Run from the project root directory:

    python -u src/evaluate_temporal_stability.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    / "temporal_stability"
)

TARGET = "headcount_time_weighted"

MIN_COVERAGE = 0.80

# Space 3 is excluded because its available data
# does not extend into the later evaluation periods.

EXCLUDED_SPACE = (
    "3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8"
)

# Preserve the original space numbering.

SPACE_MAPPING = {
    "0f3f37f8-0335-41e0-ae87-a1d677dfc7d7": 1,
    "227f1068-9e41-469f-bef0-275f27a0bdba": 2,
    "941f0352-bb69-47d0-a9c5-ac29eb4f7319": 4,
    "cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b": 5
}

FEATURES = [
    "space_number",
    "day_number",
    "hour_of_day",
    "month"
]


# ==========================================
# 2. SPACE MEAN PREDICTION
# ==========================================

def predict_space_mean(data, space_means, global_mean):
    """
    Predict occupancy using historical space averages.

    Each observation receives the mean training headcount
    for its corresponding building space.

    Unknown spaces use the global training average.

    Parameters
    ----------
    data : pandas.DataFrame
        Data requiring predictions. Must contain
        the 'floorspaceid' column.

    space_means : pandas.Series
        Historical mean headcount indexed by floorspaceid.

    global_mean : float
        Mean headcount across all training observations.

    Returns
    -------
    numpy.ndarray
        Predicted hourly headcounts in people.
    """

    return (
        data["floorspaceid"]
        .map(space_means)
        .fillna(global_mean)
        .to_numpy()
    )


# ==========================================
# 3. WEEKLY PATTERN PREDICTION
# ==========================================

def predict_weekly(
    data,
    weekly_means,
    space_hour_means,
    space_means,
    global_mean
):
    """
    Predict occupancy using recurring weekly patterns.

    The model uses the historical average headcount for
    each combination of building space, weekday and hour.

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

    weekly_means : pandas.Series
        Historical mean headcount indexed by
        floorspaceid, day_number and hour_of_day.

    space_hour_means : pandas.Series
        Historical mean headcount indexed by
        floorspaceid and hour_of_day.

    space_means : pandas.Series
        Historical mean headcount indexed by floorspaceid.

    global_mean : float
        Overall training mean headcount.

    Returns
    -------
    numpy.ndarray
        Predicted hourly headcounts in people.
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

    # Fallback to space-hour averages.

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

    # Fallback to space or global averages.

    missing = np.isnan(predictions)

    if missing.any():

        predictions[missing] = predict_space_mean(
            data.loc[missing],
            space_means,
            global_mean
        )

    return predictions


# ==========================================
# 4. FEATURE ENCODING
# ==========================================

def encode(data, columns=None):
    """
    Encode calendar and building-space features.

    Building-space identifiers are one-hot encoded
    to avoid imposing an artificial numerical order.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing the model feature columns.

    columns : pandas.Index or list, optional
        Training feature columns used to align
        validation and testing datasets.

    Returns
    -------
    pandas.DataFrame
        Encoded feature matrix.

    Notes
    -----
    When columns are provided, missing columns are
    filled with zero and extra columns are removed.
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
# 5. GENERATE MODEL PREDICTIONS
# ==========================================

def make_predictions(
    data,
    X,
    space_means,
    global_mean,
    weekly_means,
    space_hour_means,
    rf
):
    """
    Generate predictions using all three models.

    Parameters
    ----------
    data : pandas.DataFrame
        Observations requiring predictions.

    X : pandas.DataFrame
        Encoded feature matrix for the Random Forest.

    space_means : pandas.Series
        Historical mean headcount for each space.

    global_mean : float
        Overall training mean headcount.

    weekly_means : pandas.Series
        Historical average for each space, weekday and hour.

    space_hour_means : pandas.Series
        Historical average for each space and hour.

    rf : sklearn.ensemble.RandomForestRegressor
        Trained Random Forest regression model.

    Returns
    -------
    dict of str to numpy.ndarray
        Non-negative occupancy predictions for the
        Space Mean, Weekly Pattern and Random Forest models.
    """

    return {
        "Space Mean": np.maximum(
            predict_space_mean(
                data,
                space_means,
                global_mean
            ),
            0
        ),

        "Weekly Pattern": np.maximum(
            predict_weekly(
                data,
                weekly_means,
                space_hour_means,
                space_means,
                global_mean
            ),
            0
        ),

        "Random Forest": np.maximum(
            rf.predict(X),
            0
        )
    }


# ==========================================
# 6. MAIN ANALYSIS FUNCTION
# ==========================================

def main():
    """
    Run the complete temporal stability analysis.

    Loads and filters hourly occupancy data, excludes
    Space 3, creates chronological training and evaluation
    datasets, trains three prediction models and calculates
    monthly and space-specific errors.

    Returns
    -------
    None
        CSV results and PNG figures are saved to OUTPUT_DIR.

    Raises
    ------
    FileNotFoundError
        If the hourly occupancy CSV is missing.

    Notes
    -----
    The Random Forest uses 150 trees, maximum depth 12,
    minimum leaf size 10 and random_state=42.

    Existing output files are overwritten when the
    analysis is executed.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==========================================
    # 7. LOAD AND FILTER DATA
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

    # Preserve original five-space numbering.

    df["space_number"] = (
        df["floorspaceid"].map(SPACE_MAPPING)
    )

    df["space"] = (
        "Space "
        + df["space_number"].astype(str)
    )

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
        "Spaces:",
        sorted(df["space"].unique())
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

    train_dates = dates[:train_end]

    val_dates = dates[
        train_end:val_end
    ]

    test_dates = dates[val_end:]

    train = df[
        df["date"].isin(train_dates)
    ].copy()

    validation = df[
        df["date"].isin(val_dates)
    ].copy()

    test = df[
        df["date"].isin(test_dates)
    ].copy()

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
    # 9. TRAIN BASELINE MODELS
    # ==========================================

    space_means = (
        train.groupby("floorspaceid")[TARGET]
        .mean()
    )

    global_mean = train[TARGET].mean()

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

    # ==========================================
    # 10. ENCODE FEATURES
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

    # ==========================================
    # 11. TRAIN RANDOM FOREST
    # ==========================================

    print(
        "\nTraining Random Forest...",
        flush=True
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
    # 12. GENERATE PREDICTIONS
    # ==========================================

    val_predictions = make_predictions(
        validation,
        X_val,
        space_means,
        global_mean,
        weekly_means,
        space_hour_means,
        rf
    )

    test_predictions = make_predictions(
        test,
        X_test,
        space_means,
        global_mean,
        weekly_means,
        space_hour_means,
        rf
    )

    # ==========================================
    # 13. EVALUATE MODELS
    # ==========================================

    print("\nMODEL COMPARISON")

    results = []

    for period, subset, predictions in [
        ("Validation", validation, val_predictions),
        ("Test", test, test_predictions)
    ]:

        for model, predicted in (
            predictions.items()
        ):

            actual = (
                subset[TARGET].to_numpy()
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

            results.append({
                "Period": period,
                "Model": model,
                "MAE": mae,
                "RMSE": rmse,
                "Hours": len(subset)
            })

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    # ==========================================
    # 14. MONTHLY AND SPACE-SPECIFIC ERRORS
    # ==========================================

    detailed_results = []

    for period, subset, predictions in [
        ("Validation", validation, val_predictions),
        ("Test", test, test_predictions)
    ]:

        for model, predicted in (
            predictions.items()
        ):

            temp = subset[
                ["date", "space", TARGET]
            ].copy()

            temp["predicted"] = predicted

            temp["absolute_error"] = (
                temp[TARGET]
                - temp["predicted"]
            ).abs()

            temp["year_month"] = (
                temp["date"]
                .dt.to_period("M")
                .astype(str)
            )

            for (month, space), group in (
                temp.groupby(
                    ["year_month", "space"]
                )
            ):

                detailed_results.append({
                    "Period": period,
                    "Month": month,
                    "Space": space,
                    "Model": model,
                    "MAE": (
                        group["absolute_error"].mean()
                    ),
                    "Hours": len(group)
                })

    detail_df = pd.DataFrame(
        detailed_results
    )

    detail_df.to_csv(
        OUTPUT_DIR / "monthly_space_errors.csv",
        index=False
    )

    # ==========================================
    # 15. MONTHLY ERROR SUMMARY
    # ==========================================

    # Weight each space-specific monthly MAE by
    # its number of observations.

    monthly_errors = (
        detail_df.groupby(
            ["Month", "Model"]
        )
        .apply(
            lambda g: np.average(
                g["MAE"],
                weights=g["Hours"]
            ),
            include_groups=False
        )
        .reset_index(name="MAE")
    )

    # ==========================================
    # 16. MONTHLY ERROR GRAPH
    # ==========================================

    plt.figure(
        figsize=(13, 6)
    )

    sns.lineplot(
        data=monthly_errors,
        x="Month",
        y="MAE",
        hue="Model",
        marker="o"
    )

    plt.title(
        "Monthly Occupancy Prediction Error"
    )

    plt.xlabel("Month")

    plt.ylabel("MAE (people)")

    plt.xticks(
        rotation=45
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "monthly_prediction_error.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 17. ERROR BY SPACE GRAPH
    # ==========================================

    plt.figure(
        figsize=(11, 6)
    )

    sns.barplot(
        data=detail_df,
        x="Space",
        y="MAE",
        hue="Model",
        errorbar=None
    )

    plt.title(
        "Prediction Error by Space and Model"
    )

    plt.xlabel("Space")

    plt.ylabel(
        "Monthly mean MAE (people)"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "prediction_error_by_space.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 18. SAVE MONTHLY SUMMARY
    # ==========================================

    monthly_errors.to_csv(
        OUTPUT_DIR / "monthly_model_errors.csv",
        index=False
    )

    # ==========================================
    # 19. FINAL SUMMARY
    # ==========================================

    print("\n===================================")
    print("TEMPORAL STABILITY ANALYSIS COMPLETE")
    print("===================================")

    print("\nMonthly prediction errors:")

    print(
        monthly_errors.to_string(
            index=False
        )
    )

    print("\nFiles saved to:")

    print(OUTPUT_DIR)


# ==========================================
# 20. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
