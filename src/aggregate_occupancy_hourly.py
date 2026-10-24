"""
Hourly Occupancy Aggregation
============================

Processes irregularly sampled building occupancy observations
and converts them into hourly occupancy statistics.

The primary output is a time-weighted mean headcount for each
building space and hour.

Methodology
-----------
Each sensor observation represents occupancy from its timestamp
until the next observation or a maximum of 600 seconds,
whichever occurs first.

Observation intervals crossing hour boundaries are split
between the corresponding hours.

The time-weighted average is calculated as:

    sum(headcount * duration) / sum(duration)

Hourly temporal coverage is calculated as:

    covered_seconds / 3600

Hours with at least 80% coverage are flagged as reliable.

Inputs
------
data files/processed/occupancy_clean_full.csv
    Cleaned occupancy observations containing:
    - floorspaceid
    - timestamp_utc
    - headcount

Outputs
-------
data files/processed/occupancy_hourly.csv
    Hourly occupancy statistics including:
    - time-weighted headcount (people)
    - mean and maximum headcount (people)
    - observation count
    - covered seconds
    - coverage fraction
    - reliability flag
    - Melbourne local calendar features

Assumptions
-----------
- Headcount represents the estimated number of people present.
- Sensor observations remain representative for no more
  than 10 minutes.
- Missing observations are not automatically treated as
  zero occupancy.
- Timestamps are interpreted in UTC before conversion to
  Australia/Melbourne time.

Limitations
-----------
The calculated hourly occupancy depends on the accuracy of
the original sensor readings and the assumed observation
duration.

The 80% coverage threshold is a configurable criterion rather
than a guarantee of sensor accuracy.

Usage
-----
Run from the project root directory:

    python -u src/aggregate_occupancy_hourly.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data files"
    / "processed"
    / "occupancy_clean_full.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data files" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "occupancy_hourly.csv"

# ==========================================
# 2. SETTINGS
# ==========================================

# Maximum time a reading can be carried forward
MAX_HOLD_SECONDS = 600  # 10 minutes

# Minimum hourly coverage for reliable analysis
MIN_COVERAGE = 0.80

def main():
    """
    Load, aggregate and save hourly occupancy statistics.

    Reads cleaned occupancy observations from the configured
    CSV file, calculates hourly statistics and time-weighted
    headcounts, and writes the aggregated results to disk.

    Returns
    -------
    None
        Results are saved to OUTPUT_FILE.

    Notes
    -----
    The function loads the cleaned occupancy dataset into
    memory, which may require substantial RAM.

    The output file is overwritten if it already exists.
    """

    print("Loading cleaned occupancy data...")

    # ==========================================
    # 3. LOAD RELEVANT COLUMNS
    # ==========================================

    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "floorspaceid",
            "timestamp_utc",
            "headcount"
        ],
        dtype={"floorspaceid": "string"}
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp_utc"],
        errors="coerce",
        utc=True
    )

    df["headcount"] = pd.to_numeric(
        df["headcount"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["floorspaceid", "timestamp", "headcount"]
    )

    # Remove any remaining exact duplicate readings
    df = df.drop_duplicates(
        subset=["floorspaceid", "timestamp", "headcount"]
    )

    df = df.sort_values(
        ["floorspaceid", "timestamp"]
    ).reset_index(drop=True)

    print("Records loaded:", f"{len(df):,}")

    # ==========================================
    # 4. CALCULATE HOURLY BASIC STATISTICS
    # ==========================================

    print("Calculating hourly statistics...")

    df["hour"] = df["timestamp"].dt.floor("h")

    basic_hourly = (
        df.groupby(["floorspaceid", "hour"])
        .agg(
            headcount_mean=("headcount", "mean"),
            headcount_max=("headcount", "max"),
            observation_count=("headcount", "count")
        )
        .reset_index()
    )

    # ==========================================
    # 5. CALCULATE DURATION OF EACH READING
    # ==========================================

    print("Calculating time-weighted occupancy...")

    # Find the next observation within each space
    df["next_timestamp"] = (
        df.groupby("floorspaceid")["timestamp"].shift(-1)
    )

    # Cap each reading at 10 minutes
    df["end_timestamp"] = df["timestamp"] + pd.Timedelta(
        seconds=MAX_HOLD_SECONDS
    )

    # Use the earlier of the next reading or 10-min limit
    df["end_timestamp"] = df[
        ["next_timestamp", "end_timestamp"]
    ].min(axis=1)

    # For final reading in each space, use 10-min limit
    missing_end = df["end_timestamp"].isna()

    df.loc[missing_end, "end_timestamp"] = (
        df.loc[missing_end, "timestamp"]
        + pd.Timedelta(seconds=MAX_HOLD_SECONDS)
    )

    # ==========================================
    # 6. SPLIT READINGS AT HOUR BOUNDARIES
    # ==========================================

    # Every reading lasts at most 10 minutes,
    # so it can cross at most one hour boundary.

    df["next_hour"] = df["hour"] + pd.Timedelta(hours=1)

    df["first_end"] = df[
        ["end_timestamp", "next_hour"]
    ].min(axis=1)

    df["duration_1"] = (
        df["first_end"] - df["timestamp"]
    ).dt.total_seconds().clip(lower=0)

    df["duration_2"] = (
        df["end_timestamp"] - df["next_hour"]
    ).dt.total_seconds().clip(lower=0)

    # ==========================================
    # 7. AGGREGATE WEIGHTED VALUES
    # ==========================================

    # Portion of reading within its starting hour
    first_part = df[
        ["floorspaceid", "hour", "headcount", "duration_1"]
    ].copy()

    first_part["weighted_sum"] = (
        first_part["headcount"] * first_part["duration_1"]
    )

    first_part = (
        first_part.groupby(["floorspaceid", "hour"])
        .agg(
            weighted_sum=("weighted_sum", "sum"),
            covered_seconds=("duration_1", "sum")
        )
        .reset_index()
    )

    # Portion crossing into the following hour
    second_part = df[df["duration_2"] > 0][
        ["floorspaceid", "next_hour", "headcount", "duration_2"]
    ].copy()

    second_part["weighted_sum"] = (
        second_part["headcount"] * second_part["duration_2"]
    )

    second_part = second_part.rename(
        columns={"next_hour": "hour"}
    )

    second_part = (
        second_part.groupby(["floorspaceid", "hour"])
        .agg(
            weighted_sum=("weighted_sum", "sum"),
            covered_seconds=("duration_2", "sum")
        )
        .reset_index()
    )

    # Combine portions of readings
    weighted_parts = pd.concat(
        [first_part, second_part],
        ignore_index=True
    )

    weighted_hourly = (
        weighted_parts.groupby(["floorspaceid", "hour"])
        .agg(
            weighted_sum=("weighted_sum", "sum"),
            covered_seconds=("covered_seconds", "sum")
        )
        .reset_index()
    )

    weighted_hourly["headcount_time_weighted"] = (
        weighted_hourly["weighted_sum"]
        / weighted_hourly["covered_seconds"]
    )

    weighted_hourly["coverage_fraction"] = (
        weighted_hourly["covered_seconds"] / 3600
    )

    # ==========================================
    # 8. MERGE HOURLY STATISTICS
    # ==========================================

    hourly = weighted_hourly.merge(
        basic_hourly,
        on=["floorspaceid", "hour"],
        how="left"
    )

    hourly["reliable_hour"] = (
        hourly["coverage_fraction"] >= MIN_COVERAGE
    )

    # Convert UTC hours to Melbourne local time
    hourly["hour_melbourne"] = (
        hourly["hour"].dt.tz_convert("Australia/Melbourne")
    )

    # Add calendar features for weekly analysis
    hourly["date"] = (
        hourly["hour_melbourne"].dt.strftime("%Y-%m-%d")
    )

    hourly["day_of_week"] = (
        hourly["hour_melbourne"].dt.day_name()
    )

    hourly["day_number"] = (
        hourly["hour_melbourne"].dt.dayofweek
    )

    hourly["hour_of_day"] = (
        hourly["hour_melbourne"].dt.hour
    )

    hourly["month"] = (
        hourly["hour_melbourne"].dt.month
    )

    hourly = hourly.sort_values(
        ["floorspaceid", "hour"]
    )

    # ==========================================
    # 9. SAVE RESULTS
    # ==========================================

    columns_to_save = [
        "floorspaceid",
        "hour",
        "hour_melbourne",
        "date",
        "day_of_week",
        "day_number",
        "hour_of_day",
        "month",
        "headcount_mean",
        "headcount_time_weighted",
        "headcount_max",
        "observation_count",
        "covered_seconds",
        "coverage_fraction",
        "reliable_hour"
    ]

    hourly = hourly[columns_to_save]

    hourly.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ==========================================
    # 10. PRINT SUMMARY
    # ==========================================

    print("\n===================================")
    print("HOURLY AGGREGATION COMPLETE")
    print("===================================")

    print("\nTotal hourly rows:", len(hourly))

    print("\nHours per space:")
    print(hourly["floorspaceid"].value_counts())

    print("\nCoverage statistics:")
    print(hourly["coverage_fraction"].describe())

    print("\nReliable hours (>= 80% coverage):")
    print(hourly["reliable_hour"].value_counts())

    print("\nTime-weighted headcount statistics:")
    print(hourly["headcount_time_weighted"].describe())

    print("\nFirst five rows:")
    print(hourly.head().to_string(index=False))

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()