
"""
Exploratory Analysis of Weekly Building Occupancy Patterns
=========================================================

Analyse recurring occupancy patterns across monitored
building spaces using historical hourly occupancy data.

The analysis investigates how average occupancy varies
between spaces, days of the week, hours of the day and months.

Methodology
-----------
Only hourly observations with at least 80% temporal
coverage and valid time-weighted headcounts are included.

For each monitored space, the script generates:

1. Weekly occupancy heatmaps showing average headcount
   by weekday and hour.
2. Daily occupancy profiles comparing hourly patterns
   across the seven days of the week.

Additional visualisations compare average occupancy
between spaces and examine monthly occupancy trends.

Inputs
------
data files/processed/occupancy_hourly.csv
    Hourly occupancy data containing:
    - floorspaceid
    - date
    - day_of_week
    - hour_of_day
    - coverage_fraction
    - headcount_time_weighted

Outputs
-------
results/weekly_heatmap_Space_*.png
    Weekly occupancy heatmaps for each monitored space.

results/daily_profile_Space_*.png
    Daily occupancy profiles for each monitored space.

results/weekday_comparison.png
    Average occupancy by weekday and building space.

results/monthly_occupancy_trends.png
    Monthly average occupancy trends by space.

Assumptions
-----------
- Time-weighted headcount represents estimated occupancy.
- Hours with at least 80% coverage are suitable for analysis.
- Historical averages can reveal recurring occupancy patterns.
- Space identifiers represent distinct monitored spaces.
- Missing hourly observations are excluded rather than
  automatically interpreted as zero occupancy.

Limitations
-----------
The analysis is descriptive and does not establish
the causes of observed occupancy patterns.

Historical averages may conceal short-term fluctuations,
special events and seasonal changes.

Periods with missing or low-coverage observations may
not be equally represented in the calculated averages.

Usage
-----
Run from the project root directory:

    python -u src/analyse_weekly_patterns.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


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

DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


# ==========================================
# 2. MAIN ANALYSIS FUNCTION
# ==========================================

def main():
    """
    Analyse and visualise historical occupancy patterns.

    Loads hourly occupancy observations, filters records
    using the minimum coverage threshold and calculates
    historical average headcounts for different calendar
    groupings.

    The function generates weekly heatmaps, daily profiles,
    weekday comparisons and monthly trend figures.

    Returns
    -------
    None
        Figures are saved as PNG files in RESULTS_DIR.
        Summary statistics are printed to the terminal.

    Raises
    ------
    FileNotFoundError
        If the hourly occupancy CSV does not exist.

    Notes
    -----
    Average headcounts are calculated from the available
    reliable hourly observations.

    The figures are descriptive visualisations rather
    than forecasts of future occupancy.

    Existing figures with matching filenames are
    overwritten when the script is executed.
    """

    # ==========================================
    # 3. LOAD DATA
    # ==========================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading hourly occupancy data...")

    df = pd.read_csv(INPUT_FILE)

    # Only use hours with at least 80% coverage.

    df = df[
        df["coverage_fraction"] >= MIN_COVERAGE
    ].copy()

    # Remove missing time-weighted headcounts.

    df = df.dropna(
        subset=[TARGET]
    )

    # ==========================================
    # 4. CREATE SPACE LABELS
    # ==========================================

    # Assign consistent labels based on sorted
    # building-space identifiers.

    space_ids = sorted(
        df["floorspaceid"].unique()
    )

    space_labels = {
        space_id: f"Space {i + 1}"
        for i, space_id in enumerate(space_ids)
    }

    df["space"] = (
        df["floorspaceid"].map(space_labels)
    )

    # Parse local calendar dates.

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["year_month"] = (
        df["date"]
        .dt.to_period("M")
        .astype(str)
    )

    # ==========================================
    # 5. WEEKLY OCCUPANCY HEATMAPS
    # ==========================================

    print("Creating weekly heatmaps...")

    for space in sorted(
        df["space"].unique()
    ):

        space_df = df[
            df["space"] == space
        ]

        # Calculate mean headcount for each
        # weekday and hour combination.

        weekly = (
            space_df.groupby(
                ["day_of_week", "hour_of_day"]
            )[TARGET]
            .mean()
            .reset_index()
        )

        # Convert grouped results into a matrix
        # with weekdays as rows and hours as columns.

        heatmap_data = weekly.pivot(
            index="day_of_week",
            columns="hour_of_day",
            values=TARGET
        )

        # Ensure the same weekday and hour ordering
        # for every building space.

        heatmap_data = heatmap_data.reindex(
            index=DAYS,
            columns=range(24)
        )

        plt.figure(
            figsize=(14, 5)
        )

        sns.heatmap(
            heatmap_data,
            cmap="YlGnBu",
            linewidths=0.3,
            cbar_kws={
                "label": "Average headcount"
            }
        )

        plt.title(
            f"Weekly Occupancy Heatmap - {space}"
        )

        plt.xlabel("Hour of day")

        plt.ylabel("Day of week")

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR
            / f"weekly_heatmap_{space.replace(' ', '_')}.png",
            dpi=200
        )

        plt.close()

    # ==========================================
    # 6. DAILY OCCUPANCY PROFILES
    # ==========================================

    print("Creating daily occupancy profiles...")

    for space in sorted(
        df["space"].unique()
    ):

        space_df = df[
            df["space"] == space
        ]

        # Average occupancy for each weekday
        # and hour of the day.

        daily_profile = (
            space_df.groupby(
                ["day_of_week", "hour_of_day"]
            )[TARGET]
            .mean()
            .reset_index()
        )

        plt.figure(
            figsize=(12, 6)
        )

        # Plot one line for each weekday.

        for day in DAYS:

            day_data = daily_profile[
                daily_profile["day_of_week"] == day
            ].sort_values(
                "hour_of_day"
            )

            plt.plot(
                day_data["hour_of_day"],
                day_data[TARGET],
                label=day,
                linewidth=2
            )

        plt.xlabel("Hour of day")

        plt.ylabel("Average headcount")

        plt.title(
            f"Daily Occupancy Profiles - {space}"
        )

        plt.xticks(
            range(0, 24, 2)
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR
            / f"daily_profile_{space.replace(' ', '_')}.png",
            dpi=200
        )

        plt.close()

    # ==========================================
    # 7. WEEKDAY COMPARISON
    # ==========================================

    print("Creating weekday comparison...")

    # Calculate mean occupancy for each space
    # and weekday.

    weekday_average = (
        df.groupby(
            ["space", "day_of_week"]
        )[TARGET]
        .mean()
        .reset_index()
    )

    plt.figure(
        figsize=(12, 6)
    )

    sns.barplot(
        data=weekday_average,
        x="day_of_week",
        y=TARGET,
        hue="space",
        order=DAYS
    )

    plt.xlabel("Day of week")

    plt.ylabel("Average headcount")

    plt.title(
        "Average Occupancy by Weekday and Space"
    )

    plt.xticks(
        rotation=30
    )

    plt.legend(
        title="Space"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "weekday_comparison.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 8. MONTHLY OCCUPANCY TRENDS
    # ==========================================

    print("Creating monthly occupancy trends...")

    # Calculate mean occupancy for each space
    # and calendar month.

    monthly_average = (
        df.groupby(
            ["space", "year_month"]
        )[TARGET]
        .mean()
        .reset_index()
    )

    plt.figure(
        figsize=(14, 6)
    )

    sns.lineplot(
        data=monthly_average,
        x="year_month",
        y=TARGET,
        hue="space",
        marker="o"
    )

    plt.xlabel("Month")

    plt.ylabel("Average headcount")

    plt.title(
        "Monthly Occupancy Trends by Space"
    )

    plt.xticks(
        rotation=60
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "monthly_occupancy_trends.png",
        dpi=200
    )

    plt.close()

    # ==========================================
    # 9. PRINT SUMMARY
    # ==========================================

    print("\n===================================")
    print("WEEKLY PATTERN ANALYSIS COMPLETE")
    print("===================================")

    print(
        "\nReliable hourly records:",
        len(df)
    )

    print("\nSpace ID mapping:")

    for space_id, label in (
        space_labels.items()
    ):

        print(
            f"{label}: {space_id}"
        )

    print(
        "\nAverage headcount by weekday:"
    )

    print(
        df.groupby(
            "day_of_week"
        )[TARGET]
        .mean()
        .reindex(DAYS)
    )

    print(
        "\nAverage headcount by space:"
    )

    print(
        df.groupby(
            "space"
        )[TARGET]
        .mean()
    )

    print(
        "\nReliable hours per space:"
    )

    print(
        df["space"].value_counts()
    )

    print("\nGraphs saved in:")

    print(RESULTS_DIR)


# ==========================================
# 10. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
