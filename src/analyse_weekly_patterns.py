
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data files"
    / "processed"
    / "occupancy_hourly.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading hourly occupancy data...")

df = pd.read_csv(INPUT_FILE)

# Only use hours with at least 80% coverage
df = df[df["coverage_fraction"] >= 0.80].copy()

# Remove missing headcounts
df = df.dropna(subset=["headcount_time_weighted"])

# Create shorter space labels
space_ids = sorted(df["floorspaceid"].unique())

space_labels = {
    space_id: f"Space {i+1}"
    for i, space_id in enumerate(space_ids)
}

df["space"] = df["floorspaceid"].map(space_labels)

# Parse local dates
df["date"] = pd.to_datetime(df["date"])

df["year_month"] = df["date"].dt.to_period("M").astype(str)

DAYS = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

# ==========================================
# 3. WEEKLY OCCUPANCY HEATMAPS
# ==========================================

print("Creating weekly heatmaps...")

for space in sorted(df["space"].unique()):

    space_df = df[df["space"] == space]

    weekly = (
        space_df.groupby(["day_of_week", "hour_of_day"])
        ["headcount_time_weighted"]
        .mean()
        .reset_index()
    )

    heatmap_data = weekly.pivot(
        index="day_of_week",
        columns="hour_of_day",
        values="headcount_time_weighted"
    )

    heatmap_data = heatmap_data.reindex(
        index=DAYS,
        columns=range(24)
    )

    plt.figure(figsize=(14, 5))

    sns.heatmap(
        heatmap_data,
        cmap="YlGnBu",
        linewidths=0.3,
        cbar_kws={"label": "Average headcount"}
    )

    plt.title(f"Weekly Occupancy Heatmap - {space}")
    plt.xlabel("Hour of day")
    plt.ylabel("Day of week")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / f"weekly_heatmap_{space.replace(' ', '_')}.png",
        dpi=200
    )

    plt.close()

# ==========================================
# 4. DAILY OCCUPANCY PROFILES
# ==========================================

print("Creating daily occupancy profiles...")

for space in sorted(df["space"].unique()):

    space_df = df[df["space"] == space]

    daily_profile = (
        space_df.groupby(["day_of_week", "hour_of_day"])
        ["headcount_time_weighted"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    for day in DAYS:

        day_data = daily_profile[
            daily_profile["day_of_week"] == day
        ].sort_values("hour_of_day")

        plt.plot(
            day_data["hour_of_day"],
            day_data["headcount_time_weighted"],
            label=day,
            linewidth=2
        )

    plt.xlabel("Hour of day")
    plt.ylabel("Average headcount")
    plt.title(f"Daily Occupancy Profiles - {space}")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / f"daily_profile_{space.replace(' ', '_')}.png",
        dpi=200
    )

    plt.close()

# ==========================================
# 5. WEEKDAY COMPARISON
# ==========================================

print("Creating weekday comparison...")

weekday_average = (
    df.groupby(["space", "day_of_week"])
    ["headcount_time_weighted"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=weekday_average,
    x="day_of_week",
    y="headcount_time_weighted",
    hue="space",
    order=DAYS
)

plt.xlabel("Day of week")
plt.ylabel("Average headcount")
plt.title("Average Occupancy by Weekday and Space")
plt.xticks(rotation=30)
plt.legend(title="Space")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "weekday_comparison.png",
    dpi=200
)

plt.close()

# ==========================================
# 6. MONTHLY OCCUPANCY TRENDS
# ==========================================

print("Creating monthly occupancy trends...")

monthly_average = (
    df.groupby(["space", "year_month"])
    ["headcount_time_weighted"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(14, 6))

sns.lineplot(
    data=monthly_average,
    x="year_month",
    y="headcount_time_weighted",
    hue="space",
    marker="o"
)

plt.xlabel("Month")
plt.ylabel("Average headcount")
plt.title("Monthly Occupancy Trends by Space")
plt.xticks(rotation=60)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "monthly_occupancy_trends.png",
    dpi=200
)

plt.close()

# ==========================================
# 7. PRINT SUMMARY
# ==========================================

print("\n===================================")
print("WEEKLY PATTERN ANALYSIS COMPLETE")
print("===================================")

print("\nReliable hourly records:", len(df))

print("\nSpace ID mapping:")
for space_id, label in space_labels.items():
    print(f"{label}: {space_id}")

print("\nAverage headcount by weekday:")
print(
    df.groupby("day_of_week")["headcount_time_weighted"]
    .mean()
    .reindex(DAYS)
)

print("\nAverage headcount by space:")
print(
    df.groupby("space")["headcount_time_weighted"]
    .mean()
)

print("\nReliable hours per space:")
print(df["space"].value_counts())

print("\nGraphs saved in:")
print(RESULTS_DIR)
