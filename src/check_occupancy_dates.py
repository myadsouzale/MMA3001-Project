
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data files"
    / "processed"
    / "occupancy_clean_full.csv"
)

monthly_counts = {}

print("Checking monthly occupancy coverage...")

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=["timestamp_melbourne", "floorspaceid"],
    chunksize=200_000
):
    timestamps = pd.to_datetime(
        chunk["timestamp_melbourne"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert("Australia/Melbourne")

    months = timestamps.dt.strftime("%Y-%m")

    counts = months.value_counts()

    for month, count in counts.items():
        monthly_counts[month] = (
            monthly_counts.get(month, 0) + count
        )

print("\nRECORDS BY MONTH")
print("----------------")

for month in sorted(monthly_counts):
    print(f"{month}: {monthly_counts[month]:,}")

print("\nTOTAL:", sum(monthly_counts.values()))
