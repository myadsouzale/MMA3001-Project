
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. LOAD CLEANED OCCUPANCY DATA
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data files"
    / "processed"
    / "occupancy_clean_full.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only load columns needed for investigation
df = pd.read_csv(
    INPUT_FILE,
    usecols=[
        "floorspaceid",
        "timestamp_utc",
        "occupancystatus",
        "headcount"
    ],
    dtype={"floorspaceid": "string"}
)

df["timestamp"] = pd.to_datetime(
    df["timestamp_utc"],
    utc=True
).dt.tz_convert("Australia/Melbourne")

# ==========================================
# 2. CHOOSE ONE SPACE AND ONE DAY
# ==========================================

# Choose the space with the most records
space_id = df["floorspaceid"].value_counts().idxmax()

# Choose a day with many observations
space_df = df[df["floorspaceid"] == space_id].copy()

space_df["date"] = space_df["timestamp"].dt.date

selected_date = (
    space_df["date"].value_counts().idxmax()
)

day_df = space_df[
    space_df["date"] == selected_date
].sort_values("timestamp").copy()

print("\nSELECTED SPACE:")
print(space_id)

print("\nSELECTED DATE:")
print(selected_date)

print("\nNUMBER OF RECORDS:")
print(len(day_df))

# ==========================================
# 3. DISPLAY SAMPLE RECORDS
# ==========================================

print("\nFIRST 30 RECORDS:")
print(
    day_df[
        ["timestamp", "occupancystatus", "headcount"]
    ].head(30).to_string(index=False)
)

print("\nHEADCOUNT STATISTICS:")
print(day_df["headcount"].describe())

print("\nSTATUS VS HEADCOUNT:")
print(
    day_df.groupby("occupancystatus")["headcount"]
    .agg(["count", "mean", "min", "max"])
)

# ==========================================
# 4. CHECK TIME BETWEEN READINGS
# ==========================================

day_df["seconds_between_readings"] = (
    day_df["timestamp"].diff().dt.total_seconds()
)

print("\nSECONDS BETWEEN READINGS:")
print(day_df["seconds_between_readings"].describe())

# ==========================================
# 5. PLOT HEADCOUNT OVER TIME
# ==========================================

plt.figure(figsize=(13, 5))

plt.step(
    day_df["timestamp"],
    day_df["headcount"],
    where="post",
    linewidth=1
)

plt.xlabel("Time")
plt.ylabel("Headcount")
plt.title(
    f"Occupancy headcount over time\n"
    f"Space: {space_id}\n"
    f"Date: {selected_date}"
)

plt.grid(alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()

output_file = (
    OUTPUT_DIR / "headcount_one_day.png"
)

plt.savefig(output_file, dpi=200)

plt.show()

print("\nGraph saved to:", output_file)
