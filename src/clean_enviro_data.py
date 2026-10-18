
import pandas as pd
import json
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data files" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data files" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Search the entire project for the environmental CSV
env_files = [
    file for file in PROJECT_ROOT.rglob("*.csv")
    if "envsensor" in file.name.lower()
]

print("Environmental CSV files found:")
for file in env_files:
    print(file)

if not env_files:
    raise FileNotFoundError(
        "Environmental CSV not found anywhere in the project."
    )

env_file = env_files[0]

print("\nUsing environmental file:", env_file)


# ==========================================
# 2. SENSOR AND DATE RANGE
# ==========================================

SENSOR_ID = "6012002000869"

START_DATE = pd.Timestamp(
    "2024-05-01", tz="Australia/Melbourne"
)

END_DATE = pd.Timestamp(
    "2024-06-13", tz="Australia/Melbourne"
)

# Includes all of June 12, excludes June 13

# ==========================================
# 3. LOAD AND FILTER SENSOR
# ==========================================

df = pd.read_csv(
    env_file,
    dtype={"sensorid": "string"}
)

print("Original rows:", len(df))

df = df[df["sensorid"] == SENSOR_ID].copy()

print("Rows for selected sensor:", len(df))

# ==========================================
# 4. EXTRACT MEASUREMENTS FROM JSON
# ==========================================

def extract_measurements(json_text):
    """
    Extract individual environmental readings
    from the nested JSON structure.

    Each reading has:
    - variable.name
    - values[].time
    - values[].value
    """

    results = []

    variable_mapping = {
        "Carbon dioxide": "co2",
        "Temperature": "temperature",
        "Humidity": "humidity"
    }

    try:
        data = json.loads(json_text)

        for item in data:

            variable = item.get("variable", {})
            variable_name = variable.get("name")

            if variable_name not in variable_mapping:
                continue

            column_name = variable_mapping[variable_name]

            for reading in item.get("values", []):

                results.append({
                    "variable": column_name,
                    "timestamp": reading.get("time"),
                    "value": reading.get("value")
                })

    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    return results


# Extract all measurements
records = []

for row in df.itertuples(index=False):

    measurements = extract_measurements(row.jsondata)

    for measurement in measurements:
        records.append({
            "sensorid": str(row.sensorid),
            **measurement
        })

long_df = pd.DataFrame(
    records,
    columns=["sensorid", "variable", "timestamp", "value"]
)

print("Extracted measurement records:", len(long_df))

if long_df.empty:
    raise ValueError("No measurements extracted from JSON.")

# ==========================================
# 5. CONVERT MEASUREMENT TIMESTAMPS
# ==========================================

# The JSON time field contains Unix timestamps
# (seconds since 1970-01-01 UTC).

long_df["timestamp_utc"] = pd.to_datetime(
    pd.to_numeric(long_df["timestamp"], errors="coerce"),
    unit="s",
    utc=True,
    errors="coerce"
)

long_df["timestamp_melbourne"] = (
    long_df["timestamp_utc"]
    .dt.tz_convert("Australia/Melbourne")
)

# Filter to May 1 - June 12 in Melbourne time
long_df = long_df[
    (long_df["timestamp_melbourne"] >= START_DATE) &
    (long_df["timestamp_melbourne"] < END_DATE)
].copy()

print("Measurements within date range:", len(long_df))

# ==========================================
# 6. CLEAN INVALID MEASUREMENTS
# ==========================================

long_df["value"] = pd.to_numeric(
    long_df["value"], errors="coerce"
)

long_df = long_df.dropna(
    subset=["timestamp_utc", "value"]
)

# Flag implausible values as missing
valid_ranges = {
    "co2": (0, 10000),
    "temperature": (-20, 60),
    "humidity": (0, 100)
}

for variable, (minimum, maximum) in valid_ranges.items():

    invalid = (
        (long_df["variable"] == variable) &
        (~long_df["value"].between(minimum, maximum))
    )

    long_df.loc[invalid, "value"] = float("nan")

long_df = long_df.dropna(subset=["value"])

# Remove exact duplicate readings
long_df = long_df.drop_duplicates(
    subset=["sensorid", "timestamp_utc", "variable", "value"]
)

# ==========================================
# 7. CONVERT TO WIDE FORMAT
# ==========================================

# One row per timestamp, with separate columns
# for CO2, temperature and humidity.

clean_df = long_df.pivot_table(
    index=["sensorid", "timestamp_utc"],
    columns="variable",
    values="value",
    aggfunc="mean"
).reset_index()

clean_df.columns.name = None

# Ensure all three columns exist
for column in ["co2", "temperature", "humidity"]:
    if column not in clean_df.columns:
        clean_df[column] = float("nan")

clean_df["timestamp_melbourne"] = (
    clean_df["timestamp_utc"]
    .dt.tz_convert("Australia/Melbourne")
)

clean_df = clean_df[
    [
        "sensorid",
        "timestamp_utc",
        "timestamp_melbourne",
        "co2",
        "temperature",
        "humidity"
    ]
]

clean_df = clean_df.sort_values("timestamp_utc")

# ==========================================
# 8. SAVE CLEANED DATA
# ==========================================

output_file = (
    OUTPUT_DIR / "environmental_clean_may_june_2024.csv"
)

clean_df.to_csv(output_file, index=False)

# ==========================================
# 9. PRINT SUMMARY
# ==========================================

print("\nCLEANING COMPLETE")
print("Cleaned rows:", len(clean_df))

print("\nMissing values:")
print(clean_df.isna().sum())

print("\nDate range:")
print(clean_df["timestamp_melbourne"].min())
print(clean_df["timestamp_melbourne"].max())

print("\nFirst five rows:")
print(clean_df.head().to_string(index=False))

print("\nDescriptive statistics:")
print(
    clean_df[["co2", "temperature", "humidity"]].describe()
)

print("\nSaved to:", output_file)
