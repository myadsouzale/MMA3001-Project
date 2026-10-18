
import pandas as pd
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data files" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find occupancy CSV anywhere in the project
occ_files = [
    file for file in PROJECT_ROOT.rglob("*.csv")
    if "occupancysensor" in file.name.lower()
    and "processed" not in file.parts
]

if not occ_files:
    raise FileNotFoundError("Occupancy CSV not found.")

occ_file = occ_files[0]

print("Using occupancy file:", occ_file)

# ==========================================
# 2. SELECT DATE RANGE
# ==========================================

START_DATE = pd.Timestamp(
    "2024-05-01", tz="Australia/Melbourne"
)

END_DATE = pd.Timestamp(
    "2024-06-13", tz="Australia/Melbourne"
)

# Includes May 1 through June 12

# ==========================================
# 3. READ OCCUPANCY DATA IN CHUNKS
# ==========================================

CHUNK_SIZE = 100_000

total_rows = 0
filtered_rows = 0
chunks_to_keep = []

columns_needed = [
    "deviceid",
    "floorspaceid",
    "occupancystatus",
    "headcount",
    "collecteddate"
]

print("\nProcessing occupancy data...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        occ_file,
        chunksize=CHUNK_SIZE,
        usecols=columns_needed,
        dtype={
            "deviceid": "string",
            "floorspaceid": "string"
        }
    ),
    start=1
):

    total_rows += len(chunk)

    # Convert timestamps from UTC to Melbourne time
    chunk["timestamp_utc"] = pd.to_datetime(
        chunk["collecteddate"],
        errors="coerce",
        utc=True
    )

    chunk["timestamp_melbourne"] = (
        chunk["timestamp_utc"]
        .dt.tz_convert("Australia/Melbourne")
    )

    # Filter to matching environmental date range
    chunk = chunk[
        (chunk["timestamp_melbourne"] >= START_DATE) &
        (chunk["timestamp_melbourne"] < END_DATE)
    ].copy()

    filtered_rows += len(chunk)

    if not chunk.empty:
        chunks_to_keep.append(chunk)

    if chunk_number % 10 == 0:
        print(
            f"Processed {total_rows:,} rows, "
            f"kept {filtered_rows:,}"
        )

# ==========================================
# 4. COMBINE FILTERED DATA
# ==========================================

if not chunks_to_keep:
    raise ValueError(
        "No occupancy records found in selected date range."
    )

occ_df = pd.concat(chunks_to_keep, ignore_index=True)

print("\nRows within date range:", len(occ_df))

# ==========================================
# 5. CLEAN OCCUPANCY VALUES
# ==========================================

# Convert headcount to numeric
occ_df["headcount"] = pd.to_numeric(
    occ_df["headcount"],
    errors="coerce"
)

# Remove negative headcounts
occ_df.loc[
    occ_df["headcount"] < 0,
    "headcount"
] = float("nan")

# Remove rows without valid timestamps or space IDs
occ_df = occ_df.dropna(
    subset=[
        "timestamp_utc",
        "floorspaceid"
    ]
)

# Remove exact duplicate records
occ_df = occ_df.drop_duplicates(
    subset=[
        "deviceid",
        "floorspaceid",
        "timestamp_utc",
        "occupancystatus",
        "headcount"
    ]
)

# Sort chronologically
occ_df = occ_df.sort_values(
    ["floorspaceid", "timestamp_utc"]
)

# ==========================================
# 6. SELECT FINAL COLUMNS
# ==========================================

clean_df = occ_df[
    [
        "deviceid",
        "floorspaceid",
        "timestamp_utc",
        "timestamp_melbourne",
        "occupancystatus",
        "headcount"
    ]
].copy()

# ==========================================
# 7. SAVE CLEANED DATA
# ==========================================

output_file = (
    OUTPUT_DIR / "occupancy_clean_may_june_2024.csv"
)

clean_df.to_csv(output_file, index=False)

# ==========================================
# 8. DATA QUALITY SUMMARY
# ==========================================

print("\nCLEANING COMPLETE")

print("Original rows processed:", total_rows)
print("Rows within date range:", filtered_rows)
print("Final cleaned rows:", len(clean_df))

print("\nUnique occupancy spaces:")
print(clean_df["floorspaceid"].value_counts())

print("\nUnique devices:", clean_df["deviceid"].nunique())

print("\nMissing values:")
print(clean_df.isna().sum())

print("\nOccupancy status values:")
print(clean_df["occupancystatus"].value_counts(dropna=False))

print("\nHeadcount statistics:")
print(clean_df["headcount"].describe())

print("\nDate range:")
print(clean_df["timestamp_melbourne"].min())
print(clean_df["timestamp_melbourne"].max())

print("\nFirst five rows:")
print(clean_df.head().to_string(index=False))

print("\nSaved to:", output_file)
