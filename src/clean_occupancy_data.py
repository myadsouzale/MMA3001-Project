
import pandas as pd
from pathlib import Path
from collections import Counter

# ==========================================
# 1. FILE PATHS
# ==========================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data files" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data files" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "5occupancySensor_MayToDec2024_9MRows.csv"
OUTPUT_FILE = OUTPUT_DIR / "occupancy_clean_full.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Occupancy CSV not found: {INPUT_FILE}"
    )

print("Using occupancy file:", INPUT_FILE)

# ==========================================
# 2. SETTINGS
# ==========================================

CHUNK_SIZE = 100_000

COLUMNS = [
    "deviceid",
    "floorspaceid",
    "occupancystatus",
    "headcount",
    "collecteddate"
]

VALID_STATUSES = [
    "CurrentlyOccupied",
    "RecentlyOccupied",
    "NotOccupied"
]

# ==========================================
# 3. CLEAN DATA IN CHUNKS
# ==========================================

total_rows = 0
cleaned_rows = 0
invalid_rows = 0
duplicate_rows = 0

space_counts = Counter()
status_counts = Counter()

first_chunk = True
min_date = None
max_date = None

print("\nProcessing occupancy data...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        usecols=COLUMNS,
        dtype={
            "deviceid": "string",
            "floorspaceid": "string",
            "occupancystatus": "string"
        }
    ),
    start=1
):

    total_rows += len(chunk)
    original_chunk_size = len(chunk)

    # --------------------------------------
    # Convert timestamps
    # --------------------------------------

    # Assumes collecteddate is recorded in UTC
    chunk["timestamp_utc"] = pd.to_datetime(
        chunk["collecteddate"],
        errors="coerce",
        utc=True
    )

    chunk["timestamp_melbourne"] = (
        chunk["timestamp_utc"]
        .dt.tz_convert("Australia/Melbourne")
    )

    # --------------------------------------
    # Convert headcount to numbers
    # --------------------------------------

    chunk["headcount"] = pd.to_numeric(
        chunk["headcount"],
        errors="coerce"
    )

    # --------------------------------------
    # Remove invalid records
    # --------------------------------------

    valid = (
        chunk["timestamp_utc"].notna()
        & chunk["floorspaceid"].notna()
        & chunk["deviceid"].notna()
        & chunk["occupancystatus"].isin(VALID_STATUSES)
        & chunk["headcount"].notna()
        & (chunk["headcount"] >= 0)
    )

    chunk = chunk.loc[valid].copy()

    invalid_rows += original_chunk_size - len(chunk)

    # --------------------------------------
    # Remove duplicate records
    # --------------------------------------

    before_duplicates = len(chunk)

    chunk = chunk.drop_duplicates(
        subset=[
            "deviceid",
            "floorspaceid",
            "timestamp_utc",
            "occupancystatus",
            "headcount"
        ]
    )

    duplicate_rows += before_duplicates - len(chunk)

    # --------------------------------------
    # Keep relevant columns
    # --------------------------------------

    chunk = chunk[
        [
            "deviceid",
            "floorspaceid",
            "timestamp_utc",
            "timestamp_melbourne",
            "occupancystatus",
            "headcount"
        ]
    ]

    # --------------------------------------
    # Collect summary statistics
    # --------------------------------------

    cleaned_rows += len(chunk)

    space_counts.update(
        chunk["floorspaceid"].value_counts().to_dict()
    )

    status_counts.update(
        chunk["occupancystatus"].value_counts().to_dict()
    )

    if not chunk.empty:

        chunk_min = chunk["timestamp_melbourne"].min()
        chunk_max = chunk["timestamp_melbourne"].max()

        if min_date is None or chunk_min < min_date:
            min_date = chunk_min

        if max_date is None or chunk_max > max_date:
            max_date = chunk_max

    # --------------------------------------
    # Save incrementally
    # --------------------------------------

    if not chunk.empty:
        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )
        first_chunk = False

    if chunk_number % 10 == 0:
        print(
            f"Processed {total_rows:,} rows | "
            f"Kept {cleaned_rows:,}"
        )

# ==========================================
# 4. PRINT FINAL SUMMARY
# ==========================================

print("\n===================================")
print("CLEANING COMPLETE")
print("===================================")

print("\nOriginal rows:", total_rows)
print("Invalid rows removed:", invalid_rows)
print("Within-chunk duplicates removed:", duplicate_rows)
print("Final cleaned rows:", cleaned_rows)

print("\nDate range:")
print("First:", min_date)
print("Last:", max_date)

print("\nNumber of unique spaces:", len(space_counts))

print("\nRecords per space:")
for space, count in space_counts.most_common():
    print(space, ":", count)

print("\nOccupancy statuses:")
for status, count in status_counts.items():
    print(status, ":", count)

print("\nSaved to:")
print(OUTPUT_FILE)
