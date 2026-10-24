
"""
Occupancy Sensor Data Cleaning
==============================

Clean a large historical building occupancy dataset using
chunk-based processing.

The original dataset contains approximately nine million
occupancy sensor observations.

The cleaning process validates sensor identifiers, timestamps,
occupancy statuses and headcounts, removes exact duplicate
records within each chunk, and saves the cleaned observations
incrementally.

Methodology
-----------
The raw CSV is processed in chunks of 100,000 observations
to avoid loading the complete dataset into memory.

Each chunk undergoes the following operations:

1. Convert collected timestamps to UTC.
2. Convert timestamps to Melbourne local time.
3. Convert headcounts to numeric values.
4. Remove observations with missing or invalid fields.
5. Remove duplicate records within the chunk.
6. Save valid observations to the processed CSV.
7. Update summary statistics.

Inputs
------
data files/raw/5occupancySensor_MayToDec2024_9MRows.csv
    Raw occupancy sensor observations containing:
    - deviceid
    - floorspaceid
    - occupancystatus
    - headcount
    - collecteddate

Outputs
-------
data files/processed/occupancy_clean_full.csv
    Cleaned occupancy observations containing:
    - deviceid
    - floorspaceid
    - timestamp_utc
    - timestamp_melbourne
    - occupancystatus
    - headcount

Assumptions
-----------
- collecteddate represents a timestamp in UTC.
- headcount represents the estimated number of people present.
- Negative headcounts are invalid.
- Only the three specified occupancy statuses are accepted.
- Duplicate records are identified using device ID, space ID,
  timestamp, occupancy status and headcount.

Limitations
-----------
Duplicate removal is performed independently within each
chunk. Duplicate observations appearing in separate chunks
may therefore remain in the cleaned dataset.

The cleaning process does not independently verify the
accuracy of sensor measurements.

Usage
-----
Run from the project root directory:

    python -u src/clean_occupancy_data.py
"""

import pandas as pd

from pathlib import Path
from collections import Counter


# ==========================================
# 1. FILE PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = (
    PROJECT_ROOT
    / "data files"
    / "raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data files"
    / "processed"
)

INPUT_FILE = (
    RAW_DIR
    / "5occupancySensor_MayToDec2024_9MRows.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "occupancy_clean_full.csv"
)


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
# 3. MAIN CLEANING FUNCTION
# ==========================================

def main():
    """
    Clean the raw occupancy dataset and save the results.

    The raw CSV is read in fixed-size chunks. Each chunk
    undergoes timestamp conversion, numerical validation,
    invalid-record filtering and duplicate removal.

    Cleaned observations are written incrementally to the
    processed output CSV.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The cleaned dataset is saved to OUTPUT_FILE.
        Processing statistics are printed to the terminal.

    Raises
    ------
    FileNotFoundError
        If the raw occupancy CSV cannot be found.

    Notes
    -----
    Duplicate removal is limited to records within the
    same chunk.

    The output CSV is overwritten when the first non-empty
    chunk is saved. Later chunks are appended.

    This function processes the complete raw dataset and
    may take several minutes to execute.
    """

    # --------------------------------------
    # Check input file
    # --------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Occupancy CSV not found: {INPUT_FILE}"
        )

    # Create output directory if required.

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Using occupancy file:",
        INPUT_FILE
    )

    # ==========================================
    # 4. INITIALISE SUMMARY STATISTICS
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

    print(
        "\nProcessing occupancy data..."
    )

    # ==========================================
    # 5. PROCESS CSV IN CHUNKS
    # ==========================================

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

        # Assumes collecteddate is recorded in UTC.

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
        # Convert headcount to numeric
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
            & chunk["occupancystatus"].isin(
                VALID_STATUSES
            )
            & chunk["headcount"].notna()
            & (chunk["headcount"] >= 0)
        )

        chunk = chunk.loc[valid].copy()

        invalid_rows += (
            original_chunk_size - len(chunk)
        )

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

        duplicate_rows += (
            before_duplicates - len(chunk)
        )

        # --------------------------------------
        # Select output columns
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
        # Update summary statistics
        # --------------------------------------

        cleaned_rows += len(chunk)

        space_counts.update(
            chunk["floorspaceid"]
            .value_counts()
            .to_dict()
        )

        status_counts.update(
            chunk["occupancystatus"]
            .value_counts()
            .to_dict()
        )

        # --------------------------------------
        # Update date range
        # --------------------------------------

        if not chunk.empty:

            chunk_min = (
                chunk["timestamp_melbourne"].min()
            )

            chunk_max = (
                chunk["timestamp_melbourne"].max()
            )

            if (
                min_date is None
                or chunk_min < min_date
            ):

                min_date = chunk_min

            if (
                max_date is None
                or chunk_max > max_date
            ):

                max_date = chunk_max

        # --------------------------------------
        # Save cleaned records
        # --------------------------------------

        if not chunk.empty:

            chunk.to_csv(
                OUTPUT_FILE,
                mode=(
                    "w"
                    if first_chunk
                    else "a"
                ),
                header=first_chunk,
                index=False
            )

            first_chunk = False

        # --------------------------------------
        # Display processing progress
        # --------------------------------------

        if chunk_number % 10 == 0:

            print(
                f"Processed {total_rows:,} rows | "
                f"Kept {cleaned_rows:,}"
            )

    # ==========================================
    # 6. PRINT FINAL SUMMARY
    # ==========================================

    print("\n===================================")
    print("CLEANING COMPLETE")
    print("===================================")

    print(
        "\nOriginal rows:",
        total_rows
    )

    print(
        "Invalid rows removed:",
        invalid_rows
    )

    print(
        "Within-chunk duplicates removed:",
        duplicate_rows
    )

    print(
        "Final cleaned rows:",
        cleaned_rows
    )

    print("\nDate range:")

    print(
        "First:",
        min_date
    )

    print(
        "Last:",
        max_date
    )

    print(
        "\nNumber of unique spaces:",
        len(space_counts)
    )

    print("\nRecords per space:")

    for space, count in (
        space_counts.most_common()
    ):

        print(
            space,
            ":",
            count
        )

    print("\nOccupancy statuses:")

    for status, count in (
        status_counts.items()
    ):

        print(
            status,
            ":",
            count
        )

    print("\nSaved to:")

    print(OUTPUT_FILE)


# ==========================================
# 7. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
