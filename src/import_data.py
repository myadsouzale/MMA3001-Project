
import pandas as pd
from pathlib import Path

# Locate the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Locate the raw datasets
DATA_DIR = PROJECT_ROOT / "data files" 

# Find CSV files in the folder
csv_files = list(DATA_DIR.glob("*.csv"))

print("CSV files found:")
for file in csv_files:
    print(file.name)

# Read the environmental dataset
env_file = DATA_DIR / "5EnvSensor_MayToDec2024_180kRows.csv"

env_df = pd.read_csv(env_file)

print("\nENVIRONMENTAL DATA")
print(env_df.head())
print("Rows and columns:", env_df.shape)
print("Column names:", env_df.columns.tolist())


# ==========================================
# OCCUPANCY DATA IMPORT
# ==========================================

# Locate occupancy CSV
occ_file = DATA_DIR / "5occupancySensor_MayToDec2024_9MRows.csv"

# Check file exists
if not occ_file.exists():
    raise FileNotFoundError(
        f"Occupancy CSV not found: {occ_file}"
    )

print("\nIMPORTING OCCUPANCY DATA")

total_rows = 0
unique_spaces = set()
unique_devices = set()
first_rows = None

# Read in chunks to reduce memory usage
for chunk in pd.read_csv(occ_file, chunksize=100_000):

    total_rows += len(chunk)

    # Store first five rows
    if first_rows is None:
        first_rows = chunk.head()

    # Identify unique spaces
    unique_spaces.update(
        chunk["floorspaceid"].dropna().unique()
    )

    # Identify unique devices
    unique_devices.update(
        chunk["deviceid"].dropna().unique()
    )

print("\nOCCUPANCY DATA")
print(first_rows)

print("\nTotal rows:", total_rows)
print("Number of occupancy spaces:", len(unique_spaces))
print("Number of devices:", len(unique_devices))

print("\nFloorSpace IDs:")
for space in sorted(unique_spaces):
    print(space)


# ==========================================
# SENSOR LOCATION EXCEL IMPORT
# ==========================================

# Find Excel files in the raw data folder
excel_files = list(DATA_DIR.glob("*.xlsx"))

print("\nEXCEL FILES FOUND:")
for file in excel_files:
    print(file.name)

if not excel_files:
    raise FileNotFoundError("No Excel files found in raw data folder.")

# Select Excel file
excel_file = excel_files[0]

# Check available worksheets
workbook = pd.ExcelFile(excel_file)

print("\nWORKSHEETS:")
print(workbook.sheet_names)

# Import every worksheet separately
location_sheets = pd.read_excel(
    excel_file,
    sheet_name=None
)

for sheet_name, df in location_sheets.items():

    print(f"\nWORKSHEET: {sheet_name}")
    print("Rows and columns:", df.shape)
    print("Column names:", df.columns.tolist())
    print(df.head())
