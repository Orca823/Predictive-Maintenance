import pandas as pd
from pathlib import Path

EXPECTED_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
    "Engine Condition",
]

DATA_PATH = Path("engine_data.csv")

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
