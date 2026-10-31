import pandas as pd
EXPECTED_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
    "Engine Condition",
]

# Loading the dataset from Google Drive
#DATA_PATH = '/content/data/engine_data.csv'
DATA_PATH = Path("/content/drive/MyDrive/AIML_Dataset/engine_data.csv")

df = pd.read_csv(DATA_PATH)
labels = df.copy()
print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Failure distribution:")
print(df["Failure"].value_counts())
