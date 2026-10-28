import pandas as pd
from pathlib import Path

file_path = Path(__file__).resolve().parent / "data" / "phishing_email_subset.csv"

df = pd.read_csv(file_path)

print("DATASET LOADED SUCCESSFULLY!")
print("Total rows and columns:", df.shape)
print("Columns:", df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel counts:")
print(df["label"].value_counts(dropna=False))

print("\nUnique email texts:", df["text"].nunique())
print("Duplicate email texts:", df["text"].duplicated().sum())
print("Exact duplicate rows:", df.duplicated().sum())

conflicts = df.groupby("text")["label"].nunique()
print("Texts with conflicting labels:", (conflicts > 1).sum())

print("\nEmpty email texts:", df["text"].fillna("").astype(str).str.strip().eq("").sum())