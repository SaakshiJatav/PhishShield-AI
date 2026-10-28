import pandas as pd
from pathlib import Path

# Load the dataset
file_path = Path(__file__).parent / "data" / "Phishing_validation_emails.csv"

df = pd.read_csv(file_path)

# Check the dataset
print("DATASET LOADED SUCCESSFULLY!")
print("\nTotal rows and columns:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())
print("\nEmail categories:")
print(df["Email Type"].value_counts())
print("\nDuplicate emails:", df["Email Text"].duplicated().sum())
print("Missing email texts:", df["Email Text"].isnull().sum())
print("Missing labels:", df["Email Type"].isnull().sum())
print("\nUnique emails:", df["Email Text"].nunique())

print("Exact duplicate rows:", df.duplicated().sum())

conflicting = df.groupby("Email Text")["Email Type"].nunique()
print("Emails with conflicting labels:", (conflicting > 1).sum())