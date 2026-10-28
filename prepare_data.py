from pathlib import Path
import pandas as pd

# File paths
base_dir = Path(__file__).resolve().parent
input_file = base_dir / "data" / "phishing_email_subset.csv"
output_file = base_dir / "data" / "emails_clean.csv"

# Load dataset
df = pd.read_csv(input_file, usecols=["text", "label"])

# Clean text
df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str).str.strip()

# Remove empty emails
df = df[df["text"].str.len() >= 10].copy()

# Validate labels
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

# Remove texts with conflicting labels
conflicts = df.groupby("text")["label"].nunique()
bad_texts = conflicts[conflicts > 1].index
df = df[~df["text"].isin(bad_texts)].copy()

# Remove duplicate emails
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

# Save clean dataset
df.to_csv(output_file, index=False, encoding="utf-8")

print("CLEAN DATASET SAVED SUCCESSFULLY!")
print("Total clean emails:", len(df))
print("Label counts:")
print(df["label"].value_counts())
print("Duplicate texts:", df["text"].duplicated().sum())
print("Saved file:", output_file)