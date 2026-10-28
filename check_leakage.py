from pathlib import Path
import pandas as pd
import re

base = Path(__file__).resolve().parent / "data"

train = pd.read_csv(base / "train.csv")
val = pd.read_csv(base / "validation.csv")
test = pd.read_csv(base / "test.csv")


def normalize(text):
    text = str(text).lower()
    return re.sub(r"\W+", "", text)


train_texts = set(train["text"].map(normalize))
val_texts = set(val["text"].map(normalize))
test_texts = set(test["text"].map(normalize))

print("DATA LEAKAGE CHECK")

print("Train vs Validation:",
      len(train_texts & val_texts))

print("Train vs Test:",
      len(train_texts & test_texts))

print("Validation vs Test:",
      len(val_texts & test_texts))

print("\nNormalized-text overlap check completed!")