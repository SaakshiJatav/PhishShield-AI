from pathlib import Path
import pandas as pd
import joblib

base = Path(__file__).resolve().parent

# Load model and validation data
model = joblib.load(base / "models" / "phishing_model.joblib")
df = pd.read_csv(base / "data" / "validation.csv")

# Find incorrect predictions
df["prediction"] = model.predict(df["text"].fillna(""))
errors = df[df["label"] != df["prediction"]].copy()

print("Total validation emails:", len(df))
print("Incorrect predictions:", len(errors))

print("Safe emails flagged as phishing:",
      ((errors["label"] == 0) & (errors["prediction"] == 1)).sum())

print("Phishing emails missed:",
      ((errors["label"] == 1) & (errors["prediction"] == 0)).sum())

# Save errors for investigation
output = base / "data" / "validation_errors.csv"
errors.to_csv(output, index=False)

print("Error analysis saved successfully!")
print("Saved to:", output)