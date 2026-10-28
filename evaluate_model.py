from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

base_dir = Path(__file__).resolve().parent

# Load selected trained model
model = joblib.load(
    base_dir / "models" / "phishing_model.joblib"
)

# Load untouched test dataset
test_df = pd.read_csv(
    base_dir / "data" / "test.csv"
)

X_test = test_df["text"].fillna("")
y_test = test_df["label"]

# Predict
predictions = model.predict(X_test)

# Calculate results
results = (
    "FINAL TEST RESULTS\n"
    f"Test emails: {len(test_df)}\n"
    f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n"
    f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}\n"
    f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}\n"
    f"F1-score: {f1_score(y_test, predictions, zero_division=0):.4f}\n\n"
    "Classification Report:\n"
    + classification_report(
        y_test, predictions, zero_division=0
    )
    + "\nConfusion Matrix:\n"
    + str(confusion_matrix(y_test, predictions))
)

print(results)

# Save results for GitHub documentation
output_file = base_dir / "models" / "test_results.txt"
output_file.write_text(results, encoding="utf-8")

print("\nFINAL TEST COMPLETED!")
print("Results saved to:", output_file)