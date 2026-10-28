from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Project folders
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
models_dir = base_dir / "models"
models_dir.mkdir(exist_ok=True)

# Load training and validation datasets
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "validation.csv")

X_train = train_df["text"].fillna("")
y_train = train_df["label"]

X_val = val_df["text"].fillna("")
y_val = val_df["label"]

print("Training emails:", len(X_train))
print("Validation emails:", len(X_val))

# Build NLP + Machine Learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

print("MODEL TRAINING COMPLETED!")

# Validate model performance
predictions = model.predict(X_val)

print("\nVALIDATION RESULTS")
print("Accuracy:", round(accuracy_score(y_val, predictions), 4))
print("Precision:", round(precision_score(y_val, predictions, zero_division=0), 4))
print("Recall:", round(recall_score(y_val, predictions, zero_division=0), 4))
print("F1-score:", round(f1_score(y_val, predictions, zero_division=0), 4))

print("\nClassification Report:")
print(classification_report(y_val, predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predictions))

# Save trained model
model_path = models_dir / "phishing_model.joblib"
joblib.dump(model, model_path)

print("\nMODEL SAVED SUCCESSFULLY!")
print("Saved to:", model_path)