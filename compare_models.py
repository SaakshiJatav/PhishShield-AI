from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Project paths
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
models_dir = base_dir / "models"
models_dir.mkdir(exist_ok=True)

# Load training and validation data
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "validation.csv")

X_train = train_df["text"].fillna("")
y_train = train_df["label"]

X_val = val_df["text"].fillna("")
y_val = val_df["label"]

# Models to compare
classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(
        random_state=42, max_iter=5000
    )
}

results = []

for name, classifier in classifiers.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)
        )),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_val, predictions, zero_division=0
        )
    })

    print(f"{name} completed!")

# Compare results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score", ascending=False
).reset_index(drop=True)

print("\nMODEL COMPARISON RESULTS")
print(results_df.round(4).to_string(index=False))

# Save comparison table
output_file = models_dir / "model_comparison.csv"
results_df.to_csv(output_file, index=False)

print("\nBEST MODEL BY VALIDATION F1:")
print(results_df.loc[0, "Model"])

print("\nComparison saved successfully!")
print("Saved to:", output_file)