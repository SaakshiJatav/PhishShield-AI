from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).resolve().parent / "data"

train = pd.read_csv(DATA / "train.csv")
val = pd.read_csv(DATA / "validation.csv")
test = pd.read_csv(DATA / "test.csv")

all_text = pd.concat(
    [train["text"], val["text"], test["text"]],
    ignore_index=True
).fillna("").astype(str)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=30000,
    sublinear_tf=True
)

matrix = vectorizer.fit_transform(all_text)

n_train = len(train)
n_val = len(val)

train_x = matrix[:n_train]
val_x = matrix[n_train:n_train + n_val]
test_x = matrix[n_train + n_val:]


def check(name, first, second):
    high_similarity = 0
    very_high_similarity = 0
    highest = 0.0

    for start in range(0, first.shape[0], 100):
        batch = first[start:start + 100]
        similarities = (batch @ second.T).toarray()
        maximums = similarities.max(axis=1)

        high_similarity += (maximums >= 0.90).sum()
        very_high_similarity += (maximums >= 0.98).sum()
        highest = max(highest, float(maximums.max()))

    print(f"\n{name}")
    print("Emails with similarity >= 90%:", high_similarity)
    print("Emails with similarity >= 98%:", very_high_similarity)
    print("Highest similarity:", round(highest, 4))


print("NEAR-DUPLICATE SIMILARITY CHECK")

check("Validation vs Training", val_x, train_x)
check("Testing vs Training", test_x, train_x)
check("Testing vs Validation", test_x, val_x)

print("\nCHECK COMPLETED!")