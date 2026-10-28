from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data_dir = Path(__file__).resolve().parent / "data"

train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")

texts = pd.concat(
    [train["text"], test["text"]],
    ignore_index=True
).fillna("").astype(str)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=30000,
    sublinear_tf=True
)

matrix = vectorizer.fit_transform(texts)

train_x = matrix[:len(train)]
test_x = matrix[len(train):]

matches = []

for start in range(0, len(test), 100):
    similarities = (
        test_x[start:start + 100] @ train_x.T
    ).toarray()

    for i, row in enumerate(similarities):
        train_index = int(row.argmax())
        test_index = start + i
        matches.append(
            (float(row[train_index]), test_index, train_index)
        )

print("\nTOP 5 MOST SIMILAR EMAIL PAIRS")

for score, test_index, train_index in sorted(
    matches, reverse=True
)[:5]:
    print("\n" + "=" * 50)
    print("Similarity:", round(score, 4))
    print("Train label:", train.iloc[train_index]["label"])
    print("Test label:", test.iloc[test_index]["label"])
    print("TRAIN EMAIL:", repr(train.iloc[train_index]["text"][:300]))
    print("TEST EMAIL:", repr(test.iloc[test_index]["text"][:300]))

print("\nINSPECTION COMPLETE!")