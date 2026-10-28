from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent / "data"

df = pd.read_csv(DATA / "emails_clean.csv")[
    ["text", "label"]
].dropna().reset_index(drop=True)

df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)

if not df["label"].isin([0, 1]).all():
    raise ValueError("Labels must be 0 or 1.")

parent = list(range(len(df)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def join(i, j):
    a, b = find(i), find(j)
    if a != b:
        parent[b] = a


# Group identical emails despite formatting differences.
seen = {}

for i, text in enumerate(df["text"]):
    key = re.sub(r"\W+", "", text.lower())

    if key in seen:
        join(i, seen[key])
    else:
        seen[key] = i


# Also group emails with 98% or higher similarity.
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=30000,
    sublinear_tf=True
)

x = vectorizer.fit_transform(df["text"])

for start in range(0, len(df), 100):
    scores = (x[start:start + 100] @ x.T).toarray()

    for local in range(scores.shape[0]):
        i = start + local

        matches = (
            np.flatnonzero(scores[local, i + 1:] >= 0.98)
            + i + 1
        )

        for j in matches:
            join(i, int(j))


roots = [find(i) for i in range(len(df))]

group_info = pd.DataFrame({
    "group": roots,
    "label": df["label"]
})

group_info = group_info.groupby(
    "group", as_index=False
)["label"].agg(lambda s: int(s.mode().iloc[0]))

train_g, rest_g = train_test_split(
    group_info,
    test_size=0.30,
    random_state=42,
    stratify=group_info["label"]
)

val_g, test_g = train_test_split(
    rest_g,
    test_size=0.50,
    random_state=42,
    stratify=rest_g["label"]
)

group_sets = [
    set(g["group"])
    for g in (train_g, val_g, test_g)
]

assert group_sets[0].isdisjoint(group_sets[1])
assert group_sets[0].isdisjoint(group_sets[2])
assert group_sets[1].isdisjoint(group_sets[2])

# Preserve the previous splits before replacing them.
backup = DATA / "backup_original_split"
backup.mkdir(exist_ok=True)

for name in ["train.csv", "validation.csv", "test.csv"]:
    source = DATA / name
    destination = backup / name

    if source.exists() and not destination.exists():
        shutil.copy2(source, destination)


for name, selected_groups in zip(
    ["train.csv", "validation.csv", "test.csv"],
    group_sets
):
    mask = pd.Series(roots).isin(selected_groups)
    part = df.loc[mask, ["text", "label"]]

    part.to_csv(DATA / name, index=False)

    print(
        name,
        "Rows:", len(part),
        "Labels:", part["label"].value_counts().to_dict()
    )

print("SIMILARITY-AWARE SPLIT COMPLETED!")
print("Emails with 98%+ similarity stay in the same group.")