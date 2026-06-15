"""
train_emotion_model.py
======================
Trains the multi-emotion classifier that replaces the old binary
(positive/negative) Sentiment140 model.

Dataset : dair-ai/emotion  (~20k English tweets, 6 single-label emotions:
          sadness, joy, love, anger, fear, surprise)
Model   : TF-IDF  ->  LogisticRegression  (same family as before, now multiclass)

Run once:
    pip install datasets scikit-learn nltk
    python train_emotion_model.py

Produces three artifacts next to app.py:
    trained_model.sav   the fitted LogisticRegression
    vectorizer.sav      the fitted TfidfVectorizer
    labels.json         index -> emotion-name mapping (NEW: the app needs this)
"""

import re
import json
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Preprocessing  — MUST stay identical to the preprocess() used in app.py,
# otherwise the saved vectorizer won't line up with what the app feeds it.
# ---------------------------------------------------------------------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def preprocess(text: str) -> str:
    s = re.sub("[^a-zA-Z]", " ", str(text))
    s = s.lower().split()
    s = [STEMMER.stem(w) for w in s if w not in STOP_WORDS]
    return " ".join(s)


# ---------------------------------------------------------------------------
# 1) Load the emotion dataset (replaces Sentiment140)
# ---------------------------------------------------------------------------
print("Loading dair-ai/emotion …")
ds = load_dataset("dair-ai/emotion")            # default 'split' config: train/validation/test

# Pull the canonical label names straight from the dataset, e.g.
# ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']  (index = class id)
LABELS = ds["train"].features["label"].names
print("Emotions:", LABELS)

# ---------------------------------------------------------------------------
# 2) Vectorize + train
# ---------------------------------------------------------------------------
print("Preprocessing …")
X_train = [preprocess(t) for t in ds["train"]["text"]]
y_train = list(ds["train"]["label"])
X_test = [preprocess(t) for t in ds["test"]["text"]]
y_test = list(ds["test"]["label"])

print("Fitting TF-IDF …")
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

print("Training LogisticRegression (multiclass) …")
clf = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",   # emotion classes are imbalanced (joy/sadness >> love/surprise)
    n_jobs=-1,
)
clf.fit(Xtr, y_train)

# ---------------------------------------------------------------------------
# 3) Evaluate
# ---------------------------------------------------------------------------
print("\n=== Test-set performance ===")
print(classification_report(y_test, clf.predict(Xte), target_names=LABELS, digits=3))

# ---------------------------------------------------------------------------
# 4) Save artifacts  (NOTE: labels.json is new)
# ---------------------------------------------------------------------------
with open("trained_model.sav", "wb") as f:
    pickle.dump(clf, f)
with open("vectorizer.sav", "wb") as f:
    pickle.dump(vectorizer, f)
with open("labels.json", "w") as f:
    json.dump(LABELS, f)

print("\nSaved: trained_model.sav, vectorizer.sav, labels.json")