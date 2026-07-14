# Sentiment Style Transfer System

Twitter/X sentiment classification and negative-to-positive style transfer.

Type a topic. Gemini generates tweets about it. A TF-IDF + LogisticRegression classifier
trained on Sentiment140 labels each one Positive or Negative. Every negative tweet is then
rewritten as a positive one that keeps the original content.

Based on **Delete, Retrieve, Generate** (Li et al., NAACL 2018) — https://arxiv.org/abs/1804.06437

---

## Method

There is no parallel corpus of (negative tweet, same tweet made positive), so seq2seq is not an
option. DRG splits a sentence into *attribute* (sentiment) and *content*, swaps only the attribute,
then regenerates.

```
Input     the battery on this phone is terrible and the support team is useless

DELETE    strip the negative sentiment words
          -> the battery on this phone is ___ and the support team is ___

RETRIEVE  find the positive tweet with the nearest content,
          borrow the positive words it was using

GENERATE  write a fluent positive tweet from template + borrowed words
```

Implementation:

| Step | How |
|---|---|
| DELETE | Sentiment markers taken from the LogisticRegression coefficients, filtered by the paper's salience ratio |
| RETRIEVE | TF-IDF nearest neighbour over 30k positive tweets, each indexed by its content template |
| GENERATE | Gemini, prompted with the template and the borrowed markers |
| RERANK | k candidates scored by P(positive) x content similarity; best one wins |

---

## Layout

```
├── app.py                                Streamlit demo
├── requirements.txt
│
├── Twitter_Sentiment_NLP_updated.ipynb   Part 1: classifier.  Part 2: builds the DRG artifacts.
├── Style_Transfer_DRG.ipynb              Evaluation on held-out tweets. Produces the results table.
│
├── trained_model.sav                     classifier
├── vectorizer.sav                        TF-IDF vectorizer
├── drg_markers.json                      sentiment markers  -> DELETE
└── drg_retrieval.sav                     retrieval index    -> RETRIEVE
```

The four artifacts must sit next to `app.py`. Notebook 1 produces them as `project_bundle.zip` —
unzip it and copy them into the repository root.

Only notebook 1 feeds the app. Notebook 2 produces nothing the app needs; it is the evaluation.

---

## Setup

```
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
streamlit run app.py
```

Get a key at https://aistudio.google.com/apikey. Alternatives to the env var: put it in
`.streamlit/secrets.toml`, or paste it into the field the app shows on startup. Do not hardcode it
in `app.py` or a notebook cell.

---

## Evaluation

`Style_Transfer_DRG.ipynb` scores each method on 100 held-out negative tweets.

- **Style accuracy** — fraction of outputs the classifier now calls Positive.
- **self-BLEU** — BLEU against the input. Measures content preservation.

They trade off. "I love it" for every input scores perfectly on style and zero on content; copying
the input does the reverse. The table includes a `Copy input` row as that floor, and a zero-shot row
(the LLM asked only to "make this positive", with no template or markers) as the control.

**Limitations.** Style accuracy is judged by the same classifier family that produced the markers,
so it reads optimistically. Sentiment140 has no human reference rewrites, so self-BLEU rewards
copying. Its labels are emoticon-derived and noisy, which drags topic words like `phone` and
`battery` toward one polarity and makes marker extraction harder than on Yelp.

---

## Preprocessing

Two pipelines, not interchangeable:

- `stemming()` — what the **classifier** was trained on. Lossy and irreversible.
- `light_clean()` — human-readable. What **DELETE and RETRIEVE** run on.

The marker vectorizer uses `token_pattern=r"(?u)\b\w+\b"` so its vocabulary matches a plain
whitespace split, which is what DELETE performs. Both notebooks and `app.py` carry identical copies
of `light_clean()` and `delete_markers()` — do not edit one without the others.

---

## Note

Pickled scikit-learn objects are version-sensitive. If unpickling warns or fails, match the training
version: check `import sklearn; print(sklearn.__version__)` in the notebook, then
`pip install scikit-learn==<that_version>`.
