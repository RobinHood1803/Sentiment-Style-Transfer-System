# 🐦 Tweet Sentiment Analyzer

A Streamlit front end for your Twitter sentiment model. You type a topic →
**Gemini** generates a batch of realistic tweets about it → **your trained
model** (`trained_model.sav` + `vectorizer.sav`) classifies each one as
**Positive** or **Negative**.

## Folder layout

```
your-folder/
├── app.py
├── requirements.txt
├── trained_model.sav      ← from your notebook
└── vectorizer.sav         ← from your notebook
```

> The two `.sav` files must sit next to `app.py` (or point to them in the
> sidebar). They are **not** included here — copy them from your project.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Add your Gemini API key

Get a free key at https://aistudio.google.com/apikey, then either:

- paste it into the **sidebar** when the app runs, **or**
- set an env var:  `export GEMINI_API_KEY=your_key_here`, **or**
- create `.streamlit/secrets.toml`:
  ```toml
  GEMINI_API_KEY = "your_key_here"
  ```

## 3. Run

```bash
streamlit run app.py
```

## How it matches your notebook

The app reproduces your `stemming()` preprocessing **exactly** before calling
`vectorizer.transform()`:

1. strip everything that isn't a–z / A–Z
2. lowercase + split
3. Porter-stem each word, dropping English stopwords (NLTK)
4. re-join with spaces

Labels follow your training: **1 = Positive, 0 = Negative**. Confidence is taken
from `model.predict_proba`.

## ⚠️ Version note

Pickled scikit-learn objects are sensitive to the scikit-learn version. If you
hit an "incompatible version" warning or an unpickling error, install the same
scikit-learn version you used to train (check with `pip show scikit-learn` in
your notebook environment) - e.g. `pip install scikit-learn==<that_version>`.
