# Sentiment Style Transfer System

Twitter/X sentiment classification plus negative-to-positive sentiment style transfer.

You type a topic. Gemini generates a batch of realistic tweets about it. A TF-IDF +
LogisticRegression classifier trained on Sentiment140 labels each tweet Positive or Negative.
Every tweet labelled Negative is then rewritten into a positive one that preserves the original
content, using **Delete, Retrieve, Generate** (Li et al., NAACL 2018).

Paper: https://arxiv.org/abs/1804.06437

---

## The method

DRG solves a problem with no parallel data: there is no corpus of (negative tweet, same tweet
rewritten positive) pairs, so a sequence-to-sequence model cannot simply be trained on it. DRG
sidesteps this by separating a sentence into *attribute* (sentiment) and *content* (everything
else), swapping only the attribute, and regenerating.

```
Input   the battery on this phone is terrible and the support team is useless

DELETE      strip the negative attribute markers
            -> content template:  the battery on this phone is ___ and the support team is ___

RETRIEVE    find the positive tweet whose CONTENT is nearest to that template,
            and borrow the positive markers it was using

GENERATE    write a fluent positive tweet from (template + borrowed markers)

RERANK      generate k candidates, score each as P(positive) x content similarity,
            keep the argmax
```

**How each step is implemented here:**

| Step | Implementation |
|---|---|
| DELETE | Attribute markers mined from the LogisticRegression coefficients, filtered by Li et al.'s salience ratio |
| RETRIEVE | TF-IDF nearest neighbour over an index of 30k positive tweets, each indexed by its *content template* |
| GENERATE | Gemini, prompted with the template and the borrowed markers |
| RERANK | The project's own sentiment classifier scores the candidates |

The paper used an LSTM in the GENERATE slot. Substituting a stronger pretrained generator follows
Sudhakar et al., EMNLP 2019 (https://arxiv.org/abs/1908.09368). The rerank step follows
Prompt-and-Rerank, Suzgun et al., EMNLP 2022 (https://arxiv.org/abs/2205.11503).

---

## Repository layout

```
Sentiment-Style-Transfer-System/
├── app.py                            Streamlit front end (the demo)
├── requirements.txt
├── README.md
│
├── Twitter_Sentiment_NLP_updated.ipynb   Part 1: classifier. Part 2: builds the DRG artifacts.
├── Style_Transfer_DRG.ipynb              The experiment: runs DRG on held-out tweets, produces the results table.
│
├── trained_model.sav                 classifier            (produced by notebook 1)
├── vectorizer.sav                    TF-IDF vectorizer     (produced by notebook 1)
├── drg_markers.json                  attribute markers     (produced by notebook 1) -> DELETE
└── drg_retrieval.sav                 retrieval index       (produced by notebook 1) -> RETRIEVE
```

`app.py` expects those four artifacts **in the same directory as itself**. They ship inside
`project_bundle.zip`, which notebook 1 downloads at the end. Unzip it and copy the four files
into the repository root. (To keep them in a subfolder instead, edit `MODEL_PATH`,
`VECTORIZER_PATH`, `MARKERS_PATH` and `RETRIEVAL_PATH` at the top of `app.py`.)

The two notebooks play different roles and only one of them feeds the app:

- **`Twitter_Sentiment_NLP_updated.ipynb`** produces every artifact the app consumes. Run it first.
- **`Style_Transfer_DRG.ipynb`** produces nothing the app needs. It is the *evaluation*: it runs
  the pipeline over 100 held-out negative tweets and reports how well the method actually works.
  The app is the demo; this notebook is the evidence.

---

## Setup

**1. Install dependencies**

```
pip install -r requirements.txt
```

**2. Provide a Gemini API key**

Get one at https://aistudio.google.com/apikey. Then either:

- set an environment variable: `export GEMINI_API_KEY=your_key_here`, or
- create `.streamlit/secrets.toml` containing `GEMINI_API_KEY = "your_key_here"`, or
- paste it into the password field the app shows on startup.

Do not hardcode the key in `app.py` or in a notebook cell. Notebook JSON stores cell source, so a
pasted key gets committed. On Colab, use the Secrets panel (the key icon in the left sidebar) and
read it with `userdata.get('GEMINI_API_KEY')`.

**3. Run**

```
streamlit run app.py
```

---

## Reproducing the artifacts

1. Open `Twitter_Sentiment_NLP_updated.ipynb`. Part 1 trains the sentiment classifier on
   Sentiment140. Part 2 mines the attribute markers and builds the retrieval index.
2. The final cell writes and downloads `project_bundle.zip`.
3. Unzip; copy `trained_model.sav`, `vectorizer.sav`, `drg_markers.json` and `drg_retrieval.sav`
   next to `app.py`.
4. Optionally run `Style_Transfer_DRG.ipynb` (upload the same zip) to regenerate the results table.

### A note on preprocessing

There are two separate text pipelines, and they must not be mixed up:

- `stemming()` — regex strip to a-z, lowercase, drop stopwords, Porter-stem. This is what the
  **classifier** was trained on, so every input to `vectorizer.transform()` must pass through it.
  It is lossy and irreversible: `"disappointed"` becomes `"disappoint"`, which can never be turned
  back into a fluent tweet.
- `light_clean()` — lowercase, drop URLs, mentions, apostrophes and punctuation. Human-readable
  output. This is what **DELETE and RETRIEVE** operate on.

The marker vectorizer uses `token_pattern=r"(?u)\b\w+\b"` rather than scikit-learn's default so
that its vocabulary matches a plain whitespace split exactly. The DELETE step *is* a whitespace
split, so a mismatch here would cause markers to silently fail to match. Both notebooks and
`app.py` carry identical copies of `light_clean()` and `delete_markers()`; do not edit one without
the others.

---

## Evaluation

`Style_Transfer_DRG.ipynb` scores every method on 100 held-out negative tweets that were not used
to mine the markers.

- **Style accuracy** — the fraction of outputs the classifier now labels Positive. Higher is better.
- **self-BLEU** — BLEU of the output against the input. Measures content preservation. Higher is better.

These two trade off against each other, and neither is meaningful alone. A system that answers
"I love it" to every input scores perfectly on style and near zero on content. A system that copies
its input scores perfectly on content and near zero on style. The `Copy input` row in the results
table is included as exactly that floor.

The table also contains a zero-shot row: the same model, same temperature, asked only to "make this
tweet positive", with no template and no borrowed markers. That is the control condition. It is what
answers the obvious question — *why not just prompt the LLM directly?*

### Known limitations

- Style accuracy is scored by the same classifier family that produced the attribute markers, so it
  is self-referential and reads optimistically. A neutral judge (a separate BERT sentiment model, or
  an LLM-as-judge) would be the correct fix.
- self-BLEU against the input rewards copying. The DRG paper reports BLEU against *human*
  references, which Sentiment140 does not have. This is the main reason the Yelp benchmark exists.
- No fluency metric. GPT-2 perplexity would complete the standard three-column evaluation.
- Sentiment140's labels are emoticon-derived and noisy, which pushes topic words (`phone`,
  `battery`) toward one polarity and makes marker extraction harder than it is on Yelp.

---

## Version note

Pickled scikit-learn objects are sensitive to the scikit-learn version. If unpickling raises an
`InconsistentVersionWarning` or fails outright, install the version used for training. Check it in
the notebook environment with:

```python
import sklearn; print(sklearn.__version__)
```

then locally: `pip install scikit-learn==<that_version>`

---

## References

- Li, Jia, He, Liang. *Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer.* NAACL 2018. https://arxiv.org/abs/1804.06437
- Sudhakar, Upadhyay, Maheswaran. *"Transforming" Delete, Retrieve, Generate Approach for Controlled Text Style Transfer.* EMNLP 2019. https://arxiv.org/abs/1908.09368
- Suzgun, Melas-Kyriazi, Jurafsky. *Prompt-and-Rerank: A Method for Zero-Shot and Few-Shot Arbitrary Textual Style Transfer with Small Language Models.* EMNLP 2022. https://arxiv.org/abs/2205.11503
- Go, Bhayani, Huang. *Twitter Sentiment Classification using Distant Supervision.* (Sentiment140)
- Hu, Liu. *Mining and Summarizing Customer Reviews.* KDD 2004. (opinion lexicon)
