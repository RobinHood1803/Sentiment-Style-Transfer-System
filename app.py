"""
Twitter Sentiment Analysis — Streamlit front end
-------------------------------------------------
1. You enter a topic.
2. Gemini generates realistic tweets about it AND labels each Positive/Negative.
3. Your own trained model (trained_model.sav + vectorizer.sav) also classifies
   each tweet — shown side by side so you can see how often they agree.
"""

import os
import re
import json
import pickle

import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------------------------------------------------------------------
# Configuration (embedded — no sidebar needed)
# ---------------------------------------------------------------------------
# API key is embedded here. An env var overrides it if set.
# NOTE: keep this file private — don't push it to a public repo with the key in it.
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "AIzaSyDeRoHBI2mO5UFScMq5ScLjnEjN-C6JEWE",
)
GEMINI_MODEL = "gemini-2.5-flash"
MODEL_PATH = "trained_model.sav"
VECTORIZER_PATH = "vectorizer.sav"

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="🐦", layout="centered")

st.markdown(
    """
    <style>
      .tweet-card {
        border: 1px solid #e6e9ef; border-radius: 14px; padding: 14px 16px;
        margin-bottom: 12px; background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
      }
      .tweet-text { font-size: 0.97rem; line-height: 1.45; color: #14171a; }
      .verdicts { margin-top: 10px; font-size: 0.85rem; }
      .lbl { color: #657786; font-weight: 600; margin-right: 4px; }
      .badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.76rem; font-weight: 700; letter-spacing: .2px;
      }
      .badge-pos { background: #e6f7ec; color: #128a3e; }
      .badge-neg { background: #fdeaea; color: #c0392b; }
      .match { float: right; font-weight: 700; font-size: 0.8rem; }
      .match-yes { color: #128a3e; }
      .match-no  { color: #d68910; }
      .sep { color: #cfd9de; margin: 0 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_nltk():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    return set(stopwords.words("english")), PorterStemmer()


@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str, vectorizer_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def preprocess(text: str, stop_words, stemmer) -> str:
    """Exact replica of the notebook's stemming() function."""
    s = re.sub("[^a-zA-Z]", " ", text)
    s = s.lower().split()
    s = [stemmer.stem(w) for w in s if w not in stop_words]
    return " ".join(s)


def _norm_sentiment(value: str) -> str:
    return "Positive" if str(value).strip().lower().startswith("pos") else "Negative"


def generate_tweets(topic: str, n: int):
    """Ask Gemini for n tweets, each WITH Gemini's own sentiment label.
    Returns list of {"text": str, "gemini": "Positive"/"Negative"}."""
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    class GeneratedTweet(BaseModel):
        text: str
        sentiment: str  # "Positive" or "Negative"

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        f'Generate {n} short, realistic tweets about "{topic}".\n'
        "Make them sound like real people: casual tone, opinions, some slang, "
        "hashtags or emojis. Deliberately MIX sentiments — include clearly "
        "positive ones and clearly negative ones.\n"
        "Each tweet must be under 280 characters.\n"
        'For EACH tweet, also classify its sentiment as exactly "Positive" or '
        '"Negative" (no neutral — pick the closest).\n'
        "Return ONLY a JSON array of objects, each with keys: text, sentiment."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[GeneratedTweet],
            temperature=1.1,
        ),
    )

    raw = (response.text or "").strip()
    items = []
    try:
        data = json.loads(raw)
        for d in data:
            text = str(d.get("text", "")).strip()
            if text:
                items.append({"text": text, "gemini": _norm_sentiment(d.get("sentiment", ""))})
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return items


def analyze(items, model, vectorizer, stop_words, stemmer):
    """Run YOUR model on each tweet and compare to Gemini's label."""
    texts = [it["text"] for it in items]
    X = vectorizer.transform([preprocess(t, stop_words, stemmer) for t in texts])
    preds = model.predict(X)
    try:
        confidences = model.predict_proba(X).max(axis=1)
    except Exception:
        confidences = [None] * len(texts)

    results = []
    for it, pred, conf in zip(items, preds, confidences):
        model_label = "Positive" if int(pred) == 1 else "Negative"  # 1=pos, 0=neg
        results.append(
            {
                "tweet": it["text"],
                "gemini": it["gemini"],
                "model": model_label,
                "conf": conf,
                "match": it["gemini"] == model_label,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🐦 Tweet Sentiment Analyzer")

topic = st.text_input("Topic", placeholder="e.g. the new iPhone, electric cars, Mondays…")
num_tweets = st.slider("How many tweets to generate", 3, 20, 8)

go = st.button("✨ Generate & Classify", type="primary", use_container_width=True)

if go:
    if not topic.strip():
        st.warning("Type a topic first.")
        st.stop()

    try:
        model, vectorizer = load_artifacts(MODEL_PATH, VECTORIZER_PATH)
    except FileNotFoundError:
        st.error(
            f"Couldn't find `{MODEL_PATH}` or `{VECTORIZER_PATH}`. "
            "Put them next to app.py."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load the model/vectorizer: {e}")
        st.stop()

    stop_words, stemmer = load_nltk()

    try:
        with st.spinner("Asking Gemini for tweets…"):
            items = generate_tweets(topic.strip(), num_tweets)
    except Exception as e:
        st.error(f"Gemini request failed: {e}")
        st.stop()

    if not items:
        st.warning("Gemini returned no tweets. Try again or tweak the topic.")
        st.stop()

    st.session_state["results"] = analyze(items, model, vectorizer, stop_words, stemmer)

# ---------------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------------
results = st.session_state.get("results")
if results:
    total = len(results)
    matches = sum(1 for r in results if r["match"])
    model_pos = sum(1 for r in results if r["model"] == "Positive")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total tweets", total)
    c2.metric("✅ Gemini ↔ Model agree", f"{matches}/{total}")
    c3.metric("Your model: 😊 / 😠", f"{model_pos} / {total - model_pos}")

    st.subheader("Tweets")
    for r in results:
        g_cls = "badge-pos" if r["gemini"] == "Positive" else "badge-neg"
        m_cls = "badge-pos" if r["model"] == "Positive" else "badge-neg"
        conf_txt = f" {r['conf']*100:.0f}%" if r["conf"] is not None else ""
        match_cls = "match-yes" if r["match"] else "match-no"
        match_txt = "✓ match" if r["match"] else "✗ differ"
        safe = r["tweet"].replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f"""
            <div class="tweet-card">
              <div class="tweet-text">{safe}</div>
              <div class="verdicts">
                <span class="lbl">Gemini:</span>
                <span class="badge {g_cls}">{r['gemini']}</span>
                <span class="sep">|</span>
                <span class="lbl">Your model:</span>
                <span class="badge {m_cls}">{r['model']}{conf_txt}</span>
                <span class="match {match_cls}">{match_txt}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("Enter a topic and hit **Generate & Classify** to see results.")