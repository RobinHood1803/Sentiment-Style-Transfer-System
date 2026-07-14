"""
Twitter Sentiment Analysis + Sentiment Style Transfer — Streamlit front end
---------------------------------------------------------------------------
1. You enter a topic.
2. Gemini generates realistic tweets about it AND labels each Positive/Negative.
3. Your own trained model (trained_model.sav + vectorizer.sav) also classifies
   each tweet — shown side by side so you can see how often they agree.
4. Every tweet YOUR model calls Negative is rewritten Negative -> Positive using
   Delete-Retrieve-Generate (Li et al., NAACL 2018):

       DELETE    strip the negative attribute markers      -> content template
       RETRIEVE  nearest positive tweet by content         -> borrow its positive markers
       GENERATE  k candidate rewrites from Gemini, reranked with YOUR OWN classifier

   Step 4 needs drg_markers.json + drg_retrieval.sav, produced by
   Twitter_Sentiment_NLP_updated.ipynb. Without them the app still runs — it just
   skips the rewrite and behaves exactly like the old version.

Files expected next to this script:
    trained_model.sav      vectorizer.sav          (part 1)
    drg_markers.json       drg_retrieval.sav       (part 2, optional)
"""

import os
import re
import json
import pickle

import numpy as np
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-3.1-flash-lite"

MODEL_PATH = "trained_model.sav"
VECTORIZER_PATH = "vectorizer.sav"
MARKERS_PATH = "drg_markers.json"
RETRIEVAL_PATH = "drg_retrieval.sav"

N_CANDIDATES = 4        # rewrites Gemini produces per negative tweet, for the reranker
TOKEN_PATTERN = r"(?u)\b\w+\b"   # must match the notebook


def _get_api_key() -> str:
    """Env var first, then .streamlit/secrets.toml. Never hardcode the key in this file."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="🐦", layout="centered")

st.markdown(
    """
    <style>
      .tweet-card {
        border: 1px solid #e6e9ef; border-radius: 14px; padding: 14px 16px;
        margin-bottom: 4px; background: #ffffff;
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

      .rewrite {
        margin-top: 12px; padding: 10px 12px;
        border-left: 3px solid #128a3e; background: #f4faf6;
        border-radius: 0 8px 8px 0;
      }
      .rewrite-label {
        font-size: 0.7rem; font-weight: 800; color: #128a3e;
        letter-spacing: .6px; text-transform: uppercase; margin-bottom: 5px;
      }
      .rewrite-text { font-size: 0.95rem; line-height: 1.45; color: #14171a; }
      .rewrite-meta { margin-top: 7px; font-size: 0.75rem; color: #657786; }
      .flip-yes { color: #128a3e; font-weight: 700; }
      .flip-no  { color: #d68910; font-weight: 700; }
      .spacer { margin-bottom: 12px; }
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


@st.cache_resource(show_spinner=False)
def load_drg(markers_path: str, retrieval_path: str):
    """Returns (neg_markers, pos_markers, drg_bundle) or None if the files aren't there."""
    if not (os.path.exists(markers_path) and os.path.exists(retrieval_path)):
        return None
    with open(markers_path, "r") as f:
        markers = json.load(f)
    with open(retrieval_path, "rb") as f:
        drg = pickle.load(f)
    return set(markers["neg_markers"]), set(markers["pos_markers"]), drg


# ---------------------------------------------------------------------------
# Text processing — both functions are exact replicas of the notebook's
# ---------------------------------------------------------------------------
def preprocess(text: str, stop_words, stemmer) -> str:
    """Replica of the notebook's stemming(). Feeds the CLASSIFIER only."""
    s = re.sub("[^a-zA-Z]", " ", text)
    s = s.lower().split()
    s = [stemmer.stem(w) for w in s if w not in stop_words]
    return " ".join(s)


def light_clean(content: str) -> str:
    """Replica of the notebook's light_clean(). Feeds DELETE / RETRIEVE."""
    content = content.lower()
    content = re.sub(r"http\S+|www\.\S+", " ", content)
    content = re.sub(r"@\w+", " ", content)
    content = re.sub(r"'", "", content)
    content = re.sub(r"[^a-z\s]", " ", content)
    content = re.sub(r"\s+", " ", content).strip()
    return content


def _norm_sentiment(value: str) -> str:
    return "Positive" if str(value).strip().lower().startswith("pos") else "Negative"


# ---------------------------------------------------------------------------
# DRG — DELETE / RETRIEVE
# ---------------------------------------------------------------------------
def delete_markers(sentence: str, marker_set):
    """DELETE step. Bigrams greedily first, then leftover unigrams."""
    words = sentence.split()
    n = len(words)
    drop = [False] * n
    found = []

    i = 0
    while i < n - 1:
        bg = words[i] + " " + words[i + 1]
        if bg in marker_set:
            drop[i] = drop[i + 1] = True
            found.append(bg)
            i += 2
        else:
            i += 1

    for i in range(n):
        if not drop[i] and words[i] in marker_set:
            drop[i] = True
            found.append(words[i])

    template = " ".join(w for i, w in enumerate(words) if not drop[i])
    return template, found


def retrieve(template: str, drg, top_n: int = 1):
    """RETRIEVE step. Nearest positive tweet by content -> its positive markers."""
    q = drg["retrieval_vectorizer"].transform([template])
    sims = (drg["pos_template_matrix"] @ q.T).toarray().ravel()
    idx = np.argsort(sims)[::-1][:top_n]
    return [
        {
            "markers": drg["pos_template_markers"][i],
            "neighbour": drg["pos_corpus"][i],
            "sim": float(sims[i]),
        }
        for i in idx
    ]


def prepare(tweet: str, tweet_id: int, neg_markers, drg):
    """DELETE + RETRIEVE. Everything GENERATE needs, for one tweet."""
    clean = light_clean(tweet)
    template, deleted = delete_markers(clean, neg_markers)
    hits = retrieve(template, drg)
    borrowed = hits[0]["markers"][:3] if hits else []
    return {
        "id": tweet_id,
        "input": tweet,
        "template": template,
        "deleted": deleted,
        "borrowed": borrowed,
        "neighbour": hits[0]["neighbour"] if hits else "",
        "sim": hits[0]["sim"] if hits else 0.0,
        "template_based": (" ".join(borrowed[:2]) + " " + template).strip(),
    }


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def generate_tweets(topic: str, n: int, api_key: str):
    """Ask Gemini for n tweets, each WITH Gemini's own sentiment label."""
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    class GeneratedTweet(BaseModel):
        text: str
        sentiment: str  # "Positive" or "Negative"

    client = genai.Client(api_key=api_key)

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


REWRITE_PROMPT = """You are doing sentiment style transfer on tweets: NEGATIVE -> POSITIVE.

For each item you get:
- "original"  : the negative tweet
- "template"  : the same tweet with its negative sentiment words deleted. This is the CONTENT
                that must survive the rewrite.
- "borrowed"  : positive sentiment words lifted from a real positive tweet about similar content

Rewrite each tweet so that:
- the sentiment is clearly POSITIVE
- every topic, entity and fact in "template" is preserved -- do not invent new facts, do not drop any
- the "borrowed" words are worked in where they fit naturally (ignore any that do not fit)
- it still reads like a real tweet: casual, under 280 characters, keep any hashtags/emojis in spirit

Give {k} DIFFERENT rewrites per item.
Return ONLY a JSON array of objects with keys: id, rewrites.

Items:
{items}
"""


def gemini_rewrite(records, api_key: str, k: int = N_CANDIDATES):
    """GENERATE step. records: list of prepare() dicts. Returns {id: [candidate, ...]}."""
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    class Rewrite(BaseModel):
        id: int
        rewrites: list[str]

    client = genai.Client(api_key=api_key)

    items = json.dumps(
        [
            {
                "id": r["id"],
                "original": r["input"],
                "template": r["template"],
                "borrowed": r["borrowed"],
            }
            for r in records
        ],
        ensure_ascii=False,
        indent=1,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=REWRITE_PROMPT.format(k=k, items=items),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[Rewrite],
            temperature=1.0,
        ),
    )

    out = {}
    try:
        for d in json.loads((response.text or "").strip()):
            cands = [str(c).strip() for c in d.get("rewrites", []) if str(c).strip()]
            if cands:
                out[int(d["id"])] = cands
    except (json.JSONDecodeError, AttributeError, TypeError, KeyError, ValueError):
        pass
    return out


# ---------------------------------------------------------------------------
# Rerank — your own classifier picks the winner
# ---------------------------------------------------------------------------
def positivity(texts, model, vectorizer, stop_words, stemmer):
    X = vectorizer.transform([preprocess(t, stop_words, stemmer) for t in texts])
    return model.predict_proba(X)[:, 1]  # P(class 1) = P(positive)


def content_sim(original: str, candidate: str, drg) -> float:
    vec = drg["retrieval_vectorizer"]
    va = vec.transform([light_clean(original)])
    vb = vec.transform([light_clean(candidate)])
    return float((va @ vb.T).toarray().ravel()[0])


def rerank(original, candidates, drg, model, vectorizer, stop_words, stemmer):
    """score = P(positive) x content similarity. Style AND content, not just style."""
    p = positivity(candidates, model, vectorizer, stop_words, stemmer)
    sims = np.array([content_sim(original, c, drg) for c in candidates])
    scores = p * sims
    best = int(np.argmax(scores))
    return candidates[best], {
        "p_pos": float(p[best]),
        "sim": float(sims[best]),
        "all": [
            (c, round(float(pp), 3), round(float(ss), 3))
            for c, pp, ss in zip(candidates, p, sims)
        ],
    }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
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
                "transfer": None,
            }
        )
    return results


def run_style_transfer(results, drg_pack, api_key, model, vectorizer, stop_words, stemmer):
    """Rewrite every tweet YOUR model called Negative. Mutates results in place."""
    neg_markers, _pos_markers, drg = drg_pack

    targets = [r for r in results if r["model"] == "Negative"]
    if not targets:
        return results

    records = [prepare(r["tweet"], i, neg_markers, drg) for i, r in enumerate(targets)]

    candidates = {}
    if api_key:
        try:
            candidates = gemini_rewrite(records, api_key)
        except Exception as e:
            st.warning(f"Gemini rewrite failed, falling back to the TemplateBased baseline: {e}")

    for r, rec in zip(targets, records):
        cands = candidates.get(rec["id"], [])
        if cands:
            best, info = rerank(rec["input"], cands, drg, model, vectorizer, stop_words, stemmer)
            rec["output"] = best
            rec["method"] = "DRG + Gemini + Rerank"
            rec["rerank"] = info
        else:
            rec["output"] = rec["template_based"]
            rec["method"] = "DRG TemplateBased (no LLM)"
            rec["rerank"] = {}

        # Does YOUR classifier agree the tone actually flipped?
        p = positivity([rec["output"]], model, vectorizer, stop_words, stemmer)[0]
        rec["p_pos_out"] = float(p)
        rec["flipped"] = bool(p >= 0.5)
        r["transfer"] = rec

    return results


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🐦 Tweet Sentiment Analyzer")
st.caption("Classify tweets — then flip the negative ones positive with Delete-Retrieve-Generate.")

api_key = _get_api_key()
if not api_key:
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        help="Or set the GEMINI_API_KEY env var / add it to .streamlit/secrets.toml.",
    )

drg_pack = load_drg(MARKERS_PATH, RETRIEVAL_PATH)

topic = st.text_input("Topic", placeholder="e.g. the new iPhone, electric cars, Mondays…")
num_tweets = st.slider("How many tweets to generate", 3, 20, 8)

if drg_pack:
    do_transfer = st.checkbox("🔁 Rewrite negative tweets → positive (DRG)", value=True)
else:
    do_transfer = False
    st.info(
        f"Style transfer is off: `{MARKERS_PATH}` / `{RETRIEVAL_PATH}` not found. "
        "Run Part 2 of the notebook and drop them next to app.py."
    )

go = st.button("✨ Generate & Classify", type="primary", use_container_width=True)

if go:
    if not topic.strip():
        st.warning("Type a topic first.")
        st.stop()
    if not api_key:
        st.warning("Add your Gemini API key first.")
        st.stop()

    try:
        model, vectorizer = load_artifacts(MODEL_PATH, VECTORIZER_PATH)
    except FileNotFoundError:
        st.error(
            f"Couldn't find `{MODEL_PATH}` or `{VECTORIZER_PATH}`. Put them next to app.py."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load the model/vectorizer: {e}")
        st.stop()

    stop_words, stemmer = load_nltk()

    try:
        with st.spinner("Asking Gemini for tweets…"):
            items = generate_tweets(topic.strip(), num_tweets, api_key)
    except Exception as e:
        st.error(f"Gemini request failed: {e}")
        st.stop()

    if not items:
        st.warning("Gemini returned no tweets. Try again or tweak the topic.")
        st.stop()

    results = analyze(items, model, vectorizer, stop_words, stemmer)

    if do_transfer and drg_pack:
        with st.spinner("Delete → Retrieve → Generate…"):
            results = run_style_transfer(
                results, drg_pack, api_key, model, vectorizer, stop_words, stemmer
            )

    st.session_state["results"] = results

# ---------------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------------
results = st.session_state.get("results")
if results:
    total = len(results)
    matches = sum(1 for r in results if r["match"])
    model_pos = sum(1 for r in results if r["model"] == "Positive")

    transferred = [r for r in results if r.get("transfer")]
    flipped = sum(1 for r in transferred if r["transfer"]["flipped"])

    cols = st.columns(4 if transferred else 3)
    cols[0].metric("Total tweets", total)
    cols[1].metric("✅ Gemini ↔ Model agree", f"{matches}/{total}")
    cols[2].metric("Your model: 😊 / 😠", f"{model_pos} / {total - model_pos}")
    if transferred:
        cols[3].metric("🔁 Flipped to positive", f"{flipped}/{len(transferred)}")

    st.subheader("Tweets")
    for r in results:
        g_cls = "badge-pos" if r["gemini"] == "Positive" else "badge-neg"
        m_cls = "badge-pos" if r["model"] == "Positive" else "badge-neg"
        conf_txt = f" {r['conf']*100:.0f}%" if r["conf"] is not None else ""
        match_cls = "match-yes" if r["match"] else "match-no"
        match_txt = "✓ match" if r["match"] else "✗ differ"
        safe = r["tweet"].replace("<", "&lt;").replace(">", "&gt;")

        t = r.get("transfer")
        rewrite_html = ""
        if t:
            safe_out = t["output"].replace("<", "&lt;").replace(">", "&gt;")
            flip_cls = "flip-yes" if t["flipped"] else "flip-no"
            flip_txt = (
                f"✓ your model now reads this as Positive ({t['p_pos_out']*100:.0f}%)"
                if t["flipped"]
                else f"⚠ still Negative to your model ({t['p_pos_out']*100:.0f}% positive)"
            )
            rewrite_html = f"""
              <div class="rewrite">
                <div class="rewrite-label">↻ Style transfer — {t['method']}</div>
                <div class="rewrite-text">{safe_out}</div>
                <div class="rewrite-meta"><span class="{flip_cls}">{flip_txt}</span></div>
              </div>
            """

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
              {rewrite_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if t:
            with st.expander("🔍 How that rewrite was produced"):
                st.markdown(f"**1. DELETE** — negative markers found: `{t['deleted'] or '—'}`")
                st.markdown(f"**Content template:** _{t['template'] or '—'}_")
                st.markdown(
                    f"**2. RETRIEVE** — nearest positive tweet (cosine `{t['sim']:.2f}`):  \n"
                    f"_{t['neighbour'] or '—'}_"
                )
                st.markdown(f"**Borrowed positive markers:** `{t['borrowed'] or '—'}`")
                st.markdown(f"**TemplateBased baseline (no LLM):** _{t['template_based'] or '—'}_")
                if t.get("rerank", {}).get("all"):
                    st.markdown("**3. GENERATE + RERANK** — `score = P(positive) × content similarity`")
                    ranked = sorted(t["rerank"]["all"], key=lambda x: -x[1] * x[2])
                    st.table(
                        {
                            "candidate": [c for c, _, _ in ranked],
                            "P(pos)": [p for _, p, _ in ranked],
                            "content sim": [s for _, _, s in ranked],
                            "score": [round(p * s, 3) for _, p, s in ranked],
                        }
                    )
        else:
            st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
else:
    st.info("Enter a topic and hit **Generate & Classify** to see results.")
