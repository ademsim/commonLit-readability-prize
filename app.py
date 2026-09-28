import re
from pathlib import Path

import pickle
import streamlit as st

st.set_page_config(page_title="Text Readability Scorer")

BASE = Path(__file__).parent
LOW, HIGH = -1.69, -0.20  # 25th and 75th percentile of the training scores

EXAMPLES = {
    "Simple story": "The dog ran to the park. It was a sunny day. The children played with a red ball and laughed. "
                    "Then they went home to eat lunch with their mom.",
    "Academic text": "The epistemological ramifications of probabilistic inference necessitate a reconsideration of "
                     "deterministic paradigms, particularly insofar as empirical observations underdetermine the "
                     "theoretical constructs purporting to explain them.",
}


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "readability_model.pkl", "rb"))


@st.cache_resource
def load_vectorizer():
    return pickle.load(open(BASE / "readability_vectorizer.pkl", "rb"))


model = load_model()
vectorizer = load_vectorizer()

st.title("Text Readability Scorer")
st.write(
    "A TF-IDF + Ridge regression model (trained on the Kaggle CommonLit Readability dataset, about 2,800 passages "
    "rated by teachers) predicts how easy a passage is to read. Higher scores mean easier text; lower scores "
    "mean harder text."
)

choice = st.selectbox("Try an example (optional)", ["Write my own"] + list(EXAMPLES))
text = st.text_area("Passage", "" if choice == "Write my own" else EXAMPLES[choice], height=180)

if st.button("Score readability") and text.strip():
    score = float(model.predict(vectorizer.transform([text]))[0])
    if score < LOW:
        st.error(f"Hard to read: score **{score:.2f}**")
    elif score > HIGH:
        st.success(f"Easy to read: score **{score:.2f}**")
    else:
        st.warning(f"Medium difficulty: score **{score:.2f}**")
    st.progress(min(1.0, max(0.0, (score + 3.7) / 5.4)))

    words = re.findall(r"[A-Za-z']+", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if words and sentences:
        c1, c2, c3 = st.columns(3)
        c1.metric("Words", len(words))
        c2.metric("Avg. word length", f"{sum(len(w) for w in words) / len(words):.1f}")
        c3.metric("Words per sentence", f"{len(words) / len(sentences):.1f}")
    if len(words) < 50:
        st.info("Short texts give unreliable scores: the training passages have about 170 words each.")

st.caption(
    "Model: TF-IDF (unigrams and bigrams) + Ridge regression (validation RMSE ≈ 0.71; the scores range from about "
    "-3.7 (hardest) to +1.7 (easiest) with an average of -0.96). The model tends to pull extreme texts toward "
    "the middle. Difficulty bands use the quartiles of the training scores."
)
