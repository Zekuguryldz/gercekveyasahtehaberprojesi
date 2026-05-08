import re
from pathlib import Path

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"


@st.cache_resource
def load_model():
    """Model ve vectorizer'ı bir kez yükler, önbellekte tutar."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def preprocess(text: str) -> str:
    """Ham metni temizleyip modele hazır hale getirir."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


def predict(text: str, model, vectorizer) -> tuple[str, float]:
    """Metni sınıflandırır; etiket ve güven yüzdesi döner."""
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    label = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][label]
    return ("Real News" if label == 1 else "Fake News"), round(prob * 100, 1)


# ── Sayfa ayarları ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")
st.title("🔍 Fake News Detector")
st.caption("Bir haber metni gir ya da .txt dosyası yükle — model gerçek mi sahte mi söylesin.")

model, vectorizer = load_model()

# ── Giriş ───────────────────────────────────────────────────────────────────
tab_text, tab_file = st.tabs(["Metin Gir", "Dosya Yükle"])

with tab_text:
    user_title = st.text_input("Başlık (opsiyonel):")
    user_input = st.text_area("Haber metnini buraya yapıştır:", height=200)
    analyze_btn = st.button("Analiz Et", key="btn_text")

with tab_file:
    uploaded = st.file_uploader("Bir .txt dosyası seç:", type=["txt"])
    file_btn = st.button("Analiz Et", key="btn_file")

# ── Tahmin ──────────────────────────────────────────────────────────────────
text_to_analyze = None

if analyze_btn and user_input.strip():
    text_to_analyze = (user_title.strip() + " " + user_input.strip()).strip()

if file_btn and uploaded is not None:
    text_to_analyze = uploaded.read().decode("utf-8", errors="ignore")

if text_to_analyze:
    if len(text_to_analyze.split()) < 5:
        st.warning("Lütfen en az birkaç kelimeden oluşan bir metin gir.")
    else:
        label, confidence = predict(text_to_analyze, model, vectorizer)

        if label == "Real News":
            st.success(f"✅ **{label}** — Güven: %{confidence}")
        else:
            st.error(f"❌ **{label}** — Güven: %{confidence}")

        with st.expander("Ön işlenmiş metin"):
            st.code(preprocess(text_to_analyze)[:500])
