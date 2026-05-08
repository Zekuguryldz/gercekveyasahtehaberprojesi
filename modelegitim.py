import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


def preprocess(text):
    """Ham metni temizleyip modele hazır hale getirir."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)


# Veriyi yükle ve etiketle
fake = pd.read_csv("/Users/uguryildiz/Desktop/bulutbilisimandaiproje/data/Fake.csv")
true = pd.read_csv("/Users/uguryildiz/Desktop/bulutbilisimandaiproje/data/True.csv")
fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Başlık ve gövdeyi birleştirerek daha zengin özellik seti oluştur
df["combined"] = df["title"].fillna("") + " " + df["text"].fillna("")

print("Ön işleme uygulanıyor...")
df["clean_text"] = df["combined"].apply(preprocess)

# Eğitim / Test ayrımı (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42
)
print(f"Eğitim: {len(X_train)} | Test: {len(X_test)}")

# TF-IDF: metni sayısal vektöre çevir
print("TF-IDF uygulanıyor...")
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model eğitimi
print("Model eğitiliyor...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Değerlendirme
y_pred = model.predict(X_test_tfidf)
print("\n=== MODEL SONUÇLARI ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# Modeli kaydet
joblib.dump(model, "/Users/uguryildiz/Desktop/bulutbilisimandaiproje/models/model.pkl")
joblib.dump(vectorizer, "/Users/uguryildiz/Desktop/bulutbilisimandaiproje/models/vectorizer.pkl")
print("\nModel kaydedildi.")
