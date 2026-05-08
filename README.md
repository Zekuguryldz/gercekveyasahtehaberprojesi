# Yapay Zeka Destekli Sahte Haber Tespit Sistemi

Haber metinlerini **Fake News** veya **Real News** olarak sınıflandıran makine öğrenmesi tabanlı bir web uygulaması.

---

## Özellikler

- Metin yapıştırma veya `.txt` dosyası yükleme
- Tahmin sonucu ile birlikte güven yüzdesi gösterimi
- Streamlit tabanlı sade arayüz

---

## Kurulum

```bash
pip install pandas scikit-learn nltk joblib streamlit
```

---

## Çalıştırma

### 1. Modeli Eğit
```bash
python modelegitim.py
```
Bu adım `models/model.pkl` ve `models/vectorizer.pkl` dosyalarını oluşturur.

### 2. Uygulamayı Başlat
```bash
streamlit run app.py
```
Tarayıcıda `http://localhost:8501` adresini aç.

---

## Nasıl Çalışır?

```
Başlık + Haber Metni
   ↓
Ön işleme (küçük harf, noktalama temizleme, stopword kaldırma)
   ↓
TF-IDF Vectorizer (metni sayısal vektöre çevirir)
   ↓
Logistic Regression (sınıflandırır)
   ↓
Fake News / Real News + Güven Yüzdesi
```

**TF-IDF (Term Frequency–Inverse Document Frequency):** Her kelimenin o metinde ne kadar önemli olduğunu sayısal olarak ifade eder. Sık geçen ama anlamsız kelimeler (the, is, a) otomatik olarak düşük ağırlık alır.

**Logistic Regression:** TF-IDF çıktısını girdi olarak alır ve metni iki sınıftan birine atar.

> Haber başlığı ve gövde metni birleştirilerek modele verilir. Bu sayede model hem başlıktaki hem de içerikteki örüntüleri öğrenir.

---

## Model Performansı

Test seti: 8.980 haber (%20 ayırma)

| Metrik    | Değer  |
|-----------|--------|
| Accuracy  | 98.74% |
| Precision | 98.28% |
| Recall    | 99.09% |
| F1 Score  | 98.68% |

---

## Bilinen Sınırlılıklar

- Model yalnızca İngilizce metinlerde çalışır.
- Eğitim verisi büyük ölçüde siyasi haberlerden oluşmaktadır. Bilim, spor veya ekonomi haberleri gibi farklı konularda doğruluk düşebilir. Bu durum modelin kötü eğitildiğini değil, eğitim verisinin dar bir dağılıma sahip olduğunu gösterir (*distribution shift*).

---

## Veri Seti

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) — Kaggle (clmentbisaillon)

- `Fake.csv`: 23.481 sahte haber
- `True.csv`: 21.417 gerçek haber
- Toplam: 44.898 haber

---

## Proje Yapısı

```
bulutbilisimandaiproje/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
├── modelegitim.py   # Veri yükleme, ön işleme, eğitim, değerlendirme
├── app.py           # Streamlit arayüzü
└── README.md
```
