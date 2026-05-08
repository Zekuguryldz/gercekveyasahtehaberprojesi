# Yapay Zeka Destekli Sahte Haber Tespit Sistemi

Haber metinlerini **Fake News** veya **Real News** olarak sınıflandıran makine öğrenmesi tabanlı bir web uygulaması.

---

## Özellikler

- Metin yapıştırma veya `.txt` dosyası yükleme
- Tahmin sonucu ile birlikte güven yüzdesi gösterimi
- Streamlit tabanlı sade arayüz

---

## Dataset Kurulumu

Bu repoda veri seti dosyaları yer almamaktadır. Kaggle üzerinden [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) indirilmelidir.

İndirilen dosyalar şu klasöre konulmalıdır:

```
data/
├── Fake.csv
└── True.csv
```

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
| Accuracy  | 99.08% |
| Precision | 98.88% |
| Recall    | 99.18% |
| F1 Score  | 99.03% |

### Confusion Matrix Yorumu

Confusion matrix, modelin Fake ve Real sınıflarında yaptığı doğru ve yanlış tahminleri gösterir. Bu proje için özellikle sahte haberlerin gerçek olarak tahmin edilmesi kritik bir hatadır; çünkü bu durumda kullanıcı sahte bir haberi gerçek sanabilir. Model test setinde yalnızca 35 sahte haberi gerçek olarak işaretlemiştir (4.284 gerçek haberden 35'i kaçırılmış, 4.696 sahte haberden 48'i yanlış sınıflandırılmıştır).

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
