# 📌 Proyek Klasifikasi Gambar: Vegetable

## 📖 Deskripsi Proyek

Proyek ini bertujuan untuk melakukan klasifikasi gambar sayuran menggunakan model deep learning berbasis TensorFlow dan Keras. Dataset yang digunakan berasal dari Kaggle: [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset).

## 👤 Informasi Pengguna

-   **Nama**: Joshua Palti Sinaga
-   **Email**: josua123690707@gmail.com
-   **ID Dicoding**: jo_sua_07

## ⚙️ Instalasi Paket

Sebelum menjalankan proyek ini, pastikan untuk menginstal paket yang dibutuhkan dengan menjalankan perintah berikut:

```bash
pip install tensorflow tensorflowjs kagglehub matplotlib numpy pandas scikit-learn
```

## 📂 Struktur Dataset

Dataset dibagi menjadi tiga bagian utama:

-   **Train**: Data pelatihan
-   **Test**: Data pengujian
-   **Validation**: Data validasi

### 📌 Jenis Sayuran yang Diklasifikasikan

-   Cabbage
-   Cauliflower
-   Carrot
-   Bitter Gourd
-   Bean
-   Broccoli
-   Pumpkin
-   Tomato
-   Capsicum
-   Potato
-   Bottle Gourd
-   Cucumber
-   Papaya
-   Radish
-   Brinjal

## 🛠️ Langkah-langkah dalam Notebook

1. **Mengimpor Library**: Memuat pustaka yang diperlukan seperti TensorFlow, Keras, Matplotlib, dan NumPy.
2. **Mengunduh Dataset**: Menggunakan `kagglehub` untuk mengunduh dataset dari Kaggle.
3. **Pemrosesan Data**:
    - Menyalin dataset ke direktori proyek
    - Mengatur ulang struktur folder
    - Normalisasi gambar
    - Labeling dataset
4. **Visualisasi Data**:
    - Menampilkan distribusi dataset
    - Menampilkan contoh gambar dari setiap kelas
5. **Pelatihan Model**:
    - Menggunakan arsitektur CNN dengan beberapa lapisan Conv2D, MaxPooling, dan Dense
    - Menggunakan optimizer Adam dengan learning rate 0.0001
    - Callback untuk early stopping
6. **Evaluasi Model**:
    - Menggunakan Confusion Matrix dan Classification Report
    - Menampilkan akurasi dan loss dari proses training
7. **Konversi Model**:
    - Model disimpan dalam format `.keras`, TensorFlow.js, dan TFLite

## 🎯 Menjalankan Inference

Model dapat digunakan untuk melakukan prediksi terhadap gambar baru dengan cara berikut:

### 🔹 Menggunakan Model TensorFlow:

```python
from tensorflow import keras
import numpy as np
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

model = keras.models.load_model('submission/saved_model.keras')
predicted_label, confidence = predict_image('path/to/image.jpg', model, labels)
print(f'Predicted: {predicted_label} ({confidence:.2f})')
```

### 🔹 Menggunakan Model TFLite:

```python
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='submission/tflite/model.tflite')
interpreter.allocate_tensors()
```

## 📊 Hasil Akhir

Model berhasil mencapai akurasi sekitar **96%** pada dataset validasi dengan f1-score yang tinggi untuk setiap kelas.

## 📜 Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan penelitian. Bebas digunakan dengan tetap mencantumkan sumbernya.

---

> _🚀 Dibuat oleh Joshua Palti Sinaga, 2025_
