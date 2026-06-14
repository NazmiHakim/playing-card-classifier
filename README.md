# Klasifikasi Citra Kartu Remi (Playing Cards Classification)

Proyek ini merupakan implementasi Tugas Akhir (UAS) mata kuliah **Machine Learning** yang berfokus pada klasifikasi citra kartu remi ke dalam 4 kelas suit utama: **CLUB (Keriting)**, **DIAMOND (Wajik)**, **HEART (Hati)**, dan **SPADE (Sko/Sekop)**.

Repository ini membandingkan performa antara pendekatan **Machine Learning Tradisional (HOG + 5 Model ML Classifier)** dan **Deep Learning (MobileNetV2 Transfer Learning + TTA + Grad-CAM)** menggunakan dataset terstandardisasi.

---

## Fitur Utama & Keunggulan Proyek (UAS Bonus Points)
*   **Perbandingan Multi-Model (Lebih dari 5 Model)**: Membandingkan 5 model Machine Learning Tradisional (SVM Linear, SVM RBF C=1, SVM RBF C=10, KNN, Random Forest) dengan 2 model Deep Learning (MobileNetV2 Baseline dan MobileNetV2 + Test-Time Augmentation).
*   **Data Augmentation & TTA**: Menggunakan augmentasi gambar dinamis *on-the-fly* (6 jenis transformasi) pada saat training dan *Test-Time Augmentation* (TTA) berbasis *flip averaging* saat inferensi untuk meningkatkan stabilitas dan akurasi model.
*   **Explainable AI (Interpretabilitas Model)**: Visualisasi **Grad-CAM** untuk menganalisis area fokus model CNN dalam menentukan keputusan klasifikasi.
*   **Aplikasi Web Interaktif (Streamlit)**: Antarmuka sederhana untuk menguji prediksi model secara langsung menggunakan gambar baru (upload file) atau real-time kamera laptop.

---

## Dataset & Preprocessing

### Informasi Dataset
*   **Nama Dataset**: CARD DATASET 2
*   **Sumber**: https://drive.google.com/drive/folders/1XoD3fgfPbUTNtlstRuw-H-UOgT_Iwxhn?usp=sharing
*   **Total Data**: 800 gambar (200 gambar per kelas).
*   **Pembagian Data (Split)**: 80% Training (640 gambar) & 20% Validation/Testing (160 gambar), terstratifikasi agar distribusi kelas seimbang.

### Preprocessing Data
1.  **Machine Learning Tradisional**:
    *   Resize gambar ke ukuran $224 \times 224$ piksel.
    *   Ekstraksi Fitur Warna: Nilai rata-rata warna BGR (3 fitur).
    *   Ekstraksi Fitur Bentuk: **HOG (Histogram of Oriented Gradients)** dengan parameter orientasi=9, piksel per sel=(8,8), dan sel per blok=(2,2).
    *   Standardisasi fitur menggunakan `StandardScaler`.
2.  **Deep Learning (CNN)**:
    *   Resize gambar ke ukuran $224 \times 224$ piksel.
    *   Normalisasi piksel ke skala $[-1, 1]$ (sesuai standardisasi MobileNetV2).
    *   **Data Augmentation (On-the-Fly)**: *Random Flip*, *Random Rotation* (0.1), *Random Zoom* (0.15), *Random Translation* (0.1, 0.1), *Random Brightness* (0.15), dan *Random Contrast* (0.1).

---

## Hasil Evaluasi & Perbandingan (7 Model)

Model dilatih menggunakan seed terstandardisasi (`SEED=123` untuk Deep Learning, `SEED=42` untuk ML) guna menjamin aspek reproduksibilitas. Berikut adalah performa dari 7 konfigurasi model yang diuji pada data pengujian (160 gambar):

| Kategori | Model Classifier | Accuracy (%) | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Keterangan |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Deep Learning** | **MobileNetV2 + TTA (Flip Avg)** | **82.50%** | **0.84** | **0.83** | **0.83** | **Model Terbaik (Solusi Akhir)** |
| **Deep Learning** | MobileNetV2 (Baseline - Tanpa TTA) | 78.75% | 0.81 | 0.79 | 0.79 | Sangat cepat dan akurat |
| **Classic ML** | SVM — Kernel RBF (C=10, $\gamma$=0.01) | 81.25% | 0.82 | 0.81 | 0.81 | Performa ML Tradisional Terbaik |
| **Classic ML** | Random Forest (n=200) | 80.00% | 0.81 | 0.80 | 0.80 | Cukup stabil |
| **Classic ML** | SVM — Kernel Linear (C=1) | 80.00% | 0.81 | 0.80 | 0.80 | Sederhana dan efisien |
| **Classic ML** | SVM — Kernel RBF (C=1, scale) | 78.12% | 0.80 | 0.78 | 0.78 | Baseline RBF |
| **Classic ML** | K-Nearest Neighbors (k=7) | 67.50% | 0.69 | 0.67 | 0.67 | Akurasi terendah |

### Analisis Singkat
*   **Deep Learning vs ML Tradisional**: Deep Learning dengan transfer learning MobileNetV2 memberikan representasi fitur hierarkis yang lebih baik secara otomatis. Grad-CAM membuktikan model memfokuskan ekstraksi pada pola simbol kartu (*suits*) di tengah gambar, bukan pada latar belakang.
*   **Dampak TTA**: Penerapan *Test-Time Augmentation* (TTA) menstabilkan prediksi model saat inferensi dengan merata-ratakan probabilitas gambar asli dan versi *horizontal flip*, menaikkan akurasi dari **78.75% ke 82.50% (+3.75%)**.

---

## Interpretasi Model (Grad-CAM)
Visualisasi Grad-CAM diimplementasikan untuk menganalisis daerah penekanan berat (*focal point*) dari Convolutional Layer terakhir pada MobileNetV2. Heatmap menunjukkan bahwa model secara konsisten memfokuskan perhatian pada pola bentuk suit kartu (misalnya lekukan Heart atau sudut Diamond) dan mengabaikan noise latar belakang.

---

## Cara Menjalankan Program (Panduan Penggunaan)

### 1. Prasyarat Sistem
Pastikan Anda memiliki Python versi 3.10 ke atas.

### 2. Instalasi Dependensi
Clone repositori ini, lalu masuk ke direktori proyek dan instal library yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi Web Streamlit
Aplikasi web Streamlit memuat model `model_kartu_mobilenet.keras` untuk memprediksi citra kartu remi baru:
```bash
streamlit run app.py
```
Setelah dijalankan, buka browser Anda di alamat default `http://localhost:8501`. Anda dapat mengunggah gambar baru untuk memprediksi jenis kartu remi beserta tingkat kepercayaan (*confidence score*).

### 4. Menjalankan Prediksi Kamera Real-time
Untuk mencoba klasifikasi kartu remi secara langsung menggunakan Webcam laptop:
```bash
python realtime_kamera.py
```

