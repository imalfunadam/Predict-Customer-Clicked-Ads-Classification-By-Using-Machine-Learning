# 🖱 Predict Clicked Ads Customer Classification

**Tool :** Jupyter Notebook | [Link Notebook](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/Predict-Customer-Clicked-Ads-Classification.ipynb).<br>
**Programming Language :** Python<br>
**Libraries :** Pandas, NumPy, Scikit Learn, shap<br>
**Visualization :** Matplotlib, Seaborn<br>

**Table of Contents**

- STAGE 0: Introduction
    - [Background](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning?tab=readme-ov-file#backgorund)
    - [Goal](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning?tab=readme-ov-file#goal)
    - [Objective](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning?tab=readme-ov-file#objective)
    - [Business Metric](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning?tab=readme-ov-file#business-metric)
- STAGE 1: Exploratory Data Analysis
    - Data Overview
    - Data Quality Assessment
    - Data Exploration
- STAGE 2: Data Pre-processing
- STAGE 3: Data Modeling and Evaluation
    - Model Experimet
    - Evaluation: Confussion Matrix
    - Evaluation: Feature Importance
- STAGE 4: Business Recommendation

## 📂 STAGE 0: Introduction

### Backgorund
Seiring dengan perkembangan zaman, perusahaan dituntut untuk mengoptimalkan metode iklan mereka di platform digital. Hal ini bertujuan untuk menarik calon pelanggan potensial dengan biaya yang minimal. Peningkatan konversi, yang didefinisikan sebagai jumlah pelanggan potensial yang melakukan pembelian setelah mengklik iklan, menjadi fokus utama.

Namun, untuk mencapai tujuan tersebut, perusahaan harus mampu melakukan prediksi click-through rate (CTR) yang akurat. CTR merupakan persentase orang yang mengklik iklan setelah melihatnya. Prediksi CTR yang akurat sangat penting dalam menentukan keberhasilan kampanye iklan digital. Tanpa prediksi yang akurat, perusahaan berpotensi mengeluarkan biaya yang besar tanpa menghasilkan keuntungan yang signifikan.

### Goal
Membuat model machine learning yang mampu mendeteksi pengguna potensial untuk konversi atau tertarik pada sebuah iklan. Hal ini memungkinkan perusahaan untuk:

- **Mengoptimalkan biaya iklan:** Hanya menargetkan iklan kepada pengguna yang memiliki kemungkinan besar untuk melakukan konversi, sehingga meminimalkan pemborosan anggaran.
- **Meningkatkan ROI (Return on Investment):** Meningkatkan keuntungan dari investasi iklan dengan menargetkan pengguna yang tepat.
- **Meningkatkan efektivitas kampanye iklan:** Meningkatkan peluang konversi dengan menargetkan pengguna yang berminat.

### Objective
1. Prediksi Potensi Klik Iklan:
    - Membangun model machine learning untuk memprediksi pengguna yang memiliki potensi untuk mengklik iklan dengan akurasi minimal 90%.
    - Model ini akan membantu perusahaan dalam menargetkan iklan mereka dengan lebih efektif dan meningkatkan ROI (Return on Investment).
2. Pemahaman Pola Pengguna:
    - Mengidentifikasi pola dan karakteristik pengguna yang memiliki potensi untuk mengklik iklan.
    - Insight ini akan membantu perusahaan dalam memahami target audience mereka dengan lebih baik dan mengembangkan strategi iklan yang lebih efektif.
3. Rekomendasi Bisnis:
    - Memberikan rekomendasi bisnis berdasarkan hasil analisis dan model.
    - Rekomendasi ini dapat membantu perusahaan dalam meningkatkan strategi marketing mereka dan mencapai tujuan bisnis mereka.

### Business Metric
Click-through rate

## 📂 STAGE 1: Exploratory Data Analysis
### Data Overview
Dataset memiliki 1000 baris dan 9 fitur dengan 1 target. Berikut informasi fitur pada dataset:
Tabel 1 — Deskripsi Fitur
  | Fitur | Deskripsi | 
  | --- | --- | 
  | Daily Time Spent on Site | Lamanya tinggal di suatu situs (harian) dalam satuan menit |
  | Age | Umur user dalam satuan tahun |
  | Area Income | Pendapatan user dalam satuan rupiah |
  | Daily Internet Usage | Penggunaan internet harian dalam satuan menit |
  | Male | Gender user |
  | Timestamp | Kapan user mengunjungi sebuah situs |
  | Clicked on Ad |	Click atau tidak iklan yang ditampilkan |
  | City | Kota asal user |
  | Province | Provinsi asal user |
  | Category | Kategori produk yang dikunjungi | 

### Data Quality Assesment
Penilaian kualitas data (DQA) dilakukan untuk memastikan bahwa data yang digunakan untuk analisis selanjutnya sudah siap dan sesuai dengan kebutuhan analisis. DQA membantu mengidentifikasi dan mengatasi masalah pada data, sehingga hasil analisis menjadi lebih akurat dan andal.

Langkah-langkah:

1. Memeriksa Missing Value:
    - Menemukan data yang hilang pada setiap fitur.
    - Menentukan apakah missing value acak atau terpola.
    - Menghapus data dengan missing value yang signifikan atau mengisi missing value dengan metode yang tepat (misalnya, mean imputation, median imputation, regression imputation).
2. Memeriksa Duplikasi Data:
    - Menemukan data yang duplikat (data yang sama muncul lebih dari sekali).
    - Menghapus data duplikat untuk menghindari bias dalam analisis.
3. Memeriksa Tipe dan Konsistensi Nilai:
    - Memastikan bahwa setiap fitur memiliki tipe data yang sesuai (misalnya, numerik, kategorikal).
    - Memeriksa konsistensi format data (misalnya, tanggal, waktu).
    - Mengubah format data yang tidak konsisten agar sesuai dengan format yang umum digunakan.
4. Memeriksa Outlier:
    - Menemukan data yang jauh dari nilai rata-rata (outlier).
    - Menentukan apakah outlier valid atau merupakan kesalahan dalam pengumpulan data.
    - Menghapus outlier yang tidak valid atau menggantinya dengan nilai yang lebih sesuai.

Tabel 2 — Hasil Data Quality Assessment
  | Data Assesment | Finding | Handling | 
  | --- | --- | --- |
  | **Missing value** | Terdapat missing value pada fitur `Daily Time Spent on Site, Area Income, Daily Internet Usage,` dan `Male.` | - Tipe data numerik: Diatasi dengan imputasi menggunakan nilai median.<br> - Tipe data kategorikal (Age): Diatasi dengan imputasi menggunakan nilai modus.
  | **Duplikat** | Tidak ada duplikat data. | Tidak diperlukan handling karena tidak ada duplikat data. |
  | **Fitur atau nilai yang tidak sesuai** | - Fitur yang tidak digunakan: Unnamed: 0 <br> Tipe data tidak sesuai: Timestamp <br> Outlier: Fitur Area Income memiliki outlier, namun masih dapat ditoleransi.| - Fitur yang tidak digunakan (Unnamed: 0): Dihapus dengan drop().<br>- Tipe data tidak sesuai (Timestamp): Diubah tipe datanya menjadi datetime dan dapat dilakukan ekstraksi bulan, minggu, hari, dan jam.<br>Outlier (Area Income): Tidak dilakukan handling karena nilai outlier masih dapat ditoleransi. |