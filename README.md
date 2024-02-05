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