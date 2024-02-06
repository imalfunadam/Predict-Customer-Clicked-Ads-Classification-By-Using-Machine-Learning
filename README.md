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
  | **Fitur atau nilai yang tidak sesuai** | - Fitur yang tidak digunakan: **Unnamed: 0** <br> - Tipe data tidak sesuai: `Timestamp` <br> Outlier: Fitur Area `Income` memiliki **outlier**, namun masih dapat ditoleransi.| - Fitur yang tidak digunakan **(Unnamed: 0):** Dihapus dengan drop().<br> - Tipe data tidak sesuai **(Timestamp)**: Diubah tipe datanya menjadi **datetime** dan dapat dilakukan **ekstraksi bulan, minggu, hari, dan jam.**<br> - Outlier (Area Income): Tidak dilakukan handling karena nilai outlier masih dapat ditoleransi. |

### Data Exploration

Analisis Jenis dan Perilaku Pelanggan pada Iklan bertujuan untuk memahami karakteristik dan kebiasaan pelanggan terkait iklan dengan cara yang mudah dipahami oleh semua orang. Dalam analisis ini, kami mengumpulkan dan menyelidiki data tentang siapa pelanggan, apa yang mereka lakukan secara online, dan bagaimana mereka merespons iklan. Fitur yang digunakan dalam analisis ini diantaranya adalah `Daily Internet Usage`, `Daily Time Spent`, dan `Age.`

Data tentang penggunaan internet harian memberikan wawasan tentang sejauh mana pelanggan terlibat dalam aktivitas online. Informasi ini membantu kami mengenali kelompok pelanggan yang lebih aktif online dan menggunakan internet dalam kehidupan sehari-hari mereka.

Selanjutnya, waktu harian yang dihabiskan online mencerminkan seberapa lama pelanggan terlibat dalam konten digital. Ini membantu kami memahami sejauh mana mereka mungkin melihat atau berinteraksi dengan iklan.

Faktor usia juga menjadi pertimbangan penting dalam analisis ini. Usia memberikan petunjuk tentang preferensi dan minat khusus dari kelompok pelanggan. Misalnya, generasi yang lebih muda mungkin lebih terbuka terhadap inovasi teknologi dan aktif di media sosial, sementara generasi yang lebih tua mungkin lebih tertarik pada konten yang relevan dengan kehidupan sehari-hari mereka.

Dengan menganalisis informasi ini, kami dapat memahami lebih baik siapa pelanggan kami, bagaimana mereka berinteraksi dengan dunia digital, dan bagaimana iklan dapat menjadi lebih efektif bagi mereka.

![Alt text](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/assets/plot%20distribusi.png) 

![Alt text](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/assets/plot%20distribusi2.png)
<h5 align="center">Gambar 1 — Plot Distribusi Daily Internet Usage, Daily Time Spent, dan Age terhadap Clicked on Ads</h5>

Berdasarkan analisis plot Daily Time Spent, ditemukan bahwa **pengguna yang menghabiskan waktu di sebuah situs kurang dari 1 jam memiliki potensi lebih besar untuk mengklik iklan.** Ini mungkin disebabkan oleh keterbukaan mereka dalam mengeksplorasi iklan, seiring dengan waktu yang terbatas di situs, yang membuat mereka lebih mudah tergoda untuk mengeklik iklan.

Dalam analisis Daily Internet Usage, terungkap bahwa **pengguna yang jarang menggunakan internet memiliki potensi lebih besar untuk mengklik iklan dibandingkan dengan mereka yang sering menggunakan internet.** Pengguna yang terbatas dalam penggunaan internet mungkin memiliki tingkat rasa ingin tahu yang lebih tinggi terhadap produk atau layanan yang diiklankan. Hal ini dapat disebabkan oleh kurangnya kebiasaan mereka dalam menjelajahi internet, membuat mereka merasa tertarik untuk mendapatkan informasi lebih lanjut melalui iklan. Selain itu, keterbatasan akses internet juga menjadi faktor yang berpengaruh, di mana pengguna yang jarang menggunakan internet cenderung lebih tertarik untuk mengklik iklan yang menarik guna memperoleh informasi tambahan.

Dalam konteks analisis usia, ditemukan bahwa **pengguna yang lebih tua memiliki potensi lebih besar untuk mengklik iklan.** Fakta ini mungkin karena pengguna internet yang lebih muda lebih terampil dalam menggunakan teknologi dan internet, membuat mereka lebih cenderung mencari informasi dari sumber lain selain iklan. Mereka juga biasanya lebih kritis dalam menilai iklan dan memilih untuk menghindari iklan yang dianggap terlalu mengganggu atau tidak relevan. Di sisi lain, pengguna yang lebih tua mungkin memiliki ketertarikan yang lebih besar terhadap iklan yang sesuai dengan kehidupan sehari-hari mereka, menjadikan mereka lebih mungkin untuk mengklik 

![Alt text](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/assets/Plot%20Korelasi%20.png)
<h5 align="center">Gambar 2 — Plot Korelasi Daily Time Spent on Site dengan Internet Usage terhadap Clicked on Ads</h5>

Dari plot korelasi antara Daily Time Spent on Site dengan Internet Usage terhadap Target, ditemukan bahwa distribusi pengguna dapat dibagi menjadi dua segmen, yaitu pengguna aktif dan non-aktif. **Pengguna aktif cenderung menghabiskan lebih banyak waktu di situs dan lebih terlibat dalam penggunaan internet secara keseluruhan.** Namun, menariknya, **pengguna aktif cenderung tidak terlalu suka mengklik iklan yang ditampilkan.**

Dengan temuan ini, perusahaan dapat **mengoptimalkan sistem iklannya dengan memfokuskan target kepada pengguna non-aktif.** Pengguna non-aktif ini mungkin memiliki waktu yang lebih sedikit dihabiskan di situs dan penggunaan internet secara umum. Oleh karena itu, mereka mungkin lebih rentan terhadap iklan dan memiliki kemungkinan yang lebih tinggi untuk mengklik iklan yang ditampilkan. Dengan memfokuskan strategi iklan kepada pengguna non-aktif, perusahaan dapat meningkatkan efektivitas kampanye iklannya. Hal ini dapat dilakukan dengan menyesuaikan konten iklan agar lebih menarik dan relevan bagi pengguna non-aktif serta memilih situs pemasaran yang tepat untuk mencapai mereka. Dengan melakukan pendekatan yang lebih spesifik terhadap pengguna non-aktif, perusahaan dapat mengoptimalkan sistem iklannya dan meningkatkan peluang untuk mendapatkan respons yang lebih baik dari target audiens yang dituju.

Selanjutnya, **Time Analysis of User Clicks on Ads** digunakan untuk **menganalisis pola waktu pengguna saat mengklik iklan dengan mengidentifikasi tren dan pola yang dapat memberikan insight.** Dengan analisis ini, perusahaan dapat menentukan jam-jam atau periode tertentu di mana pengguna cenderung lebih aktif dalam mengklik iklan dan mengoptimalkan strategi penempatannya. Hal tersebut dilakukan agar dapat mencapai audiens yang lebih responsif dan meningkatkan peluang untuk mendapatkan klik yang lebih banyak. Selain itu, analisis ini juga dapat membantu perusahaan untuk mengalokasikan anggaran iklan dengan lebih efisien dengan mengarahkan sumber daya ke periode waktu yang paling menguntungkan.

![Alt text](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/assets/Tren%20Harian.png)
<h5 align="center">Gambar 3 — Tren Harian Clicked on Ads</h5>




