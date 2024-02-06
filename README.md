# 🖱 Predict Clicked Ads Customer Classification

**Tool :** Jupyter Notebook | [Link Notebook](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/Predict-Customer-Clicked-Ads-Classification.ipynb).<br>
**Programming Language :** Python<br>
**Libraries :** Pandas, NumPy, Scikit Learn, shap<br>
**Visualization :** Matplotlib, Seaborn<br>

**Table of Contents**

- STAGE 0 : [Introduction](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/tree/main?tab=readme-ov-file#-stage-0-introduction)
    - [Background](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning?tab=readme-ov-file#backgorund)
    - [Goal](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning?tab=readme-ov-file#goal)
    - [Objective](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning?tab=readme-ov-file#objective)
    - [Business Metric](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning?tab=readme-ov-file#business-metric)
- STAGE 1  : [Exploratory Data Analysis](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/tree/main?tab=readme-ov-file#-stage-1-exploratory-data-analysis)
    - [Data Overview](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/tree/main?tab=readme-ov-file#data-overview)
    - [Data Quality Assessment](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/tree/main?tab=readme-ov-file#data-quality-assesment)
    - [Data Exploration](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/tree/main?tab=readme-ov-file#data-exploration)
- STAGE 2 : [Data Pre-processing](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/tree/main?tab=readme-ov-file#-stage-2-data-pre-processing)
- STAGE 3 : [Data Modeling and Evaluation](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/tree/main?tab=readme-ov-file#-stage-3-data-modeling-and-evaluation)
    - [Model Experiment](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/tree/main?tab=readme-ov-file#model-experiment)
    - [Evaluation : Confussion Matrix](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/tree/main?tab=readme-ov-file#evaluation-confusion-matrix)
    - [Evaluation : Feature Importance](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/tree/main?tab=readme-ov-file#evaluation-feature-importance)
- STAGE 4 : [Business Recommendation](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/tree/main?tab=readme-ov-file#-stage-4-business-recommendation)

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

Dalam analisis mengenai perilaku pengguna terhadap klik iklan pada hari-hari tertentu, **terlihat bahwa pengguna cenderung kurang aktif dalam mengklik iklan pada hari Senin dan Jumat.** Hari-hari ini sering dianggap sebagai awal dan akhir minggu kerja, di mana konsentrasi pengguna cenderung lebih terfokus pada pekerjaan dan kurang pada aktivitas online seperti mengklik iklan. Faktor ini dapat menjelaskan mengapa jumlah pengguna yang mengklik iklan pada hari-hari ini cenderung rendah.

Sebaliknya, **pada hari Rabu terlihat adanya konversi klik iklan yang paling baik.** Jumlah pengguna yang mengklik iklan relatif tinggi, sementara jumlah pengguna yang tidak mengklik iklan rendah. Ini mungkin disebabkan oleh fakta bahwa hari Rabu sering dianggap sebagai titik tengah minggu di mana orang merasa lebih rileks dan memiliki lebih banyak waktu untuk menghabiskan waktu online serta melakukan aktivitas seperti berbelanja.

Terdapat juga data menarik bahwa hari **Selasa dan Sabtu memiliki tingkat lalu lintas yang tinggi, dengan sekitar 50% pengguna cenderung mengklik iklan.** Ini menunjukkan bahwa pada hari-hari ini, ada peluang yang cukup baik untuk mencapai pengguna dengan iklan yang relevan. Oleh karena itu, perusahaan dapat memanfaatkan pola ini dengan menyesuaikan strategi iklan mereka untuk lebih efektif pada hari-hari tersebut dan meningkatkan potensi konversi.

![Alt text](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/assets/tren%20setiap%20jam.png)
<h5 align="center">Gambar 4 — Tren Setiap Jam Clicked on Ads</h5>

Analisis berdasarkan waktu jam menunjukkan bahwa terdapat **potensi pengguna untuk mengklik iklan dan memiliki tingkat konversi pembelian yang tinggi pada jam-jam tertentu, yaitu pukul 00.00, 09.00, 11.00, dan 18.00.**

Analisis berdasarkan waktu menunjukkan bahwa terdapat potensi tinggi untuk meningkatkan klik dan konversi iklan pada jam-jam tertentu:

1. **00.00:**
    - **Asumsi:** Orang-orang lebih santai dan memiliki waktu luang untuk menjelajahi internet.
    - **Peluang:** Menampilkan iklan menarik untuk meningkatkan kemungkinan klik.

2. **09.00 & 11.00:**
    - **Asumsi:** Orang-orang memiliki jeda dalam pekerjaan atau mengambil istirahat.
    - **Peluang:** Menampilkan iklan relevan dan menarik untuk meningkatkan klik dan konversi.

3. 18.00:
    - **Asumsi:** Orang-orang selesai bekerja dan memiliki waktu luang untuk bersantai dan menjelajahi internet.
    - **Peluang:** Menampilkan iklan kepada target audiens yang responsif untuk meningkatkan klik dan konversi.
Dengan memanfaatkan pola waktu ini, perusahaan dapat mengoptimalkan strategi iklan mereka dan meningkatkan efektivitas kampanye pemasaran.

## 📂 STAGE 2: Data Pre-processing

Berikut tahapan-tahapan dalam Data Pre-processing yang telah dilakukan.

![Alt text](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/assets/Tahap%20Data%20Pre-processing.png)
<h5 align="center">Gambar 5 — Tahap Data Pre-processing</h5><br>

Fitur yang digunakan untuk model.

![Alt text](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/assets/Fitur%20yang%20digunakan%20untuk%20Model.png)
<h5 align="center">Gambar 6 — Fitur yang digunakan untuk Model</h5>

## 📂 STAGE 3: Data Modeling and Evaluation

### Model Experiment
Untuk melakukan prediksi pada klik iklan, dilakukan dua eksperimen yang berbeda. Pada eksperimen pertama, data train default digunakan untuk melatih model. Eksperimen ini memanfaatkan data train dalam bentuk default atau tanpa adanya penyesuaian tambahan. Sementara itu, pada eksperimen kedua, data distandardisasi menggunakan StandardScaler. Hal ini dilakukan karena distribusi data cenderung mendekati normal, sehingga perlu dilakukan standardisasi agar data memiliki skala yang serupa.

Dalam kedua eksperimen ini, matriks akurasi digunakan sebagai metrik evaluasi. Matriks akurasi memberikan gambaran tentang seberapa baik model dapat mengklasifikasikan data dengan benar. Penggunaan matriks akurasi ini dipilih karena jumlah kategori pada target (Clicked on Ads) yang digunakan dalam analisis seimbang, yaitu memiliki jumlah pengguna yang mengklik iklan dan tidak mengklik yang relatif setara.

Pemilihan matriks akurasi dijustifikasi oleh kesetaraan jumlah pengguna yang mengklik iklan dan yang tidak mengklik, sehingga memberikan gambaran yang seimbang tentang kinerja model. Metode evaluasi ini memungkinkan untuk menilai sejauh mana model dapat memprediksi dengan akurat tanpa memihak pada salah satu kelas target.

<h5 align="center">Tabel 1 — Hasil Eksperimen Pertama (Tanpa Standardization)</h5>

![Alt text](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/assets/Hasil%20Eksperimen%20Pertama.png)

<h5 align="center">Tabel 2 — Hasil Eksperimen Kedua (Standardization)</h5>

![Alt text](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/assets/Hasil%20Eksperimen%20Kedua.png)

Pada hasil eksperimen, terlihat bahwa algoritma **Random Forest memiliki akurasi tertinggi baik pada eksperimen pertama maupun kedua, dengan nilai akurasi mencapai 96%.** Selain itu, algoritma-algoritma lain seperti Gradient Boosting, XGBoost, dan LGBM juga menunjukkan akurasi yang tinggi pada eksperimen pertama, dengan nilai akurasi sebesar 95%. Pada eksperimen kedua, ketiga algoritma tersebut juga memberikan hasil akurasi yang hampir sama. Menariknya, terlihat bahwa penggunaan metode standardization tidak memberikan perubahan yang signifikan pada nilai akurasi untuk algoritma-algoritma tersebut. Hal ini mengindikasikan bahwa model tidak terlalu sensitif terhadap perbedaan skala fitur dalam data. Dengan kata lain, perbedaan skala fitur tidak memiliki pengaruh yang signifikan pada kinerja model.

Selain itu, algoritma seperti Random Forest, XGBoost, Gradient Boosting, dan LGBM termasuk dalam kategori algoritma yang robust dan memiliki kemampuan yang kuat dalam menangani berbagai jenis data. Mereka dapat menyesuaikan dengan baik terhadap data yang tidak distandardisasi, sehingga tidak memerlukan proses preprocessing yang rumit. Oleh karena itu, nilai akurasi mereka tidak banyak berubah ketika fitur-fitur distandardisasi atau tidak distandardisasi. Hasil ini menunjukkan bahwa algoritma-algoritma tersebut dapat diandalkan dan efektif dalam melakukan prediksi tanpa memerlukan tahap standardisasi yang rumit pada fitur-fitur data.


### Evaluation: Confusion Matrix
![Alt text](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/assets/confusionmatrix.png)
<h5 align="center">Gambar 7 — Confussion Matrix Random Forest</h5>

Berdasarkan model Random Forest, performa model secara mendetail dievaluasi menggunakan confusion matrix. Hasil dari confusion matrix menunjukkan bahwa model Random Forest menunjukkan **performa yang sangat baik dalam memprediksi pengguna yang mengklik iklan atau tidak.** Jumlah kesalahan prediksi, yang terdiri dari False Positive (prediksi salah bahwa pengguna mengklik iklan) dan False Negative (prediksi salah bahwa pengguna tidak mengklik iklan), sangat kecil. Kesalahan prediksi yang minim ini menandakan tingkat akurasi yang tinggi pada model.

Dengan nilai kesalahan prediksi yang kecil, model Random Forest dapat dianggap sebagai model prediksi yang akurat. Hal ini memberikan keyakinan kepada perusahaan untuk menggunakan model ini dalam mengidentifikasi dengan akurat pengguna yang memiliki potensi untuk mengklik iklan. Dengan demikian, perusahaan dapat mengoptimalkan strategi pemasarannya dengan lebih efektif, mengarah pada peningkatan hasil kampanye iklan dan penggunaan sumber daya pemasaran dengan lebih efisien. Model ini dapat menjadi alat yang berharga bagi perusahaan untuk meningkatkan targeting iklan dan memperoleh respons yang lebih baik dari target audiens.

### Evaluation: Feature Importance
![Alt text](https://github.com/imalfunadam/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/blob/main/assets/Feature%20Importance.png)
<h5 align="center">Gambar 8 — Feature Importance Random Forest</h5>

Analisis Feature Importance digunakan untuk mengidentifikasi fitur yang paling penting dalam membangun model. Dalam analisis menggunakan plot SHAP, beberapa fitur menonjol sebagai pengaruh utama terhadap prediksi klik pada iklan. Fitur-fitur yang menunjukkan pengaruh signifikan antara lain adalah Daily Internet Usage, Daily Time Spent on Site, Area Income, dan Age.

Fitur-fitur seperti **Daily Internet Usage, Daily Time Spent on Site, dan Area Income memiliki korelasi negatif** terhadap klik iklan, ditandai dengan warna merah pada sisi kiri plot. Hal ini menunjukkan bahwa pengguna dengan kebiasaan penggunaan internet yang kurang aktif dan pengguna dengan pendapatan menengah ke bawah memiliki potensi yang lebih tinggi untuk mengklik iklan. Di sisi lain, **fitur Age memiliki korelasi positif** dengan klik iklan. Artinya, semakin tua usia pengguna, semakin tinggi potensi mereka untuk mengklik iklan yang ditampilkan.

Informasi mengenai Feature Importance ini memberikan wawasan berharga untuk mengoptimalkan strategi pemasaran. Dengan mempertimbangkan karakteristik pengguna berdasarkan fitur-fitur yang memiliki pengaruh signifikan dalam model, perusahaan dapat menyusun iklan yang lebih efektif dan lebih sesuai dengan preferensi serta perilaku pengguna.

## 📂 STAGE 4: Business Recommendation
Rekomendasi berdasarkan Feature Importance dan insight yang telah ditemukan:

#### **Targeting Pengguna Internet Non-Aktif:**
1. **Iklan Singkat dan Menarik:**
    - Karena pengguna non-aktif memiliki keterbatasan waktu, penting untuk menciptakan iklan yang singkat dan menarik. Pesan yang padat dan jelas dengan pemilihan kata yang tepat dapat menarik perhatian mereka dalam waktu singkat.

2. **Retargeting:**
    - Manfaatkan strategi retargeting untuk terus berkomunikasi dengan pengguna non-aktif. Tampilkan iklan yang relevan secara berulang kali di berbagai platform yang mereka kunjungi untuk meningkatkan awareness pengguna.

3. **Konten Relevan:**
    - Pastikan konten iklan Anda relevan dengan minat dan kebutuhan pengguna non-aktif. Pahami preferensi mereka untuk mengoptimalkan respons terhadap iklan.

#### **Targeting Kelompok Usia di Atas 40 Tahun:**
1. Kampanye yang Relevan:
    - Fokuskan kampanye iklan yang memiliki dampak atau relevansi dengan kehidupan dan kebutuhan kelompok usia di atas 40 tahun.

2. Desain Sederhana:
    - Desain iklan yang mudah dibaca dan sederhana akan lebih efektif untuk kelompok usia di atas 40 tahun.

3. Pilih Platform yang Sesuai:
    - Gunakan platform iklan yang sesuai, seperti Facebook, karena kelompok usia di atas 40 tahun cenderung lebih sedikit terlibat dalam media sosial dibandingkan dengan kelompok usia yang lebih muda.

#### **Targeting Kelompok Pendapatan Menengah Kebawah:**
1. Penawaran Harga Terjangkau:
    - Berikan iklan dengan penawaran harga yang terjangkau dan sesuai dengan anggaran pengguna dalam kisaran. Diskon khusus, bundel, atau harga promo dapat mendorong mereka untuk mengklik iklan.

#### **Optimalisasi Waktu Penayangan Iklan:**
1. Manfaatkan Hari dan Jam Tertentu:
    - Manfaatkan hari Rabu yang menunjukkan konversi klik iklan yang baik. Selain itu, tingkat lalu lintas tinggi pada Selasa dan Sabtu dapat dimanfaatkan dengan penayangan iklan yang relevan.

2.  Jadwal Penayangan yang Tepat:
    - Gunakan jam-jam dengan potensi klik iklan tinggi, seperti pukul 00.00, 09.00, 11.00, dan 18.00. Pastikan iklan ditayangkan secara tepat pada saat-saat tersebut.

#### **Strategi Softselling untuk Pengguna Aktif:**
1. Pendekatan Softselling:
    - Jika perusahaan ingin menargetkan kelompok pengguna aktif, strategi iklan dengan pendekatan softselling dapat menjadi pilihan yang efektif. Fokus pada membangun hubungan yang baik dengan calon konsumen dan memberikan informasi bermanfaat tentang produk atau layanan.
2. Pilih Platform Media Sosial:
    - Gunakan media sosial sebagai platform penayangan iklan, mengingat kelompok pengguna aktif cenderung menggunakan media sosial secara intensif.
Dengan mengimplementasikan rekomendasi ini, perusahaan dapat lebih tepat sasaran dalam pemasaran dan meningkatkan efektivitas kampanye iklannya sesuai dengan karakteristik pengguna yang telah diidentifikasi melalui analisis data.



















