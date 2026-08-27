<div align="center">

# 🦾 Prosthetic-Inn
### Multi-Stream 3D CNN untuk Klasifikasi Gestur Tangan Prostetik Mioelektrik Berbasis Sinyal EMG

_"Mengembalikan gerak, membangun kemandirian."_

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![GEMASTIK](https://img.shields.io/badge/GEMASTIK-XVII%20%26%20XVIII-003366?style=for-the-badge)]()
[![FILKOM UB](https://img.shields.io/badge/FILKOM-Universitas%20Brawijaya-blue?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Selesai-brightgreen?style=flat-square)]()
[![License](https://img.shields.io/badge/License-Academic%20Use-blue?style=flat-square)]()

<img src="https://img.shields.io/badge/🎯%20Akurasi%20Terbaik-98%25-blue?style=flat-square" />
<img src="https://img.shields.io/badge/✋%20Gestur%20Terklasifikasi-10-blue?style=flat-square" />
<img src="https://img.shields.io/badge/🧠%20Arsitektur-Multi--Stream%203D%20CNN-blue?style=flat-square" />

</div>

---

## 📌 Daftar Isi
- [Tentang Project](#-tentang-project)
- [Dua Iterasi Penelitian](#-dua-iterasi-penelitian)
- [Alur Kerja Sistem](#-alur-kerja-sistem)
- [Tech Stack](#️-tech-stack)
- [Struktur Direktori](#-struktur-direktori)
- [Ringkasan Hasil](#-ringkasan-hasil)
- [Tim Peneliti](#-tim-peneliti)
- [Sitasi](#-sitasi)

---

## 📖 Tentang Project

**Prosthetic-Inn** adalah rangkaian riset yang mengembangkan **framework Multi-Stream 3D Convolutional Neural Network (CNN)** untuk mengklasifikasikan gestur tangan pada **tangan prostetik mioelektrik** berbasis sinyal *Electromyography* (EMG).

Latar belakangnya: dari ±3 juta penyandang amputasi lengan bawah di dunia (2,4 juta di antaranya di negara berkembang termasuk Indonesia), tingkat penerimaan tangan prostetik di Indonesia baru berkisar **27%–56%** karena keterbatasan teknis dan sosial — salah satunya akurasi klasifikasi gestur yang masih rendah dan penggunaan sensor EMG *wired* yang menimbulkan noise serta ketidaknyamanan.

Project ini dikembangkan di **Laboratorium Robotika dan Embedded System (RES), Fakultas Ilmu Komputer, Universitas Brawijaya**, sebagai bagian dari riset berkelanjutan tangan prostetik mioelektrik *low-cost* milik lab tersebut, dan diikutsertakan pada kompetisi **GEMASTIK (Pagelaran Mahasiswa Nasional Bidang Teknologi Informasi dan Komunikasi)** cabang Karya Tulis Ilmiah (KTI) TIK.

> 🎓 Riset ini selaras dengan **Sustainable Development Goals (SDGs)** poin 3 (Kesehatan yang Baik dan Kesejahteraan), poin 10 (Mengurangi Kesenjangan), dan poin 11 (Kota dan Permukiman yang Berkelanjutan), serta UU No. 8 Tahun 2016 tentang Penyandang Disabilitas.

---

## 🔬 Dua Iterasi Penelitian

Repository ini memuat **dua iterasi** penelitian yang saling berkelanjutan (iterasi 2025 merupakan pengembangan lanjutan dari iterasi 2024):

| | GEMASTIK XVII (2024) | GEMASTIK XVIII (2025) |
|---|---|---|
| **Judul** | Pengembangan Framework Baru Multi-Stream 3D CNN dalam Optimalisasi Akurasi Gestur Tangan Prostetik Mioelektrik untuk Peningkatan Pelayanan Kesejahteraan Sosial bagi Penyandang Disabilitas | Inovasi Framework Multi-Stream 3D CNN Berbasis Wearable Wireless EMG Sensor untuk Peningkatan Performansi Klasifikasi Gestur Tangan Prostetik Mioelektrik guna Mendukung Kemandirian Penyandang Disabilitas di Indonesia |
| **Dataset** | Sekunder — Kaggle *"EMG Signal for Gesture Recognition"* (MYO Thalmic bracelet, 36 subjek) | Primer — akuisisi langsung via **Oymotion gForce200** wearable wireless EMG armband (8 kanal, 15 subjek) |
| **Jumlah Gestur** | 8 gestur | 10 gestur |
| **Akurasi** | 95.87% | 98% |
| **Precision / Recall / F1** | 95.87% / 100% / 96.32% | 98% / 98% / 98% |
| **Fokus Peningkatan** | Pelayanan kesejahteraan sosial | Kemandirian penyandang disabilitas (sensor *wireless*, kurangi noise & tingkatkan kenyamanan) |
| **Tim** | Hasan, Arvin Mulia Fernanda, Zulfiana Aulia Syafa | Hasan, Zulfiana Aulia Syafa, Husain |

📂 Detail lengkap masing-masing iterasi ada di folder [`2024-gemastik-xvii/`](./2024-gemastik-xvii) dan [`2025-gemastik-xviii/`](./2025-gemastik-xviii).

---

## 🧠 Alur Kerja Sistem

Kedua iterasi menggunakan pipeline yang sama, hanya berbeda sumber data dan jumlah gestur:

```text
            Sinyal EMG (8 kanal)
                     │
                     ▼
    ┌───────────────────────────────┐
    │  1. Filtering                  │
    │  Exponential Filter (α=0.5)    │
    │  + Notch Filter (50 Hz)        │
    └────────────────┬────────────────┘
                     ▼
    ┌───────────────────────────────┐
    │  2. Segmentasi                 │
    │  Adjacent Windowing (3s)       │
    └────────────────┬────────────────┘
                     ▼
    ┌───────────────────────────────┐
    │  3. Ekstraksi Fitur (3 stream) │
    │  MAV · RMS · Amplitudo         │
    └────────────────┬────────────────┘
                     ▼
    ┌───────────────────────────────┐
    │  4. Multi-Stream 3D CNN        │
    │  3x Conv3D (32,64,64; 3x3x3)   │
    │  → 2x FC (512,128) → Softmax   │
    └────────────────┬────────────────┘
                     ▼
          Klasifikasi Gestur Tangan
```

Ketiga fitur (MAV, RMS, Amplitudo) diproses sebagai *stream* independen melalui arsitektur 3D CNN yang identik, lalu digabungkan (*concatenation*) sebelum masuk ke *fully connected layers* dan lapisan output *softmax*. Performa dibandingkan terhadap baseline **1D CNN** dan **Compact CNN**.

---

## 🛠️ Tech Stack

<div align="center">

| Kategori | Teknologi |
|---|---|
| **Bahasa** | ![Python](https://img.shields.io/badge/-Python%203.10+-3776ab?logo=python&logoColor=white) |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/-TensorFlow-ff6f00?logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/-Keras-d00000?logo=keras&logoColor=white) ![Scikit--Learn](https://img.shields.io/badge/-Scikit--Learn-f7931e?logo=scikitlearn&logoColor=white) |
| **Signal Processing** | ![SciPy](https://img.shields.io/badge/-SciPy-8caae6?logo=scipy&logoColor=white) |
| **Data & Visualisasi** | ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c) ![Seaborn](https://img.shields.io/badge/-Seaborn-3776ab) |
| **Sensor EMG** | Oymotion gForce200 Gesture Armband (wearable wireless, 2025) · MYO Thalmic Bracelet (dataset Kaggle, 2024) |

</div>

---

## 📂 Struktur Direktori

```text
Prosthetic-Inn/
├── README.md                                  # Overview & perbandingan kedua iterasi (file ini)
│
├── 2024-gemastik-xvii/
│   ├── README.md                              # Ringkasan khusus iterasi 2024
│   ├── KTI_GEMASTIK-XVII_Prosthetic-Inn.pdf
│   └── Prosthetic-Inn_Multi-Stream-3D-CNN.ipynb
│
└── 2025-gemastik-xviii/
    ├── README.md                              # Ringkasan khusus iterasi 2025
    ├── KTI_GEMASTIK-XVIII_Prosthetic-Inn.pdf
    └── Prosthetic-Inn_Multi-Stream-3D-CNN_v2.ipynb
```

---

## 📊 Ringkasan Hasil

**Hasil Evaluasi Iterasi 2025 (10 Gestur):**

| Gestur | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Open | 1.00 | 0.97 | 1.00 | 0.98 |
| Close | 0.98 | 0.98 | 0.98 | 0.98 |
| Fine Grip | 0.97 | 0.98 | 0.97 | 0.97 |
| Pointer | 0.95 | 0.97 | 0.95 | 0.96 |
| Agree | 1.00 | 1.00 | 1.00 | 1.00 |
| Two | 0.95 | 0.98 | 0.95 | 0.97 |
| Three | 1.00 | 0.97 | 1.00 | 0.98 |
| Four | 1.00 | 0.95 | 1.00 | 0.98 |
| Pinch | 0.95 | 1.00 | 0.95 | 0.97 |
| Half Close | 1.00 | 1.00 | 1.00 | 1.00 |
| **Overall** | **0.98** | **0.98** | **0.98** | **0.98** |

Dibandingkan riset sebelumnya oleh Adani & Widasari (2023) yang hanya mencapai akurasi **87.74%** pada 4 gestur menggunakan sensor EMG *wired*, framework ini berhasil meningkatkan akurasi ke **98% pada 10 gestur** sekaligus beralih ke sensor *wireless* yang lebih nyaman digunakan.

---

## 👥 Tim Peneliti

**GEMASTIK XVII (2024):**
1. Muhammad Hasan Fadhlillah (225150207111026) — Ketua
2. Arvin Mulia Fernanda (225150300111018)
3. Zulfiana Aulia Syafa (225150401111020)

**GEMASTIK XVIII (2025):**
1. Muhammad Hasan Fadhlillah (225150207111026) — Ketua
2. Zulfiana Aulia Syafa (225150401111020)
3. Muhammad Husain Fadhlillah (225150207111027)

**Dosen Pembimbing:** Edita Rosana Widasari, S.T., M.T., M.Eng., Ph.D

*Fakultas Ilmu Komputer, Universitas Brawijaya — Laboratorium Robotika dan Embedded System*

---

## 📚 Sitasi
