# GEMASTIK XVIII (2025) — Inovasi Framework Multi-Stream 3D CNN Berbasis Wearable Wireless EMG Sensor

## Abstrak

Pengembangan lanjutan dari iterasi 2024: framework **Multi-Stream 3D CNN** kini diuji dengan **data primer** hasil akuisisi langsung menggunakan sensor **wearable wireless EMG**, mengklasifikasikan **10 gestur tangan** guna mendukung kemandirian penyandang disabilitas di Indonesia — sekaligus menghilangkan noise dan ketidaknyamanan yang muncul dari sensor EMG *wired*.

## Dataset

Data **primer**, diakuisisi langsung menggunakan sensor **Oymotion gForce200 Gesture Armband** (wearable wireless, 8 kanal elektroda) dari **15 subjek** berusia 21–22 tahun (10 subjek data latih, 5 subjek data uji). 10 gestur yang diklasifikasikan: *Hand Open, Hand Close, Fine Grip, Pointer, Agree, Two, Three, Four, Pinch,* dan *Half Close*.

## Metode

1. **Filtering:** Exponential filter (α = 0.5) + Notch filter (50 Hz)
2. **Segmentasi:** Adjacent Windowing 3 detik → 100 segmen per kelas gestur
3. **Ekstraksi Fitur:** MAV, RMS, Amplitudo (3 stream independen)
4. **Klasifikasi:** Multi-Stream 3D CNN — 3× Conv3D (32, 64, 64 kernel; 3×3×3) → 2× Fully Connected (512, 128 neuron) → Softmax (10 kelas)
5. **Training:** 72 epoch, batch size 16, validation split 20%, optimizer Adam
6. **Baseline Pembanding:** 1D CNN dan Compact CNN

## Hasil

| Metrik | Nilai |
|---|---|
| Akurasi | 98% |
| Presisi | 98% |
| Recall | 98% |
| F1-Score | 98% |

Gestur *Open, Agree, Three, Four,* dan *Half Close* mencapai recall sempurna (1.00). Hasil ini meningkat signifikan dari riset sebelumnya oleh Adani & Widasari (2023) yang mencapai 87.74% pada 4 gestur menggunakan sensor *wired*.

## Tim

- Muhammad Hasan Fadhlillah (225150207111026) — Ketua
- Zulfiana Aulia Syafa (225150401111020)
- Muhammad Husain Fadhlillah (225150207111027)
- Pembimbing: Edita Rosana Widasari, S.T., M.T., M.Eng., Ph.D

## File

- `KTI_GEMASTIK-XVIII_Prosthetic-Inn.pdf` — Naskah karya tulis ilmiah lengkap
- `Prosthetic-Inn_Multi-Stream-3D-CNN_v2.ipynb` — Notebook training & evaluasi model
