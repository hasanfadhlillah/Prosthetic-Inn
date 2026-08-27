# GEMASTIK XVII (2024) — Pengembangan Framework Baru Multi-Stream 3D CNN

## Abstrak

Riset ini mengembangkan **framework baru Multi-Stream 3D CNN** untuk mengoptimalkan akurasi klasifikasi **8 gestur tangan** pada tangan prostetik mioelektrik, sebagai kelanjutan riset Laboratorium Robotika dan Embedded System FILKOM UB yang sebelumnya hanya mencapai akurasi 87.74% pada 4 gestur (Adani & Widasari, 2023).

## Dataset

Dataset **sekunder** dari Kaggle: *EMG Signal for Gesture Recognition*, direkam menggunakan **MYO Thalmic Bracelet** (wearable wireless) dari 36 subjek. 8 gestur yang diklasifikasikan: *open hand, hand at rest, hand clenched in a fist, wrist flexion, wrist extension, radial deviations, ulnar deviations,* dan *extended palm*.

## Metode

1. **Filtering:** Exponential filter (α = 0.5) + Notch filter (50 Hz)
2. **Segmentasi:** Adjacent Windowing 3 detik (overlap 50%) → 7.408 segmen
3. **Ekstraksi Fitur:** MAV, RMS, Amplitudo (3 stream independen)
4. **Klasifikasi:** Multi-Stream 3D CNN — 3× Conv3D (32, 64, 64 kernel; 3×3×3) → 2× Fully Connected (512, 128 neuron) → Softmax (8 kelas)
5. **Training:** 50 epoch, batch size 32, validation split 20%, optimizer Adam

## Hasil

| Metrik | Nilai |
|---|---|
| Akurasi | 95.87% |
| Presisi | 95.87% |
| Recall | 100% |
| F1-Score | 96.32% |

## Tim

- Muhammad Hasan Fadhlillah (225150207111026) — Ketua
- Arvin Mulia Fernanda (225150300111018)
- Zulfiana Aulia Syafa (225150401111020)
- Pembimbing: Edita Rosana Widasari, S.T., M.T., M.Eng., Ph.D

## File

- `KTI_GEMASTIK-XVII_Prosthetic-Inn.pdf` — Naskah karya tulis ilmiah lengkap
- `Prosthetic-Inn_Multi-Stream-3D-CNN.ipynb` — Notebook training & evaluasi model
