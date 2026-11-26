# Solar Panel Fault Classification

Klasifikasi gambar panel surya ke dalam 6 kondisi: bersih, berdebu, tertutup salju, terkena kotoran burung, rusak secara fisik, dan rusak secara elektrik. Proyek ini membandingkan dua pendekatan: transfer learning dengan MobileNetV2, dan model machine learning klasik (Random Forest, SVM, Gradient Boosting, Stacking) yang memakai fitur hasil ekstraksi VGG16.

Seluruh kode ada di satu notebook: `solar-panel-fault-classification.ipynb`.

## Dataset

Sumber: [Solar Panel Images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images) di Kaggle. Lisensi dataset tertulis "unknown" di halaman Kaggle, jadi repo ini tidak menyertakan gambarnya. Notebook mengunduh dataset langsung dari Kaggle.

| Kelas | Jumlah gambar |
|---|---|
| Clean | 194 |
| Bird-drop | 193 |
| Dusty | 191 |
| Snow-Covered | 124 |
| Electrical-damage | 104 |
| Physical-Damage | 70 |

Kelas Physical-Damage punya gambar paling sedikit, sekitar sepertiga dari kelas terbesar.

## Pendekatan

### 1. CNN end-to-end (MobileNetV2)

- Split data 80/10/10 (train 693, validation 85, test 91 gambar) dengan seed 42.
- Resize ke 224 x 224 dan normalisasi piksel ke 0-1.
- Augmentasi pada data train: rotasi 20 derajat, pergeseran horizontal dan vertikal 20%, shear 20%, zoom 20%, flip horizontal.
- Model: MobileNetV2 pretrained ImageNet (bobot dibekukan), Global Average Pooling, Dense 128 unit (ReLU), Dropout 0,5, Dense 6 unit (softmax). Optimizer Adam, 10 epoch.
- Fine-tuning: 30 layer terakhir MobileNetV2 dibuka dengan learning rate 1e-5, memakai EarlyStopping dan ReduceLROnPlateau.

### 2. Ekstraksi fitur VGG16 + model klasik

- VGG16 pretrained ImageNet tanpa lapisan klasifikasi mengubah tiap gambar menjadi vektor 25.088 fitur.
- Split 80/20 dengan stratifikasi (177 gambar test) dan `random_state=42`.
- Model: Random Forest (100 pohon), SVM RBF (C=10, dengan StandardScaler), Gradient Boosting (100 estimator, learning rate 0,1, kedalaman 3).
- Stacking: ketiga model di atas sebagai base learner, Logistic Regression sebagai meta-model, cross-validation 5 lipatan.

## Hasil

| Model | Accuracy | F1 (weighted) | Jumlah gambar test |
|---|---|---|---|
| MobileNetV2 | 0,7473 | 0,7325 | 91 |
| MobileNetV2 (fine-tuning) | 0,7582 | 0,7427 | 91 |
| Random Forest (fitur VGG16) | 0,7288 | 0,7268 | 177 |
| SVM (fitur VGG16) | 0,6497 | 0,6405 | 177 |
| Gradient Boosting (fitur VGG16) | 0,7232 | 0,7199 | 177 |
| Stacking (fitur VGG16) | 0,7514 | 0,7464 | 177 |

Temuan utama:

- Lima dari enam model berada di kisaran 72-76% akurasi. SVM paling rendah dengan 65%.
- Pada test set 91 gambar, satu gambar bernilai sekitar 1,1 poin akurasi. Selisih antara MobileNetV2 sebelum dan sesudah fine-tuning sama dengan satu gambar.
- Fine-tuning berhenti di epoch 6 karena validation loss tidak membaik setelah epoch 1.
- Snow-Covered adalah kelas paling mudah dikenali (F1 0,92 pada MobileNetV2 dan Stacking).
- Physical-Damage adalah kelas tersulit. MobileNetV2 hanya mengenali 1 dari 7 gambar kelas ini (recall 0,14). Stacking menaikkan recall menjadi 0,43 (6 dari 14 gambar).

## Keterbatasan

- Dataset kecil (sekitar 870 gambar) dan tidak seimbang antar kelas.
- Test set kecil dan evaluasi memakai satu kali split tanpa cross-validation, jadi selisih beberapa poin antar model belum tentu bermakna.
- Pendekatan 1 dan 2 memakai split yang berbeda (91 vs 177 gambar test), sehingga angkanya tidak bisa dibandingkan langsung.

## Pengembangan lanjutan

- Pakai class weight atau tambah data Physical-Damage untuk memperbaiki recall kelas itu.
- Evaluasi dengan stratified k-fold cross-validation pada kedua pendekatan, dengan split yang sama.
- Coba arsitektur lain seperti EfficientNet atau fine-tuning VGG16.

## Cara menjalankan

1. Buka notebook di Google Colab atau Jupyter dengan GPU.
2. Siapkan kredensial Kaggle sendiri: simpan `kaggle.json` di `~/.kaggle/` atau pakai Colab Secrets. Jangan menulis username dan API key di dalam notebook.
3. Jalankan semua sel secara berurutan. Notebook memasang `kaggle`, `split-folders`, dan memakai library berikut: TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn.

## Struktur repo

```
solar-panel-fault-classification/
├── solar-panel-fault-classification.ipynb
└── README.md
```
