# Cek Risiko Gagal Bayar Kartu Kredit

Aplikasi berbasis **Streamlit** untuk memprediksi risiko gagal bayar kartu kredit pada bulan berikutnya menggunakan model machine learning yang telah dilatih sebelumnya.

Pengguna dapat memasukkan informasi nasabah, limit kartu kredit, usia, serta riwayat status pembayaran, jumlah tagihan, dan jumlah pembayaran selama enam bulan terakhir untuk memperoleh estimasi risiko gagal bayar.

## Fitur

- Prediksi risiko gagal bayar kartu kredit menggunakan model machine learning.
- Input karakteristik nasabah dan riwayat transaksi selama enam bulan terakhir.
- Visualisasi persentase estimasi gagal bayar dan kemungkinan lancar membayar.
- Penentuan hasil prediksi berdasarkan threshold model.
- Informasi mengenai keterbatasan model dan detail teknis input prediksi.
- Validasi kategori input agar sesuai dengan kategori yang dikenali model.

## Teknologi

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- Joblib

## Struktur Proyek

```text
credit-streamlit/
├── app.py
├── README.md
├── requirements.txt
├── model/
│   ├── model_final.joblib
│   └── model_final.json
└── notebooks/
    └── credit_scoring.ipynb
```

## Menjalankan Secara Lokal

### 1. Clone repository

```bash
git clone <URL_REPOSITORY_GITHUB>
cd credit-streamlit
```

Ganti `<URL_REPOSITORY_GITHUB>` dengan URL repository GitHub milikmu.

### 2. Buat virtual environment

```bash
python -m venv .venv
```

### 3. Aktifkan virtual environment

**Windows (Command Prompt):**

```bat
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 4. Instal dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Jalankan aplikasi

```bash
streamlit run app.py
```

Setelah aplikasi berjalan, buka alamat lokal yang ditampilkan di terminal.

## Deploy ke Streamlit Community Cloud

1. Push project ke repository GitHub.
2. Buka [Streamlit Community Cloud](https://share.streamlit.io/) dan login menggunakan GitHub.
3. Pilih **Create app**.
4. Pilih repository, branch, dan file utama `app.py`.
5. Tentukan versi Python yang kompatibel dengan model dan dependencies.
6. Klik **Deploy**.

Pastikan file model dan metadata tersedia pada lokasi yang sesuai dengan konfigurasi path di `app.py`.

## Tentang Model

Aplikasi menggunakan model machine learning yang disimpan dalam format `.joblib`. Metadata model dalam format JSON digunakan untuk membaca informasi seperti fitur model, rentang numerik, threshold, dan metrik evaluasi apabila tersedia.

Preprocessing input pada aplikasi mencakup penyesuaian kategori pendidikan dan status pembayaran agar konsisten dengan proses training model.

## Dataset dan Interpretasi

Dataset yang digunakan berasal dari nasabah kartu kredit di Taiwan. Nilai uang yang ditampilkan menggunakan NT$ (dolar Taiwan), sesuai dengan dataset yang digunakan untuk melatih model.

Hasil prediksi merupakan estimasi model, bukan kepastian bahwa nasabah akan gagal bayar. Karena model menggunakan pembobotan kelas, probabilitas yang ditampilkan dapat lebih tinggi daripada peluang gagal bayar yang sebenarnya.

Aplikasi ini ditujukan untuk demonstrasi dan eksplorasi machine learning, bukan sebagai satu-satunya dasar pengambilan keputusan kredit.

## Notebook

Notebook yang berisi proses eksplorasi data, preprocessing, training, dan evaluasi model tersedia di folder [`notebooks/`](./notebooks/).

Notebook digunakan untuk mendokumentasikan proses pengembangan model, sedangkan `app.py` digunakan untuk menjalankan model yang sudah dilatih melalui aplikasi Streamlit.
