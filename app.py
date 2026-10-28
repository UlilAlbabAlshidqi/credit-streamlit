"""Aplikasi Streamlit: cek risiko gagal bayar kartu kredit (untuk pengguna awam).

Catatan penting soal konsistensi dengan training:
1. Pipeline training mengubah kategori menjadi string dari nilai FLOAT ("2.0", bukan "2").
   Karena OneHotEncoder(handle_unknown='ignore'), nilai int akan diam-diam jadi nol semua.
   Maka semua input di sini dipaksa float, lalu dicek ulang terhadap kategori encoder.
2. Feature engineering saat training (EDUCATION & PAY digabung) harus diulang saat inference.
"""
import base64
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cek Risiko Gagal Bayar Kartu Kredit", page_icon="💳", layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model_final.joblib"
META_PATH = MODEL_DIR / "model_final.json"


# ----------------------------------------------------------------------------
# Fungsi custom yang ikut di-pickle di pipeline.
# Di Colab fungsi ini didefinisikan di __main__, jadi joblib mencarinya di __main__.
# ----------------------------------------------------------------------------
def categorical_to_string(X):
    X = pd.DataFrame(X).copy()
    return X.map(lambda value: str(value) if pd.notna(value) else np.nan).to_numpy()


import __main__  # noqa: E402

__main__.categorical_to_string = categorical_to_string


@st.cache_resource(show_spinner="Memuat model...")
def muat_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def muat_metadata():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}


# ----------------------------------------------------------------------------
# Konstanta
# ----------------------------------------------------------------------------
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
FITUR_MODEL = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_COLS + BILL_COLS + PAYAMT_COLS
)

# Label ramah pengguna -> kode asli dataset (kategori 0 MARRIAGE sengaja tidak ditampilkan)
OPSI_SEX = {"Laki-laki": 1, "Perempuan": 2}
OPSI_EDUCATION = {"Pascasarjana": 1, "Universitas / S1": 2, "SMA": 3, "Lainnya": 4}
OPSI_MARRIAGE = {"Menikah": 1, "Belum menikah": 2, "Lainnya": 3}
OPSI_STATUS = {
    "Tidak ada pemakaian / tidak ada tagihan": -2,
    "Lunas tepat waktu": -1,
    "Bayar sebagian / cicilan minimum": 0,
    "Terlambat 1 bulan": 1,
    "Terlambat 2 bulan": 2,
    "Terlambat 3 bulan atau lebih": 3,
}
LABEL_BULAN = [
    "Bulan ke-1 (paling baru)", "Bulan ke-2", "Bulan ke-3",
    "Bulan ke-4", "Bulan ke-5", "Bulan ke-6 (paling lama)",
]


# ----------------------------------------------------------------------------
# Preprocessing sisi aplikasi (harus identik dengan feature engineering training)
# ----------------------------------------------------------------------------
def ke_float(v):
    return np.nan if v is None else float(v)


def terapkan_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0.0: 4.0, 5.0: 4.0, 6.0: 4.0})
    for col in PAY_COLS:
        df[col] = df[col].replace({4.0: 3.0, 5.0: 3.0, 6.0: 3.0, 7.0: 3.0, 8.0: 3.0})
    # Saat training kategori 1 digabung ke 2 untuk PAY_2..PAY_4. Pada PAY_5/PAY_6 kategori 1
    # memang tidak pernah muncul di data latih, jadi juga dipetakan ke 2 agar tidak "unknown".
    for col in ["PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        df[col] = df[col].replace({1.0: 2.0})
    return df.astype(float)


def cek_kategori_dikenal(model, df: pd.DataFrame) -> list[str]:
    """Pastikan setiap nilai kategorikal (sebagai string) dikenal oleh OneHotEncoder."""
    masalah = []
    prep = model.named_steps["preprocessor"]
    for nama, trf, kolom in prep.transformers_:
        if nama not in ("kategorik_ordinal", "kategorik_nominal"):
            continue
        enc = trf.named_steps["encoder"]
        for col, cats in zip(kolom, enc.categories_):
            v = df[col].iloc[0]
            s = "KOSONG" if pd.isna(v) else str(v)
            if s not in {str(c) for c in cats}:
                masalah.append(f"{col}: nilai '{s}' tidak dikenal (kategori model: {list(cats)})")
    return masalah


# ----------------------------------------------------------------------------
# Tampilan kartu hasil
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');
.stApp h1 { font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif; font-weight: 800; letter-spacing: -0.02em; }
.hasil, .hasil * { font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif; box-sizing: border-box; }
.hasil { border-radius: 20px; padding: 30px 28px 24px; color: #fff; }
.hasil.ok { background: #0E5A52; }
.hasil.bad { background: #8F2B22; }
.hasil.kosong { background: #1B2B3C; }
.hasil .kepala { display: flex; align-items: center; gap: 14px; }
.hasil .ikon { width: 56px; height: 56px; flex: none; }
.hasil .judul { font-size: 1.1rem; font-weight: 700; margin: 0; line-height: 1.3; }
.hasil .angka { font-size: 4.6rem; line-height: 1; font-weight: 800; letter-spacing: -0.04em; margin: 14px 0 6px; }
.hasil .ket { font-size: .92rem; opacity: .88; margin: 0 0 24px; line-height: 1.45; }
.hasil .barwrap { position: relative; margin: 0 0 6px; }
.hasil .bar { display: flex; height: 16px; border-radius: 999px; overflow: hidden; background: rgba(255,255,255,.15); }
.hasil .seg-ok { background: #5FD6C2; }
.hasil .seg-bad { background: #FF8A7A; }
.hasil .tanda { position: absolute; top: -5px; bottom: -5px; width: 3px; margin-left: -1px; background: #fff; border-radius: 2px; }
.hasil .catatan { font-size: .78rem; opacity: .75; margin: 8px 0 18px; }
.hasil .baris { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-top: 1px solid rgba(255,255,255,.2); font-size: .95rem; }
.hasil .baris b { font-weight: 800; font-size: 1.05rem; }
.hasil .titik { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
.hasil .istilah { font-size: .78rem; opacity: .75; margin: 14px 0 0; line-height: 1.45; }
</style>
"""


def ikon(jenis: str) -> str:
    """Ikon kartu kredit. SVG di-embed sebagai data URI agar tidak disaring sanitizer HTML."""
    latar = {"ok": "#0E5A52", "bad": "#8F2B22", "kosong": "#1B2B3C"}[jenis]
    kartu = (
        '<rect x="5" y="14" width="46" height="32" rx="6" fill="none" stroke="#fff" stroke-width="3"/>'
        '<rect x="5" y="22" width="46" height="7" fill="#fff"/>'
        '<rect x="11" y="35" width="11" height="6" rx="2" fill="#fff" opacity=".9"/>'
    )
    if jenis == "ok":
        lencana = (
            f'<circle cx="47" cy="44" r="13" fill="#5FD6C2" stroke="{latar}" stroke-width="3"/>'
            f'<path d="M41 44.5l4.2 4.2L53 39.5" fill="none" stroke="{latar}" '
            'stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    elif jenis == "bad":
        lencana = (
            f'<circle cx="47" cy="44" r="13" fill="#FF8A7A" stroke="{latar}" stroke-width="3"/>'
            f'<path d="M47 37v8" stroke="{latar}" stroke-width="3.6" stroke-linecap="round"/>'
            f'<circle cx="47" cy="50.5" r="2" fill="{latar}"/>'
        )
    else:
        lencana = ""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="56" height="56">'
        f'{kartu}{lencana}</svg>'
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img class="ikon" src="data:image/svg+xml;base64,{b64}" alt="">'


def kartu_kosong() -> str:
    return "".join([
        '<div class="hasil kosong">',
        f'<div class="kepala">{ikon("kosong")}<p class="judul">Hasil prediksi</p></div>',
        '<div class="angka">–</div>',
        '<p class="ket">Isi data di sebelah kiri, lalu klik <b>Hitung risiko</b>. '
        'Persentase kemungkinan akan muncul di sini.</p>',
        '<p class="istilah">Gagal bayar berarti tagihan kartu kredit bulan depan tidak terbayar.</p>',
        '</div>',
    ])


def kartu_hasil(p_gagal: float, threshold: float) -> str:
    gagal = p_gagal >= threshold
    p_menang = p_gagal if gagal else 1 - p_gagal
    kelas = "bad" if gagal else "ok"
    judul = "Diprediksi gagal bayar" if gagal else "Diprediksi lancar membayar"
    ket = (
        "Model menilai tagihan kartu kredit bulan depan kemungkinan besar tidak terbayar."
        if gagal else
        "Model menilai tagihan kartu kredit bulan depan kemungkinan besar terbayar."
    )
    lancar_pct = (1 - p_gagal) * 100
    gagal_pct = p_gagal * 100
    posisi_batas = (1 - threshold) * 100
    return "".join([
        f'<div class="hasil {kelas}">',
        f'<div class="kepala">{ikon(kelas)}<p class="judul">{judul}</p></div>',
        f'<div class="angka">{p_menang * 100:.1f}%</div>',
        f'<p class="ket">{ket}</p>',
        '<div class="barwrap"><div class="bar">',
        f'<div class="seg-ok" style="width:{lancar_pct:.2f}%"></div>',
        f'<div class="seg-bad" style="width:{gagal_pct:.2f}%"></div>',
        '</div>',
        f'<div class="tanda" style="left:{posisi_batas:.2f}%"></div></div>',
        f'<p class="catatan">Garis putih = batas penentuan. Diprediksi gagal bayar jika '
        f'persentase gagal bayar mencapai {threshold * 100:.0f}%.</p>',
        f'<div class="baris"><span><span class="titik" style="background:#5FD6C2"></span>'
        f'Lancar membayar</span><b>{lancar_pct:.1f}%</b></div>',
        f'<div class="baris"><span><span class="titik" style="background:#FF8A7A"></span>'
        f'Gagal bayar</span><b>{gagal_pct:.1f}%</b></div>',
        '<p class="istilah">Gagal bayar berarti tagihan kartu kredit bulan depan tidak terbayar.</p>',
        '</div>',
    ])


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
model = muat_model()
meta = muat_metadata()
rentang = meta.get("rentang_numerik", {})
threshold = meta.get("evaluasi_test", {}).get("threshold", 0.5)

st.markdown(CSS, unsafe_allow_html=True)


def batas(col, default_min, default_max):
    r = rentang.get(col, {})
    return float(r.get("min", default_min)), float(r.get("max", default_max))


st.title("Cek risiko gagal bayar kartu kredit")
st.write(
    "Isi data kartu kredit dan riwayat 6 bulan terakhir. Kolom boleh dikosongkan jika tidak "
    "tahu, tetapi makin lengkap datanya, makin masuk akal hasilnya. "
    "Semua nilai uang dalam NT$ (dolar Taiwan), sesuai data yang dipakai melatih model."
)

lim_min, lim_max = batas("LIMIT_BAL", 10000, 1_000_000)
age_min, age_max = batas("AGE", 21, 79)

kiri, kanan = st.columns([1.8, 1], gap="large")

with kiri:
    with st.form("form_prediksi"):
        st.subheader("Data nasabah")
        c1, c2 = st.columns(2)
        limit = c1.number_input(
            "Limit kartu kredit (NT$) *", min_value=lim_min, max_value=lim_max, value=None,
            step=10000.0, format="%.0f", placeholder="mis. 50000",
        )
        umur = c2.number_input(
            "Usia (tahun) *", min_value=age_min, max_value=age_max, value=None,
            step=1.0, format="%.0f", placeholder="mis. 35",
        )
        c3, c4, c5 = st.columns(3)
        sex = c3.selectbox("Jenis kelamin", list(OPSI_SEX), index=None, placeholder="Pilih (opsional)")
        edu = c4.selectbox("Pendidikan", list(OPSI_EDUCATION), index=None, placeholder="Pilih (opsional)")
        mar = c5.selectbox(
            "Status pernikahan", list(OPSI_MARRIAGE), index=None,
            placeholder="Pilih status pernikahan (opsional)",
        )

        st.subheader("Riwayat 6 bulan terakhir")
        status, tagihan, bayar = [], [], []
        for i in range(6):
            st.markdown(f"**{LABEL_BULAN[i]}**")
            a, b, c = st.columns(3)
            status.append(a.selectbox(
                "Status pembayaran", list(OPSI_STATUS), index=None,
                placeholder="Pilih status", key=f"pay_{i}",
            ))
            bmin, bmax = batas(BILL_COLS[i], -200_000, 1_000_000)
            pmin, pmax = batas(PAYAMT_COLS[i], 0, 1_000_000)
            tagihan.append(b.number_input(
                "Jumlah tagihan (NT$)", min_value=bmin, max_value=bmax, value=None,
                step=1000.0, format="%.0f", placeholder="mis. 20000", key=f"bill_{i}",
            ))
            bayar.append(c.number_input(
                "Jumlah dibayar (NT$)", min_value=pmin, max_value=pmax, value=None,
                step=1000.0, format="%.0f", placeholder="mis. 5000", key=f"payamt_{i}",
            ))

        kirim = st.form_submit_button("Hitung risiko", type="primary", use_container_width=True)

with kanan:
    if not kirim:
        st.markdown(kartu_kosong(), unsafe_allow_html=True)
    elif limit is None or umur is None:
        st.markdown(kartu_kosong(), unsafe_allow_html=True)
        st.error("Limit kartu kredit dan usia wajib diisi.")
    else:
        # Kosong -> NaN (model dilatih dengan NaN: median untuk angka, 'KOSONG' untuk kategori)
        row = {
            "LIMIT_BAL": ke_float(limit),
            "SEX": ke_float(OPSI_SEX.get(sex)),
            "EDUCATION": ke_float(OPSI_EDUCATION.get(edu)),
            "MARRIAGE": ke_float(OPSI_MARRIAGE.get(mar)),
            "AGE": ke_float(umur),
        }
        for i in range(6):
            row[PAY_COLS[i]] = ke_float(OPSI_STATUS.get(status[i]))
            row[BILL_COLS[i]] = ke_float(tagihan[i])
            row[PAYAMT_COLS[i]] = ke_float(bayar[i])

        fitur = meta.get("fitur_model", FITUR_MODEL)
        df_input = terapkan_feature_engineering(pd.DataFrame([row])[fitur])

        # Pengaman: pastikan tidak ada kategori yang diam-diam jadi nol semua
        masalah = cek_kategori_dikenal(model, df_input)
        if masalah:
            st.error("Format input tidak cocok dengan model, prediksi dibatalkan.")
            st.code("\n".join(masalah))
        else:
            p_gagal = float(model.predict_proba(df_input)[0, 1])
            st.markdown(kartu_hasil(p_gagal, threshold), unsafe_allow_html=True)
            st.caption(
                "Persentase ini hasil hitungan model, bukan jaminan. Model dilatih dengan "
                "pembobotan kelas, jadi persentase gagal bayar cenderung lebih tinggi dari "
                "peluang sebenarnya."
            )

            kosong = [c for c in fitur if pd.isna(df_input[c].iloc[0])]
            if len(kosong) > 6:
                st.info(f"{len(kosong)} kolom dikosongkan, hasil bisa kurang akurat.")

            di_luar = [
                c for c, r in rentang.items()
                if c in df_input and pd.notna(df_input[c].iloc[0])
                and not (r["min"] <= df_input[c].iloc[0] <= r["max"])
            ]
            if di_luar:
                st.warning("Beberapa nilai di luar rentang data latih: " + ", ".join(di_luar))

            with st.expander("Detail teknis (data yang dikirim ke model)"):
                st.dataframe(df_input.T.rename(columns={0: "nilai"}))
                st.write("Pengecekan kategori terhadap encoder: semua dikenal")

with st.expander("Tentang model dan keterbatasannya"):
    m = meta.get("evaluasi_test", {}).get("metrics", {})
    if m:
        st.write(
            f"Model CatBoost ini diuji pada 6.000 nasabah yang tidak dipakai untuk melatih. "
            f"Dari 100 nasabah yang benar-benar gagal bayar, model berhasil menandai sekitar "
            f"{m.get('recall', 0) * 100:.0f}. Dari 100 nasabah yang ditandai gagal bayar, sekitar "
            f"{m.get('precision', 0) * 100:.0f} memang gagal bayar dan sisanya salah tandai."
        )
    st.markdown(
        "- Hasil adalah perkiraan, bukan kepastian bahwa seseorang akan gagal bayar.\n"
        "- Data latih berasal dari nasabah kartu kredit di Taiwan, sehingga nilai uang dalam NT$.\n"
        "- Status pembayaran \"cicilan minimum\" (kode 0) dan \"tidak ada pemakaian\" (kode -2) "
        "adalah interpretasi umum dari dataset ini, bukan dari codebook resmi.\n"
        "- Jangan dipakai sebagai satu-satunya dasar keputusan kredit."
    )
