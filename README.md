# ⚽ Regresi Berganda — Prediksi Gol Tim Sepak Bola

Implementasi **Multiple Linear Regression** menggunakan statistik pertandingan untuk memprediksi jumlah gol sebuah tim. Dibangun dengan Python + Streamlit, siap deploy ke Streamlit Community Cloud.

🔗 **Live Demo**: *(isi setelah deploy)*

---

## 📸 Screenshot

> Tambahkan screenshot app di sini setelah deploy.

---

## 📌 Variabel

| Peran | Variabel | Deskripsi |
|---|---|---|
| **Y (Target)** | `goals` | Jumlah gol yang dicetak |
| **X₁** | `possession` | Penguasaan bola (%) |
| **X₂** | `shots` | Total tembakan |
| **X₃** | `shots_on_target` | Tembakan tepat sasaran |
| **X₄** | `assists` | Jumlah assist |
| **X₅** | `fouls` | Pelanggaran dilakukan |

---

## 🚀 Cara Run Lokal

### 1. Clone repo
```bash
git clone https://github.com/USERNAME/REPO-NAME.git
cd REPO-NAME
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan app
```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## ☁️ Deploy ke Streamlit Community Cloud

1. Push semua file ke GitHub (pastikan `app.py`, `requirements.txt`, dan `football_regression_data.csv` ada)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **"New app"** → pilih repo ini
4. Main file path: `app.py`
5. Klik **Deploy** — selesai!

---

## 📁 Struktur File

```
├── app.py                        # Streamlit app utama
├── football_regression_data.csv  # Dataset 500 records
├── requirements.txt              # Dependencies
└── README.md                     # Dokumentasi ini
```

---

## 📊 Fitur Aplikasi

- **Sidebar Kalkulator** — input statistik tim secara interaktif, prediksi gol tampil real-time
- **Tab Dataset** — preview data, statistik deskriptif, persamaan regresi
- **Tab Heatmap** — korelasi antar variabel + bar chart korelasi terhadap target
- **Tab Scatter Plot** — hubungan tiap fitur X dengan Y + panel Actual vs Predicted
- **Tab Evaluasi** — koefisien, metrik (R², RMSE, MAE), residual plot, distribusi residual

---

## 🧪 Hasil Model

| Metrik | Nilai |
|---|---|
| R² Score | ~0.69 |
| RMSE | ~1.53 gol |
| MAE | ~1.25 gol |

---

## 🛠️ Tech Stack

- Python 3.x
- [Streamlit](https://streamlit.io)
- scikit-learn
- pandas / numpy
- matplotlib / seaborn / scipy
