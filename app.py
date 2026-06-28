"""
=============================================================
  app.py — Streamlit App
  Regresi Berganda: Prediksi Gol Tim Sepak Bola
  Deploy: streamlit run app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Regresi Berganda — Prediksi Gol",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global background */
  .stApp { background-color: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] { background-color: #161b22; }

  /* Metric cards */
  [data-testid="stMetric"] {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 14px 18px;
  }
  [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 13px; }
  [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 26px; font-weight: 700; }
  [data-testid="stMetricDelta"] { font-size: 12px; }

  /* Headers */
  h1 { color: #e6edf3 !important; }
  h2 { color: #58a6ff !important; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
  h3 { color: #3fb950 !important; }

  /* Prediction result box */
  .pred-box {
      background: linear-gradient(135deg, #1f2937, #161b22);
      border: 2px solid #58a6ff;
      border-radius: 14px;
      padding: 24px;
      text-align: center;
      margin-top: 16px;
  }
  .pred-number { font-size: 56px; font-weight: 800; color: #58a6ff; line-height: 1.1; }
  .pred-label  { font-size: 14px; color: #8b949e; margin-top: 4px; }

  /* Equation box */
  .eq-box {
      background: #161b22;
      border-left: 4px solid #3fb950;
      border-radius: 0 8px 8px 0;
      padding: 14px 18px;
      font-family: monospace;
      font-size: 13px;
      color: #e6edf3;
      margin: 12px 0;
      word-break: break-all;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { color: #8b949e; border-radius: 6px; }
  .stTabs [data-baseweb="tab"][aria-selected="true"] {
      background: #21262d; color: #58a6ff; font-weight: 600;
  }

  /* Divider */
  hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PALETTE (untuk matplotlib)
# ══════════════════════════════════════════════════════════════
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#f0883e"
PURPLE  = "#bc8cff"
RED     = "#ff7b72"
MUTED   = "#8b949e"
WHITE   = "#e6edf3"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#30363d", "axes.labelcolor": WHITE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": WHITE, "grid.color": "#21262d", "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
})


# ══════════════════════════════════════════════════════════════
#  DATA & MODEL  (cache supaya tidak re-train tiap interaksi)
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_and_train():
    """Load dataset, generate 500 rows jika perlu, train model."""
    try:
        df = pd.read_csv("football_regression_data.csv")
    except FileNotFoundError:
        # ── Generate synthetic dataset berbasis distribusi realistis ──
        np.random.seed(42)
        n = 500
        possession       = np.random.normal(50, 12, n).clip(20, 80)
        shots            = np.random.normal(35, 15, n).clip(5, 85)
        shots_on_target  = (shots * np.random.uniform(0.25, 0.65, n)).clip(0, shots)
        assists          = np.random.exponential(2.5, n).clip(0, 12)
        fouls            = np.random.normal(30, 9, n).clip(5, 65)

        goals = (
            -0.02
            + 0.005  * possession
            - 0.003  * shots
            + 0.153  * shots_on_target
            + 0.613  * assists
            + 0.005  * fouls
            + np.random.normal(0, 1.4, n)
        ).clip(0, 14).round().astype(int)

        df = pd.DataFrame({
            "goals": goals, "possession": possession.round(2),
            "shots": shots.round(2), "shots_on_target": shots_on_target.round(2),
            "assists": assists.round(2), "fouls": fouls.round(2),
        })

    features = ["possession", "shots", "shots_on_target", "assists", "fouls"]
    target   = "goals"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "r2":   r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae":  mean_absolute_error(y_test, y_pred),
    }
    return df, model, features, target, X_train, X_test, y_train, y_test, y_pred, metrics


df, model, features, target, X_train, X_test, y_train, y_test, y_pred, metrics = load_and_train()


# ══════════════════════════════════════════════════════════════
#  SIDEBAR — Kalkulator Prediksi
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚽ Kalkulator Prediksi")
    st.markdown("Masukkan statistik tim untuk memprediksi jumlah gol.")
    st.divider()

    inp_possession      = st.slider("🔵 Ball Possession (%)",      10, 80, 55)
    inp_shots           = st.slider("🎯 Total Shots",               1, 90, 35)
    inp_shots_on_target = st.slider("✅ Shots on Target",           0, int(inp_shots), min(12, inp_shots))
    inp_assists         = st.slider("🤝 Assists",                   0, 15,  3)
    inp_fouls           = st.slider("🟨 Fouls Committed",           5, 70, 30)

    inp_array = np.array([[inp_possession, inp_shots,
                           inp_shots_on_target, inp_assists, inp_fouls]])
    pred_goals = float(model.predict(inp_array)[0])
    pred_goals = max(0, pred_goals)

    st.divider()
    st.markdown(f"""
    <div class="pred-box">
      <div class="pred-number">{pred_goals:.1f}</div>
      <div class="pred-label">⚽ Prediksi Gol</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Model: Multiple Linear Regression")
    st.caption(f"R² = {metrics['r2']:.4f} | RMSE = {metrics['rmse']:.4f}")


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("# ⚽ Regresi Berganda — Prediksi Gol Tim Sepak Bola")
st.markdown(
    "Implementasi **Multiple Linear Regression** menggunakan statistik pertandingan "
    "untuk memprediksi jumlah gol yang dicetak sebuah tim."
)

# ── Metrik utama ──────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📊 Total Data",   f"{len(df):,} records")
c2.metric("🎯 Variabel X",   f"{len(features)} fitur")
c3.metric("📈 R² Score",     f"{metrics['r2']:.4f}",   f"{metrics['r2']*100:.1f}% explained")
c4.metric("📉 RMSE",         f"{metrics['rmse']:.4f}", "gol")
c5.metric("📉 MAE",          f"{metrics['mae']:.4f}",  "gol")

st.divider()


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Dataset & Statistik",
    "🔥 Heatmap Korelasi",
    "📈 Scatter Plots",
    "🧪 Evaluasi Model",
])


# ── TAB 1: Dataset ────────────────────────────────────────────
with tab1:
    st.markdown("## 📋 Dataset")

    col_info, col_stat = st.columns([1, 2])
    with col_info:
        st.markdown("### Informasi Dataset")
        st.markdown(f"""
        | | |
        |---|---|
        | **Sumber** | Statistik Tim FIFA/AFCON |
        | **Total Records** | {len(df):,} |
        | **Kolom** | {len(df.columns)} |
        | **Train Set** | {len(X_train)} (80%) |
        | **Test Set** | {len(X_test)} (20%) |
        | **Target (Y)** | `goals` |
        """)

        st.markdown("### Variabel X (Fitur)")
        feat_desc = {
            "possession":       "Penguasaan bola (%)",
            "shots":            "Total tembakan",
            "shots_on_target":  "Tembakan tepat sasaran",
            "assists":          "Jumlah assist",
            "fouls":            "Pelanggaran",
        }
        for f, d in feat_desc.items():
            st.markdown(f"- **`{f}`** — {d}")

    with col_stat:
        st.markdown("### Statistik Deskriptif")
        st.dataframe(
            df[features + [target]].describe().round(3),
            use_container_width=True,
        )

    st.markdown("### Persamaan Regresi")
    eq = f"goals = {model.intercept_:.4f}"
    for feat, coef in zip(features, model.coef_):
        sign = "+" if coef >= 0 else "-"
        eq  += f" {sign} {abs(coef):.4f}×{feat}"
    st.markdown(f'<div class="eq-box">{eq}</div>', unsafe_allow_html=True)

    st.markdown("### Preview Data (50 baris pertama)")
    st.dataframe(df[features + [target]].head(50), use_container_width=True)


# ── TAB 2: Heatmap ────────────────────────────────────────────
with tab2:
    st.markdown("## 🔥 Heatmap Korelasi")
    st.markdown(
        "Heatmap menunjukkan seberapa kuat hubungan linear antara setiap pasang variabel. "
        "Nilai mendekati **+1** = korelasi positif kuat, **-1** = negatif kuat."
    )

    fig_heat, axes_heat = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK_BG)

    # Heatmap penuh
    corr = df[features + [target]].corr()
    cmap = sns.diverging_palette(220, 15, sep=10, as_cmap=True)
    sns.heatmap(
        corr, ax=axes_heat[0], annot=True, fmt=".2f", cmap=cmap,
        vmin=-1, vmax=1, linewidths=0.6, linecolor="#21262d",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10, "color": WHITE}, square=True,
    )
    axes_heat[0].set_title("Korelasi Semua Variabel",
                            fontsize=13, fontweight="bold", color=WHITE, pad=10)
    axes_heat[0].tick_params(colors=MUTED, labelsize=9)
    axes_heat[0].set_xticklabels(
        axes_heat[0].get_xticklabels(), rotation=30, ha="right")

    # Bar korelasi dengan target
    corr_target = corr[target].drop(target).sort_values()
    colors_bar  = [GREEN if v > 0 else RED for v in corr_target]
    axes_heat[1].barh(corr_target.index, corr_target.values,
                      color=colors_bar, edgecolor="#21262d", height=0.6)
    axes_heat[1].axvline(0, color=MUTED, linewidth=1.2, linestyle="--", alpha=0.7)
    for i, val in enumerate(corr_target.values):
        axes_heat[1].text(val + (0.01 if val >= 0 else -0.01), i,
                          f"{val:+.3f}", va="center",
                          ha="left" if val >= 0 else "right",
                          fontsize=10, color=WHITE, fontweight="bold")
    axes_heat[1].set_title(f"Korelasi Fitur dengan '{target}'",
                            fontsize=13, fontweight="bold", color=WHITE, pad=10)
    axes_heat[1].set_xlabel("Koefisien Korelasi (r)", fontsize=10)
    axes_heat[1].grid(True, axis="x", alpha=0.3)
    axes_heat[1].spines[["top", "right"]].set_visible(False)
    axes_heat[1].legend(
        handles=[
            plt.Rectangle((0,0),1,1, fc=GREEN, label="Korelasi Positif"),
            plt.Rectangle((0,0),1,1, fc=RED,   label="Korelasi Negatif"),
        ],
        facecolor=CARD_BG, edgecolor="#30363d", labelcolor=WHITE, fontsize=9,
    )

    plt.tight_layout()
    st.pyplot(fig_heat, use_container_width=True)
    plt.close(fig_heat)

    # Interpretasi otomatis
    st.markdown("### 💡 Interpretasi")
    corr_target_sorted = corr[target].drop(target).abs().sort_values(ascending=False)
    top_feat = corr_target_sorted.index[0]
    top_val  = corr[target][top_feat]
    st.info(
        f"**`{top_feat}`** memiliki korelasi tertinggi dengan `goals` "
        f"(r = {top_val:+.3f}), artinya fitur ini paling berpengaruh "
        f"dalam memprediksi jumlah gol."
    )


# ── TAB 3: Scatter Plots ──────────────────────────────────────
with tab3:
    st.markdown("## 📈 Scatter Plot: Setiap X terhadap Y")
    st.markdown(
        "Setiap panel menampilkan hubungan satu fitur dengan `goals`, "
        "dilengkapi **garis regresi linear sederhana**."
    )

    plot_colors = [ACCENT, GREEN, ORANGE, PURPLE, RED]
    fig_sc, axes_sc = plt.subplots(2, 3, figsize=(17, 10), facecolor=DARK_BG)
    axes_flat = axes_sc.flatten()

    for i, (feat, col) in enumerate(zip(features, plot_colors)):
        ax = axes_flat[i]
        ax.scatter(df[feat], df[target], alpha=0.4, color=col,
                   edgecolors="none", s=18)
        m_s = LinearRegression()
        m_s.fit(df[[feat]], df[target])
        x_line = np.linspace(df[feat].min(), df[feat].max(), 150).reshape(-1, 1)
        ax.plot(x_line, m_s.predict(x_line),
                color=WHITE, linewidth=2, linestyle="--", alpha=0.85)
        r_val = df[[feat, target]].corr().iloc[0, 1]
        ax.set_title(f"{feat}  (r = {r_val:.2f})",
                     fontsize=10, fontweight="bold", color=col, pad=7)
        ax.set_xlabel(feat, fontsize=9, color=MUTED)
        ax.set_ylabel("Goals", fontsize=9, color=MUTED)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.97, 0.05, f"r = {r_val:.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, color=WHITE,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG,
                          edgecolor=col, alpha=0.8))

    # Panel 6: Actual vs Predicted
    ax_pva = axes_flat[5]
    ax_pva.scatter(y_test, y_pred, alpha=0.55, color=PURPLE,
                   edgecolors="none", s=22)
    lim = (min(y_test.min(), y_pred.min()) - 0.5,
           max(y_test.max(), y_pred.max()) + 0.5)
    ax_pva.plot(lim, lim, color=WHITE, linewidth=2,
                linestyle="--", alpha=0.85, label="Ideal (y=x)")
    ax_pva.set_xlim(lim); ax_pva.set_ylim(lim)
    ax_pva.set_xlabel("Actual Goals",    fontsize=9, color=MUTED)
    ax_pva.set_ylabel("Predicted Goals", fontsize=9, color=MUTED)
    ax_pva.set_title(f"Actual vs Predicted  (R²={metrics['r2']:.3f})",
                     fontsize=10, fontweight="bold", color=PURPLE, pad=7)
    ax_pva.grid(True, alpha=0.3)
    ax_pva.spines[["top", "right"]].set_visible(False)
    ax_pva.legend(facecolor=CARD_BG, edgecolor="#30363d",
                  labelcolor=WHITE, fontsize=9)

    plt.tight_layout()
    st.pyplot(fig_sc, use_container_width=True)
    plt.close(fig_sc)


# ── TAB 4: Evaluasi ───────────────────────────────────────────
with tab4:
    st.markdown("## 🧪 Evaluasi Model")

    col_l, col_r = st.columns(2)

    # Koefisien tabel
    with col_l:
        st.markdown("### Koefisien Regresi")
        coef_df = pd.DataFrame({
            "Fitur":      features,
            "Koefisien":  model.coef_,
            "|Koefisien|": np.abs(model.coef_),
            "Arah":       ["▲ Positif" if c > 0 else "▼ Negatif" for c in model.coef_],
        }).sort_values("|Koefisien|", ascending=False)

        st.dataframe(
            coef_df[["Fitur", "Koefisien", "Arah"]].style.format({"Koefisien": "{:+.4f}"}),
            use_container_width=True, hide_index=True,
        )

        st.markdown("### Metrik Evaluasi")
        eval_df = pd.DataFrame({
            "Metrik":       ["R² Score", "RMSE", "MAE", "Intercept"],
            "Nilai":        [f"{metrics['r2']:.4f}", f"{metrics['rmse']:.4f}",
                             f"{metrics['mae']:.4f}", f"{model.intercept_:.4f}"],
            "Keterangan":   [
                f"{metrics['r2']*100:.1f}% variance explained",
                "Root Mean Squared Error",
                "Mean Absolute Error",
                "Nilai baseline (semua X=0)",
            ],
        })
        st.dataframe(eval_df, use_container_width=True, hide_index=True)

    # Residual + QQ plot
    with col_r:
        st.markdown("### Diagnostik Residual")
        residuals = y_test.values - y_pred

        fig_diag, axes_diag = plt.subplots(1, 2, figsize=(10, 4), facecolor=DARK_BG)

        # Residual vs Fitted
        axes_diag[0].scatter(y_pred, residuals, alpha=0.5, color=ACCENT, s=18)
        axes_diag[0].axhline(0, color=RED, linewidth=1.5, linestyle="--")
        axes_diag[0].set_xlabel("Fitted Values", fontsize=9, color=MUTED)
        axes_diag[0].set_ylabel("Residuals",     fontsize=9, color=MUTED)
        axes_diag[0].set_title("Residual vs Fitted",
                                fontsize=10, fontweight="bold", color=ACCENT, pad=7)
        axes_diag[0].grid(True, alpha=0.3)
        axes_diag[0].spines[["top", "right"]].set_visible(False)

        # Histogram residual
        axes_diag[1].hist(residuals, bins=22, color=GREEN,
                          edgecolor="#21262d", alpha=0.85, density=True)
        xr = np.linspace(residuals.min(), residuals.max(), 200)
        axes_diag[1].plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()),
                          color=WHITE, linewidth=2, label="Normal Fit")
        axes_diag[1].set_xlabel("Residual", fontsize=9, color=MUTED)
        axes_diag[1].set_ylabel("Density",  fontsize=9, color=MUTED)
        axes_diag[1].set_title("Distribusi Residual",
                                fontsize=10, fontweight="bold", color=GREEN, pad=7)
        axes_diag[1].legend(facecolor=CARD_BG, edgecolor="#30363d",
                             labelcolor=WHITE, fontsize=8)
        axes_diag[1].grid(True, alpha=0.3)
        axes_diag[1].spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_diag, use_container_width=True)
        plt.close(fig_diag)

    # Kesimpulan
    st.divider()
    st.markdown("### 📝 Kesimpulan")
    top_coef = sorted(zip(features, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    st.success(
        f"Model **Multiple Linear Regression** berhasil menjelaskan **{metrics['r2']*100:.1f}%** "
        f"variasi jumlah gol (R² = {metrics['r2']:.4f}). "
        f"Fitur paling berpengaruh adalah **`{top_coef[0][0]}`** "
        f"(koefisien = {top_coef[0][1]:+.4f}) dan **`{top_coef[1][0]}`** "
        f"(koefisien = {top_coef[1][1]:+.4f}). "
        f"RMSE sebesar {metrics['rmse']:.4f} berarti prediksi model rata-rata meleset sekitar "
        f"{metrics['rmse']:.2f} gol dari nilai aktual."
    )


# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#8b949e; font-size:12px;'>"
    "Multiple Linear Regression · scikit-learn · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
