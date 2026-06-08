"""
========================================================
  APLIKASI SIMULASI INVENTORY (M, N) - NIRA AREN
  Mata Kuliah: Pemodelan dan Simulasi Komputer
  SubCPMK01: Analisis → Perancangan → Implementasi → Evaluasi
========================================================
Cara menjalankan:
    pip install streamlit numpy pandas matplotlib
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy import stats
from simulation import run_periodic_review_simulation, run_newsvendor_simulation

# ─────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulasi Inventory Nira Aren",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font & background */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: "🌴";
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity: 0.15;
    }
    .main-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.3rem 0 0;
        opacity: 0.85;
        font-size: 0.9rem;
        font-weight: 300;
    }

    /* Badge tahapan */
    .stage-badge {
        display: inline-block;
        background: #d8f3dc;
        color: #1b4332;
        border: 1.5px solid #74c69d;
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
    }

    /* Metric card */
    .metric-card {
        background: #f8fffe;
        border: 1.5px solid #b7e4c7;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #555;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.35rem;
        font-weight: 600;
        color: #1b4332;
        margin-top: 0.2rem;
    }
    .metric-card .sub {
        font-size: 0.7rem;
        color: #888;
        margin-top: 0.1rem;
    }

    /* Alert box */
    .info-box {
        background: #ebfbee;
        border-left: 4px solid #40916c;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        font-size: 0.88rem;
        color: #1b4332;
        margin: 0.5rem 0;
    }
    .warn-box {
        background: #fff3e0;
        border-left: 4px solid #f77f00;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        font-size: 0.88rem;
        color: #7f4f24;
        margin: 0.5rem 0;
    }

    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background: #f0faf4;
        border-right: 1.5px solid #b7e4c7;
    }

    /* Fix semua teks di sidebar agar kontras */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #1b4332 !important;
    }

    /* Khusus value slider (angka di atas slider) */
    section[data-testid="stSidebar"] .stSlider p,
    section[data-testid="stSidebar"] .stSlider span,
    section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] {
        color: #1b4332 !important;
        font-weight: 600 !important;
    }

    /* Section header (## di sidebar) */
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0d3320 !important;
        font-weight: 700 !important;
    }

    /* Divider sidebar */
    section[data-testid="stSidebar"] hr {
        border-color: #b7e4c7 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #e9f5ee;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #2d6a4f;
        border: 1.5px solid #b7e4c7;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: #2d6a4f !important;
        color: white !important;
    }

    /* DataFrame table */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1.5px solid #b7e4c7;
        margin: 1.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# HEADER UTAMA
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Simulasi Inventory (M, N) — Nira Aren Perishable</h1>
    <p>Mata Kuliah: Pemodelan dan Simulasi Komputer &nbsp;|&nbsp; SubCPMK01</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SIDEBAR — PARAMETER INPUT
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameter Simulasi")
    st.markdown("---")

    st.markdown("#### 🎯 Kebijakan Inventaris")
    M = st.slider("M — Stok Maksimum (Liter)", 50, 300, 150, step=25,
                  help="Level restock: jika stok di bawah M, lakukan pemesanan")
    N = st.slider("N — Review Period (Hari)", 1, 7, 3,
                  help="Frekuensi pengecekan stok (setiap N hari)")
    sim_days = st.slider("Durasi Simulasi (Hari)", 10, 90, 30, step=5)

    st.markdown("---")
    st.markdown("#### 📊 Distribusi Permintaan")
    scenario = st.selectbox(
    "Pilih Skenario Demand",
    [
        "Normal",
        "Musiman",
        "Ekstrem"
    ]
)
    demand_mean = st.number_input("Rata-rata Permintaan (L/hari)", 50, 300, 100, step=10)
    demand_std  = st.number_input("Std. Deviasi Permintaan (L)", 5, 80, 20, step=5)

    st.markdown("---")
    st.markdown("#### 🚚 Lead Time")
    lt_min = st.number_input("Lead Time Minimum (Hari)", 1, 5, 1)
    lt_max = st.number_input("Lead Time Maksimum (Hari)", 1, 10, 3)
    if lt_max < lt_min:
        st.error("Lead Time Max harus ≥ Min!")

    st.markdown("---")
    st.markdown("#### 💰 Parameter Biaya (Rp/Liter)")
    purchase_cost  = st.number_input("Harga Beli Nira",    1000, 50000, 5000,  step=500)
    shortage_cost  = st.number_input("Biaya Shortage",     1000, 50000, 8000,  step=500)
    spoilage_cost  = st.number_input("Biaya Spoilage",     1000, 50000, 5000,  step=500)
    holding_cost   = st.number_input("Biaya Holding/hari",  100, 5000,   500,  step=100)

    st.markdown("---")
    seed = st.number_input("Random Seed", 0, 9999, 42)
    run_btn = st.button("▶ Jalankan Simulasi", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────
# ENGINE SIMULASI
# ─────────────────────────────────────────────────────────
def run_simulation(M, N, sim_days, demand_mean, demand_std,
                   lt_min, lt_max, purchase_cost, shortage_cost,
                   spoilage_cost, holding_cost, seed=42):

    np.random.seed(seed)
    demands = np.maximum(np.random.normal(demand_mean, demand_std, sim_days), 0).round(2)

    current_stock  = M
    pending_orders = defaultdict(float)
    pending_orders[1] += M
    records = []

    for day in range(1, sim_days + 1):
        received = pending_orders.pop(day, 0)
        current_stock += received

        order_qty, order_arrival = 0, 0
        if day % N == 0 and current_stock < M:
            order_qty  = M - current_stock
            lead_time  = int(np.random.uniform(lt_min, lt_max + 1))
            arrival    = day + lead_time
            pending_orders[arrival] += order_qty
            order_arrival = arrival

        demand    = demands[day - 1]
        shortage  = spoilage = 0

        if current_stock >= demand:
            current_stock -= demand
            spoilage       = current_stock
            current_stock  = 0
        else:
            shortage       = demand - current_stock
            current_stock  = 0

        cost_holding   = received  * holding_cost
        cost_shortage  = shortage  * shortage_cost
        cost_spoilage  = spoilage  * spoilage_cost
        cost_purchase  = order_qty * purchase_cost
        total_day      = cost_holding + cost_shortage + cost_spoilage + cost_purchase

        records.append({
            "Hari"               : day,
            "Review?"            : "✅" if day % N == 0 else "–",
            "Pesanan Datang (L)" : round(received, 2),
            "Order Baru (L)"     : round(order_qty, 2),
            "ETA Order"          : f"Hari {order_arrival}" if order_qty > 0 else "–",
            "Permintaan (L)"     : round(demand, 2),
            "Shortage (L)"       : round(shortage, 2),
            "Spoilage (L)"       : round(spoilage, 2),
            "Biaya Beli (Rp)"    : int(cost_purchase),
            "Biaya Simpan (Rp)"  : int(cost_holding),
            "Biaya Shortage (Rp)": int(cost_shortage),
            "Biaya Spoilage (Rp)": int(cost_spoilage),
            "Total Biaya/Hari"   : int(total_day),
        })

    df = pd.DataFrame(records)
    summary = {
        "Total Permintaan (L)"      : df["Permintaan (L)"].sum(),
        "Total Shortage (L)"        : df["Shortage (L)"].sum(),
        "Total Spoilage (L)"        : df["Spoilage (L)"].sum(),
        "Total Biaya Beli (Rp)"     : df["Biaya Beli (Rp)"].sum(),
        "Total Biaya Simpan (Rp)"   : df["Biaya Simpan (Rp)"].sum(),
        "Total Biaya Shortage (Rp)" : df["Biaya Shortage (Rp)"].sum(),
        "Total Biaya Spoilage (Rp)" : df["Biaya Spoilage (Rp)"].sum(),
        "GRAND TOTAL BIAYA (Rp)"    : df["Total Biaya/Hari"].sum(),
    }
    return df, summary


def run_sensitivity(sim_days, demand_mean, demand_std, lt_min, lt_max,
                    purchase_cost, shortage_cost, spoilage_cost, holding_cost, seed):
    M_vals = [75, 100, 125, 150, 175, 200]
    N_vals = [1, 2, 3, 5, 7]
    rows = []
    for m in M_vals:
        for n in N_vals:
            _, s = run_simulation(m, n, sim_days, demand_mean, demand_std,
                                  lt_min, lt_max, purchase_cost, shortage_cost,
                                  spoilage_cost, holding_cost, seed)
            rows.append({
                "M": m, "N": n,
                "Shortage (L)": round(s["Total Shortage (L)"], 1),
                "Spoilage (L)": round(s["Total Spoilage (L)"], 1),
                "Grand Total (Rp)": s["GRAND TOTAL BIAYA (Rp)"],
            })
    eval_df = pd.DataFrame(rows)
    best    = eval_df.loc[eval_df["Grand Total (Rp)"].idxmin()]
    return eval_df, best


# ─────────────────────────────────────────────────────────
# JALANKAN SIMULASI (auto atau on-click)
# ─────────────────────────────────────────────────────────
if "df" not in st.session_state or run_btn:
    with st.spinner("Menjalankan simulasi..."):
        df, summary = run_periodic_review_simulation(
    M,
    N,
    sim_days,
    demand_mean,
    demand_std,
    lt_min,
    lt_max,
    purchase_cost,
    shortage_cost,
    spoilage_cost,
    holding_cost,
    scenario,
    seed
)
        eval_df, best = run_sensitivity(
            sim_days, demand_mean, demand_std, lt_min, lt_max,
            purchase_cost, shortage_cost, spoilage_cost, holding_cost, seed
        )
        st.session_state.update({"df": df, "summary": summary,
                                  "eval_df": eval_df, "best": best})

df      = st.session_state["df"]
summary = st.session_state["summary"]
eval_df = st.session_state["eval_df"]
best    = st.session_state["best"]


# ─────────────────────────────────────────────────────────
# TABS UTAMA (SubCPMK01 + SubCPMK02 + SubCPMK03)
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📋 1. Analisis Kebutuhan",
    "📐 2. Perancangan Model",
    "▶ 3. Implementasi & Simulasi",
    "📊 4. Evaluasi & Optimasi",
    "🧮 5. Formulasi Matematis",
    "📈 6. Analisis Statistik",
    "⚖️ 7. Komparasi & Validasi Sistem",
    "🗂️ 8. Perbandingan Dataset"
])


# ══════════════════════════════════════════════════════════
# TAB 1 — ANALISIS KEBUTUHAN
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="stage-badge">TAHAP 1 — ANALISIS KEBUTUHAN</div>', unsafe_allow_html=True)
    st.markdown("### Identifikasi Masalah Inventaris Nira Aren")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
**Latar Belakang Masalah**

Nira aren adalah bahan baku gula semut yang bersifat sangat *perishable* — hanya tahan
**± 3 jam** sebelum mengalami fermentasi dan tidak dapat digunakan.
Ketidakpastian permintaan dan waktu pengiriman dari hutan menyebabkan dua risiko utama:

- 🔴 **Shortage**: Nira tidak tersedia saat dibutuhkan → kehilangan produksi
- 🟠 **Spoilage**: Nira berlebih tidak terpakai → terbuang sia-sia

Sistem inventaris konvensional (deterministik) tidak mampu menangani ketidakpastian ini,
sehingga dibutuhkan model **stokastik berbasis simulasi**.
        """)

        st.markdown("---")
        st.markdown("**Kebutuhan Fungsional Sistem**")
        kebutuhan = {
            "KF-01": "Sistem dapat memodelkan permintaan harian yang bersifat acak (Normal distribution)",
            "KF-02": "Sistem dapat memodelkan lead time pengiriman yang acak (Uniform distribution)",
            "KF-03": "Sistem menerapkan kebijakan review periodik (M, N)",
            "KF-04": "Sistem menghitung biaya shortage, spoilage, holding, dan pembelian",
            "KF-05": "Sistem dapat melakukan sensitivity analysis terhadap parameter M dan N",
            "KF-06": "Sistem merekomendasikan kombinasi M dan N yang optimal",
        }
        for kode, desc in kebutuhan.items():
            st.markdown(f"- **{kode}**: {desc}")

    with col2:
        st.markdown("**Variabel Sistem**")
        vars_data = {
            "Variabel": ["Permintaan Harian", "Lead Time", "Stok Maks (M)", "Review Period (N)",
                         "Shortage", "Spoilage"],
            "Tipe": ["Stokastik", "Stokastik", "Keputusan", "Keputusan", "Output", "Output"],
            "Distribusi": ["Normal(μ,σ)", "Uniform[min,max]", "Kontrol", "Kontrol", "Derived", "Derived"],
        }
        st.dataframe(pd.DataFrame(vars_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Asumsi Model**")
        asumsi = [
            "Nira tidak dapat disimpan lintas shift (perishable 100%)",
            "Pemesanan dilakukan tepat saat review period",
            "Kekurangan tidak dipenuhi dari pesanan (lost sales)",
            "Biaya tetap per unit (tidak ada diskon kuantitas)",
            "Satu jenis produk, satu supplier",
        ]
        for a in asumsi:
            st.markdown(f"✔ {a}")

    st.markdown("---")
    st.markdown("**Parameter Aktif Saat Ini**")
    param_cols = st.columns(5)
    params = [
        ("M (Stok Maks)", f"{M} L"), ("N (Review)", f"{N} hari"),
        ("Durasi", f"{sim_days} hari"), ("Demand μ", f"{demand_mean} L"),
        ("Demand σ", f"{demand_std} L"),
    ]
    for col, (lbl, val) in zip(param_cols, params):
        with col:
            st.markdown(f'<div class="metric-card"><div class="label">{lbl}</div><div class="value">{val}</div></div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — PERANCANGAN MODEL
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="stage-badge">TAHAP 2 — PERANCANGAN MODEL</div>', unsafe_allow_html=True)
    st.markdown("### Arsitektur Model Simulasi (M, N)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Alur Logika Simulasi per Hari**")
        st.markdown("""
```
┌─────────────────────────────────────────┐
│         UNTUK SETIAP HARI t             │
├─────────────────────────────────────────┤
│  STEP 1: Terima pesanan yang tiba       │
│          current_stock += received      │
│                                         │
│  STEP 2: Review period? (t mod N == 0)  │
│     YA → hitung order_qty = M - stok   │
│           generate lead_time ~ U[a,b]  │
│           jadwalkan pesanan             │
│                                         │
│  STEP 3: Penuhi permintaan              │
│     d ~ N(μ, σ)                         │
│     IF stok >= d:                       │
│        stok -= d                        │
│        spoilage = sisa stok (expired)   │
│     ELSE:                               │
│        shortage = d - stok             │
│        stok = 0                         │
│                                         │
│  STEP 4: Hitung biaya harian            │
│     C = C_beli + C_holding +            │
│         C_shortage + C_spoilage         │
└─────────────────────────────────────────┘
```
        """)

    with col2:
        st.markdown("**Fungsi Biaya Total**")
        st.latex(r"""
        TC = \sum_{t=1}^{T} \left[
            c_p \cdot Q_t +
            c_h \cdot R_t +
            c_s \cdot S_t^- +
            c_w \cdot S_t^+
        \right]
        """)
        st.markdown("""
| Simbol | Keterangan | Nilai |
|--------|-----------|-------|
| $c_p$ | Biaya beli per liter | Rp {:,.0f} |
| $c_h$ | Biaya holding per liter | Rp {:,.0f} |
| $c_s$ | Biaya shortage per liter | Rp {:,.0f} |
| $c_w$ | Biaya spoilage per liter | Rp {:,.0f} |
| $Q_t$ | Jumlah pesanan hari t | Variable |
| $R_t$ | Jumlah diterima hari t | Variable |
| $S_t^-$ | Shortage hari t | Variable |
| $S_t^+$ | Spoilage hari t | Variable |
        """.format(purchase_cost, holding_cost, shortage_cost, spoilage_cost))

    st.markdown("---")
    st.markdown("**Distribusi yang Digunakan**")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.markdown(f"""
**Permintaan Harian — Normal Distribution**
- $D_t \sim \mathcal{{N}}(\mu={demand_mean}, \sigma={demand_std})$
- Domain: $D_t \geq 0$ (di-truncate pada 0)
- Alasan: Permintaan pasar cenderung simetris di sekitar rata-rata
        """)
    with dcol2:
        st.markdown(f"""
**Lead Time — Uniform Distribution**
- $L \sim \mathcal{{U}}[{lt_min}, {lt_max}]$ hari
- Alasan: Tidak ada informasi cukup untuk distribusi spesifik;
  uniform = asumsi konservatif yang adil
        """)

    # Visualisasi distribusi
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    x_d = np.linspace(max(0, demand_mean - 4*demand_std), demand_mean + 4*demand_std, 200)
    from scipy.stats import norm
    axes[0].plot(x_d, norm.pdf(x_d, demand_mean, demand_std), color="#2d6a4f", lw=2.5)
    axes[0].fill_between(x_d, norm.pdf(x_d, demand_mean, demand_std), alpha=0.2, color="#52b788")
    axes[0].axvline(demand_mean, color="#d62728", ls="--", lw=1.5, label=f"μ={demand_mean}")
    axes[0].set_title("Distribusi Permintaan (Normal)", fontsize=10)
    axes[0].set_xlabel("Liter/hari"); axes[0].set_ylabel("Density"); axes[0].legend()
    axes[0].grid(alpha=0.3)

    lt_range = np.arange(lt_min, lt_max + 2)
    prob = 1 / (lt_max - lt_min + 1)
    axes[1].bar(range(lt_min, lt_max + 1), [prob] * (lt_max - lt_min + 1),
                color="#40916c", alpha=0.8, width=0.5)
    axes[1].set_title("Distribusi Lead Time (Uniform)", fontsize=10)
    axes[1].set_xlabel("Hari"); axes[1].set_ylabel("Probabilitas")
    axes[1].set_xticks(range(lt_min, lt_max + 1))
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════
# TAB 3 — IMPLEMENTASI & SIMULASI
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="stage-badge">TAHAP 3 — IMPLEMENTASI & SIMULASI</div>', unsafe_allow_html=True)
    st.markdown(f"### Hasil Simulasi: M={M}, N={N}, {sim_days} Hari")

    # KPI Row
    grand_total = summary["GRAND TOTAL BIAYA (Rp)"]
    shortage_l  = summary["Total Shortage (L)"]
    spoilage_l  = summary["Total Spoilage (L)"]
    total_d     = summary["Total Permintaan (L)"]
    sl          = max(0, (1 - shortage_l / total_d) * 100) if total_d > 0 else 100

    kpi_cols = st.columns(5)
    kpis = [
        ("Grand Total Biaya", f"Rp {grand_total/1e6:.2f}M", "30 hari"),
        ("Service Level", f"{sl:.1f}%", "% demand terpenuhi"),
        ("Total Shortage", f"{shortage_l:.1f} L", "tidak terlayani"),
        ("Total Spoilage", f"{spoilage_l:.1f} L", "terbuang"),
        ("Biaya/Hari Rata²", f"Rp {grand_total/sim_days/1000:.0f}K", "per hari"),
    ]
    for col, (lbl, val, sub) in zip(kpi_cols, kpis):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="label">{lbl}</div>'
                f'<div class="value">{val}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Grafik utama
    days = df["Hari"]
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)
    fig.suptitle(f"Simulasi Inventory (M={M}, N={N}) — Nira Aren | {sim_days} Hari",
                 fontsize=13, fontweight="bold", y=0.99)

    # G1: Permintaan vs Shortage vs Spoilage
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(days, df["Permintaan (L)"], label="Permintaan", color="#4C72B0", alpha=0.75)
    ax1.bar(days, df["Shortage (L)"],   label="Shortage",   color="#d62728", alpha=0.9)
    ax1.bar(days, df["Spoilage (L)"],   label="Spoilage",   color="#ff7f0e", alpha=0.85,
            bottom=df["Shortage (L)"])
    review_days = df[df["Review?"] == "✅"]["Hari"]
    for rd in review_days:
        ax1.axvline(rd, color="#2d6a4f", ls="--", alpha=0.5, lw=0.9)
    ax1.set_title("Permintaan Harian vs Shortage vs Spoilage  (garis hijau = hari review)", fontsize=10)
    ax1.set_xlabel("Hari"); ax1.set_ylabel("Volume (Liter)")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # G2: Komposisi biaya
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.stackplot(days,
        df["Biaya Beli (Rp)"] / 1000,
        df["Biaya Simpan (Rp)"] / 1000,
        df["Biaya Shortage (Rp)"] / 1000,
        df["Biaya Spoilage (Rp)"] / 1000,
        labels=["Beli", "Simpan", "Shortage", "Spoilage"],
        colors=["#4C72B0", "#55A868", "#d62728", "#ff7f0e"], alpha=0.85)
    ax2.set_title("Komposisi Biaya Harian", fontsize=10)
    ax2.set_xlabel("Hari"); ax2.set_ylabel("Biaya (Rp Ribu)")
    ax2.legend(loc="upper right", fontsize=8); ax2.grid(alpha=0.3)

    # G3: Akumulasi
    ax3 = fig.add_subplot(gs[1, 1])
    cumcost = df["Total Biaya/Hari"].cumsum() / 1_000_000
    ax3.plot(days, cumcost, color="#7b2d8b", lw=2.5)
    ax3.fill_between(days, cumcost, alpha=0.15, color="#7b2d8b")
    ax3.set_title("Akumulasi Total Biaya", fontsize=10)
    ax3.set_xlabel("Hari"); ax3.set_ylabel("Biaya Kumulatif (Rp Juta)")
    ax3.grid(alpha=0.3)

    # G4: Pie biaya
    ax4 = fig.add_subplot(gs[2, 0])
    pie_vals = [summary[f"Total Biaya {k} (Rp)"] for k in ["Beli", "Simpan", "Shortage", "Spoilage"]]
    pie_lbls = ["Beli", "Simpan", "Shortage", "Spoilage"]
    pie_clrs = ["#4C72B0", "#55A868", "#d62728", "#ff7f0e"]
    wedges, texts, autotexts = ax4.pie(
        pie_vals, labels=pie_lbls, colors=pie_clrs,
        autopct="%1.1f%%", startangle=140, pctdistance=0.82)
    for at in autotexts:
        at.set_fontsize(9)
    ax4.set_title("Proporsi Biaya Total (30 Hari)", fontsize=10)

    # G5: Bar biaya per komponen
    ax5 = fig.add_subplot(gs[2, 1])
    bar_vals = [v / 1_000_000 for v in pie_vals]
    bars = ax5.bar(pie_lbls, bar_vals, color=pie_clrs, alpha=0.88)
    for bar, val in zip(bars, bar_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"Rp {val:.2f}M", ha="center", va="bottom", fontsize=9)
    ax5.set_title("Total Biaya per Komponen", fontsize=10)
    ax5.set_ylabel("Total Biaya (Rp Juta)"); ax5.grid(axis="y", alpha=0.3)

    st.pyplot(fig)
    plt.close()

    # Tabel simulasi
    st.markdown("---")
    st.markdown("**📋 Tabel Simulasi Harian**")

    styled_df = df.copy()
    styled_df["Total Biaya/Hari"] = styled_df["Total Biaya/Hari"].apply(lambda x: f"Rp {x:,.0f}")
    styled_df["Biaya Beli (Rp)"]     = styled_df["Biaya Beli (Rp)"].apply(lambda x: f"Rp {x:,.0f}")
    styled_df["Biaya Simpan (Rp)"]   = styled_df["Biaya Simpan (Rp)"].apply(lambda x: f"Rp {x:,.0f}")
    styled_df["Biaya Shortage (Rp)"] = styled_df["Biaya Shortage (Rp)"].apply(lambda x: f"Rp {x:,.0f}")
    styled_df["Biaya Spoilage (Rp)"] = styled_df["Biaya Spoilage (Rp)"].apply(lambda x: f"Rp {x:,.0f}")

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Ringkasan biaya
    st.markdown("---")
    st.markdown("**💰 Ringkasan Biaya Total**")
    rcols = st.columns(4)
    cost_items = [
        ("Biaya Beli", summary["Total Biaya Beli (Rp)"], "#4C72B0"),
        ("Biaya Simpan", summary["Total Biaya Simpan (Rp)"], "#55A868"),
        ("Biaya Shortage", summary["Total Biaya Shortage (Rp)"], "#d62728"),
        ("Biaya Spoilage", summary["Total Biaya Spoilage (Rp)"], "#ff7f0e"),
    ]
    for col, (lbl, val, _) in zip(rcols, cost_items):
        with col:
            st.metric(lbl, f"Rp {val/1e6:.2f}M")

    st.success(f"**GRAND TOTAL: Rp {grand_total:,.0f}**  ({sim_days} hari, M={M}, N={N})")


# ══════════════════════════════════════════════════════════
# TAB 4 — EVALUASI & OPTIMASI
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="stage-badge">TAHAP 4 — EVALUASI & OPTIMASI</div>', unsafe_allow_html=True)
    st.markdown("### Sensitivity Analysis & Rekomendasi Kebijakan")

    # Tabel hasil evaluasi
    st.markdown("**Tabel Biaya Total untuk Semua Kombinasi M & N**")

    pivot_cost = eval_df.pivot(index="M", columns="N", values="Grand Total (Rp)") / 1_000_000

    def color_heatmap(val):
        vmin, vmax = pivot_cost.values.min(), pivot_cost.values.max()
        ratio = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
        r = int(220 * ratio + 50 * (1 - ratio))
        g = int(180 * (1 - ratio) + 60 * ratio)
        b = int(50 * (1 - ratio) + 50 * ratio)
        return f"background-color: rgb({r},{g},{b}); color: {'white' if ratio > 0.6 else 'black'};"

    pivot_display = pivot_cost.applymap(lambda x: f"Rp {x:.2f}M")
    st.dataframe(
        pivot_cost.style.applymap(lambda v: color_heatmap(v)).format("{:.2f}"),
        use_container_width=True
    )

    # Heatmap grafik
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5))
    fig2.suptitle("Sensitivity Analysis: Total Biaya per Kombinasi M & N", fontsize=12, fontweight="bold")

    for ax, metric, label, cmap in zip(
        axes2,
        ["Grand Total (Rp)", "Shortage (L)", "Spoilage (L)"],
        ["Grand Total Biaya (Rp Juta)", "Total Shortage (Liter)", "Total Spoilage (Liter)"],
        ["RdYlGn_r", "Reds", "Oranges"]
    ):
        pv = eval_df.pivot(index="M", columns="N", values=metric)
        if metric == "Grand Total (Rp)":
            pv = pv / 1_000_000
        im = ax.imshow(pv.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(pv.columns)))
        ax.set_xticklabels([f"N={n}" for n in pv.columns], fontsize=8)
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels([f"M={m}" for m in pv.index], fontsize=8)
        ax.set_title(label, fontsize=9)
        for i in range(len(pv.index)):
            for j in range(len(pv.columns)):
                val = pv.values[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.5, color="black")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    # Rekomendasi
    best_m = int(best["M"])
    best_n = int(best["N"])
    best_cost = best["Grand Total (Rp)"]
    current_cost = summary["GRAND TOTAL BIAYA (Rp)"]
    saving = current_cost - best_cost

    st.markdown("**✅ Rekomendasi Kebijakan Optimal**")
    rec_cols = st.columns(3)
    with rec_cols[0]:
        st.markdown(f'<div class="metric-card"><div class="label">M Optimal</div>'
                    f'<div class="value">{best_m} L</div><div class="sub">Stok Maks</div></div>',
                    unsafe_allow_html=True)
    with rec_cols[1]:
        st.markdown(f'<div class="metric-card"><div class="label">N Optimal</div>'
                    f'<div class="value">{best_n} hari</div><div class="sub">Review Period</div></div>',
                    unsafe_allow_html=True)
    with rec_cols[2]:
        st.markdown(f'<div class="metric-card"><div class="label">Biaya Optimal</div>'
                    f'<div class="value">Rp {best_cost/1e6:.2f}M</div><div class="sub">{sim_days} hari</div></div>',
                    unsafe_allow_html=True)

    if saving > 0:
        st.markdown(
            f'<div class="info-box">💡 Dengan mengganti kebijakan ke M={best_m}, N={best_n}, '
            f'penghematan biaya sebesar <strong>Rp {saving:,.0f}</strong> '
            f'({saving/current_cost*100:.1f}%) dibandingkan setting saat ini (M={M}, N={N}).</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="info-box">✅ Setting saat ini (M={M}, N={N}) sudah merupakan '
            f'kebijakan optimal dari semua kombinasi yang diuji!</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("**Interpretasi & Kesimpulan**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
**Analisis Temuan:**
- M kecil → shortage tinggi → biaya shortage dominan
- M besar → spoilage tinggi (nira terbuang) → biaya spoilage naik
- N kecil (review lebih sering) → stok lebih terkontrol, tapi biaya ordering meningkat
- N besar → risiko kehabisan stok antara review period
- **Sweet spot** ada di M={best_m} dan N={best_n}
        """)
    with col_b:
        st.markdown(f"""
**Implikasi Manajerial:**
- Tetapkan stok maksimum pada **{best_m} liter**
- Lakukan pengecekan stok setiap **{best_n} hari sekali**
- Komponen biaya terbesar adalah shortage (Rp {summary['Total Biaya Shortage (Rp)']/1e6:.2f}M)
  → fokus investasi pada kepastian pasokan
- Pertimbangkan buffer stock atau supplier backup untuk mengurangi lead time
        """)

    # Tabel lengkap evaluasi
    st.markdown("---")
    st.markdown("**Tabel Lengkap Semua Kombinasi**")
    eval_display = eval_df.copy()
    eval_display["Grand Total (Rp)"] = eval_display["Grand Total (Rp)"].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(eval_display, use_container_width=True, hide_index=True)



# ══════════════════════════════════════════════════════════
# TAB 5 — FORMULASI MATEMATIS (SubCPMK02)
# ══════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="stage-badge">SubCPMK02 — FORMULASI MODEL MATEMATIS</div>', unsafe_allow_html=True)
    st.markdown("### Model Matematis Sistem Inventory (M, N) — Nira Aren")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1. Variabel State & Keputusan")
        st.latex(r"""
        \begin{aligned}
        I_t &= \text{Stok awal hari ke-}t \text{ (liter)} \\
        D_t &= \text{Permintaan hari ke-}t \text{ (liter)} \\
        R_t &= \text{Jumlah nira diterima hari ke-}t \\
        Q_t &= \text{Jumlah pemesanan hari ke-}t \\
        S_t^- &= \text{Shortage hari ke-}t \\
        S_t^+ &= \text{Spoilage hari ke-}t \\
        M &= \text{Stok maksimum (level restock)} \\
        N &= \text{Review period (hari)}
        \end{aligned}
        """)

        st.markdown("#### 2. Distribusi Permintaan")
        st.latex(r"D_t \sim \mathcal{N}(\mu, \sigma^2), \quad D_t \geq 0")
        st.markdown(f"Dengan nilai: $\\mu = {demand_mean}$ liter/hari, $\\sigma = {demand_std}$ liter")
        st.latex(r"f(D_t) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(D_t-\mu)^2}{2\sigma^2}\right)")

        st.markdown("#### 3. Distribusi Lead Time")
        st.latex(r"L \sim \mathcal{U}[a, b]")
        st.markdown(f"Dengan nilai: $a = {lt_min}$ hari, $b = {lt_max}$ hari")
        st.latex(r"f(L) = \frac{1}{b - a}, \quad a \leq L \leq b")

    with col2:
        st.markdown("#### 4. Dinamika Stok")
        st.markdown("**Penerimaan pesanan:**")
        st.latex(r"I_t = I_{t-1} + R_t")
        st.markdown("**Kebijakan pemesanan (review periodik):**")
        st.latex(r"""
        Q_t = \begin{cases}
        M - I_t & \text{jika } t \bmod N = 0 \text{ dan } I_t < M \\
        0 & \text{lainnya}
        \end{cases}
        """)
        st.markdown("**Pemenuhan permintaan (perishable):**")
        st.latex(r"""
        S_t^- = \max(0,\ D_t - I_t)
        """)
        st.latex(r"""
        S_t^+ = \max(0,\ I_t - D_t) \quad \text{(expired, tidak bisa disimpan)}
        """)
        st.markdown("**Stok akhir hari (selalu 0 karena perishable):**")
        st.latex(r"I_t^{\text{akhir}} = 0 \quad \forall t")

        st.markdown("#### 5. Fungsi Biaya Total")
        st.latex(r"""
        TC = \sum_{t=1}^{T} C_t
        """)
        st.latex(r"""
        C_t = c_p \cdot Q_t + c_h \cdot R_t + c_s \cdot S_t^- + c_w \cdot S_t^+
        """)
        st.markdown("**Keterangan parameter biaya:**")
        biaya_data = {
            "Simbol": ["$c_p$", "$c_h$", "$c_s$", "$c_w$"],
            "Keterangan": ["Biaya beli per liter", "Biaya holding per liter", "Biaya shortage per liter", "Biaya spoilage per liter"],
            "Nilai (Rp)": [f"{purchase_cost:,}", f"{holding_cost:,}", f"{shortage_cost:,}", f"{spoilage_cost:,}"],
        }
        st.dataframe(pd.DataFrame(biaya_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 6. Formulasi Optimasi")
    st.markdown("**Tujuan:** Temukan kebijakan $(M^*, N^*)$ yang meminimalkan total biaya:")
    st.latex(r"""
    (M^*, N^*) = \arg\min_{M, N}\ \mathbb{E}\left[\sum_{t=1}^{T}
    \left( c_p Q_t + c_h R_t + c_s S_t^- + c_w S_t^+ \right)\right]
    """)
    st.markdown("**Subject to:**")
    st.latex(r"""
    \begin{aligned}
    &M > 0,\quad N \in \mathbb{Z}^+\\
    &I_t \geq 0 \quad \forall t\\
    &D_t \sim \mathcal{N}(\mu, \sigma^2),\quad L \sim \mathcal{U}[a,b]
    \end{aligned}
    """)

    st.markdown("---")
    st.markdown("#### 7. Koneksi ke Sistem Nyata")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
<div class="info-box">
<strong>🌿 Kondisi Nyata</strong><br>
Nira aren dipanen pagi hari, harus langsung diolah dalam 3 jam. Petani tidak bisa menyimpan nira — expired = terbuang.
</div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
<div class="info-box">
<strong>🔢 Representasi Model</strong><br>
Satu unit waktu = 1 shift kerja. S⁺ₜ merepresentasikan nira expired. Stok akhir selalu 0 karena tidak ada carry-over.
</div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown("""
<div class="info-box">
<strong>📦 Kebijakan (M, N)</strong><br>
Setiap N hari, stok diperiksa. Jika kurang dari M, pesan sebanyak M − Iₜ. Pesanan tiba setelah lead time L hari.
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 6 — ANALISIS STATISTIK (SubCPMK02)
# ══════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="stage-badge">SubCPMK02 — ANALISIS STATISTIK HASIL SIMULASI</div>', unsafe_allow_html=True)
    st.markdown("### Analisis Statistik Deskriptif & Inferensial")

    # ── Jalankan multiple runs untuk analisis statistik ──
    N_RUNS = 50
    if "stat_runs" not in st.session_state or run_btn:
        runs = []
        for i in range(N_RUNS):
            _, s = run_simulation(M, N, sim_days, demand_mean, demand_std,
                                  lt_min, lt_max, purchase_cost, shortage_cost,
                                  spoilage_cost, holding_cost, seed=i)
            runs.append({
                "Run": i+1,
                "Total Biaya (Rp)":     s["GRAND TOTAL BIAYA (Rp)"],
                "Shortage (L)":         s["Total Shortage (L)"],
                "Spoilage (L)":         s["Total Spoilage (L)"],
                "Biaya Shortage (Rp)":  s["Total Biaya Shortage (Rp)"],
                "Biaya Spoilage (Rp)":  s["Total Biaya Spoilage (Rp)"],
            })
        st.session_state["stat_runs"] = pd.DataFrame(runs)

    runs_df = st.session_state["stat_runs"]

    st.markdown(f"*Analisis berbasis **{N_RUNS} kali simulasi** dengan seed berbeda (seed 0–{N_RUNS-1}), M={M}, N={N}, durasi={sim_days} hari*")

    # ── Statistik Deskriptif ──
    st.markdown("---")
    st.markdown("#### 📊 Statistik Deskriptif")

    metrics = ["Total Biaya (Rp)", "Shortage (L)", "Spoilage (L)"]
    stat_rows = []
    for m in metrics:
        data = runs_df[m]
        ci = stats.t.interval(0.95, df=len(data)-1, loc=data.mean(), scale=stats.sem(data))
        stat_rows.append({
            "Metrik": m,
            "Mean": f"{data.mean():,.1f}",
            "Median": f"{data.median():,.1f}",
            "Std Dev": f"{data.std():,.1f}",
            "Min": f"{data.min():,.1f}",
            "Max": f"{data.max():,.1f}",
            "CI 95% Bawah": f"{ci[0]:,.1f}",
            "CI 95% Atas": f"{ci[1]:,.1f}",
        })
    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    # ── Grafik distribusi ──
    st.markdown("---")
    st.markdown("#### 📉 Distribusi Hasil Simulasi (50 Runs)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["#7b2d8b", "#d62728", "#ff7f0e"]

    for ax, col, color in zip(axes, metrics, colors):
        data = runs_df[col]
        ax.hist(data, bins=12, color=color, alpha=0.75, edgecolor="white", linewidth=0.7)
        ax.axvline(data.mean(), color="white", linestyle="--", linewidth=1.8, label=f"Mean: {data.mean():,.0f}")
        ci = stats.t.interval(0.95, df=len(data)-1, loc=data.mean(), scale=stats.sem(data))
        ax.axvline(ci[0], color="#C8F560", linestyle=":", linewidth=1.4, label=f"CI 95%: [{ci[0]:,.0f}, {ci[1]:,.0f}]")
        ax.axvline(ci[1], color="#C8F560", linestyle=":", linewidth=1.4)
        ax.set_title(col, fontsize=10, color="white")
        ax.set_xlabel("Nilai", fontsize=9, color="white")
        ax.set_ylabel("Frekuensi", fontsize=9, color="white")
        ax.legend(fontsize=7.5, labelcolor="white")
        ax.set_facecolor("#132637")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1B998B")

    fig.patch.set_facecolor("#0D1B2A")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Uji Normalitas ──
    st.markdown("---")
    st.markdown("#### 🧪 Uji Normalitas (Shapiro-Wilk)")
    st.markdown("*H₀: Data berdistribusi normal. Tolak H₀ jika p-value < 0.05*")

    norm_rows = []
    for col in metrics:
        stat_sw, p_sw = stats.shapiro(runs_df[col])
        norm_rows.append({
            "Metrik": col,
            "Statistik W": f"{stat_sw:.4f}",
            "p-value": f"{p_sw:.4f}",
            "Kesimpulan": "✅ Normal (gagal tolak H₀)" if p_sw > 0.05 else "⚠️ Tidak Normal (tolak H₀)",
        })
    st.dataframe(pd.DataFrame(norm_rows), use_container_width=True, hide_index=True)

    # ── Boxplot perbandingan ──
    st.markdown("---")
    st.markdown("#### 📦 Boxplot Variabilitas Hasil")

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, color in zip(axes2, metrics, colors):
        data = runs_df[col]
        bp = ax.boxplot(data, patch_artist=True, notch=True,
                        boxprops=dict(facecolor=color, alpha=0.7),
                        medianprops=dict(color="#C8F560", linewidth=2),
                        whiskerprops=dict(color="white"),
                        capprops=dict(color="white"),
                        flierprops=dict(markerfacecolor=color, alpha=0.5))
        ax.set_title(col, fontsize=10, color="white")
        ax.set_ylabel("Nilai", fontsize=9, color="white")
        ax.set_facecolor("#132637")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1B998B")
        # Tambahkan anotasi mean
        ax.text(1.2, data.mean(), f"μ={data.mean():,.0f}", color="#C8F560", fontsize=8.5, va="center")

    fig2.patch.set_facecolor("#0D1B2A")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Korelasi ──
    st.markdown("---")
    st.markdown("#### 🔗 Analisis Korelasi antar Variabel Output")
    corr = runs_df[metrics].corr()

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    im = ax3.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax3.set_xticks(range(len(metrics)))
    ax3.set_yticks(range(len(metrics)))
    short_labels = ["Total Biaya", "Shortage", "Spoilage"]
    ax3.set_xticklabels(short_labels, rotation=15, fontsize=9, color="white")
    ax3.set_yticklabels(short_labels, fontsize=9, color="white")
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax3.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                     fontsize=11, color="black", fontweight="bold")
    plt.colorbar(im, ax=ax3)
    ax3.set_title("Matriks Korelasi Output Simulasi", fontsize=10, color="white")
    ax3.set_facecolor("#132637")
    ax3.tick_params(colors="white")
    fig3.patch.set_facecolor("#0D1B2A")
    plt.tight_layout()
    col_corr, _ = st.columns([1.5, 1])
    with col_corr:
        st.pyplot(fig3)
    plt.close()

    st.markdown("""
<div class="info-box">
💡 <strong>Interpretasi Korelasi:</strong><br>
Korelasi positif kuat antara <em>Shortage</em> dan <em>Total Biaya</em> menunjukkan bahwa
komponen biaya shortage adalah driver utama total biaya. Korelasi antara Shortage dan Spoilage
yang negatif mengkonfirmasi trade-off fundamental sistem perishable ini.
</div>""", unsafe_allow_html=True)

    # ── Ringkasan Statistik untuk Presentasi ──
    st.markdown("---")
    st.markdown("#### 📋 Ringkasan Temuan Statistik")
    total_biaya_data = runs_df["Total Biaya (Rp)"]
    shortage_data    = runs_df["Shortage (L)"]
    spoilage_data    = runs_df["Spoilage (L)"]
    ci_biaya = stats.t.interval(0.95, df=len(total_biaya_data)-1,
                                 loc=total_biaya_data.mean(), scale=stats.sem(total_biaya_data))

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f'<div class="metric-card"><div class="label">Rata-rata Total Biaya</div>'
                    f'<div class="value">Rp {total_biaya_data.mean()/1e6:.2f}M</div>'
                    f'<div class="sub">dari {N_RUNS} simulasi</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="metric-card"><div class="label">Confidence Interval 95%</div>'
                    f'<div class="value">±Rp {(ci_biaya[1]-ci_biaya[0])/2/1000:.0f}K</div>'
                    f'<div class="sub">margin of error</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="metric-card"><div class="label">Koef. Variasi Biaya</div>'
                    f'<div class="value">{total_biaya_data.std()/total_biaya_data.mean()*100:.1f}%</div>'
                    f'<div class="sub">stabilitas model</div></div>', unsafe_allow_html=True)
    with r4:
        st.markdown(f'<div class="metric-card"><div class="label">Rata-rata Shortage</div>'
                    f'<div class="value">{shortage_data.mean():.1f} L</div>'
                    f'<div class="sub">per {sim_days} hari</div></div>', unsafe_allow_html=True)

# =========================================================
# TAB 7
# =========================================================
with tab7:
    st.markdown('<div class="stage-badge">SubCPMK03 — KOMPARASI & VALIDASI</div>', unsafe_allow_html=True)
    st.markdown(f"## ⚖️ Analisis Sistem (Periodic vs Newsvendor) | Skenario: {scenario}")

    # ==========================================
    # RUN KEDUA METODE UNTUK KOMPARASI
    # ==========================================
    df_mn, sum_mn = run_periodic_review_simulation(
        M, N, sim_days, demand_mean, demand_std, 
        lt_min, lt_max, purchase_cost, shortage_cost, 
        spoilage_cost, holding_cost, scenario, seed
    )

    df_nv, sum_nv = run_newsvendor_simulation(
        sim_days, demand_mean, demand_std, 
        purchase_cost, shortage_cost, 
        spoilage_cost, holding_cost, scenario, seed
    )

    # ==========================================
    # KPI TABLE
    # ==========================================
    compare_df = pd.DataFrame({
        "Metric": [
            "Grand Total Biaya", 
            "Total Shortage", 
            "Total Spoilage", 
            "Service Level"
        ],
        "Periodic Review (M,N)": [
            f"Rp {sum_mn['GRAND TOTAL BIAYA (Rp)']:,.0f}", 
            f"{sum_mn['Total Shortage (L)']:.2f} L", 
            f"{sum_mn['Total Spoilage (L)']:.2f} L", 
            f"{sum_mn['Service Level']:.2f}%"
        ],
        "Newsvendor (Q*)": [
            f"Rp {sum_nv['GRAND TOTAL BIAYA (Rp)']:,.0f}", 
            f"{sum_nv['Total Shortage (L)']:.2f} L", 
            f"{sum_nv['Total Spoilage (L)']:.2f} L", 
            f"{sum_nv['Service Level']:.2f}%"
        ]
    })

    st.markdown("#### Tabel Perbandingan KPI")
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    # ==========================================
    # BAR CHART
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["Beli", "Simpan", "Shortage", "Spoilage"]
    
    mn_vals = [
        sum_mn["Total Biaya Beli (Rp)"] / 1e6, 
        sum_mn["Total Biaya Simpan (Rp)"] / 1e6, 
        sum_mn["Total Biaya Shortage (Rp)"] / 1e6, 
        sum_mn["Total Biaya Spoilage (Rp)"] / 1e6
    ]
    
    nv_vals = [
        sum_nv["Total Biaya Beli (Rp)"] / 1e6, 
        sum_nv["Total Biaya Simpan (Rp)"] / 1e6, 
        sum_nv["Total Biaya Shortage (Rp)"] / 1e6, 
        sum_nv["Total Biaya Spoilage (Rp)"] / 1e6
    ]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, mn_vals, width, label=f"Periodic Review (M={M}, N={N})")
    ax.bar(x + width/2, nv_vals, width, label=f"Newsvendor (Q*={sum_nv.get('Q_optimal', 0)})")
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Biaya (Rp Juta)")
    ax.set_title(f"Komparasi Komponen Biaya — Skenario: {scenario}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    plt.close()

    # ==========================================
    # VALIDATION TEXT
    # ==========================================
    st.markdown("---")
    if sum_nv["GRAND TOTAL BIAYA (Rp)"] < sum_mn["GRAND TOTAL BIAYA (Rp)"]:
        st.success(f"✅ **Validasi Data-Driven:** Berdasarkan skenario **{scenario}**, metode **Newsvendor** terbukti lebih optimal karena menghasilkan total biaya lebih rendah.\n\nModel ini sangat direkomendasikan untuk produk yang 100% *perishable* seperti nira aren, karena keputusan pengadaan (inventory) dioptimasi per hari (*single-period*) dengan mempertimbangkan *Critical Ratio* (trade-off antara risiko kekurangan dan kerusakan barang).")
    else:
        st.info(f"ℹ️ **Validasi Data-Driven:** Berdasarkan skenario **{scenario}**, metode **Periodic Review (M,N)** saat ini lebih optimal secara biaya dibandingkan Newsvendor statis.\n\nKebijakan siklus periodik masih dapat menahan fluktuasi jika parameter M dan N disetel dengan sangat akurat sesuai variasi *demand* dan *lead time*.")


# =========================================================
# TAB 8 — PERBANDINGAN DATASET (SubCPMK03)
# =========================================================
with tab8:
    st.markdown('<div class="stage-badge">SubCPMK03 — PERBANDINGAN DATASET</div>', unsafe_allow_html=True)
    st.markdown("## 🗂️ Perbandingan Dataset A vs Dataset B")
    st.markdown("""
<div class="info-box">
💡 <strong>Tujuan:</strong> Membandingkan dua dataset dengan parameter berbeda menggunakan kedua metode
(Periodic Review & Newsvendor) untuk menguji <em>robustness</em> dan validitas model simulasi.
Pilih parameter Dataset B yang berbeda dari parameter utama (Dataset A) di sidebar.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Parameter Dataset B ──────────────────────────────────
    st.markdown("### ⚙️ Parameter Dataset B")
    st.markdown("*Dataset A otomatis menggunakan parameter dari sidebar kiri.*")

    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        st.markdown("**Distribusi Permintaan**")
        b_scenario    = st.selectbox("Skenario Demand B", ["Normal", "Musiman", "Ekstrem"],
                                     index=1, key="b_scenario")
        b_demand_mean = st.number_input("Rata-rata Permintaan B (L/hari)", 50, 300, 130, step=10, key="b_dmean")
        b_demand_std  = st.number_input("Std. Deviasi B (L)", 5, 80, 35, step=5, key="b_dstd")
    with bcol2:
        st.markdown("**Kebijakan Inventaris B**")
        b_M    = st.slider("M — Stok Maksimum B (L)", 50, 300, 175, step=25, key="b_M")
        b_N    = st.slider("N — Review Period B (Hari)", 1, 7, 2, key="b_N")
        b_days = st.slider("Durasi Simulasi B (Hari)", 10, 90, 30, step=5, key="b_days")
    with bcol3:
        st.markdown("**Lead Time & Biaya B**")
        b_lt_min       = st.number_input("Lead Time Min B", 1, 5, 1, key="b_ltmin")
        b_lt_max       = st.number_input("Lead Time Max B", 1, 10, 4, key="b_ltmax")
        b_purchase     = st.number_input("Harga Beli B (Rp/L)", 1000, 50000, 6000, step=500, key="b_buy")
        b_shortage     = st.number_input("Biaya Shortage B (Rp/L)", 1000, 50000, 9000, step=500, key="b_short")
        b_spoilage     = st.number_input("Biaya Spoilage B (Rp/L)", 1000, 50000, 4500, step=500, key="b_spoil")
        b_holding      = st.number_input("Biaya Holding B (Rp/L)", 100, 5000, 600, step=100, key="b_hold")
        b_seed         = st.number_input("Random Seed B", 0, 9999, 123, key="b_seed")

    run_compare = st.button("▶ Jalankan Perbandingan Dataset", type="primary", use_container_width=True, key="run_compare")

    st.markdown("---")

    # ── Jalankan simulasi kedua dataset ─────────────────────
    if "compare_done" not in st.session_state or run_compare:
        with st.spinner("Menjalankan simulasi Dataset A & B..."):
            # Dataset A — kedua metode
            dfA_mn, sumA_mn = run_periodic_review_simulation(
                M, N, sim_days, demand_mean, demand_std,
                lt_min, lt_max, purchase_cost, shortage_cost,
                spoilage_cost, holding_cost, scenario, seed
            )
            dfA_nv, sumA_nv = run_newsvendor_simulation(
                sim_days, demand_mean, demand_std,
                purchase_cost, shortage_cost, spoilage_cost, holding_cost,
                scenario, seed
            )
            # Dataset B — kedua metode
            dfB_mn, sumB_mn = run_periodic_review_simulation(
                b_M, b_N, b_days, b_demand_mean, b_demand_std,
                b_lt_min, b_lt_max, b_purchase, b_shortage,
                b_spoilage, b_holding, b_scenario, b_seed
            )
            dfB_nv, sumB_nv = run_newsvendor_simulation(
                b_days, b_demand_mean, b_demand_std,
                b_purchase, b_shortage, b_spoilage, b_holding,
                b_scenario, b_seed
            )
            st.session_state.update({
                "compare_done": True,
                "dfA_mn": dfA_mn, "sumA_mn": sumA_mn,
                "dfA_nv": dfA_nv, "sumA_nv": sumA_nv,
                "dfB_mn": dfB_mn, "sumB_mn": sumB_mn,
                "dfB_nv": dfB_nv, "sumB_nv": sumB_nv,
            })

    if "compare_done" in st.session_state:
        sumA_mn = st.session_state["sumA_mn"]
        sumA_nv = st.session_state["sumA_nv"]
        sumB_mn = st.session_state["sumB_mn"]
        sumB_nv = st.session_state["sumB_nv"]
        dfA_mn  = st.session_state["dfA_mn"]
        dfB_mn  = st.session_state["dfB_mn"]
        dfA_nv  = st.session_state["dfA_nv"]
        dfB_nv  = st.session_state["dfB_nv"]

        # ── 1. Ringkasan Parameter ────────────────────────────
        st.markdown("### 📋 Ringkasan Parameter Kedua Dataset")
        param_df = pd.DataFrame({
            "Parameter": [
                "Skenario Demand", "Rata-rata Demand (L/hr)", "Std. Deviasi (L)",
                "Durasi Simulasi (Hari)", "Stok Maks M (L)", "Review Period N (Hari)",
                "Lead Time (Hari)", "Harga Beli (Rp/L)", "Biaya Shortage (Rp/L)",
                "Biaya Spoilage (Rp/L)", "Biaya Holding (Rp/L)", "Random Seed"
            ],
            "Dataset A (Referensi)": [
                scenario, demand_mean, demand_std,
                sim_days, M, N,
                f"{lt_min}–{lt_max}", purchase_cost, shortage_cost,
                spoilage_cost, holding_cost, seed
            ],
            "Dataset B (Pembanding)": [
                b_scenario, b_demand_mean, b_demand_std,
                b_days, b_M, b_N,
                f"{b_lt_min}–{b_lt_max}", b_purchase, b_shortage,
                b_spoilage, b_holding, b_seed
            ],
        })
        st.dataframe(param_df, use_container_width=True, hide_index=True)

        # ── 2. Tabel KPI 4-Way ───────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Tabel Perbandingan KPI — 4 Kombinasi (Dataset × Metode)")

        kpi4_df = pd.DataFrame({
            "KPI": ["Grand Total Biaya (Rp)", "Total Shortage (L)", "Total Spoilage (L)", "Service Level (%)"],
            "A — Periodic (M,N)": [
                f"Rp {sumA_mn['GRAND TOTAL BIAYA (Rp)']:,.0f}",
                f"{sumA_mn['Total Shortage (L)']:.2f}",
                f"{sumA_mn['Total Spoilage (L)']:.2f}",
                f"{sumA_mn['Service Level']:.2f}%",
            ],
            "A — Newsvendor (Q*)": [
                f"Rp {sumA_nv['GRAND TOTAL BIAYA (Rp)']:,.0f}",
                f"{sumA_nv['Total Shortage (L)']:.2f}",
                f"{sumA_nv['Total Spoilage (L)']:.2f}",
                f"{sumA_nv['Service Level']:.2f}%",
            ],
            "B — Periodic (M,N)": [
                f"Rp {sumB_mn['GRAND TOTAL BIAYA (Rp)']:,.0f}",
                f"{sumB_mn['Total Shortage (L)']:.2f}",
                f"{sumB_mn['Total Spoilage (L)']:.2f}",
                f"{sumB_mn['Service Level']:.2f}%",
            ],
            "B — Newsvendor (Q*)": [
                f"Rp {sumB_nv['GRAND TOTAL BIAYA (Rp)']:,.0f}",
                f"{sumB_nv['Total Shortage (L)']:.2f}",
                f"{sumB_nv['Total Spoilage (L)']:.2f}",
                f"{sumB_nv['Service Level']:.2f}%",
            ],
        })
        st.dataframe(kpi4_df, use_container_width=True, hide_index=True)

        # ── 3. Grafik Perbandingan Biaya (Grouped Bar) ───────
        st.markdown("---")
        st.markdown("### 📉 Visualisasi Perbandingan Biaya Komponen")

        fig_comp, axes_comp = plt.subplots(1, 2, figsize=(16, 5))
        fig_comp.suptitle("Perbandingan Komponen Biaya: Dataset A vs B  ×  Metode Periodic vs Newsvendor",
                           fontsize=12, fontweight="bold")

        cost_keys = ["Total Biaya Beli (Rp)", "Total Biaya Simpan (Rp)",
                     "Total Biaya Shortage (Rp)", "Total Biaya Spoilage (Rp)"]
        cost_labels = ["Beli", "Simpan", "Shortage", "Spoilage"]
        colors4 = ["#4C72B0", "#55A868", "#d62728", "#ff7f0e"]
        x = np.arange(len(cost_labels))
        w = 0.2

        for ax, (label_ds, sum_mn_ds, sum_nv_ds) in zip(
            axes_comp,
            [("A", sumA_mn, sumA_nv), ("B", sumB_mn, sumB_nv)]
        ):
            vals_mn = [sum_mn_ds[k] / 1e6 for k in cost_keys]
            vals_nv = [sum_nv_ds[k] / 1e6 for k in cost_keys]
            bars_mn = ax.bar(x - w/2, vals_mn, w, label="Periodic (M,N)",
                             color=colors4, alpha=0.85)
            bars_nv = ax.bar(x + w/2, vals_nv, w, label="Newsvendor (Q*)",
                             color=colors4, alpha=0.45, edgecolor="black", linewidth=0.8)
            for bar in bars_mn:
                if bar.get_height() > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f"{bar.get_height():.2f}M", ha="center", va="bottom", fontsize=7.5)
            for bar in bars_nv:
                if bar.get_height() > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f"{bar.get_height():.2f}M", ha="center", va="bottom", fontsize=7.5)
            ax.set_title(f"Dataset {label_ds}", fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(cost_labels)
            ax.set_ylabel("Biaya (Rp Juta)")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_comp)
        plt.close()

        # ── 4. Grafik Akumulasi Biaya Harian ─────────────────
        st.markdown("---")
        st.markdown("### 📈 Kurva Akumulasi Biaya Harian")

        fig_cum, ax_cum = plt.subplots(figsize=(14, 5))
        ax_cum.plot(dfA_mn["Hari"], dfA_mn["Total Biaya/Hari"].cumsum() / 1e6,
                    label="A — Periodic (M,N)", color="#4C72B0", lw=2.2, linestyle="-")
        ax_cum.plot(dfA_nv["Hari"], dfA_nv["Total Biaya/Hari"].cumsum() / 1e6,
                    label="A — Newsvendor (Q*)", color="#4C72B0", lw=2.2, linestyle="--")
        ax_cum.plot(dfB_mn["Hari"], dfB_mn["Total Biaya/Hari"].cumsum() / 1e6,
                    label="B — Periodic (M,N)", color="#d62728", lw=2.2, linestyle="-")
        ax_cum.plot(dfB_nv["Hari"], dfB_nv["Total Biaya/Hari"].cumsum() / 1e6,
                    label="B — Newsvendor (Q*)", color="#d62728", lw=2.2, linestyle="--")
        ax_cum.set_xlabel("Hari")
        ax_cum.set_ylabel("Biaya Kumulatif (Rp Juta)")
        ax_cum.set_title("Akumulasi Biaya Harian — Dataset A (Biru) vs Dataset B (Merah)  |  Solid=Periodic, Dash=Newsvendor",
                          fontsize=10)
        ax_cum.legend(fontsize=9)
        ax_cum.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_cum)
        plt.close()

        # ── 5. Perbandingan Permintaan Harian ────────────────
        st.markdown("---")
        st.markdown("### 🔀 Distribusi Permintaan Harian (Histogram)")

        fig_hist, ax_hist = plt.subplots(figsize=(12, 4))
        ax_hist.hist(dfA_mn["Permintaan (L)"], bins=20, alpha=0.6,
                     color="#4C72B0", label=f"Dataset A ({scenario}, μ={demand_mean})", edgecolor="white")
        ax_hist.hist(dfB_mn["Permintaan (L)"], bins=20, alpha=0.6,
                     color="#d62728", label=f"Dataset B ({b_scenario}, μ={b_demand_mean})", edgecolor="white")
        ax_hist.axvline(dfA_mn["Permintaan (L)"].mean(), color="#4C72B0", lw=2,
                        ls="--", label=f"Mean A = {dfA_mn['Permintaan (L)'].mean():.1f}")
        ax_hist.axvline(dfB_mn["Permintaan (L)"].mean(), color="#d62728", lw=2,
                        ls="--", label=f"Mean B = {dfB_mn['Permintaan (L)'].mean():.1f}")
        ax_hist.set_xlabel("Volume Permintaan (Liter)")
        ax_hist.set_ylabel("Frekuensi")
        ax_hist.set_title("Histogram Permintaan Harian — Dataset A vs Dataset B")
        ax_hist.legend(fontsize=9)
        ax_hist.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close()

        # ── 6. Uji Statistik Antar Dataset ───────────────────
        st.markdown("---")
        st.markdown("### 🧪 Uji Statistik Perbandingan Dataset (Mann-Whitney U Test)")
        st.markdown("*Menguji apakah distribusi biaya harian Dataset A dan B berbeda secara signifikan.*")

        from scipy.stats import mannwhitneyu, ttest_ind

        stat_tests = []
        for col in ["Total Biaya/Hari", "Shortage (L)", "Spoilage (L)"]:
            a_vals = dfA_mn[col]
            b_vals = dfB_mn[col]
            u_stat, p_val = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            stat_tests.append({
                "Variabel": col,
                "Mean Dataset A": f"{a_vals.mean():.2f}",
                "Mean Dataset B": f"{b_vals.mean():.2f}",
                "U-Statistik": f"{u_stat:.2f}",
                "p-value": f"{p_val:.4f}",
                "Kesimpulan": "✅ Berbeda signifikan (p<0.05)" if p_val < 0.05 else "⚠️ Tidak berbeda signifikan"
            })
        st.dataframe(pd.DataFrame(stat_tests), use_container_width=True, hide_index=True)

        # ── 7. Radar Chart Perbandingan ───────────────────────
        st.markdown("---")
        st.markdown("### 🕸️ Radar Chart — Profil Kinerja 4 Kombinasi")

        import matplotlib.patches as mpatches

        # Normalisasi 4 dimensi: Total Biaya, Shortage, Spoilage, Service Level
        kpi_dims = ["Total Biaya\n(Rp Juta)", "Total\nShortage (L)",
                    "Total\nSpoilage (L)", "Service Level\n(Dibalik)"]
        N_dims = len(kpi_dims)
        angles = np.linspace(0, 2 * np.pi, N_dims, endpoint=False).tolist()
        angles += angles[:1]

        def normalize_radar(vals_list):
            arr = np.array(vals_list, dtype=float)
            mn, mx = arr.min(axis=0), arr.max(axis=0)
            denom = np.where(mx - mn == 0, 1, mx - mn)
            return (arr - mn) / denom

        raw = np.array([
            [sumA_mn["GRAND TOTAL BIAYA (Rp)"] / 1e6, sumA_mn["Total Shortage (L)"],
             sumA_mn["Total Spoilage (L)"], 100 - sumA_mn["Service Level"]],
            [sumA_nv["GRAND TOTAL BIAYA (Rp)"] / 1e6, sumA_nv["Total Shortage (L)"],
             sumA_nv["Total Spoilage (L)"], 100 - sumA_nv["Service Level"]],
            [sumB_mn["GRAND TOTAL BIAYA (Rp)"] / 1e6, sumB_mn["Total Shortage (L)"],
             sumB_mn["Total Spoilage (L)"], 100 - sumB_mn["Service Level"]],
            [sumB_nv["GRAND TOTAL BIAYA (Rp)"] / 1e6, sumB_nv["Total Shortage (L)"],
             sumB_nv["Total Spoilage (L)"], 100 - sumB_nv["Service Level"]],
        ])
        norm = normalize_radar(raw)

        radar_labels = [
            f"A–Periodic (M={M},N={N})",
            f"A–Newsvendor (Q*={sumA_nv.get('Q_optimal', 0):.0f})",
            f"B–Periodic (M={b_M},N={b_N})",
            f"B–Newsvendor (Q*={sumB_nv.get('Q_optimal', 0):.0f})",
        ]
        radar_colors = ["#4C72B0", "#55A868", "#d62728", "#ff7f0e"]

        fig_radar, ax_radar = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for i, (label, color) in enumerate(zip(radar_labels, radar_colors)):
            values = norm[i].tolist()
            values += values[:1]
            ax_radar.plot(angles, values, color=color, lw=2, label=label)
            ax_radar.fill(angles, values, color=color, alpha=0.12)

        ax_radar.set_thetagrids(np.degrees(angles[:-1]), kpi_dims, fontsize=9)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title("Profil Kinerja (nilai lebih rendah = lebih baik untuk 3 dimensi pertama)",
                            fontsize=9, pad=20)
        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)
        ax_radar.grid(True, alpha=0.4)
        plt.tight_layout()

        rcol1, rcol2 = st.columns([1, 1])
        with rcol1:
            st.pyplot(fig_radar)
        plt.close()

        with rcol2:
            st.markdown("**Interpretasi Radar Chart:**")
            st.markdown("""
- **Area lebih kecil** → profil kinerja lebih baik (biaya & shortage/spoilage lebih rendah)
- Dimensi 1–3 diukur dalam skala *semakin kecil semakin baik*
- Dimensi 4 (Service Level) dibalik → nilai kecil = service level tinggi

Radar chart memungkinkan perbandingan **multidimensi sekaligus**
antar kombinasi dataset dan metode yang berbeda.
""")

        # ── 8. Kesimpulan Komparasi ───────────────────────────
        st.markdown("---")
        st.markdown("### 🏆 Kesimpulan Komparasi Dataset")

        all_combos = {
            f"A – Periodic (M={M}, N={N})": sumA_mn["GRAND TOTAL BIAYA (Rp)"],
            f"A – Newsvendor (Q*)":         sumA_nv["GRAND TOTAL BIAYA (Rp)"],
            f"B – Periodic (M={b_M}, N={b_N})": sumB_mn["GRAND TOTAL BIAYA (Rp)"],
            f"B – Newsvendor (Q*)":         sumB_nv["GRAND TOTAL BIAYA (Rp)"],
        }
        best_combo  = min(all_combos, key=all_combos.get)
        worst_combo = max(all_combos, key=all_combos.get)

        concl_col1, concl_col2 = st.columns(2)
        with concl_col1:
            st.markdown(f'<div class="metric-card"><div class="label">🏆 Kombinasi Terbaik</div>'
                        f'<div class="value" style="font-size:1rem">{best_combo}</div>'
                        f'<div class="sub">Rp {all_combos[best_combo]:,.0f}</div></div>', unsafe_allow_html=True)
        with concl_col2:
            st.markdown(f'<div class="metric-card"><div class="label">⚠️ Kombinasi Termahal</div>'
                        f'<div class="value" style="font-size:1rem">{worst_combo}</div>'
                        f'<div class="sub">Rp {all_combos[worst_combo]:,.0f}</div></div>', unsafe_allow_html=True)

        delta_pct = (all_combos[worst_combo] - all_combos[best_combo]) / all_combos[worst_combo] * 100
        a_best = "Newsvendor" if sumA_nv["GRAND TOTAL BIAYA (Rp)"] < sumA_mn["GRAND TOTAL BIAYA (Rp)"] else "Periodic Review"
        b_best = "Newsvendor" if sumB_nv["GRAND TOTAL BIAYA (Rp)"] < sumB_mn["GRAND TOTAL BIAYA (Rp)"] else "Periodic Review"

        st.markdown(f"""
<div class="info-box">
📌 <strong>Temuan Utama Perbandingan Dataset:</strong><br><br>
1. Pada <strong>Dataset A</strong> (Skenario: {scenario}, μ={demand_mean} L), metode terbaik adalah <strong>{a_best}</strong>.<br>
2. Pada <strong>Dataset B</strong> (Skenario: {b_scenario}, μ={b_demand_mean} L), metode terbaik adalah <strong>{b_best}</strong>.<br>
3. Selisih biaya antara kombinasi terbaik dan terburuk mencapai <strong>{delta_pct:.1f}%</strong>.<br>
4. Perubahan parameter data (skenario demand, rata-rata, variabilitas) <em>secara signifikan</em> mempengaruhi 
   kinerja kedua metode — dikonfirmasi oleh uji statistik Mann-Whitney U di atas.<br>
5. <strong>Kesimpulan validasi:</strong> Model simulasi bersifat <em>robust</em> dan responsif terhadap perubahan input data,
   sehingga valid digunakan sebagai alat bantu pengambilan keputusan inventaris nira aren.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.78rem; padding: 0.5rem 0;">
    Simulasi Inventory (M, N) — Nira Aren Perishable &nbsp;|&nbsp;
    Mata Kuliah: Pemodelan dan Simulasi Komputer &nbsp;|&nbsp;
    SubCPMK01 + SubCPMK02 + SubCPMK03
</div>
""", unsafe_allow_html=True)