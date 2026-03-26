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

/* Teks label di sidebar */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1b4332 !important;
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
        df, summary = run_simulation(
            M, N, sim_days, demand_mean, demand_std,
            lt_min, lt_max, purchase_cost, shortage_cost,
            spoilage_cost, holding_cost, seed
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
# TABS UTAMA (4 tahapan SubCPMK01)
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 1. Analisis Kebutuhan",
    "📐 2. Perancangan Model",
    "▶ 3. Implementasi & Simulasi",
    "📊 4. Evaluasi & Optimasi",
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


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.78rem; padding: 0.5rem 0;">
    Simulasi Inventory (M, N) — Nira Aren Perishable &nbsp;|&nbsp;
    Mata Kuliah: Pemodelan dan Simulasi Komputer &nbsp;|&nbsp;
    SubCPMK01: Analisis → Perancangan → Implementasi → Evaluasi
</div>
""", unsafe_allow_html=True)
