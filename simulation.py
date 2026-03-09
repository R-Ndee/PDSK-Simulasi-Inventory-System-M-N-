"""
Simulasi Inventory System (M, N) - Nira Aren Perishable
Mata Kuliah: Pemodelan dan Simulasi Komputer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# =============================================================================
# BLOK 1: PARAMETER & ASUMSI
# =============================================================================

np.random.seed(42)  # Seed untuk reproduksibilitas hasil

# Parameter Simulasi
SIM_DAYS        = 30        # Jumlah hari simulasi
M               = 150       # Stok maksimum (liter) - level restocking
N               = 3         # Review period (hari sekali stok dicek)

# Parameter Distribusi
DEMAND_MEAN     = 100       # Rata-rata permintaan harian (liter)
DEMAND_STD      = 20        # Standar deviasi permintaan (liter)
LEAD_TIME_MIN   = 1         # Lead time minimum (hari)
LEAD_TIME_MAX   = 3         # Lead time maksimum (hari)

# Parameter Biaya
PURCHASE_COST   = 5000      # Harga beli nira per liter (Rp)
SHORTAGE_COST   = 8000      # Biaya kekurangan stok per liter (Rp)
SPOILAGE_COST   = 5000      # Biaya nira rusak/expired per liter (Rp)
HOLDING_COST    = 500       # Biaya simpan per liter per hari (Rp)

# =============================================================================
# BLOK 2: ENGINE SIMULASI
# =============================================================================

def generate_demand(days):
    """Generate permintaan harian menggunakan distribusi Normal."""
    demands = np.random.normal(DEMAND_MEAN, DEMAND_STD, days)
    return np.maximum(demands, 0).round(2)  # Permintaan tidak boleh negatif


def generate_lead_time():
    """Generate lead time menggunakan distribusi Uniform."""
    return int(np.random.uniform(LEAD_TIME_MIN, LEAD_TIME_MAX + 1))


def run_simulation(M, N, sim_days):
    """
    Menjalankan simulasi inventory (M, N) untuk nira aren.
    
    Parameters:
        M        : Stok maksimum / order-up-to level (liter)
        N        : Review period (hari)
        sim_days : Total hari simulasi
    
    Returns:
        df       : DataFrame hasil simulasi harian
        summary  : Dictionary ringkasan biaya total
    """

    # Generate semua demand sekaligus
    demands = generate_demand(sim_days)

    # State awal - stok hari pertama langsung tersedia
    current_stock   = M
    pending_orders  = defaultdict(float)
    pending_orders[1] += M  # Stok awal dianggap "datang" di hari 1

    # Tracking hasil per hari
    records = []

    for day in range(1, sim_days + 1):

        # --- 1. Terima pesanan yang tiba hari ini ---
        received = pending_orders.pop(day, 0)
        current_stock += received

        # --- 2. Review period: cek apakah perlu pesan ---
        order_qty       = 0
        order_arrival   = 0
        if day % N == 0:
            if current_stock < M:
                order_qty = M - current_stock
                lead_time = generate_lead_time()
                arrival_day = day + lead_time
                pending_orders[arrival_day] += order_qty
                order_arrival = arrival_day

        # --- 3. Penuhi permintaan harian ---
        demand      = demands[day - 1]
        shortage    = 0
        spoilage    = 0

        if current_stock >= demand:
            current_stock -= demand
            # Nira sisa hari ini EXPIRED (tidak bisa disimpan ke hari berikutnya)
            # karena masa pakai hanya 3 jam (1 shift)
            spoilage        = current_stock
            current_stock   = 0
        else:
            # Stok tidak cukup
            shortage        = demand - current_stock
            current_stock   = 0

        # --- 4. Hitung biaya hari ini ---
        cost_holding    = received * HOLDING_COST          # Biaya simpan stok yang baru datang
        cost_shortage   = shortage * SHORTAGE_COST         # Biaya kekurangan
        cost_spoilage   = spoilage * SPOILAGE_COST         # Biaya nira rusak
        cost_purchase   = order_qty * PURCHASE_COST        # Biaya pembelian (jika ada order)
        total_cost_day  = cost_holding + cost_shortage + cost_spoilage + cost_purchase

        # --- 5. Catat hasil hari ini ---
        records.append({
            "Hari"              : day,
            "Review?"           : "YA" if day % N == 0 else "-",
            "Stok Awal (L)"     : round(received, 2),
            "Pesanan Datang (L)": round(received, 2),
            "Order Baru (L)"    : round(order_qty, 2),
            "ETA Order"         : f"Hari {order_arrival}" if order_qty > 0 else "-",
            "Permintaan (L)"    : round(demand, 2),
            "Shortage (L)"      : round(shortage, 2),
            "Spoilage (L)"      : round(spoilage, 2),
            "Biaya Beli (Rp)"   : cost_purchase,
            "Biaya Simpan (Rp)" : cost_holding,
            "Biaya Shortage (Rp)": cost_shortage,
            "Biaya Spoilage (Rp)": cost_spoilage,
            "Total Biaya/Hari (Rp)": total_cost_day,
        })

    df = pd.DataFrame(records)

    # --- Summary ---
    summary = {
        "Total Hari Simulasi"       : sim_days,
        "M (Stok Maks)"             : M,
        "N (Review Period)"         : N,
        "Total Permintaan (L)"      : df["Permintaan (L)"].sum(),
        "Total Shortage (L)"        : df["Shortage (L)"].sum(),
        "Total Spoilage (L)"        : df["Spoilage (L)"].sum(),
        "Total Biaya Beli (Rp)"     : df["Biaya Beli (Rp)"].sum(),
        "Total Biaya Simpan (Rp)"   : df["Biaya Simpan (Rp)"].sum(),
        "Total Biaya Shortage (Rp)" : df["Biaya Shortage (Rp)"].sum(),
        "Total Biaya Spoilage (Rp)" : df["Biaya Spoilage (Rp)"].sum(),
        "GRAND TOTAL BIAYA (Rp)"    : df["Total Biaya/Hari (Rp)"].sum(),
    }

    return df, summary

# =============================================================================
# BLOK 3: EVALUASI - SENSITIVITY ANALYSIS (variasi M dan N)
# =============================================================================

def evaluate_parameters():
    """
    Jalankan simulasi dengan berbagai kombinasi M dan N.
    Temukan kombinasi dengan total biaya minimum.
    """
    M_values = [100, 125, 150, 175, 200]
    N_values = [1, 2, 3, 5]

    results = []
    for m in M_values:
        for n in N_values:
            _, summary = run_simulation(m, n, SIM_DAYS)
            results.append({
                "M"                         : m,
                "N"                         : n,
                "Total Shortage (L)"        : summary["Total Shortage (L)"],
                "Total Spoilage (L)"        : summary["Total Spoilage (L)"],
                "Grand Total Biaya (Rp)"    : summary["GRAND TOTAL BIAYA (Rp)"],
            })

    eval_df = pd.DataFrame(results)
    best = eval_df.loc[eval_df["Grand Total Biaya (Rp)"].idxmin()]
    return eval_df, best

# =============================================================================
# BLOK 4: VISUALISASI
# =============================================================================

def plot_results(df, summary, eval_df, best):
    """Generate semua grafik hasil simulasi."""

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Simulasi Inventory (M={summary['M (Stok Maks)']}, N={summary['N (Review Period)']}) "
        f"— Nira Aren Perishable\n30 Hari Simulasi",
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    days = df["Hari"]

    # --- Grafik 1: Permintaan vs Shortage vs Spoilage ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(days, df["Permintaan (L)"], label="Permintaan", color="#4C72B0", alpha=0.7)
    ax1.bar(days, df["Shortage (L)"], label="Shortage", color="#DD4444", alpha=0.9)
    ax1.bar(days, df["Spoilage (L)"], label="Spoilage (Terbuang)", color="#FFA500", alpha=0.8, bottom=df["Shortage (L)"])
    ax1.set_title("Permintaan Harian vs Shortage vs Spoilage", fontsize=11)
    ax1.set_xlabel("Hari")
    ax1.set_ylabel("Volume (Liter)")
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # --- Grafik 2: Biaya Harian per Komponen ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.stackplot(
        days,
        df["Biaya Beli (Rp)"] / 1000,
        df["Biaya Simpan (Rp)"] / 1000,
        df["Biaya Shortage (Rp)"] / 1000,
        df["Biaya Spoilage (Rp)"] / 1000,
        labels=["Beli", "Simpan", "Shortage", "Spoilage"],
        colors=["#4C72B0", "#55A868", "#DD4444", "#FFA500"],
        alpha=0.8
    )
    ax2.set_title("Komposisi Biaya Harian", fontsize=11)
    ax2.set_xlabel("Hari")
    ax2.set_ylabel("Biaya (Rp Ribu)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    # --- Grafik 3: Akumulasi Total Biaya ---
    ax3 = fig.add_subplot(gs[1, 1])
    cumulative_cost = df["Total Biaya/Hari (Rp)"].cumsum() / 1_000_000
    ax3.plot(days, cumulative_cost, color="#9B59B6", linewidth=2)
    ax3.fill_between(days, cumulative_cost, alpha=0.2, color="#9B59B6")
    ax3.set_title("Akumulasi Total Biaya", fontsize=11)
    ax3.set_xlabel("Hari")
    ax3.set_ylabel("Biaya Kumulatif (Rp Juta)")
    ax3.grid(alpha=0.3)

    # --- Grafik 4: Heatmap Total Biaya per kombinasi M & N ---
    ax4 = fig.add_subplot(gs[2, 0])
    pivot = eval_df.pivot(index="M", columns="N", values="Grand Total Biaya (Rp)") / 1_000_000
    im = ax4.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
    ax4.set_xticks(range(len(pivot.columns)))
    ax4.set_xticklabels([f"N={n}" for n in pivot.columns])
    ax4.set_yticks(range(len(pivot.index)))
    ax4.set_yticklabels([f"M={m}" for m in pivot.index])
    ax4.set_title("Heatmap Total Biaya per Kombinasi M & N\n(Rp Juta, hijau = lebih murah)", fontsize=10)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax4.text(j, i, f"{pivot.values[i,j]:.1f}", ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, ax=ax4, label="Rp Juta")

    # --- Grafik 5: Bar Total Biaya per Komponen (Summary) ---
    ax5 = fig.add_subplot(gs[2, 1])
    komponen = ["Beli", "Simpan", "Shortage", "Spoilage"]
    nilai = [
        summary["Total Biaya Beli (Rp)"] / 1_000_000,
        summary["Total Biaya Simpan (Rp)"] / 1_000_000,
        summary["Total Biaya Shortage (Rp)"] / 1_000_000,
        summary["Total Biaya Spoilage (Rp)"] / 1_000_000,
    ]
    colors = ["#4C72B0", "#55A868", "#DD4444", "#FFA500"]
    bars = ax5.bar(komponen, nilai, color=colors, alpha=0.85)
    ax5.set_title("Total Biaya per Komponen (30 Hari)", fontsize=11)
    ax5.set_ylabel("Total Biaya (Rp Juta)")
    for bar, val in zip(bars, nilai):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"Rp {val:.2f}M", ha="center", va="bottom", fontsize=9)
    ax5.grid(axis='y', alpha=0.3)

    plt.savefig("hasil_simulasi.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Grafik disimpan: hasil_simulasi.png (di folder yang sama dengan simulation.py)")


# =============================================================================
# MAIN - JALANKAN SEMUA
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  SIMULASI INVENTORY (M, N) - NIRA AREN PERISHABLE")
    print("=" * 60)

    # Jalankan simulasi utama
    df, summary = run_simulation(M, N, SIM_DAYS)

    # Tampilkan tabel simulasi
    print("\n📋 TABEL SIMULASI HARIAN:")
    print(df.to_string(index=False))

    # Tampilkan ringkasan biaya
    print("\n💰 RINGKASAN BIAYA:")
    print("-" * 40)
    for key, val in summary.items():
        if "Rp" in key:
            print(f"  {key:<35}: Rp {val:>15,.0f}")
        else:
            print(f"  {key:<35}: {val}")

    # Evaluasi kombinasi M dan N
    print("\n🔍 EVALUASI KOMBINASI M & N:")
    eval_df, best = evaluate_parameters()
    print(eval_df.to_string(index=False))
    print(f"\n✅ KOMBINASI TERBAIK:")
    print(f"   M = {best['M']} L    |  N = {best['N']} hari")
    print(f"   Grand Total Biaya = Rp {best['Grand Total Biaya (Rp)']:,.0f}")

    # Plot semua grafik
    plot_results(df, summary, eval_df, best)

    print("\n✅ Simulasi selesai.")