import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import norm

# =========================================================
# DEMAND GENERATOR
# =========================================================
def generate_demand(sim_days, demand_mean, demand_std, scenario="Normal", seed=42):
    np.random.seed(seed)

    if scenario == "Normal":
        demands = np.random.normal(demand_mean, demand_std, sim_days)

    elif scenario == "Musiman":
        demands = []
        for day in range(1, sim_days + 1):
            d = max(0, np.random.normal(demand_mean, demand_std)) # Pencegahan nilai negatif sebelum multiplier
            if day % 7 in [6, 0]: # Lonjakan akhir pekan
                multiplier = np.random.uniform(1.5, 2.0)
                d *= multiplier
            demands.append(d)
        demands = np.array(demands)

    elif scenario == "Ekstrem":
        demands = np.random.poisson(demand_mean, sim_days).astype(float)
        # Random spikes (anomali)
        spike_days = np.random.choice(range(sim_days), size=max(2, sim_days // 10), replace=False)
        for s in spike_days:
            demands[s] *= np.random.uniform(2.0, 3.5)

    else:
        demands = np.random.normal(demand_mean, demand_std, sim_days)

    demands = np.maximum(demands, 0)
    return np.round(demands, 2)


# =========================================================
# PERIODIC REVIEW (M,N) - METODE LAMA
# =========================================================
def run_periodic_review_simulation(M, N, sim_days, demand_mean, demand_std, lt_min, lt_max, purchase_cost, shortage_cost, spoilage_cost, holding_cost, scenario="Normal", seed=42):
    demands = generate_demand(sim_days, demand_mean, demand_std, scenario, seed)
    current_stock = M
    pending_orders = defaultdict(float)
    pending_orders[1] += M
    records = []

    for day in range(1, sim_days + 1):
        received = pending_orders.pop(day, 0)
        current_stock += received
        order_qty = 0
        arrival = 0

        # Review Period
        if day % N == 0 and current_stock < M:
            order_qty = M - current_stock
            lead_time = int(np.random.uniform(lt_min, lt_max + 1))
            arrival = day + lead_time
            pending_orders[arrival] += order_qty

        demand = demands[day - 1]
        shortage = 0
        spoilage = 0

        # Demand Fulfillment (Perishable)
        if current_stock >= demand:
            current_stock -= demand
            spoilage = current_stock
            current_stock = 0
        else:
            shortage = demand - current_stock
            current_stock = 0

        # Costs
        cost_purchase = order_qty * purchase_cost
        cost_holding = received * holding_cost
        cost_shortage = shortage * shortage_cost
        cost_spoilage = spoilage * spoilage_cost
        total_cost = cost_purchase + cost_holding + cost_shortage + cost_spoilage

        records.append({
            "Hari": day,
            "Review?": "✅" if day % N == 0 else "–",
            "Pesanan Datang (L)": round(received, 2),
            "Order Baru (L)": round(order_qty, 2),
            "ETA Order": f"Hari {arrival}" if order_qty > 0 else "–",
            "Permintaan (L)": round(demand, 2),
            "Shortage (L)": round(shortage, 2),
            "Spoilage (L)": round(spoilage, 2),
            "Biaya Beli (Rp)": int(cost_purchase),
            "Biaya Simpan (Rp)": int(cost_holding),
            "Biaya Shortage (Rp)": int(cost_shortage),
            "Biaya Spoilage (Rp)": int(cost_spoilage),
            "Total Biaya/Hari": int(total_cost)
        })

    df = pd.DataFrame(records)
    summary = build_summary(df)
    return df, summary


# =========================================================
# NEWSVENDOR MODEL - METODE BARU
# =========================================================
def run_newsvendor_simulation(sim_days, demand_mean, demand_std, purchase_cost, shortage_cost, spoilage_cost, holding_cost, scenario="Normal", seed=42):
    demands = generate_demand(sim_days, demand_mean, demand_std, scenario, seed)

    # Hitung Critical Ratio
    Cu = shortage_cost
    Co = spoilage_cost + purchase_cost
    critical_ratio = Cu / (Cu + Co)
    
    # Hitung Optimal Order Quantity (Q*)
    z = norm.ppf(critical_ratio)
    Q_star = demand_mean + z * demand_std
    Q_star = max(0, round(Q_star, 2))

    records = []
    for day in range(1, sim_days + 1):
        demand = demands[day - 1]
        shortage = max(0, demand - Q_star)
        spoilage = max(0, Q_star - demand)

        cost_purchase = Q_star * purchase_cost
        cost_holding = Q_star * holding_cost
        cost_shortage = shortage * shortage_cost
        cost_spoilage = spoilage * spoilage_cost
        total_cost = cost_purchase + cost_holding + cost_shortage + cost_spoilage

        records.append({
            "Hari": day,
            "Permintaan (L)": demand,
            "Q*": Q_star,
            "Shortage (L)": shortage,
            "Spoilage (L)": spoilage,
            "Biaya Beli (Rp)": cost_purchase,
            "Biaya Simpan (Rp)": cost_holding,
            "Biaya Shortage (Rp)": cost_shortage,
            "Biaya Spoilage (Rp)": cost_spoilage,
            "Total Biaya/Hari": total_cost
        })

    df = pd.DataFrame(records)
    summary = build_summary(df)
    summary["Q_optimal"] = Q_star
    summary["Critical Ratio"] = critical_ratio

    return df, summary


# =========================================================
# SUMMARY BUILDER
# =========================================================
def build_summary(df):
    total_demand = df["Permintaan (L)"].sum()
    total_shortage = df["Shortage (L)"].sum()
    service_level = max(0, (1 - total_shortage / total_demand) * 100) if total_demand > 0 else 100

    return {
        "Total Permintaan (L)": total_demand,
        "Total Shortage (L)": total_shortage,
        "Total Spoilage (L)": df["Spoilage (L)"].sum(),
        "Total Biaya Beli (Rp)": df["Biaya Beli (Rp)"].sum(),
        "Total Biaya Simpan (Rp)": df["Biaya Simpan (Rp)"].sum(),
        "Total Biaya Shortage (Rp)": df["Biaya Shortage (Rp)"].sum(),
        "Total Biaya Spoilage (Rp)": df["Biaya Spoilage (Rp)"].sum(),
        "GRAND TOTAL BIAYA (Rp)": df["Total Biaya/Hari"].sum(),
        "Service Level": service_level
    }