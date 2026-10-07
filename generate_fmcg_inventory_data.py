"""
Data generator: simulates day-by-day stock movement of 40 FMCG SKUs
for a Multan-based distributor, then takes a weekly snapshot per SKU.

Features use ONLY information available on the snapshot date (past sales, current
stock, open purchase orders, planned promotions, calendar).
The target is built from what happens AFTER the snapshot (next 7 days of stock,
next 14 days of realised sales after that), so it is not a formula of the features.
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
START, END = pd.Timestamp("2024-12-01"), pd.Timestamp("2026-10-04")
days = pd.date_range(START, END)
N = len(days)

# ---- calendar events (approximate Islamic calendar dates, synthetic use) ----
ramadan = [("2025-03-01", "2025-03-30"), ("2026-02-18", "2026-03-19")]   # approximate dates
eid = [("2025-03-31", "2025-04-02"), ("2025-06-07", "2025-06-10"),
       ("2026-03-20", "2026-03-22"), ("2026-05-27", "2026-05-30")]
def flag(ranges):
    f = np.zeros(N, bool)
    for a, b in ranges: f |= (days >= a) & (days <= b)
    return f
is_ramadan, is_eid = flag(ramadan), flag(eid)
month = days.month.values
is_summer = np.isin(month, [5, 6, 7, 8, 9])
is_winter = np.isin(month, [11, 12, 1, 2])          # also wedding season

# ---- catalogue ----
categories = {   # base daily demand range, unit cost range (PKR), shelf life (days)
 "Cooking Oil":        ((25, 60), (550, 900), 365),
 "Tea":                ((20, 50), (300, 700), 540),
 "Biscuits & Snacks":  ((40, 120), (40, 150), 180),
 "Juices & Beverages": ((30, 100), (60, 250), 270),
 "Detergent":          ((20, 60), (200, 600), 720),
 "Personal Care":      ((15, 50), (150, 700), 720),
 "Dairy & Dessert Mix": ((15, 45), (100, 400), 120),
 "Rice & Pulses":      ((20, 55), (250, 650), 365),
}
suppliers = {   # mean lead time (days), on-time probability
 "Supplier A (Lahore)": (4, 0.92), "Supplier B (Karachi)": (8, 0.80),
 "Supplier C (Faisalabad)": (5, 0.88), "Supplier D (Multan local)": (2, 0.95),
 "Supplier E (Islamabad)": (6, 0.85), "Supplier F (Sahiwal)": (3, 0.90),
}
sup_names = list(suppliers)

def season_mult(cat):
    m = np.ones(N)
    if cat in ("Cooking Oil", "Rice & Pulses"): m *= np.where(is_ramadan, 1.5, 1) * np.where(is_winter, 1.15, 1)
    if cat == "Tea": m *= np.where(is_ramadan, 1.3, 1) * np.where(is_winter, 1.35, 0.9)
    if cat == "Juices & Beverages": m *= np.where(is_ramadan, 1.8, 1) * np.where(is_summer, 1.5, 0.8)
    if cat == "Dairy & Dessert Mix": m *= np.where(is_ramadan, 1.6, 1) * np.where(is_eid, 1.3, 1)
    if cat == "Biscuits & Snacks": m *= np.where(is_ramadan | is_eid, 1.25, 1)
    if cat == "Personal Care": m *= np.where(is_eid, 1.4, 1)
    return m

rows, sku_no = [], 0
for cat, (dem, cost, shelf) in categories.items():
    for k in range(5):
        sku_no += 1
        sku = f"SKU-{sku_no:03d}"
        sup = sup_names[rng.integers(len(sup_names))]
        lead_mean, ontime = suppliers[sup]
        base = rng.uniform(*dem); unit_cost = round(rng.uniform(*cost), 0)
        sigma = rng.uniform(0.15, 0.40); moq = int(base * rng.choice([7, 10, 14]))
        # promotions: planned in advance, 7 days long, ~12% of weeks
        promo = np.zeros(N, bool)
        for w in range(0, N - 7, 7):
            if rng.random() < 0.12: promo[w:w + 7] = True
        post_dip = np.zeros(N, bool); post_dip[7:] = promo[:-7] & ~promo[7:]
        post_dip = np.convolve(post_dip, np.ones(7), "full")[:N] > 0
        dow = np.array([1.0, 1.0, 1.0, 1.05, 1.15, 1.1, 0.9])[days.dayofweek.values]
        mean_d = base * dow * season_mult(cat) * np.where(promo, 1.4, 1) * np.where(post_dip, 0.85, 1)
        demand = np.maximum(0, np.round(mean_d * rng.lognormal(0, sigma, N))).astype(int)

        stock = base * 20; orders = []        # (arrival_day_index, qty)
        last_qty, last_day = base * 14, 0
        stock_hist, sales_hist, snap = np.zeros(N), np.zeros(N), []
        for t in range(N):
            for o in [o for o in orders if o[0] == t]:
                stock += o[1]; last_qty, last_day = o[1], t
            orders = [o for o in orders if o[0] > t]
            stock_hist[t] = stock                     # stock at start of day t
            s = min(stock, demand[t]); sales_hist[t] = s; stock -= s
            if t >= 28:
                avg28 = sales_hist[t - 27:t + 1].mean()
                pipeline = sum(o[1] for o in orders)
                if stock + pipeline <= avg28 * (lead_mean + 4) and not orders:
                    qty = avg28 * 21 * rng.uniform(0.9, 1.8)
                    if rng.random() < 0.06: qty *= 2.5          # bulk / discount buy
                    qty = int(np.ceil(qty / moq) * moq)
                    lead = lead_mean + (0 if rng.random() < ontime else rng.integers(2, 7))
                    orders.append((t + int(lead), qty))
            snap.append((t, stock, list(orders), last_qty, last_day))

        limit_days = min(28, 0.2 * shelf)
        for t in range(28, N - 22):
            d = days[t]
            if d.dayofweek != 0 or d < pd.Timestamp("2025-01-01"): continue   # Monday snapshots
            _, st_end, ords, lq, ld = snap[t - 1]       # state at start of day t (end of t-1)
            st_now = stock_hist[t]
            ords_now = [o for o in snap[t - 1][2] if o[0] > t]   # orders arriving today are already in stock
            s7, s28 = sales_hist[t - 7:t].mean(), sales_hist[t - 28:t].mean()
            cv28 = sales_hist[t - 28:t].std() / max(s28, 1e-9)
            open_qty = sum(o[1] for o in ords_now)
            nxt = min([o[0] - t for o in ords_now], default=99)
            arriving7 = sum(o[1] for o in ords_now if o[0] - t < 7)          # known purchase orders due within the week
            proj_cover = max(0.0, st_now + arriving7 - 7 * s28) / max(s28, 1e-9)
            # --- FUTURE (never used as a feature) ---
            fut_stock_day7 = stock_hist[t + 7]
            fut_avg_sales = sales_hist[t + 7:t + 21].mean()
            risk = int(fut_stock_day7 > limit_days * max(fut_avg_sales, 1e-9))
            rows.append(dict(
                Snapshot_Date=d.date(), SKU_ID=sku, Product_Category=cat, Supplier=sup,
                Unit_Cost_PKR=unit_cost, Shelf_Life_Days=shelf,
                Current_Stock=int(st_now), Stock_Value_PKR=int(st_now * unit_cost),
                Avg_Daily_Sales_7D=round(s7, 1), Avg_Daily_Sales_28D=round(s28, 1),
                Sales_Trend_Ratio=round(s7 / max(s28, 1e-9), 2), Demand_Variability_28D=round(cv28, 2),
                Days_Of_Cover=round(st_now / max(s28, 1e-9), 1),
                Projected_Cover_Days_7D=round(proj_cover, 1),
                Reorder_Level=int(s28 * (suppliers[sup][0] + 4)),
                Last_Replenishment_Qty=int(lq), Days_Since_Last_Replenishment=int(t - ld),
                Open_Order_Qty=int(open_qty), Days_To_Next_Arrival=int(nxt),
                Supplier_Lead_Time_Days=suppliers[sup][0], Supplier_On_Time_Rate=suppliers[sup][1],
                Promo_Planned_Next_7D=int(promo[t:t + 7].any()), Promo_Ran_Last_7D=int(promo[t - 7:t].any()),
                Ramadan_Eid_Within_14D=int((is_ramadan | is_eid)[t:t + 14].any()),
                Summer_Season=int(is_summer[t]), Winter_Wedding_Season=int(is_winter[t]),
                Month=int(d.month), Week_Of_Year=int(d.isocalendar().week),
                Excess_Inventory_Risk_7D=risk))

df = pd.DataFrame(rows).sort_values(["Snapshot_Date", "SKU_ID"]).reset_index(drop=True)
df.to_csv("fmcg_inventory_dataset.csv", index=False)
print(df.shape, "\nrisk rate:", df.Excess_Inventory_Risk_7D.mean().round(3))
print(df.groupby("Product_Category").Excess_Inventory_Risk_7D.mean().round(2))
print(df.Snapshot_Date.min(), df.Snapshot_Date.max(), df.isna().sum().sum())
