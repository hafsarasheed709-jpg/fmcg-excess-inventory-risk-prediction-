"""
FMCG Excess Inventory Risk Prediction - Dashboard (Streamlit)
Run:  streamlit run app.py
Needs in the same folder: inventory_model.pkl, fmcg_inventory_dataset.csv
(run train_inventory_model.py first if the .pkl is missing)
"""
import os
from datetime import date
import numpy as np
import pandas as pd
import altair as alt
import joblib
import streamlit as st
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

st.set_page_config(page_title="FMCG Excess Inventory Risk Prediction", page_icon="📦", layout="wide")

NAVY, TEAL, RED, AMBER, GREEN = "#1b3a5c", "#2a6f97", "#d1495b", "#edae49", "#2a9d8f"
LEVELS = ["High", "Medium", "Low"]
LEVEL_SCALE = alt.Scale(domain=LEVELS, range=[RED, AMBER, GREEN])
TARGET = "Excess_Inventory_Risk_7D"
ACTIONS = {
    "High": "Hold new orders; consider promotion or discount to move stock",
    "Medium": "Review open purchase orders; avoid extra bulk buying",
    "Low": "No action needed",
}

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(90deg, #1b3a5c, #2a6f97); padding: 18px 26px;
           border-radius: 12px; color: white; margin-bottom: 14px;}
    .hero h1 {color: white; margin: 0; font-size: 1.9rem;}
    .hero p {margin: 4px 0 0 0; opacity: 0.92;}
    </style>
    <div class="hero">
      <h1>📦 FMCG Excess Inventory Risk Prediction</h1>
      <p>Weekly early-warning system for a Multan FMCG distributor: which products will carry excess stock 7 days from now?</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data & model
@st.cache_resource
def load_bundle():
    return joblib.load("inventory_model.pkl")


@st.cache_data
def load_scored():
    b = joblib.load("inventory_model.pkl")
    d = pd.read_csv("fmcg_inventory_dataset.csv", parse_dates=["Snapshot_Date"])
    d["Risk_Probability"] = b["model"].predict_proba(d[b["features"]])[:, 1]
    d["Risk_Level"] = np.where(d["Risk_Probability"] >= 0.70, "High",
                               np.where(d["Risk_Probability"] >= 0.40, "Medium", "Low"))
    d["Period"] = np.where(d["Snapshot_Date"] < "2025-07-01", "Train",
                           np.where(d["Snapshot_Date"] < "2026-01-01", "Validation", "Test"))
    d["Recommended_Action"] = d["Risk_Level"].map(ACTIONS)
    d["Risk_Pct"] = (d["Risk_Probability"] * 100).round(0)
    return d


try:
    bundle = load_bundle()
    data = load_scored()
except Exception as e:
    st.error("The saved model could not be loaded on this computer (usually a scikit-learn version difference "
             "or a missing file). Fix: open a terminal in this folder, run  python train_inventory_model.py  "
             "once, then run  streamlit run app.py  again.")
    st.code(f"{type(e).__name__}: {e}")
    st.stop()

MODEL, FEATURES, NUM = bundle["model"], bundle["features"], bundle["num"]


def show(builder):
    """Draw an Altair chart; never let one chart crash the whole dashboard."""
    try:
        st.altair_chart(builder())
    except Exception as exc:  # pragma: no cover
        st.warning(f"Chart could not be drawn: {exc}")


def pkr(x):
    return f"PKR {x/1e6:,.1f}M" if abs(x) >= 1e6 else f"PKR {x:,.0f}"


# ------------------------------------------------------------------ sidebar filters
dates = sorted(data["Snapshot_Date"].dt.date.unique())
all_cats = sorted(data["Product_Category"].unique())
all_sups = sorted(data["Supplier"].unique())

with st.sidebar:
    st.header("🎛️ Filters")
    min_d, max_d = dates[0], dates[-1]
    pick = st.date_input("📅 Pick a date (uses the latest weekly snapshot on or before it)", value=max_d,
                         min_value=min_d, max_value=max_d)
    week = max([x for x in dates if x <= pick], default=dates[0]) if pick else dates[-1]
    st.caption(f"Showing snapshot of {week}")
    cats = st.multiselect("Product category", all_cats, default=all_cats)
    sups = st.multiselect("Supplier", all_sups, default=all_sups)
    levels = st.multiselect("Risk level (Risk Monitor)", LEVELS, default=LEVELS)
    min_p = st.selectbox("Minimum risk probability", [0.0, 0.4, 0.7],
                         format_func=lambda v: {0.0: "All (0% and above)", 0.4: "40% and above", 0.7: "70% and above"}[v])
    start_default = max(min_d, date(2026, 1, 1))
    rng = st.date_input("📆 Trend period (start and end date)", value=(start_default, max_d), min_value=min_d, max_value=max_d)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        date_range = rng
    else:
        date_range = (start_default, max_d)
        st.caption("Pick an end date to apply the period.")
    st.divider()
    st.caption(f"Model: {bundle.get('name', 'Gradient Boosting')}")
    st.caption("Risk levels: High 70%+, Medium 40-70%, Low below 40%.")
    st.caption("Data is simulated. Validate with real distributor records before use.")

f = data[data["Product_Category"].isin(cats) & data["Supplier"].isin(sups)]
if f.empty:
    st.warning("No data for the selected filters. Choose at least one category and one supplier.")
    st.stop()

week_ts = pd.Timestamp(week)
cur = f[f["Snapshot_Date"] == week_ts]
prev = f[f["Snapshot_Date"] == week_ts - pd.Timedelta(days=7)]
d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
trend = f[(f["Snapshot_Date"] >= d0) & (f["Snapshot_Date"] <= d1)]
if cur.empty or trend.empty:
    st.warning("No rows for this week or period. Adjust the filters.")
    st.stop()


def kpis(d):
    return {
        "n": len(d),
        "high": int((d["Risk_Level"] == "High").sum()),
        "avg": float(d["Risk_Probability"].mean()),
        "val_high": float(d.loc[d["Risk_Level"] == "High", "Stock_Value_PKR"].sum()),
        "val": float(d["Stock_Value_PKR"].sum()),
        "cover": float(d["Days_Of_Cover"].mean()),
    }


k = kpis(cur)
kp = kpis(prev) if not prev.empty else None


def dlt(key, fmt="{:+,.0f}"):
    return None if kp is None else fmt.format(k[key] - kp[key])


# ------------------------------------------------------------------ KPI cards
st.subheader(f"Week of {week}  |  {len(cats)} categories, {len(sups)} suppliers selected")
cards = st.columns(6)
spec = [
    ("SKUs monitored", f"{k['n']}", None, "off"),
    ("High-risk SKUs", f"{k['high']}", dlt("high"), "inverse"),
    ("Average risk", f"{k['avg']:.0%}", dlt("avg", "{:+.1%}"), "inverse"),
    ("Stock value at high risk", pkr(k["val_high"]), None if kp is None else pkr(k["val_high"] - kp["val_high"]), "inverse"),
    ("Total stock value", pkr(k["val"]), None if kp is None else pkr(k["val"] - kp["val"]), "off"),
    ("Avg days of cover", f"{k['cover']:.1f}", dlt("cover", "{:+.1f}"), "inverse"),
]
for col, (label, value, delta, colour) in zip(cards, spec):
    with col:
        with st.container(border=True):
            st.metric(label, value, delta=delta, delta_color=colour)
st.caption("Arrows compare with the previous week for the same filters (red = getting worse).")

tab_names = ["📊 Overview", "🚨 Risk Monitor", "🎯 Predict Risk", "📈 Trends", "🔍 Product Deep Dive",
             "🧪 Model Performance", "🔮 What-if", "🗂️ Data Explorer"]
t_over, t_mon, t_pred, t_trend, t_sku, t_model, t_what, t_data = st.tabs(tab_names)

# ------------------------------------------------------------------ 1. overview
with t_over:
    top_cat = cur.groupby("Product_Category")["Risk_Probability"].mean().idxmax()
    top3 = cur.sort_values("Risk_Probability", ascending=False).head(3)["SKU_ID"].tolist()
    st.info(
        f"**Key insights:** {k['high']} of {k['n']} SKUs are high risk, with {pkr(k['val_high'])} of stock "
        f"at stake. Highest average risk: **{top_cat}**. First SKUs to act on: **{', '.join(top3)}**."
    )
    a, b = st.columns(2)
    with a:
        st.markdown("**Risk level mix**")
        lv = cur.groupby("Risk_Level").size().reset_index(name="SKUs")
        show(lambda: alt.Chart(lv).mark_arc(innerRadius=60).encode(
            theta="SKUs:Q", color=alt.Color("Risk_Level:N", scale=LEVEL_SCALE),
            tooltip=["Risk_Level", "SKUs"]).properties(height=300))
    with b:
        st.markdown("**Average risk by category**")
        cg = cur.groupby("Product_Category")["Risk_Probability"].mean().reset_index()
        show(lambda: alt.Chart(cg).mark_bar(color=TEAL).encode(
            x=alt.X("Risk_Probability:Q", title="Average risk probability", axis=alt.Axis(format="%")),
            y=alt.Y("Product_Category:N", sort="-x", title=None),
            tooltip=["Product_Category", alt.Tooltip("Risk_Probability:Q", format=".0%")]).properties(height=300))
    c, d = st.columns(2)
    with c:
        st.markdown("**Stock value by category and risk level (PKR)**")
        sv = cur.groupby(["Product_Category", "Risk_Level"])["Stock_Value_PKR"].sum().reset_index()
        show(lambda: alt.Chart(sv).mark_bar().encode(
            x=alt.X("Stock_Value_PKR:Q", title="Stock value (PKR)"),
            y=alt.Y("Product_Category:N", title=None),
            color=alt.Color("Risk_Level:N", scale=LEVEL_SCALE),
            tooltip=["Product_Category", "Risk_Level", alt.Tooltip("Stock_Value_PKR:Q", format=",.0f")]).properties(height=320))
    with d:
        st.markdown("**Days of cover vs projected cover (each dot = one SKU)**")
        show(lambda: alt.Chart(cur).mark_circle(opacity=0.8).encode(
            x=alt.X("Days_Of_Cover:Q", title="Days of cover now"),
            y=alt.Y("Projected_Cover_Days_7D:Q", title="Projected cover in 7 days"),
            size=alt.Size("Stock_Value_PKR:Q", legend=None),
            color=alt.Color("Risk_Level:N", scale=LEVEL_SCALE),
            tooltip=["SKU_ID", "Product_Category", "Current_Stock", "Days_Of_Cover",
                     "Projected_Cover_Days_7D", alt.Tooltip("Risk_Probability:Q", format=".0%")]).properties(height=320))
    st.markdown("**Average risk: category x supplier (selected trend period)**")
    hm = trend.groupby(["Product_Category", "Supplier"])["Risk_Probability"].mean().reset_index()

    def heat():
        base = alt.Chart(hm).encode(x=alt.X("Supplier:N", axis=alt.Axis(labelAngle=-25)),
                                    y=alt.Y("Product_Category:N", title=None))
        rect = base.mark_rect().encode(color=alt.Color("Risk_Probability:Q", scale=alt.Scale(scheme="orangered"),
                                                       legend=alt.Legend(format=".0%")))
        text = base.mark_text(fontSize=11, color="black").encode(text=alt.Text("Risk_Probability:Q", format=".0%"))
        return (rect + text).properties(height=300)
    show(heat)

# ------------------------------------------------------------------ 2. risk monitor
with t_mon:
    view = cur[cur["Risk_Level"].isin(levels) & (cur["Risk_Probability"] >= min_p)] \
        .sort_values("Risk_Probability", ascending=False)
    st.markdown(f"**{len(view)} SKUs match the filters** (week of {week})")
    if view.empty:
        st.info("No SKUs match the risk level / probability filters in the sidebar.")
    else:
        top = view.head(10)
        show(lambda: alt.Chart(top).mark_bar().encode(
            x=alt.X("Risk_Probability:Q", title="Risk probability", axis=alt.Axis(format="%")),
            y=alt.Y("SKU_ID:N", sort="-x", title=None),
            color=alt.Color("Risk_Level:N", scale=LEVEL_SCALE),
            tooltip=["SKU_ID", "Product_Category", alt.Tooltip("Risk_Probability:Q", format=".0%")]).properties(
            height=320, title="Top 10 SKUs by risk"))
        cols = ["SKU_ID", "Product_Category", "Supplier", "Current_Stock", "Days_Of_Cover",
                "Projected_Cover_Days_7D", "Open_Order_Qty", "Days_To_Next_Arrival", "Stock_Value_PKR",
                "Risk_Pct", "Risk_Level", "Recommended_Action"]
        st.dataframe(
            view[cols], hide_index=True,
            column_config={"Risk_Pct": st.column_config.ProgressColumn("Risk %", min_value=0, max_value=100, format="%.0f"),
                           "Stock_Value_PKR": st.column_config.NumberColumn("Stock value (PKR)", format="%d")})
        st.download_button("⬇️ Download this list (CSV)", data=view[cols].to_csv(index=False).encode("utf-8"),
                           file_name=f"risk_list_{week}.csv", mime="text/csv")
        st.markdown("**Recommended actions summary**")
        summary = view.groupby(["Risk_Level", "Recommended_Action"]).agg(
            SKUs=("SKU_ID", "count"), Stock_Value_PKR=("Stock_Value_PKR", "sum")).reset_index()
        st.dataframe(summary, hide_index=True)

# ------------------------------------------------------------------ predict risk (manual entry)
with t_pred:
    st.write("Enter the details of any product to predict its risk of excess inventory 7 days ahead.")
    sup_info = data.groupby("Supplier")[["Supplier_Lead_Time_Days", "Supplier_On_Time_Rate"]].first()
    with st.form("predict_form"):
        a, b, c = st.columns(3)
        with a:
            p_cat = st.selectbox("Product category", all_cats, key="p_cat")
            p_sup = st.selectbox("Supplier", all_sups, key="p_sup")
            p_date = st.date_input("Prediction date", value=max_d, key="p_date")
            p_cost = st.number_input("Unit cost (PKR)", min_value=1, value=400)
            p_shelf = st.number_input("Shelf life (days)", min_value=1, value=365)
        with b:
            p_stock = st.number_input("Current stock (units)", min_value=0, value=400)
            p_s7 = st.number_input("Average daily sales, last 7 days", min_value=0.0, value=40.0)
            p_s28 = st.number_input("Average daily sales, last 28 days", min_value=0.1, value=40.0)
            p_var = st.number_input("Demand variability (0 = steady, 1 = very unstable)", min_value=0.0, max_value=2.0, value=0.25, step=0.05)
            p_open = st.number_input("Open order quantity (units on the way)", min_value=0, value=0)
        with c:
            p_arr = st.number_input("Days until next delivery (99 = none)", min_value=0, max_value=99, value=99)
            p_lastq = st.number_input("Last delivery quantity (units)", min_value=0, value=300)
            p_since = st.number_input("Days since last delivery", min_value=0, value=10)
            p_promo = st.selectbox("Promotion planned in the next 7 days?", ["No", "Yes"])
            p_ended = st.selectbox("Promotion ran in the last 7 days?", ["No", "Yes"])
            p_fest = st.selectbox("Ramadan / Eid within 14 days?", ["No", "Yes"])
        go_p = st.form_submit_button("🎯 Predict risk")

    if go_p:
        lead = float(sup_info.loc[p_sup, "Supplier_Lead_Time_Days"])
        proj = round(max(0.0, p_stock + (p_open if p_arr < 7 else 0) - 7 * p_s28) / p_s28, 1)
        vals = {
            "Product_Category": p_cat, "Supplier": p_sup, "Unit_Cost_PKR": p_cost, "Shelf_Life_Days": p_shelf,
            "Current_Stock": p_stock, "Stock_Value_PKR": p_stock * p_cost, "Avg_Daily_Sales_7D": p_s7,
            "Avg_Daily_Sales_28D": p_s28, "Sales_Trend_Ratio": round(p_s7 / p_s28, 2), "Demand_Variability_28D": p_var,
            "Days_Of_Cover": round(p_stock / p_s28, 1), "Projected_Cover_Days_7D": proj,
            "Reorder_Level": p_s28 * (lead + 4), "Last_Replenishment_Qty": p_lastq, "Days_Since_Last_Replenishment": p_since,
            "Open_Order_Qty": p_open, "Days_To_Next_Arrival": p_arr, "Supplier_Lead_Time_Days": lead,
            "Supplier_On_Time_Rate": float(sup_info.loc[p_sup, "Supplier_On_Time_Rate"]),
            "Promo_Planned_Next_7D": 1 if p_promo == "Yes" else 0, "Promo_Ran_Last_7D": 1 if p_ended == "Yes" else 0,
            "Ramadan_Eid_Within_14D": 1 if p_fest == "Yes" else 0,
            "Summer_Season": 1 if p_date.month in (5, 6, 7, 8, 9) else 0,
            "Winter_Wedding_Season": 1 if p_date.month in (11, 12, 1, 2) else 0,
            "Month": p_date.month, "Week_Of_Year": p_date.isocalendar()[1]}
        x = pd.DataFrame([vals])[FEATURES]
        x[NUM] = x[NUM].astype(float)
        pr = float(MODEL.predict_proba(x)[0, 1])
        lv = "High" if pr >= 0.70 else ("Medium" if pr >= 0.40 else "Low")
        st.subheader("Prediction result")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Risk probability", f"{pr:.0%}")
        q2.metric("Risk level", lv)
        q3.metric("Days of cover now", f"{vals['Days_Of_Cover']:.1f}")
        q4.metric("Projected cover in 7 days", f"{proj:.1f}")
        st.progress(min(max(pr, 0.0), 1.0))
        (st.error if lv == "High" else st.warning if lv == "Medium" else st.success)(f"{lv} risk of excess inventory. {ACTIONS[lv]}")
        st.caption("Projected cover above 28 days is the strongest warning sign the model uses. Prediction is for 7 days ahead of the chosen date.")

# ------------------------------------------------------------------ 3. trends
with t_trend:
    st.caption("Predictions before January 2026 are on weeks the model was trained on, so they look better than "
               "real performance. The dashed line marks the start of the untouched test period.")
    agg = trend.groupby("Snapshot_Date").agg(Actual=(TARGET, "mean"), Predicted=("Risk_Probability", "mean")).reset_index()
    long = agg.melt("Snapshot_Date", ["Actual", "Predicted"], var_name="Series", value_name="Rate")
    cut = pd.DataFrame({"d": [pd.Timestamp("2026-01-01")]})

    def trend_chart():
        line = alt.Chart(long).mark_line().encode(
            x=alt.X("Snapshot_Date:T", title=None), y=alt.Y("Rate:Q", title="Share of SKUs at risk", axis=alt.Axis(format="%")),
            color=alt.Color("Series:N", scale=alt.Scale(range=[NAVY, AMBER])),
            tooltip=["Snapshot_Date:T", "Series", alt.Tooltip("Rate:Q", format=".1%")])
        rule = alt.Chart(cut).mark_rule(strokeDash=[4, 4], color="gray").encode(x="d:T")
        return (line + rule).properties(height=300, title="Actual vs predicted risk rate")
    show(trend_chart)

    hv = trend.assign(HV=np.where(trend["Risk_Level"] == "High", trend["Stock_Value_PKR"], 0)) \
        .groupby("Snapshot_Date")["HV"].sum().reset_index(name="High_Risk_Stock_PKR")
    show(lambda: alt.Chart(hv).mark_area(color=RED, opacity=0.5, line={"color": RED}).encode(
        x=alt.X("Snapshot_Date:T", title=None), y=alt.Y("High_Risk_Stock_PKR:Q", title="Stock value at high risk (PKR)"),
        tooltip=["Snapshot_Date:T", alt.Tooltip("High_Risk_Stock_PKR:Q", format=",.0f")]).properties(
        height=260, title="Stock value sitting in high-risk SKUs"))

    m = trend.assign(Month=trend["Snapshot_Date"].dt.to_period("M").astype(str)) \
        .groupby(["Month", "Product_Category"])[TARGET].mean().reset_index()
    show(lambda: alt.Chart(m).mark_rect().encode(
        x=alt.X("Month:N", axis=alt.Axis(labelAngle=-60), title=None), y=alt.Y("Product_Category:N", title=None),
        color=alt.Color(f"{TARGET}:Q", scale=alt.Scale(scheme="orangered"), title="Actual risk rate"),
        tooltip=["Month", "Product_Category", alt.Tooltip(f"{TARGET}:Q", format=".0%")]).properties(
        height=300, title="Seasonality: actual risk rate by month and category"))

    rows = []
    for col, label in [("Promo_Planned_Next_7D", "Promotion planned"), ("Promo_Ran_Last_7D", "Promotion just ended"),
                       ("Ramadan_Eid_Within_14D", "Ramadan/Eid near"), ("Summer_Season", "Summer"),
                       ("Winter_Wedding_Season", "Winter / wedding")]:
        for v, g in trend.groupby(col):
            rows.append({"Factor": label, "Condition": "Yes" if v == 1 else "No", "Risk_rate": g[TARGET].mean()})
    fx = pd.DataFrame(rows)
    show(lambda: alt.Chart(fx).mark_bar().encode(
        x=alt.X("Factor:N", title=None, axis=alt.Axis(labelAngle=-20)),
        y=alt.Y("Risk_rate:Q", title="Actual risk rate", axis=alt.Axis(format="%")),
        color=alt.Color("Condition:N", scale=alt.Scale(domain=["No", "Yes"], range=["#9aa5b1", TEAL])),
        xOffset="Condition:N",
        tooltip=["Factor", "Condition", alt.Tooltip("Risk_rate:Q", format=".1%")]).properties(
        height=280, title="Actual risk rate with and without each factor"))

# ------------------------------------------------------------------ 4. product deep dive
with t_sku:
    sku = st.selectbox("Select SKU", sorted(f["SKU_ID"].unique()))
    h = f[f["SKU_ID"] == sku].sort_values("Snapshot_Date")
    r = h[h["Snapshot_Date"] <= week_ts].iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Category", r["Product_Category"])
    c2.metric("Current stock", f"{int(r['Current_Stock']):,}")
    c3.metric("Days of cover", f"{r['Days_Of_Cover']:.1f}")
    c4.metric("Risk probability", f"{r['Risk_Probability']:.0%}")
    c5.metric("Risk level", r["Risk_Level"])
    st.caption(f"{r['Supplier']} | shelf life {int(r['Shelf_Life_Days'])} days | unit cost PKR {r['Unit_Cost_PKR']:,.0f} | "
               f"open order {int(r['Open_Order_Qty'])} units")
    st.success(f"Recommended action: {r['Recommended_Action']}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Stock and 28-day sales rate**")
        st.line_chart(h.set_index("Snapshot_Date")[["Current_Stock", "Avg_Daily_Sales_28D"]])
    with right:
        st.markdown("**Predicted risk probability over time**")
        st.line_chart(h.set_index("Snapshot_Date")[["Risk_Probability"]])
    st.markdown("**Projected cover (days) over time**")
    st.area_chart(h.set_index("Snapshot_Date")[["Projected_Cover_Days_7D"]])
    st.markdown("**Last 12 weeks**")
    st.dataframe(h.tail(12)[["Snapshot_Date", "Current_Stock", "Avg_Daily_Sales_7D", "Days_Of_Cover",
                             "Open_Order_Qty", "Risk_Pct", "Risk_Level", TARGET]].sort_values("Snapshot_Date", ascending=False),
                 hide_index=True)

# ------------------------------------------------------------------ 5. model performance
with t_model:
    test = data[data["Period"] == "Test"]
    st.caption(f"Evaluated on the untouched test period (January 2026 onward, {len(test):,} rows, "
               f"{test[TARGET].mean():.0%} at risk). All categories and suppliers are included.")
    thr = st.selectbox("Decision threshold (flag a SKU when risk probability is above)", [0.3, 0.4, 0.5, 0.6, 0.7], index=2,
                       format_func=lambda v: f"{v:.0%}")
    y, p = test[TARGET].values, test["Risk_Probability"].values
    pred = (p >= thr).astype(int)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{accuracy_score(y, pred):.1%}")
    m2.metric("Precision", f"{precision_score(y, pred, zero_division=0):.1%}")
    m3.metric("Recall", f"{recall_score(y, pred):.1%}")
    m4.metric("F1 score", f"{f1_score(y, pred):.2f}")
    m5.metric("ROC AUC", f"{roc_auc_score(y, p):.3f}")
    st.caption("Precision: of the flagged SKUs, how many were truly at risk. Recall: of the truly at-risk SKUs, "
               "how many were caught. A lower threshold catches more risk but gives more false alarms.")
    a, b = st.columns(2)
    with a:
        cm = confusion_matrix(y, pred)
        cmd = pd.DataFrame({"Actual": ["No risk", "No risk", "Risk", "Risk"],
                            "Predicted": ["No risk", "Risk", "No risk", "Risk"],
                            "Count": [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]})

        def cm_chart():
            base = alt.Chart(cmd).encode(x=alt.X("Predicted:N", sort=["No risk", "Risk"]), y=alt.Y("Actual:N", sort=["No risk", "Risk"]))
            return (base.mark_rect().encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None)) +
                    base.mark_text(fontSize=20, color="black").encode(text="Count:Q")).properties(height=280, title="Confusion matrix")
        show(cm_chart)
    with b:
        fpr, tpr, _ = roc_curve(y, p)
        roc = pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr})
        diag = pd.DataFrame({"False positive rate": [0, 1], "True positive rate": [0, 1]})

        def roc_chart():
            line = alt.Chart(roc).mark_line(color=TEAL).encode(x="False positive rate:Q", y="True positive rate:Q")
            dg = alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(x="False positive rate:Q", y="True positive rate:Q")
            return (line + dg).properties(height=280, title="ROC curve")
        show(roc_chart)
    c, d = st.columns(2)
    with c:
        ths = np.arange(0.05, 0.96, 0.05)
        sweep = pd.DataFrame([{"Threshold": t, "Precision": precision_score(y, (p >= t).astype(int), zero_division=0),
                               "Recall": recall_score(y, (p >= t).astype(int)), "F1": f1_score(y, (p >= t).astype(int))} for t in ths])
        sw = sweep.melt("Threshold", ["Precision", "Recall", "F1"], var_name="Metric", value_name="Value")
        show(lambda: alt.Chart(sw).mark_line().encode(
            x=alt.X("Threshold:Q"), y=alt.Y("Value:Q", axis=alt.Axis(format="%")), color="Metric:N",
            tooltip=["Metric", "Threshold", alt.Tooltip("Value:Q", format=".1%")]).properties(height=280, title="Threshold trade-off"))
    with d:
        hist = test.assign(Actual=np.where(test[TARGET] == 1, "Actually at risk", "Actually fine"))
        show(lambda: alt.Chart(hist).mark_bar(opacity=0.7).encode(
            x=alt.X("Risk_Probability:Q", bin=alt.Bin(maxbins=20), title="Predicted risk probability"),
            y=alt.Y("count()", title="SKU-weeks", stack=None), color="Actual:N").properties(height=280, title="Probability separation"))
    left, right = st.columns(2)
    with left:
        st.markdown("**Model comparison (validation period)**")
        if os.path.exists("analysis_output/model_comparison_validation.csv"):
            st.dataframe(pd.read_csv("analysis_output/model_comparison_validation.csv"), hide_index=True)
        else:
            st.info("Run train_inventory_model.py to create the comparison table.")
    with right:
        st.markdown("**What the model relies on (permutation importance)**")
        if os.path.exists("analysis_output/top_features.csv"):
            tf = pd.read_csv("analysis_output/top_features.csv", index_col=0)
            st.bar_chart(tf["importance"])
        else:
            st.info("Run train_inventory_model.py to create the feature importance file.")
    st.caption("Importance shows what the model uses, not what causes excess stock.")

# ------------------------------------------------------------------ 6. what-if
with t_what:
    st.write("Pick a SKU and week (sidebar), change stock, open orders or promotion, and see how the risk changes.")
    sku_w = st.selectbox("SKU for what-if", sorted(data["SKU_ID"].unique()), key="whatif_sku")
    base = data[(data["SKU_ID"] == sku_w) & (data["Snapshot_Date"] == week_ts)].iloc[0]
    st.caption(f"{base['Product_Category']} | {base['Supplier']} | 28-day average sales: {base['Avg_Daily_Sales_28D']}/day")

    def make_row(stock, open_qty, arrival, sales7, promo):
        row = base[FEATURES].copy()
        a28 = max(float(base["Avg_Daily_Sales_28D"]), 1e-9)
        row["Current_Stock"] = stock
        row["Stock_Value_PKR"] = stock * base["Unit_Cost_PKR"]
        row["Open_Order_Qty"] = open_qty
        row["Days_To_Next_Arrival"] = arrival
        row["Avg_Daily_Sales_7D"] = sales7
        row["Sales_Trend_Ratio"] = round(sales7 / a28, 2)
        row["Days_Of_Cover"] = round(stock / a28, 1)
        row["Projected_Cover_Days_7D"] = round(max(0.0, stock + (open_qty if arrival < 7 else 0) - 7 * a28) / a28, 1)
        row["Promo_Planned_Next_7D"] = promo
        out = pd.DataFrame([row])
        out[NUM] = out[NUM].astype(float)
        return out

    with st.form("whatif_form"):
        a, b = st.columns(2)
        with a:
            stock = st.number_input("Current stock (units)", min_value=0, value=int(base["Current_Stock"]))
            open_qty = st.number_input("Open order quantity (units on the way)", min_value=0, value=int(base["Open_Order_Qty"]))
            arrival = st.number_input("Days until next delivery (99 = none)", min_value=0, max_value=99,
                                      value=int(base["Days_To_Next_Arrival"]))
        with b:
            sales7 = st.number_input("Average daily sales, last 7 days", min_value=0.0, value=float(base["Avg_Daily_Sales_7D"]))
            promo = st.selectbox("Promotion planned in the next 7 days?", ["No", "Yes"], index=int(base["Promo_Planned_Next_7D"]))
        go = st.form_submit_button("Predict risk")

    if go:
        promo_v = 1 if promo == "Yes" else 0
        p_now = float(MODEL.predict_proba(make_row(stock, open_qty, arrival, sales7, promo_v))[0, 1])
        lvl = "High" if p_now >= 0.70 else ("Medium" if p_now >= 0.40 else "Low")
        r1, r2 = st.columns(2)
        r1.metric("Probability of excess inventory in 7 days", f"{p_now:.0%}")
        r2.metric("Risk level", lvl)
        (st.error if lvl == "High" else st.warning if lvl == "Medium" else st.success)(f"{lvl} risk. {ACTIONS[lvl]}")
        mult = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
        sens = pd.DataFrame({
            "Stock": [int(stock * m_) for m_ in mult],
            "Risk_probability": [float(MODEL.predict_proba(make_row(int(stock * m_), open_qty, arrival, sales7, promo_v))[0, 1]) for m_ in mult]})
        show(lambda: alt.Chart(sens).mark_line(point=True, color=RED).encode(
            x=alt.X("Stock:Q", title="Current stock (units)"), y=alt.Y("Risk_probability:Q", title="Risk probability", axis=alt.Axis(format="%")),
            tooltip=["Stock", alt.Tooltip("Risk_probability:Q", format=".0%")]).properties(
            height=260, title="How risk changes as stock changes (other inputs fixed)"))

# ------------------------------------------------------------------ 7. data explorer
with t_data:
    q = st.text_input("Search SKU ID or category (optional)")
    ex = trend
    if q:
        ex = ex[ex["SKU_ID"].str.contains(q, case=False) | ex["Product_Category"].str.contains(q, case=False)]
    st.markdown(f"**{len(ex):,} rows** (selected categories, suppliers and trend period)")
    st.dataframe(ex.drop(columns=["Recommended_Action", "Risk_Pct"]).sort_values(["Snapshot_Date", "SKU_ID"], ascending=[False, True]),
                 hide_index=True)
    st.download_button("⬇️ Download these rows (CSV)", data=ex.to_csv(index=False).encode("utf-8"),
                       file_name="inventory_data_filtered.csv", mime="text/csv")
    with st.expander("Numerical summary"):
        st.dataframe(ex.select_dtypes("number").describe().T.round(2))
    with st.expander("About this data"):
        st.write("One row = one product (SKU) on one Monday. The target `Excess_Inventory_Risk_7D` is 1 if, 7 days "
                 "later, stock is above the allowed days of cover. Features use only information known on Monday. "
                 "Data is simulated from a day-by-day stock model; see docs/Industry_Project_Guide.md.")

st.divider()
st.caption("Academic project: Excess Inventory Risk Prediction for a Multan FMCG distributor. "
           "Simulated data; results must be validated with real business data before practical use.")
