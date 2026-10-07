import os, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

OUT, PL = "analysis_output", "analysis_output/plots"
os.makedirs(PL, exist_ok=True); sns.set_theme(style="whitegrid")
df = pd.read_csv("fmcg_inventory_dataset.csv", parse_dates=["Snapshot_Date"])
T = "Excess_Inventory_Risk_7D"
num = [c for c in df.select_dtypes("number").columns if c != T]

rep = []
def add(t, c): rep.append(f"\n===== {t} =====\n{c}\n")
add("1. DATASET OVERVIEW", f"Rows: {len(df)}, Columns: {df.shape[1]}\nSKUs: {df.SKU_ID.nunique()}\nPeriod: {df.Snapshot_Date.min().date()} to {df.Snapshot_Date.max().date()}\n\n{df.dtypes.to_string()}")
add("2. MISSING VALUES", df.isnull().sum().to_string())
add("3. DUPLICATES", f"Duplicate rows: {df.duplicated().sum()}\nDuplicate (date, SKU) pairs: {df.duplicated(['Snapshot_Date','SKU_ID']).sum()}")
add("4. TARGET DISTRIBUTION", f"{df[T].value_counts().to_string()}\n\n{(df[T].value_counts(normalize=True)*100).round(1).to_string()}")
add("5. NUMERICAL SUMMARY", df[num].describe().T.round(2).to_string())
add("6. RISK RATE BY CATEGORY (%)", (df.groupby('Product_Category')[T].mean()*100).round(1).sort_values().to_string())
add("7. RISK RATE BY SUPPLIER (%)", (df.groupby('Supplier')[T].mean()*100).round(1).sort_values().to_string())
corr = df[num + [T]].corr()[T].drop(T).sort_values(ascending=False)
add("8. CORRELATION WITH TARGET", corr.round(3).to_string())
open(f"{OUT}/analysis_report.txt", "w", encoding="utf-8").write("".join(rep))
df[num].describe().T.round(2).to_csv(f"{OUT}/numerical_summary.csv")

plt.figure(figsize=(5, 4)); ax = sns.countplot(x=df[T]); ax.bar_label(ax.containers[0]); ax.set_xticklabels(["No Risk", "Risk"])
plt.title("Target distribution"); plt.tight_layout(); plt.savefig(f"{PL}/eda_1_target_distribution.png", dpi=150); plt.close()

keep = ["Days_Of_Cover", "Projected_Cover_Days_7D", "Avg_Daily_Sales_28D", "Demand_Variability_28D", "Open_Order_Qty", "Current_Stock"]
df[keep].hist(figsize=(11, 6), bins=25, edgecolor="black"); plt.suptitle("Key numerical features"); plt.tight_layout()
plt.savefig(f"{PL}/eda_2_feature_histograms.png", dpi=150); plt.close()

cols = ["Days_Of_Cover", "Projected_Cover_Days_7D", "Avg_Daily_Sales_28D", "Sales_Trend_Ratio", "Demand_Variability_28D",
        "Open_Order_Qty", "Days_To_Next_Arrival", "Promo_Planned_Next_7D", "Promo_Ran_Last_7D", "Ramadan_Eid_Within_14D", T]
plt.figure(figsize=(10, 8)); sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
plt.title("Correlation heatmap (key features)"); plt.tight_layout(); plt.savefig(f"{PL}/eda_3_correlation_heatmap.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(x=df[T], y=df["Projected_Cover_Days_7D"], ax=ax[0]); ax[0].set_title("Projected cover by risk")
sns.boxplot(x=df[T], y=df["Days_Of_Cover"], ax=ax[1]); ax[1].set_title("Days of cover by risk")
plt.tight_layout(); plt.savefig(f"{PL}/eda_4_boxplots_by_risk.png", dpi=150); plt.close()

pt = df.groupby(["Promo_Planned_Next_7D", "Ramadan_Eid_Within_14D"])[T].mean().mul(100).unstack()
pt.index = ["No promo", "Promo planned"]; pt.columns = ["No Ramadan/Eid", "Ramadan/Eid near"]
plt.figure(figsize=(6, 3.5)); sns.heatmap(pt, annot=True, fmt=".0f", cmap="OrRd"); plt.title("% at risk: promotion x Ramadan/Eid")
plt.tight_layout(); plt.savefig(f"{PL}/eda_5_promo_season.png", dpi=150); plt.close()
print("".join(rep))
