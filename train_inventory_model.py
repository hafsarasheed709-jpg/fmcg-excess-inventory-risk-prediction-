import json, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

OUT = "analysis_output"; PL = f"{OUT}/plots"
df = pd.read_csv("fmcg_inventory_dataset.csv", parse_dates=["Snapshot_Date"])
T = "Excess_Inventory_Risk_7D"
CAT = ["Product_Category", "Supplier"]
# identifiers and date are NOT used as features (SKU_ID would let the model memorise products)
NUM = [c for c in df.columns if c not in CAT + [T, "Snapshot_Date", "SKU_ID"]]
FEATURES = CAT + NUM

# ---- time-based split (never shuffle time-series data) ----
train = df[df.Snapshot_Date < "2025-07-01"]
val = df[(df.Snapshot_Date >= "2025-07-01") & (df.Snapshot_Date < "2026-01-01")]
test = df[df.Snapshot_Date >= "2026-01-01"]
print(f"train {len(train)} | val {len(val)} | test {len(test)}")

def pipe(model, scale=False):
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT), ("num", num, NUM)])
    return Pipeline([("pre", pre), ("model", model)])

models = {
 "Rule: Projected_Cover_Days_7D > 28 (baseline)": None,
 "Logistic Regression": pipe(LogisticRegression(max_iter=2000), scale=True),
 "Decision Tree (depth 5)": pipe(DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)),
 "Random Forest": pipe(RandomForestClassifier(300, min_samples_leaf=5, random_state=42, n_jobs=-1)),
 "Gradient Boosting": pipe(HistGradientBoostingClassifier(random_state=42)),
}
def score(y, p, prob=None):
    r = dict(Accuracy=accuracy_score(y, p), Precision=precision_score(y, p, zero_division=0),
             Recall=recall_score(y, p), F1=f1_score(y, p))
    r["ROC_AUC"] = roc_auc_score(y, prob) if prob is not None else np.nan
    return r

res = []
for name, m in models.items():
    if m is None:
        p = (val.Projected_Cover_Days_7D > 28).astype(int); r = score(val[T], p)
    else:
        m.fit(train[FEATURES], train[T]); r = score(val[T], m.predict(val[FEATURES]), m.predict_proba(val[FEATURES])[:, 1])
    res.append(dict(Model=name, **r))
val_tbl = pd.DataFrame(res).round(3)
print("\nVALIDATION\n", val_tbl.to_string(index=False))
val_tbl.to_csv(f"{OUT}/model_comparison_validation.csv", index=False)

# pick best by validation F1 among real models, refit on train+val, test once
best = val_tbl[val_tbl.Model.str.startswith("Rule") == False].sort_values("F1", ascending=False).iloc[0].Model
print("\nSelected on validation F1:", best)
final = models[best]; trv = pd.concat([train, val])
final.fit(trv[FEATURES], trv[T])
pt = final.predict(test[FEATURES]); pp = final.predict_proba(test[FEATURES])[:, 1]
test_res = score(test[T], pt, pp)
base_res = score(test[T], (test.Projected_Cover_Days_7D > 28).astype(int))
cm = confusion_matrix(test[T], pt)
summary = dict(selected_model=best, test_rows=len(test), test_risk_rate=round(test[T].mean(), 3),
               model_test={k: round(v, 3) for k, v in test_res.items()},
               baseline_rule_test={k: round(v, 3) for k, v in base_res.items() if k != "ROC_AUC"},
               confusion_matrix=cm.tolist())
json.dump(summary, open(f"{OUT}/final_test_results.json", "w"), indent=2)
print("\nTEST", json.dumps(summary, indent=2))
joblib.dump(dict(model=final, features=FEATURES, cat=CAT, num=NUM, name=best), "inventory_model.pkl")

# ---- plots ----
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 4)); r = val_tbl.iloc[::-1]
plt.barh(r.Model, r.F1, color="#2a6f97"); plt.xlabel("F1 on validation (Jul-Dec 2025)"); plt.title("Model comparison")
plt.tight_layout(); plt.savefig(f"{PL}/model_1_model_comparison.png", dpi=150); plt.close()

plt.figure(figsize=(4.5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Risk", "Risk"], yticklabels=["No Risk", "Risk"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Test confusion matrix\n{best}"); plt.tight_layout()
plt.savefig(f"{PL}/model_2_confusion_matrix.png", dpi=150); plt.close()

from sklearn.inspection import permutation_importance
pi = permutation_importance(final, test[FEATURES], test[T], n_repeats=10, random_state=42, scoring="f1")
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values().tail(12)
plt.figure(figsize=(8, 5)); imp.plot(kind="barh", color="#2a6f97"); plt.title("Top features (permutation importance, test set)")
plt.xlabel("Drop in F1 when shuffled"); plt.tight_layout(); plt.savefig(f"{PL}/model_3_feature_importance.png", dpi=150); plt.close()
imp.iloc[::-1].round(4).to_csv(f"{OUT}/top_features.csv", header=["importance"])

m = df.groupby("Product_Category")[T].mean().sort_values()
plt.figure(figsize=(8, 4)); (m * 100).plot(kind="barh", color="#e07a5f"); plt.xlabel("% of snapshots at risk"); plt.title("Excess-risk rate by category")
plt.tight_layout(); plt.savefig(f"{PL}/model_4_risk_by_category.png", dpi=150); plt.close()

mt = df.groupby(df.Snapshot_Date.dt.to_period("M"))[T].mean() * 100
plt.figure(figsize=(9, 4)); plt.plot(mt.index.astype(str), mt.values, marker="o", color="#1b3a5c")
plt.xticks(rotation=60); plt.ylabel("% at risk"); plt.title("Excess-risk rate over time (Ramadan/Eid and seasons visible)")
plt.tight_layout(); plt.savefig(f"{PL}/model_5_risk_over_time.png", dpi=150); plt.close()
