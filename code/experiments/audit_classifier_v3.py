"""v3 classifier audit (REVISION_PLAN_v3 A3). Single scaled RF variant, five repeated stratified 80/20 random splits
(seeds 0-4), same protocol as audit_surrogate_classifier.py, plus the boundary-margin and median-range statistics
computed with the SAME scaled model, so no number mixes classifier variants. Writes classifier_audit.json and a text report.
usage: python3 audit_classifier_v3.py <database.csv> <out_dir>
Ranges are on the 160 Wh/kg database basis; the 203.3 Wh/kg basis factor is applied by make_numbers.py."""
import json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score

warnings.filterwarnings("ignore")
DB, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
KEYS = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2",
        "tip_twist", "winglet_sweep_deg", "winglet_twist_deg", "battery_mass", "cruise_speed_ms"]
df = pd.read_csv(DB); df["is_valid"] = df["is_valid"].astype(bool)
X = df[KEYS].values; y = df.is_valid.values
q90 = df[df.is_valid].range_calm_km.quantile(0.90); q98 = df[df.is_valid].range_calm_km.quantile(0.98)
sc = MinMaxScaler().fit(X)


def margin(d):  # distance to nearest observable constraint, as a fraction of the allowed interval (same definition as extra_analyses.py)
    return np.minimum.reduce([(d.SM - 0.05).abs() / 0.20, (0.25 - d.SM).abs() / 0.20, (0.10 - d.Cm_trim.abs()) / 0.10])


rows = []
for seed in range(5):
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(sc.transform(X), y, np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]; pred = p >= 0.5
    te = df.iloc[ite].copy(); te["p"] = p; te["pred"] = pred
    v = te[te.is_valid]; fn = v[~v.pred]; tp = v[v.pred]
    t10 = v[v.range_calm_km >= q90]; t2 = v[v.range_calm_km >= q98]
    rows.append(dict(seed=seed, precision=precision_score(yte, pred), recall=recall_score(yte, pred),
                     recall_top10=float(t10.pred.mean()), n_top10=len(t10), recall_top2=float(t2.pred.mean()), n_top2=len(t2),
                     mean_p_top10=float(t10.p.mean()), fn_best_range=float(fn.range_calm_km.max()),
                     n_test=len(te), n_test_valid=len(v), n_fn=len(fn), n_tp=len(tp),
                     fn_within10=float((margin(fn) < 0.10).mean()), tp_within10=float((margin(tp) < 0.10).mean()),
                     fn_median_range=float(fn.range_calm_km.median()), tp_median_range=float(tp.range_calm_km.median())))
r = pd.DataFrame(rows)
res = dict(n_rows=len(df), n_valid=int(df.is_valid.sum()), share_infeasible=float(1 - df.is_valid.mean()), q90=float(q90), q98=float(q98),
           per_seed=rows, mean={k: float(r[k].mean()) for k in r.columns if k != "seed"},
           sd={k: float(r[k].std(ddof=1)) for k in r.columns if k != "seed"},
           fn_best_range_max=float(r.fn_best_range.max()), fn_best_range_mean=float(r.fn_best_range.mean()))
json.dump(res, open(f"{OUT}/classifier_audit.json", "w"), indent=1)
with open(f"{OUT}/classifier_audit_report.txt", "w") as f:
    f.write("Classifier audit v3: RandomForest 100 trees, MinMax-scaled features, five repeated stratified 80/20 random splits (seeds 0-4)\n")
    f.write(f"Source: {DB} ({len(df)} rows, {int(df.is_valid.sum())} valid, {100*(1-df.is_valid.mean()):.1f}% infeasible)\n")
    f.write(f"Range thresholds (160 Wh/kg basis): top 10% >= {q90:.1f} km, top 2% >= {q98:.1f} km\n\n")
    f.write(r.round(4).to_string(index=False) + "\n\nmean:\n" + str({k: round(v_, 4) for k, v_ in res["mean"].items()}) + "\nsd:\n" + str({k: round(v_, 4) for k, v_ in res["sd"].items()}) + "\n")
    f.write(f"Highest false-negative range: max over splits {res['fn_best_range_max']:.1f} km, mean {res['fn_best_range_mean']:.1f} km (160 basis)\n")
print(open(f"{OUT}/classifier_audit_report.txt").read())
