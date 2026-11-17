"""Audit of the surrogate ensemble and feasibility classifier on the existing 10,087-row database.

Questions (Reviewer 1, items 3 and 5):
  1. Does the RF feasibility classifier (recall ~55%) drop competitive designs near the front?
  2. Is the MLP surrogate accurate where it matters (top of the range distribution), not just globally?
  3. What actually drives range across the valid designs?
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import precision_score, recall_score, r2_score

warnings.filterwarnings("ignore")
DB = sys.argv[1]
KEYS = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2",
        "tip_twist", "winglet_sweep_deg", "winglet_twist_deg", "battery_mass", "cruise_speed_ms"]
TARGETS = ["CL", "CD", "L_D", "range_calm_km", "penalty_score"]

df = pd.read_csv(DB)
df["is_valid"] = df["is_valid"].astype(bool)
print(f"rows={len(df)} valid={df.is_valid.sum()}")

# ---------------------------------------------------------------- 1. classifier
print("\n=== 1. Feasibility classifier (RF, 100 trees, threshold 0.5) ===")
X = df[KEYS].values; y = df.is_valid.values
sc = MinMaxScaler().fit(X)
res = []
for seed in range(5):
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(sc.transform(X), y, np.arange(len(df)),
                                                          test_size=0.2, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pred = p >= 0.5
    te = df.iloc[idx_te].copy(); te["p"] = p; te["pred"] = pred
    v = te[te.is_valid]
    q90 = df[df.is_valid].range_calm_km.quantile(0.90)
    q98 = df[df.is_valid].range_calm_km.quantile(0.98)
    top10 = v[v.range_calm_km >= q90]; top2 = v[v.range_calm_km >= q98]
    res.append(dict(seed=seed, precision=precision_score(yte, pred), recall=recall_score(yte, pred),
                    recall_top10pct=top10.pred.mean(), n_top10=len(top10),
                    recall_top2pct=top2.pred.mean(), n_top2=len(top2),
                    mean_p_top10=top10.p.mean(),
                    fn_best_range=v[~v.pred].range_calm_km.max()))
r = pd.DataFrame(res)
print(r.round(3).to_string(index=False))
print("mean:", r.drop(columns="seed").mean().round(3).to_dict())
print(f"valid-design range 90th pct = {q90:.1f} km, 98th pct = {q98:.1f} km, max = {df[df.is_valid].range_calm_km.max():.1f} km")

# Weighted by how the AL loop uses the classifier: sigmoid(10*(p-0.5)) multiplies acquisition
w = 1/(1+np.exp(-10*(p-0.5)))
print(f"(last split) mean AL weight on valid top-10% designs = {w[te.is_valid.values & (te.range_calm_km.values>=q90)].mean():.3f}")

# ---------------------------------------------------------------- 2. surrogate accuracy where it matters
print("\n=== 2. MLP ensemble accuracy: global vs top of the range distribution ===")
vd = df[df.is_valid].reset_index(drop=True)
Xv = vd[KEYS].values; Yv = vd[TARGETS].values
out = []
for seed in range(3):
    Xtr, Xte, Ytr, Yte = train_test_split(Xv, Yv, test_size=0.2, random_state=seed)
    xs = MinMaxScaler().fit(Xtr); ys = StandardScaler().fit(Ytr)
    preds = []
    for s in [42, 101, 2024, 777, 999]:
        m = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="adam", max_iter=1000, early_stopping=True,
                         random_state=s).fit(xs.transform(Xtr), ys.transform(Ytr))
        preds.append(ys.inverse_transform(m.predict(xs.transform(Xte))))
    P = np.mean(preds, axis=0)
    rt, rp = Yte[:, 3], P[:, 3]
    top = rt >= np.quantile(vd.range_calm_km, 0.90)
    top5 = np.argsort(-rt)[:max(5, int(0.02 * len(rt)))]
    out.append(dict(seed=seed, n_test=len(rt), R2_global=r2_score(rt, rp),
                    MAE_global=np.mean(np.abs(rt - rp)), MAPE_global=100 * np.mean(np.abs(rt - rp) / rt),
                    n_top10=int(top.sum()), R2_top10=r2_score(rt[top], rp[top]),
                    MAE_top10=np.mean(np.abs(rt[top] - rp[top])), bias_top10=np.mean(rp[top] - rt[top]),
                    spearman_top10=spearmanr(rt[top], rp[top])[0],
                    spearman_toppct=spearmanr(rt[top5], rp[top5])[0]))
o = pd.DataFrame(out)
print(o.round(3).to_string(index=False))
print("mean:", o.drop(columns="seed").mean().round(3).to_dict())

# ---------------------------------------------------------------- 3. what drives range
print("\n=== 3. What drives range among valid designs? (log-decomposition) ===")
g = 9.81
vd["log_eta"] = np.log(vd.eta_total)
vd["log_bfrac"] = np.log(vd.m_battery_kg / vd.m_total_kg)
vd["log_LD"] = np.log(vd.L_D)
vd["log_R"] = np.log(vd.range_calm_km)
chk = vd.log_eta + vd.log_bfrac + vd.log_LD
print("max |log R - (log eta + log bfrac + log L/D + const)| =",
      float(np.abs(vd.log_R - chk - (vd.log_R - chk).mean()).max()))
for name, sub in [("all valid", vd), ("top 10% by range", vd[vd.range_calm_km >= vd.range_calm_km.quantile(0.9)]),
                  ("top 2% by range", vd[vd.range_calm_km >= vd.range_calm_km.quantile(0.98)])]:
    var = sub.log_R.var()
    cov = {c: np.cov(sub[c], sub.log_R)[0, 1] / var for c in ["log_eta", "log_bfrac", "log_LD"]}
    print(f"{name:20s} n={len(sub):4d}  share of Var(log range):  eta={cov['log_eta']:.2f}  battery-fraction={cov['log_bfrac']:.2f}  L/D={cov['log_LD']:.2f}")

top = vd[vd.range_calm_km >= vd.range_calm_km.quantile(0.98)]
print("\nTop 2% designs: fraction within 5% of a bound")
B = dict(sweep_inner=(25, 45), sweep_outer=(25, 45), y_blend=(0.20, 0.35), c_root=(0.42, 0.52), c_blend=(0.25, 0.35),
         taper=(0.55, 0.80), theta0=(0, 4), theta1=(-3, 2), theta2=(-6, 0), tip_twist=(-12, -2),
         winglet_sweep_deg=(15, 50), winglet_twist_deg=(-10, 0), battery_mass=(1.5, 3.5), cruise_speed_ms=(36.11, 61.11))
for k, (lo, hi) in B.items():
    f = (top[k] - lo) / (hi - lo)
    print(f"  {k:18s} median position={f.median():.2f}  frac at lower 5%={np.mean(f < .05):.2f}  frac at upper 5%={np.mean(f > .95):.2f}")
