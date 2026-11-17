"""Extra analyses for the revised manuscript: (1) ablation re-analysis with planform-area control, (2) geometry monotonicity audit of the
implemented loft over the whole database, (3) classifier false-negative boundary analysis, (4) 5-fold CV of the MLP ensemble for the five targets."""
import sys, json, os, warnings; warnings.filterwarnings("ignore")
ROOT = os.environ.get("DRONE_ROOT", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone"); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scipy import stats
R = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "revision_runs/results_revision")); out = {}
KEYS = ["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist","winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"]
# (1) ablation
a = pd.read_csv(os.path.join(ROOT, "research/paper_b_flagship/results/ablation_database.csv")); a["inv"] = a.c_blend > a.c_root; v = a[a.is_valid.astype(bool)].copy()
x, y = v[v.inv].L_D, v[~v.inv].L_D
mw = stats.mannwhitneyu(x, y); ks = stats.ks_2samp(x, y)
lo, hi = max(v[v.inv].S_ref.min(), v[~v.inv].S_ref.min()), min(v[v.inv].S_ref.max(), v[~v.inv].S_ref.max()); m = v[(v.S_ref >= lo) & (v.S_ref <= hi)]
mw2 = stats.mannwhitneyu(m[m.inv].L_D, m[~m.inv].L_D)
X = np.c_[np.ones(len(v)), v.inv.astype(float), v.S_ref, v.cruise_speed_ms, v.battery_mass, v.SM]; beta = np.linalg.lstsq(X, v.L_D.values, rcond=None)[0]
res = v.L_D.values - X @ beta; cov = np.linalg.inv(X.T @ X) * res.var(ddof=X.shape[1]); se = np.sqrt(np.diag(cov))
out["ablation"] = dict(n=len(a), n_inv=int(a.inv.sum()), valid_inv=int(v.inv.sum()), valid_mono=int((~v.inv).sum()), vr_inv=float(a[a.inv].is_valid.mean()), vr_mono=float(a[~a.inv].is_valid.mean()),
    mean_inv=float(x.mean()), mean_mono=float(y.mean()), med_inv=float(x.median()), med_mono=float(y.median()), max_inv=float(x.max()), max_mono=float(y.max()),
    maxR_inv=float(v[v.inv].range_calm_km.max()), maxR_mono=float(v[~v.inv].range_calm_km.max()), U=float(mw.statistic), p=float(mw.pvalue), D=float(ks.statistic), pks=float(ks.pvalue),
    Sref_inv=float(v[v.inv].S_ref.mean()), Sref_mono=float(v[~v.inv].S_ref.mean()), AR_inv=float((1.5**2 / v[v.inv].S_ref).mean()), AR_mono=float((1.5**2 / v[~v.inv].S_ref).mean()),
    n_match_inv=int(m.inv.sum()), n_match_mono=int((~m.inv).sum()), mean_match_inv=float(m[m.inv].L_D.mean()), mean_match_mono=float(m[~m.inv].L_D.mean()),
    med_match_inv=float(m[m.inv].L_D.median()), med_match_mono=float(m[~m.inv].L_D.median()), p_match=float(mw2.pvalue), ols_inv=float(beta[1]), ols_inv_se=float(se[1]),
    rho_LD_Sref=float(stats.spearmanr(v.S_ref, v.L_D)[0]), sref_lo=float(lo), sref_hi=float(hi))
print(json.dumps(out["ablation"], indent=0)[:1500], flush=True)
# (2) monotonicity audit
from flying_wing_optimizer.geometry import build_wing_geometry
db = pd.read_csv(os.path.join(ROOT, "flying_wing_optimizer/results/database.csv")); db["v"] = db.is_valid.astype(bool)
def chords(r):
    d = r[KEYS].astype(float)
    ap, _ = build_wing_geometry(sweep_inner=d.sweep_inner, sweep_outer=d.sweep_outer, y_blend=d.y_blend, c_root=d.c_root, c_blend=d.c_blend, taper=d.taper, theta0=d.theta0, theta1=d.theta1,
        theta2=d.theta2, tip_twist=d.tip_twist, battery_mass=d.battery_mass, winglet_sweep_deg=d.winglet_sweep_deg, winglet_twist_deg=d.winglet_twist_deg)
    return None if ap is None else np.array([float(s.chord) for s in ap.wings[0].xsecs[:25]])
rows = []
for i, r in db[db.v].iterrows():
    try: c = chords(r)
    except Exception: continue
    if c is None: continue
    dc = np.diff(c); rows.append((dc.max(), c[1] > c[0]))
rows = np.array(rows); mx = rows[:, 0]
out["mono"] = dict(n=len(mx), share_nonmono=float((mx > 1e-9).mean()), share_gt1mm=float((mx > 1e-3).mean()), share_gt5mm=float((mx > 5e-3).mean()), max_mm=float(mx.max() * 1e3),
                   median_mm_nonmono=float(np.median(mx[mx > 1e-9]) * 1e3))
print(out["mono"], flush=True)
# champion
best = db[db.v].nlargest(1, "range_calm_km").iloc[0]; cb = chords(best); out["mono"]["champion_monotone"] = bool((np.diff(cb) <= 1e-9).all())
# (3) classifier false negatives vs constraint margins
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
fn_rows = []
for seed in range(5):
    tr, te = train_test_split(db, test_size=0.2, random_state=seed, stratify=db.v)
    rf = RandomForestClassifier(100, random_state=seed, n_jobs=1).fit(tr[KEYS], tr.v); p = rf.predict(te[KEYS]); t = te[te.v]
    fn = t[p[te.v.values] == 0]; tp = t[p[te.v.values] == 1]
    def margin(d):  # distance to nearest observable constraint, as a fraction of the allowed interval
        return np.minimum.reduce([(d.SM - 0.05).abs() / 0.20, (0.25 - d.SM).abs() / 0.20, (0.10 - d.Cm_trim.abs()) / 0.10]) if "Cm_trim" in d else None
    mf, mt = margin(fn), margin(tp)
    fn_rows.append(dict(seed=seed, n_fn=len(fn), n_tp=len(tp), fn_within10=float((mf < 0.10).mean()), tp_within10=float((mt < 0.10).mean()), fn_median_range=float(fn.range_calm_km.median()), tp_median_range=float(tp.range_calm_km.median()),
                        fn_max_range=float(fn.range_calm_km.max()), recall=float(len(tp) / len(t)), prec=float(((p == 1) & te.v.values).sum() / max((p == 1).sum(), 1))))
out["fn"] = pd.DataFrame(fn_rows).mean().round(4).to_dict(); print(out["fn"], flush=True)
# (4) 5-fold CV of the MLP ensemble
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
vd = db[db.v].reset_index(drop=True); cv = {}
for tgt, col in [("CL","CL"),("CD","CD"),("L_D","L_D"),("range","range_calm_km"),("SM","SM")]:
    r2s, mapes = [], []
    for k, (tri, tei) in enumerate(KFold(5, shuffle=True, random_state=0).split(vd)):
        sx, sy = StandardScaler().fit(vd.loc[tri, KEYS]), StandardScaler().fit(vd.loc[tri, [col]]); P = []
        for s in (42, 101, 2024, 777, 999):
            mm = MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True, max_iter=800, random_state=s).fit(sx.transform(vd.loc[tri, KEYS]), sy.transform(vd.loc[tri, [col]]).ravel())
            P.append(sy.inverse_transform(mm.predict(sx.transform(vd.loc[tei, KEYS])).reshape(-1, 1)).ravel())
        mu = np.mean(P, 0); y = vd.loc[tei, col].values
        r2s.append(1 - ((y - mu)**2).sum() / ((y - y.mean())**2).sum()); mapes.append(np.mean(np.abs((y - mu) / y)) * 100)
    cv[tgt] = dict(r2=float(np.mean(r2s)), r2_sd=float(np.std(r2s, ddof=1)), mape=float(np.mean(mapes))); print(tgt, cv[tgt], flush=True)
out["cv"] = cv
json.dump(out, open(f"{R}/extra_analyses.json", "w"), indent=1)
