"""B4 steps 4-5. Fit SM_Euler = a + b * SM_VLM over all Euler CFD designs (results_revision/cfd/*_h7_result.csv and results_v3/cfd/*/result.csv),
with a pairs-bootstrap 95% CI, then re-rank every arm under the calibrated stability constraint SM_cal in [0.05, 0.25] (post hoc, no new optimisation).
SM_Euler = -dCMy/dCL between the two angles of attack (same definition as make_numbers.py); CL agreement is the Euler/VLM ratio at cruise.
Re-ranking only sees designs the original run marked valid (invalid designs were stored with zero range), so the calibrated feasible set is a
subset of the original one; this is conservative for designs the VLM rejected at the lower cap.
usage: python3 b4_calibrate.py OUTDIR"""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
R, V3 = f"{ROOT}/revision_runs/results_revision", f"{ROOT}/revision_runs/results_v3"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{V3}/cfd_calibration"; os.makedirs(OUT, exist_ok=True)
SMLO, SMHI = 0.05, 0.25
rows = []
for f in sorted(glob.glob(f"{R}/cfd/*_h7_result.csv")) + sorted(glob.glob(f"{V3}/cfd/*_h7/result.csv")):
    d = pd.read_csv(f)
    if len(d) < 2: continue
    d = d.sort_values("alpha"); a0, a1 = d.iloc[0], d.iloc[1]
    rows.append(dict(design=a0.design, src=("results_v3" if "results_v3" in f else "results_revision"), SM_euler=-(a1.CMy - a0.CMy) / (a1.CL - a0.CL), SM_vlm=a0.VLM_SM,
                     CL_euler=a0.CL, CL_vlm=a0.VLM_CL_cruise, CL_ratio=a0.CL / a0.VLM_CL_cruise, nodes=int(a0.nodes)))
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/cfd_points.csv", index=False)
n = len(C); x, y = C.SM_vlm.values, C.SM_euler.values
b, a = np.polyfit(x, y, 1); res = y - (a + b * x); sd = res.std(ddof=2) if n > 2 else np.nan
rng = np.random.default_rng(0); bs = []
for _ in range(10000):
    i = rng.integers(0, n, n)
    if np.ptp(x[i]) < 1e-9: continue
    bb, aa = np.polyfit(x[i], y[i], 1); bs.append((aa, bb))
bs = np.array(bs); ci = np.percentile(bs, [2.5, 97.5], axis=0)
ratio = y / x
cal = dict(n_designs=n, a=a, b=b, a_ci=ci[:, 0].tolist(), b_ci=ci[:, 1].tolist(), resid_sd=sd, r2=1 - res.var() / y.var() if n > 2 else None,
           ratio_min=ratio.min(), ratio_max=ratio.max(), ratio_mean=ratio.mean(), CL_ratio_min=C.CL_ratio.min(), CL_ratio_max=C.CL_ratio.max(),
           CL_ratio_median=C.CL_ratio.median(), SM_vlm_cap_equiv=(SMHI - a) / b, SM_vlm_floor_equiv=(SMLO - a) / b)
# re-ranking -----------------------------------------------------------------------------------------------
KEYS = ["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist","winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"]
def load(arm, pat, rc, smc):
    out = []
    for f in sorted(glob.glob(pat)):
        d = pd.read_csv(f); d["is_valid"] = d.is_valid.astype(bool)
        out.append((os.path.basename(f), d.rename(columns={rc: "range", smc: "SM"})[["range", "SM", "is_valid"]]))
    return arm, out
arms = [load("AL (soft)", f"{R}/al/al_seed?.csv", "range_calm_km", "SM"), load("TuRBO", f"{R}/turbo/turbo_seed?.csv", "range_km", "SM"),
        load("TuRBO extra", f"{V3}/turbo_extra/turbo_seed?.csv", "range_km", "SM"), load("TuRBO + feas weight", f"{V3}/turbo_feas/turbo_seed?.csv", "range_km", "SM"),
        load("AL no weight", f"{V3}/al_nofeas/al_seed?.csv", "range_calm_km", "SM"), load("AL hard weight", f"{V3}/al_hard/al_seed?.csv", "range_calm_km", "SM")]
for c in ["adapt10", "local10", "local5", "lhs", "uniform"]:
    arms.append(load(c, f"{R}/controls/{c}_seed?.csv", "range_km", "SM")); arms.append(load(c + " (v3 extra)", f"{V3}/controls_extra/{c}_seed?.csv", "range_km", "SM"))
db = pd.read_csv(f"{ROOT}/flying_wing_optimizer/results/database.csv"); db["is_valid"] = db.is_valid.astype(bool)
arms.append(("original campaign (Phase 2)", [("db", db[~db.evaluation_type.str.startswith("Phase1")].rename(columns={"range_calm_km": "range"})[["range", "SM", "is_valid"]])]))
arms.append(("pool (Phase 1)", [("db", db[db.evaluation_type.str.startswith("Phase1")].rename(columns={"range_calm_km": "range"})[["range", "SM", "is_valid"]])]))
summ, per_run = [], []
for arm, runs in arms:
    runs = [(n_, d) for n_, d in runs if len(d) >= 287 or arm.startswith(("orig", "pool", "AL (soft)"))]
    if not runs: continue
    for n_, d in runs:
        v = d[d.is_valid]
        if v.empty:
            per_run.append(dict(arm=arm, run=n_, best_vlm=0.0, best_cal=0.0, loss=0.0, champ_SM_vlm=np.nan, champ_SM_cal=np.nan, champ_infeasible=False, n_valid=0, n_valid_cal=0)); continue
        smc = a + b * v.SM; ok = v[(smc >= SMLO) & (smc <= SMHI)]
        b0 = v.range.max(); b1 = ok.range.max() if len(ok) else 0.0
        c_sm = v.loc[v.range.idxmax(), "SM"]
        per_run.append(dict(arm=arm, run=n_, best_vlm=b0, best_cal=b1, loss=b0 - b1, champ_SM_vlm=c_sm, champ_SM_cal=a + b * c_sm, champ_infeasible=bool(a + b * c_sm > SMHI or a + b * c_sm < SMLO),
                            n_valid=len(v), n_valid_cal=len(ok)))
P = pd.DataFrame(per_run); P.to_csv(f"{OUT}/rerank_per_run.csv", index=False)
S = P.groupby("arm", sort=False).agg(n=("run", "count"), best_vlm_mean=("best_vlm", "mean"), best_cal_mean=("best_cal", "mean"), loss_mean=("loss", "mean"),
                                     champs_infeasible=("champ_infeasible", "sum"), valid_share_kept=("n_valid_cal", "sum"), n_valid_total=("n_valid", "sum")).reset_index()
S["valid_share_kept"] = S.valid_share_kept / S.n_valid_total; S = S.drop(columns="n_valid_total"); S.to_csv(f"{OUT}/rerank_summary.csv", index=False)
cal["rerank"] = S.to_dict("records"); json.dump(cal, open(f"{OUT}/calibration.json", "w"), indent=1, default=float)
print(C.round(3).to_string()); print({k: (np.round(v, 3).tolist() if isinstance(v, (list, np.ndarray)) else (round(v, 3) if isinstance(v, float) else v)) for k, v in cal.items() if k != "rerank"}); print(S.round(1).to_string())
