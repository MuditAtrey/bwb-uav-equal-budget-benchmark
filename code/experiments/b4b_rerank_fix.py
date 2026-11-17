"""B4b (2026-11-16). Post-hoc re-ranking under the calibrated stability constraint, repeated for the corrected runs.
Uses the frozen calibration of cfd_calibration/calibration.json (no refit). Same rule as b4_calibrate.py: per run, best valid range with the
lattice margin, and best valid range among designs with 0.05 <= a + b*SM_VLM <= 0.25. Arms: TuRBO (turbo_fix), TuRBO + sigmoid weight
(turbo_feas_fix), adapt10 (adapt_fix), AL (results_revision/al), AL-R (al_repair, al_repair_extra), AL hard gate, local10, local5, LHS, uniform.
Writes cfd_calibration/rerank_per_run_fix.csv. usage: python3 b4b_rerank_fix.py"""
import glob, json, os
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
R, V3 = f"{ROOT}/revision_runs/results_revision", f"{ROOT}/revision_runs/results_v3"
cal = json.load(open(f"{V3}/cfd_calibration/calibration.json")); a, b = cal["a"], cal["b"]; SMLO, SMHI = 0.05, 0.25
ARMS = [("AL (soft)", [f"{R}/al/al_seed?.csv"], "range_calm_km", 0), ("AL-R", [f"{V3}/al_repair/al_seed*.csv", f"{V3}/al_repair_extra/al_seed*.csv"], "range_calm_km", 0),
        ("TuRBO", [f"{V3}/turbo_fix/turbo_seed*.csv", f"{V3}/turbo_fix_extra/turbo_seed*.csv"], "range_km", 287), ("TuRBO + feas weight", [f"{V3}/turbo_feas_fix/turbo_seed*.csv"], "range_km", 287),
        ("AL hard weight", [f"{V3}/al_hard/al_seed?.csv"], "range_calm_km", 287), ("adapt10", [f"{V3}/adapt_fix/adapt10_seed*.csv"], "range_km", 287),
        ("local10", [f"{R}/controls/local10_seed?.csv"], "range_km", 287), ("local5", [f"{R}/controls/local5_seed?.csv", f"{V3}/controls_extra/local5_seed?.csv"], "range_km", 287),
        ("lhs", [f"{R}/controls/lhs_seed?.csv", f"{V3}/controls_extra/lhs_seed?.csv"], "range_km", 287), ("uniform", [f"{R}/controls/uniform_seed?.csv", f"{V3}/controls_extra/uniform_seed?.csv"], "range_km", 287)]
rows = []
for arm, pats, rc, nmin in ARMS:
    for f in sorted(sum((glob.glob(p) for p in pats), [])):
        d = pd.read_csv(f)
        if len(d) < nmin: continue
        d = d.iloc[:287]; v = d[d.is_valid.astype(bool)]; run = os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f)
        if v.empty: rows.append(dict(arm=arm, run=run, best_vlm=0.0, best_cal=0.0, loss=0.0, champ_SM_vlm=np.nan, champ_infeasible=False, n_valid=0, n_valid_cal=0)); continue
        smc = a + b * v.SM; ok = v[(smc >= SMLO) & (smc <= SMHI)]; b0 = v[rc].max(); b1 = ok[rc].max() if len(ok) else 0.0; c = v.loc[v[rc].idxmax(), "SM"]
        rows.append(dict(arm=arm, run=run, best_vlm=b0, best_cal=b1, loss=b0 - b1, champ_SM_vlm=c, champ_infeasible=bool(not (SMLO <= a + b * c <= SMHI)), n_valid=len(v), n_valid_cal=len(ok)))
P = pd.DataFrame(rows); P.to_csv(f"{V3}/cfd_calibration/rerank_per_run_fix.csv", index=False)
F = (48 * 5.0 * 3.6 * 0.80 / 3.400) / 160.0
print(P.groupby("arm", sort=False).agg(n=("run", "count"), vlm=("best_vlm", lambda x: x.mean() * F), cal=("best_cal", lambda x: x.mean() * F), loss=("loss", lambda x: x.mean() * F), out=("champ_infeasible", "sum")).round(1).to_string())
