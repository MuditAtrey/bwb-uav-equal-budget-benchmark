"""Sensitivity runs on the best-range design of the original campaign (database.csv): airframe-mass multiplier and battery-bay position sweep.
Runs on a patched copy of the package (/tmp/work2) where the airframe mass formula is scaled by env AF_MULT."""
import os, sys, warnings, json; warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/work2")
import numpy as np, pandas as pd
KEYS = ["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist","winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"]
db = pd.read_csv("/tmp/work/flying_wing_optimizer/results/database.csv"); best = db[db.is_valid.astype(bool)].nlargest(1, "range_calm_km").iloc[0]; v = best[KEYS].astype(float).values
SRC = os.environ.get("SENS_SRC"); SUF = ""        # e.g. SENS_SRC=al_seed0.csv:271 selects a design of this study's runs
if SRC:
    fn, idx = SRC.split(":"); d_ = pd.read_csv(f"/mnt/user-data/outputs/results_revision/al/{fn}"); v = d_.loc[int(idx), KEYS].astype(float).values; SUF = "_" + fn[:-4]
from flying_wing_optimizer.main import evaluate_design_vector
import flying_wing_optimizer.config as C, flying_wing_optimizer.geometry as G
rows = []
def ev(tag, **kw):
    r = evaluate_design_vector(v, 0, "sens"); rows.append(dict(tag=tag, **kw, valid=bool(r["is_valid"]), range_km_160=float(r["range_calm_km"]) * 160 / 208.25, L_D=float(r["L_D"]), SM=float(r["SM"]), Cm=float(r["Cm_trim"] if "Cm_trim" in r else r.get("Cm", np.nan)),
                                       m_total=float(r.get("m_total_kg", r.get("m_total", np.nan))), CL=float(r["CL"]))); print(rows[-1], flush=True)
for mult in (0.7, 1.0, 1.3, 1.5):
    os.environ["AF_MULT"] = str(mult); ev("airframe_mult", mult=mult)
os.environ["AF_MULT"] = "1"
for fr in np.round(np.arange(0.50, 0.981, 0.04), 2):
    object.__setattr__(C.MASS_PLACEMENT, "battery_centroid_fraction", float(fr))
    object.__setattr__(G.MASS_PLACEMENT, "battery_centroid_fraction", float(fr)); ev("bay_fraction", frac=float(fr))
pd.DataFrame(rows).to_csv(f"/mnt/user-data/outputs/results_revision/sensitivity_runs{SUF}.csv", index=False)
# re-balanced airframe-mass sensitivity: for each multiplier scan the bay fraction and keep the best valid design
rb = []
for mult in (0.7, 1.0, 1.3, 1.5):
    os.environ["AF_MULT"] = str(mult); best_r = None
    for fr in np.round(np.arange(0.70, 0.991, 0.02), 2):
        object.__setattr__(C.MASS_PLACEMENT, "battery_centroid_fraction", float(fr)); object.__setattr__(G.MASS_PLACEMENT, "battery_centroid_fraction", float(fr))
        r = evaluate_design_vector(v, 0, "sens")
        if r["is_valid"] and (best_r is None or r["range_calm_km"] > best_r["range_calm_km"]): best_r = dict(r); best_r["frac"] = float(fr)
    rb.append(dict(mult=mult, frac=None if best_r is None else best_r["frac"], range_km_160=None if best_r is None else float(best_r["range_calm_km"]) * 160 / 208.25,
                   SM=None if best_r is None else float(best_r["SM"]), L_D=None if best_r is None else float(best_r["L_D"]))); print(rb[-1], flush=True)
pd.DataFrame(rb).to_csv(f"/mnt/user-data/outputs/results_revision/sensitivity_rebalanced{SUF}.csv", index=False)
