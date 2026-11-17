"""Sensitivity of a design to (i) the battery-bay position (fraction of root chord; the pipeline uses 0.82) and (ii) the airframe-mass model
(multiplier on 1.0 + 1.5 S_ref), with a re-balanced variant (best bay position per multiplier). Runs on the patched copy /tmp/work2.
The Oswald-efficiency gate (e <= 1.0 in production) is relaxed to 1.3 for the mass table only, because heavier airframes raise CL and push
the strip-based Oswald estimate above 1; this is stated in the paper.
usage: python3 sens_all.py NAME [SRC]    NAME = orig | allruns ; SRC = file:index of the design (allruns), default best of database.csv"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/work2")
import numpy as np, pandas as pd
KEYS = ["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist","winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"]
R = "/mnt/user-data/outputs/results_revision"; NAME = sys.argv[1]
if NAME == "orig":
    db = pd.read_csv("/tmp/work/flying_wing_optimizer/results/database.csv"); v = db[db.is_valid.astype(bool)].nlargest(1, "range_calm_km").iloc[0][KEYS].astype(float).values
else:
    fn, idx = sys.argv[2].split(":"); v = pd.read_csv(f"{R}/al/{fn}").loc[int(idx), KEYS].astype(float).values
from flying_wing_optimizer.main import evaluate_design_vector
import flying_wing_optimizer.config as C, flying_wing_optimizer.geometry as G
E = 160.0 / 208.25

def setfrac(fr):
    object.__setattr__(C.MASS_PLACEMENT, "battery_centroid_fraction", float(fr)); object.__setattr__(G.MASS_PLACEMENT, "battery_centroid_fraction", float(fr))

def ev():
    r = evaluate_design_vector(list(v), 0, "sens"); return r

rows = []
os.environ["AF_MULT"] = "1"; os.environ["OSW_MAX"] = "1.0"
for fr in np.round(np.arange(0.50, 0.981, 0.04), 2):
    setfrac(fr); r = ev()
    rows.append(dict(frac=float(fr), valid=bool(r["is_valid"]), range_km_160=float(r["range_calm_km"]) * E if r["is_valid"] else 0.0, L_D=float(r["L_D"]), SM=float(r["SM"]), Cm=float(r["Cm_trim"])))
pd.DataFrame(rows).to_csv(f"{R}/sens_bay_{NAME}.csv", index=False)
os.environ["OSW_MAX"] = "1.3"; mrows = []
for mult in (0.7, 1.0, 1.3, 1.5):
    os.environ["AF_MULT"] = str(mult); setfrac(0.82); r0 = ev(); best = None
    for fr in np.round(np.arange(0.70, 0.991, 0.02), 2):
        setfrac(fr); r = ev()
        if r["is_valid"] and (best is None or r["range_calm_km"] > best[0]["range_calm_km"]): best = (dict(r), float(fr))
    mrows.append(dict(mult=mult, valid_082=bool(r0["is_valid"]), R082=float(r0["range_calm_km"]) * E if r0["is_valid"] else 0.0, LD082=float(r0["L_D"]), SM082=float(r0["SM"]),
                      frac_best=None if best is None else best[1], R_best=None if best is None else float(best[0]["range_calm_km"]) * E, SM_best=None if best is None else float(best[0]["SM"])))
    print(mrows[-1], flush=True)
pd.DataFrame(mrows).to_csv(f"{R}/sens_mass_{NAME}.csv", index=False)
