"""Aggregate properties of (a) the best design of the original campaign and (b) the best design over all runs of this study
(database, AL replications, controls; all in the paper bounds). Writes best_designs.json. The parameter vectors are not written out
(IP firewall); only the source file and row index, and aggregate quantities."""
import sys, os, json, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/work"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from cfd_run import vlm_ref
from cfd_geom import KEYS
R = "/mnt/user-data/outputs/results_revision"; E = 160.0 / 208.25
db = pd.read_csv("/tmp/work/flying_wing_optimizer/results/database.csv"); dv = db[db.is_valid.astype(bool)]
cands = [("original", "database.csv", dv.loc[dv.range_calm_km.idxmax()], float(dv.range_calm_km.max()))]
best = None
for f in sorted(glob.glob(f"{R}/al/al_seed?.csv")) + sorted(glob.glob(f"{R}/controls/*.csv")):
    d = pd.read_csv(f); d = d[d.is_valid.astype(bool)]
    col = "range_calm_km" if "range_calm_km" in d else "range_km"
    i = d[col].idxmax(); r = float(d.loc[i, col])
    if best is None or r > best[3]: best = ("allruns", os.path.basename(f) + f":{i}", d.loc[i], r)
if best[3] <= cands[0][3]: best = ("allruns", cands[0][1], cands[0][2], cands[0][3])
cands.append(best)
out = {}
for name, src, row, r160 in cands:
    ref = vlm_ref(row[KEYS].astype(float).values)
    out[name] = dict(source=src, range_160=r160, **{k: float(ref[k]) for k in ["CL", "CD", "CD_vlm", "CD_profile", "CD_trim", "L_D", "SM", "Cm_cruise", "S_ref", "m_total", "V_cruise_actual", "airframe_mass", "AoA_cruise"]},
                     battery_mass=float(row["battery_mass"]), cruise_speed_ms=float(row["cruise_speed_ms"]))
    from flying_wing_optimizer.main import evaluate_design_vector
    v2 = row[KEYS].astype(float).values.copy(); v2[KEYS.index("battery_mass")] = 3.400
    r2 = evaluate_design_vector(list(v2), 0, "snap")
    out[name].update(snapped_valid=bool(r2["is_valid"]), snapped_range_160=float(r2["range_calm_km"]) * E, snapped_SM=float(r2["SM"]), snapped_LD=float(r2["L_D"]), snapped_mtot=float(r2.get("m_total_kg", np.nan)))
    print(name, src, r160 * 203.294 / 160, out[name], flush=True)
json.dump(out, open(f"{R}/best_designs.json", "w"), indent=1)
