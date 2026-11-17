"""VLM panel-grid convergence for Reviewer 2. Uses the pipeline's own geometry builder and VLM call.
For several designs, runs the pipeline at fixed alpha (cruise AoA from the production grid) across grids
and records CL, CDi, Cm, x_np (SM), panel counts."""
import sys, warnings, time, os
ROOT = os.environ.get("DRONE_ROOT", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone")
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "revision_runs/results_revision"))
sys.path.insert(0, ROOT); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, aerosandbox as asb
from flying_wing_optimizer import aerodynamics as A
from flying_wing_optimizer import geometry as G
import inspect
KEYS = ["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist",
        "winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"]
db = pd.read_csv(f"{ROOT}/flying_wing_optimizer/results/database.csv")
al = pd.read_csv(f"{RESULTS_DIR}/al/al_seed0.csv")
p1 = db[db.evaluation_type.str.startswith("Phase1") & db.is_valid.astype(bool)]
designs = {"AL_champion": al[al.is_valid].nlargest(1,"range_calm_km")[KEYS].iloc[0].values,
           "P1_best": p1.nlargest(1,"range_calm_km")[KEYS].iloc[0].values,
           "P1_median": p1.sort_values("range_calm_km").iloc[len(p1)//2][KEYS].values,
           "P1_p10": p1.sort_values("range_calm_km").iloc[len(p1)//10][KEYS].values}
print(inspect.signature(G.build_wing_geometry))

GRIDS = [(2,12),(1,12),(3,12),(2,16),(2,24),(3,16),(4,16)]
orig = A._run_vlm
CUR = {"g": (2,12), "n": None}
def patched(airplane, op_point, primary_res=(2,12), fallback_res=(1,8), run_stability=False):
    g = CUR["g"]
    vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=op_point, spanwise_resolution=g[0], chordwise_resolution=g[1])
    out = vlm.run_with_stability_derivatives() if run_stability else vlm.run()
    CUR["n"] = len(vlm.vortex_centers)
    return out
A._run_vlm = patched
rows = []
for name, x in designs.items():
    vec = dict(zip(KEYS, x))
    for g in GRIDS:
        CUR["g"] = g; t = time.time()
        try:
            r = A.evaluate_aerodynamics(sweep_inner=vec["sweep_inner"], sweep_outer=vec["sweep_outer"], y_blend=vec["y_blend"],
                c_root=vec["c_root"], c_blend=vec["c_blend"], taper=vec["taper"], theta0=vec["theta0"], theta1=vec["theta1"],
                theta2=vec["theta2"], tip_twist=vec["tip_twist"], v_cruise=vec["cruise_speed_ms"], battery_mass=vec["battery_mass"],
                winglet_sweep_deg=vec["winglet_sweep_deg"], winglet_twist_deg=vec["winglet_twist_deg"])
            rows.append(dict(design=name, span_res=g[0], chord_res=g[1], panels=CUR["n"], secs=time.time()-t, **{k: v for k, v in r.items() if isinstance(v, (int, float, bool, np.floating))}))
        except Exception as e:
            rows.append(dict(design=name, span_res=g[0], chord_res=g[1], err=str(e)[:80]))
        pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/grid/grid_convergence.csv", index=False)
        print(rows[-1].get("design"), g, rows[-1].get("panels"), rows[-1].get("L_D"), rows[-1].get("SM"), flush=True)
pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/grid/grid_convergence.csv", index=False)
