"""B4: generalised cfd_run.py. Runs the half-model Euler SU2 case (cruise AoA and +2 deg) for ANY design vector.
The design comes from a JSON file {name: {14 design keys}} given by DESIGNS_JSON, so nothing is hard-coded to the 5 named designs.
usage: DESIGNS_JSON=designs.json SU2_CFD=$REPO/scratch/su2/bin/SU2_CFD python3 cfd_run_v3.py NAME OUTDIR H_WALL_MM
Skips a case whose result.csv already exists (resumable). Re-uses cfg/run_case/vlm_ref and cfd_mesh.build from results_revision/scripts (unmodified)."""
import sys, os, json, warnings
ROOT = os.environ.get("DRONE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
RS = os.path.join(ROOT, "revision_runs/results_revision/scripts")
os.environ.setdefault("SU2_CFD", os.path.join(ROOT, "scratch/su2/bin/SU2_CFD"))
sys.path.insert(0, RS); sys.path.insert(0, ROOT); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from cfd_run import vlm_ref, run_case
from cfd_geom import KEYS

if __name__ == "__main__":
    name, out, h = sys.argv[1], sys.argv[2], float(sys.argv[3])
    if os.path.exists(f"{out}/result.csv"):
        print("exists, skipping", out); sys.exit(0)
    os.makedirs(out, exist_ok=True); niter = int(os.environ.get("NITER", "800"))
    des = json.load(open(os.environ["DESIGNS_JSON"]))[name]
    v = np.array([float(des[k]) for k in KEYS])
    ref = vlm_ref(v)
    json.dump({k: float(x) for k, x in ref.items() if isinstance(x, (int, float, np.floating))}, open(f"{out}/vlm_ref.json", "w"), indent=1)
    from cfd_mesh import build
    ap, w, nn = build(v, out, h_wall=h)
    mesh = f"{out}/mesh.su2"
    a0 = float(ref["AoA_cruise"]); mach = float(ref["V_cruise_actual"]) / np.sqrt(1.4 * 287.05 * 284.9)
    sref_half = float(ap.s_ref) / 2; cref = float(ap.c_ref); xref = float(ref["x_cg"])
    res = [run_case(mesh, out, a, mach, sref_half, cref, xref, f"a{int(round(a*100))}", niter) for a in (a0, a0 + 2.0)]
    d = pd.DataFrame(res); d["design"] = name; d["h_wall_mm"] = h; d["nodes"] = nn
    d["VLM_CL_cruise"] = ref["CL"]; d["VLM_CD_induced_plus"] = ref["CD_vlm"]; d["VLM_Cm"] = ref["Cm_cruise"]; d["VLM_SM"] = ref["SM"]; d["VLM_AoA"] = a0
    d.to_csv(f"{out}/result.csv", index=False); print(d.to_string())
    for f in ("mesh.su2",):                       # meshes are large; keep only the results
        try: os.remove(f"{out}/{f}")
        except OSError: pass
