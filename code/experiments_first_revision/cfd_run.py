"""3D inviscid (Euler) CFD of a design at its VLM cruise condition and at +2 deg, compared with the pipeline VLM.
usage: cfd_run.py NAME OUTDIR h_wall_mm   (design chosen via env DESIGN='AL_champion' or 'rank:K' of AL seed0 / 'p1:Q')"""
import sys, os, json, subprocess, time, warnings, numpy as np, pandas as pd
ROOT = os.environ.get("DRONE_ROOT", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone")
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "revision_runs/results_revision"))
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, ROOT); warnings.filterwarnings("ignore")
from cfd_geom import KEYS
from flying_wing_optimizer import aerodynamics as A
SU2 = os.environ.get("SU2_CFD", "/tmp/su2dl/su2/bin/SU2_CFD")

def vlm_ref(v):
    d = dict(zip(KEYS, [float(t) for t in v]))
    return A.evaluate_aerodynamics(sweep_inner=d["sweep_inner"], sweep_outer=d["sweep_outer"], y_blend=d["y_blend"], c_root=d["c_root"],
        c_blend=d["c_blend"], taper=d["taper"], theta0=d["theta0"], theta1=d["theta1"], theta2=d["theta2"], tip_twist=d["tip_twist"],
        v_cruise=d["cruise_speed_ms"], battery_mass=d["battery_mass"], winglet_sweep_deg=d["winglet_sweep_deg"], winglet_twist_deg=d["winglet_twist_deg"])

def cfg(mesh, alpha, mach, sref_half, cref, xref, hist, niter=1500):
    return f"""SOLVER= EULER
MACH_NUMBER= {mach:.5f}
AOA= {alpha:.4f}
SIDESLIP_ANGLE= 0.0
FREESTREAM_PRESSURE= 95500.0
FREESTREAM_TEMPERATURE= 284.9
REF_ORIGIN_MOMENT_X= {xref:.5f}
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= {cref:.5f}
REF_AREA= {sref_half:.6f}
MARKER_EULER= ( wing )
MARKER_SYM= ( symmetry )
MARKER_FAR= ( farfield )
MARKER_MONITORING= ( wing )
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.2, 1.0, 50.0 )
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
TIME_DISCRE_FLOW= EULER_IMPLICIT
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-3
LINEAR_SOLVER_ITER= 10
ITER= {niter}
CONV_FIELD= RMS_DENSITY
CONV_RESIDUAL_MINVAL= -8
CONV_STARTITER= 20
MESH_FILENAME= {mesh}
MESH_FORMAT= SU2
OUTPUT_FILES= ( RESTART )
RESTART_FILENAME= {hist}_restart.dat
CONV_FILENAME= {hist}
SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, LIFT, DRAG, MOMENT_Y )
HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )
WRT_PERFORMANCE= NO
"""

def run_case(mesh, out, alpha, mach, sref_half, cref, xref, tag, niter):
    c = f"{out}/{tag}.cfg"; h = f"{out}/{tag}_hist"
    open(c, "w").write(cfg(mesh, alpha, mach, sref_half, cref, xref, h, niter))
    t = time.time(); r = subprocess.run([SU2, c], capture_output=True, text=True, cwd=out)
    open(f"{out}/{tag}.log", "w").write(r.stdout + r.stderr)
    H = pd.read_csv(h + ".csv"); H.columns = [x.strip().strip('"') for x in H.columns]
    last = H.iloc[-1]
    return dict(tag=tag, alpha=alpha, iters=int(last.get("Inner_Iter", len(H))), final_log10_rho=float(last["rms[Rho]"]),
                CL=float(last["CL"]), CD=float(last["CD"]), CMy=float(last["CMy"]), secs=time.time() - t,
                CL_std_last100=float(H["CL"].tail(100).std()), CD_std_last100=float(H["CD"].tail(100).std()))

if __name__ == "__main__":
    name, out, h = sys.argv[1], sys.argv[2], float(sys.argv[3]); os.makedirs(out, exist_ok=True)
    niter = int(os.environ.get("NITER", "800"))
    al = pd.read_csv("/mnt/user-data/outputs/results_revision/al/al_seed0.csv"); va = al[al.is_valid]
    db = pd.read_csv("/tmp/work/flying_wing_optimizer/results/database.csv"); p1 = db[db.evaluation_type.str.startswith("Phase1") & db.is_valid.astype(bool)].sort_values("range_calm_km")
    allv = db[db.is_valid.astype(bool)]
    sel = {"PAPER_champion": allv.nlargest(1, "range_calm_km").iloc[0], "AL_champion": va.nlargest(1, "range_calm_km").iloc[0], "P1_best": p1.iloc[-1], "P1_median": p1.iloc[len(p1) // 2], "P1_p10": p1.iloc[len(p1) // 10]}
    row = sel[name]; v = row[KEYS].values
    ref = vlm_ref(v)
    json.dump({k: (float(x) if isinstance(x, (int, float, np.floating)) else str(x)) for k, x in ref.items() if not isinstance(x, (list, dict))}, open(f"{out}/vlm_ref.json", "w"), indent=1)
    from cfd_mesh import build
    ap, w, nn = build(v, out, h_wall=h)
    mesh = f"{out}/mesh.su2"
    a0 = float(ref["AoA_cruise"]); mach = float(ref["V_cruise_actual"]) / np.sqrt(1.4 * 287.05 * 284.9)
    sref_half = float(ap.s_ref) / 2; cref = float(ap.c_ref); xref = float(ref["x_cg"])
    res = [run_case(mesh, out, a, mach, sref_half, cref, xref, f"a{int(round(a*100))}", niter) for a in (a0, a0 + 2.0)]
    d = pd.DataFrame(res); d["design"] = name; d["h_wall_mm"] = h; d["nodes"] = nn
    d["VLM_CL_cruise"] = ref["CL"]; d["VLM_CD_induced_plus"] = ref["CD_vlm"]; d["VLM_Cm"] = ref["Cm_cruise"]; d["VLM_SM"] = ref["SM"]; d["VLM_AoA"] = a0
    d.to_csv(f"{out}/result.csv", index=False); print(d.to_string())
