"""Compare the pipeline's Raymer strip profile drag (x_tr=0.40 fixed) with NeuralFoil section drag, per design.
NeuralFoil is evaluated at each strip's local Re (and a sweep of local cl), fully viscous with transition prediction.
Wing-only CD_wing_profile referenced to S_ref, same strip loop as aerodynamics._compute_profile_drag."""
import sys, os, math, warnings, json
ROOT = os.environ.get("DRONE_ROOT", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone")
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "revision_runs/results_revision"))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(RESULTS_DIR, "scripts")); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, aerosandbox as asb
from cfd_geom import sections, KEYS
from cfd_run import vlm_ref
from flying_wing_optimizer.aerodynamics import ATMOSPHERE

al = pd.read_csv(f"{RESULTS_DIR}/al/al_seed0.csv"); va = al[al.is_valid]
db = pd.read_csv(f"{ROOT}/flying_wing_optimizer/results/database.csv"); p1 = db[db.evaluation_type.str.startswith("Phase1") & db.is_valid.astype(bool)].sort_values("range_calm_km")
sel = {"PAPER_champion": db[db.is_valid.astype(bool)].nlargest(1,"range_calm_km").iloc[0], "AL_champion": va.nlargest(1, "range_calm_km").iloc[0], "P1_best": p1.iloc[-1], "P1_median": p1.iloc[len(p1)//2], "P1_p10": p1.iloc[len(p1)//10]}
rho, nu = ATMOSPHERE.air_density, ATMOSPHERE.kinematic_viscosity
rows = []
for name, row in sel.items():
    v = row[KEYS].values; ref = vlm_ref(v); V = float(ref["V_cruise_actual"]); CL = float(ref["CL"]); S = float(ref["S_ref"])
    ap, w, _ = sections(v); xs = w.xsecs; q = 0.5 * rho * V**2
    for dcl in (0.0, -0.1, 0.1):
        D_raymer = 0.0; D_nf = 0.0; D_nf_turb = 0.0
        for i in range(len(xs) - 1):
            a, b = xs[i], xs[i+1]; c = (a.chord + b.chord) / 2
            ds = math.hypot(abs(b.xyz_le[1]-a.xyz_le[1]), abs(b.xyz_le[2]-a.xyz_le[2]))
            if ds < 1e-6 or c < 1e-4: continue
            tc = (a.airfoil.max_thickness() + b.airfoil.max_thickness()) / 2
            Re = max(V*c/nu, 1e4); x = 0.4
            Cf = x*1.328/math.sqrt(max(Re*x,100)) + (1-x)*0.455/math.log10(Re)**2.58
            FF = 1 + 2*tc + 60*tc**4
            D_raymer += Cf*FF*2*(1+0.2*tc)*c*ds*q
            af = a.airfoil if i % 1 == 0 else b.airfoil
            cl = max(CL + dcl, 0.0)
            r = af.get_aero_from_neuralfoil(alpha=0.0 + np.degrees(0) if False else 0.0, Re=Re, mach=V/340.0) if False else None
            # find alpha that gives target cl, then read cd
            al_grid = np.linspace(-6, 10, 33)
            aer = af.get_aero_from_neuralfoil(alpha=al_grid, Re=Re, mach=V/340.0, model_size="large")
            CLs, CDs = np.array(aer["CL"]).ravel(), np.array(aer["CD"]).ravel()
            cd = float(np.interp(cl, CLs[np.argsort(CLs)], CDs[np.argsort(CLs)]))
            D_nf += cd*c*ds*q
            # forced-turbulent comparison
            aer_t = af.get_aero_from_neuralfoil(alpha=al_grid, Re=Re, mach=V/340.0, model_size="large", xtr_upper=0.0, xtr_lower=0.0)
            CLt, CDt = np.array(aer_t["CL"]).ravel(), np.array(aer_t["CD"]).ravel()
            D_nf_turb += float(np.interp(cl, CLt[np.argsort(CLt)], CDt[np.argsort(CLt)]))*c*ds*q
        k = 2.0/(q*S)
        rows.append(dict(design=name, cl_local=CL+dcl, V=V, CD_raymer=D_raymer*k, CD_neuralfoil=D_nf*k, CD_nf_forced_turb=D_nf_turb*k,
                         ratio_nf_over_raymer=D_nf/D_raymer, CD_profile_total_pipeline=float(ref["CD_profile"]), CD_induced=float(ref["CD_vlm"]), CD_total=float(ref["CD"])))
        print(rows[-1], flush=True)
d = pd.DataFrame(rows); d.to_csv(f"{RESULTS_DIR}/viscous/profile_drag_check.csv", index=False)
print(d.to_string())
