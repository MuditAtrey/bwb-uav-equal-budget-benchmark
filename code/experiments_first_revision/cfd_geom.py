"""Extract the right-half wing sections (with twist, sweep, dihedral, winglet) from the pipeline's own geometry builder."""
import sys, os, warnings, json
ROOT = os.environ.get("DRONE_ROOT", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone")
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "revision_runs/results_revision"))
sys.path.insert(0, ROOT); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from flying_wing_optimizer.geometry import build_wing_geometry
KEYS = ["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist",
        "winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"]

def sections(v):
    v = dict(zip(KEYS, [float(t) for t in v]))
    ap, _ = build_wing_geometry(sweep_inner=v["sweep_inner"], sweep_outer=v["sweep_outer"], y_blend=v["y_blend"], c_root=v["c_root"],
        c_blend=v["c_blend"], taper=v["taper"], theta0=v["theta0"], theta1=v["theta1"], theta2=v["theta2"], tip_twist=v["tip_twist"],
        battery_mass=v["battery_mass"], winglet_sweep_deg=v["winglet_sweep_deg"], winglet_twist_deg=v["winglet_twist_deg"])
    w = ap.wings[0]
    out = []
    for i, xs in enumerate(w.xsecs):
        xg, yg, zg = w._compute_frame_of_WingXSec(i)
        af = np.array(xs.airfoil.coordinates, dtype=float)       # (N,2), unit chord, LE at x=0
        P = np.array(xs.xyz_le, dtype=float) + float(xs.chord) * (af[:, [0]] * xg + af[:, [1]] * zg)
        out.append(P)
    return ap, w, out

if __name__ == "__main__":
    al = pd.read_csv(f"{RESULTS_DIR}/al/al_seed0.csv")
    b = al[al.is_valid].nlargest(1, "range_calm_km").iloc[0]
    ap, w, S = sections(b[KEYS].values)
    print("symmetric", w.symmetric, "nsec", len(S), "s_ref", ap.s_ref, "c_ref", ap.c_ref, "b_ref", ap.b_ref, "xyz_ref", ap.xyz_ref)
    for i, (xs, P) in enumerate(zip(w.xsecs, S)):
        te_gap = np.linalg.norm(P[0] - P[-1])
        print(i, np.round(P[:, 1].mean(), 3), "chord", round(float(xs.chord), 3), "N", len(P), "TEgap_mm", round(te_gap * 1e3, 2), "zrange", np.round([P[:,2].min(), P[:,2].max()], 3), xs.airfoil.name)
