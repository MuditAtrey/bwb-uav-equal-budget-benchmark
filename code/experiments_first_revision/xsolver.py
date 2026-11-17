"""Cross-solver check (AeroBuildup, viscous strip + XFOIL/NeuralFoil polars) on designs found by each arm.
For each (arm, seed): top-6 valid designs by pipeline range + 6 random valid designs. AeroBuildup L/D is matched to the pipeline CL.
Outputs xsolver.csv with pipeline L/D, AB L/D, ratio. Spearman rank correlation computed in the analysis step.
usage: DRONE_ROOT=.. NPROC=.. python3 xsolver.py IN_DIR OUT_CSV arm1,arm2 seed_lo seed_hi
"""
import os, sys, types, warnings, math
warnings.filterwarnings("ignore")
ROOT = os.environ.get("DRONE_ROOT", "/tmp/work"); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, multiprocessing as mp
import aerosandbox as asb
import flying_wing_optimizer.geometry as geom
from flying_wing_optimizer.config import ATMOSPHERE

IN, OUTCSV = sys.argv[1], sys.argv[2]
ARMS = sys.argv[3].split(","); SEEDS = range(int(sys.argv[4]), int(sys.argv[5]))
K = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2", "tip_twist",
     "winglet_sweep_deg", "winglet_twist_deg", "battery_mass", "cruise_speed_ms"]


def noclamp_builder():
    src = open(geom.__file__).read()
    a = "    c_blend = float(np.clip(c_blend, 0.25, c_root - 0.05))\n    c_tip = float(np.clip(c_tip, 0.05, c_blend - 0.05))\n"
    mod = types.ModuleType("g2"); mod.__file__ = geom.__file__
    exec(compile(src.replace(a, "    pass\n"), geom.__file__, "exec"), mod.__dict__)
    return mod.build_wing_geometry


def one(row):
    builder = noclamp_builder() if row["arm"] == "noclamp" else geom.build_wing_geometry
    out = dict(arm=row["arm"], seed=row["seed"], k=row["k"], sel=row["sel"], range_km=row["range_km"], L_D=row["L_D"],
               CL=row["CL"], inverted=row["inverted"], clamp_active=row["clamp_active"], c_blend=row["c_blend"], c_root=row["c_root"])
    try:
        res = builder(**{k: float(row[k]) for k in K if k not in ("cruise_speed_ms",)})
        ap = res[0] if isinstance(res, tuple) else res
        atm = asb.Atmosphere(altitude=ATMOSPHERE.cruise_altitude)
        al = np.arange(0.0, 9.0, 1.0); cl = []; cd = []
        for a in al:
            r = asb.AeroBuildup(airplane=ap, op_point=asb.OperatingPoint(atmosphere=atm, velocity=float(row["cruise_speed_ms"]), alpha=a)).run()
            cl.append(float(np.squeeze(r["CL"]))); cd.append(float(np.squeeze(r["CD"])))
        cl = np.array(cl); cd = np.array(cd)
        if row["CL"] < cl.min() or row["CL"] > cl.max():
            out.update(ab_LD=np.nan, note="CL out of AB sweep range")
        else:
            cdm = float(np.interp(row["CL"], cl, cd)); out.update(ab_LD=row["CL"] / cdm, ab_CD=cdm, note="")
    except Exception as e:
        out.update(ab_LD=np.nan, note=str(e)[:80])
    return out


if __name__ == "__main__":
    rows = []
    for arm in ARMS:
        for s in SEEDS:
            f = f"{IN}/{arm}_seed{s}.csv"
            if not os.path.exists(f): continue
            d = pd.read_csv(f); d["arm"] = arm; d["seed"] = s
            v = d[d.is_valid].copy()
            top = v.nlargest(6, "range_km"); top["sel"] = "top"
            rnd = v.drop(top.index).sample(min(6, max(0, len(v) - 6)), random_state=s); rnd["sel"] = "random"
            rows += [r for _, r in pd.concat([top, rnd]).iterrows()]
    done = pd.read_csv(OUTCSV) if os.path.exists(OUTCSV) else pd.DataFrame()
    todo = [r for r in rows if done.empty or not ((done.arm == r["arm"]) & (done.seed == r["seed"]) & (done.k == r["k"])).any()]
    print(f"{len(rows)} designs, {len(todo)} to do", flush=True)
    with mp.Pool(int(os.environ.get("NPROC", "2"))) as p:
        res = list(p.imap(one, todo, chunksize=1))
    out = pd.concat([done, pd.DataFrame(res)], ignore_index=True) if res else done
    out.to_csv(OUTCSV, index=False); print("saved", len(out))
