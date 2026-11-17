"""Controlled manifold vs unconstrained comparison (Reviewer 1, item 1).
Same geometry builder (flying_wing_optimizer.geometry.build_wing_geometry), same evaluator, same bounds (c_blend widened to
[0.15, 0.65] so inversion is reachable), same search (LHS gen 0 then Gaussian-perturbation elite search), same budget.
Only difference: arm 'clamp' keeps the c_blend/c_tip clips, arm 'noclamp' removes them (y_blend clip kept).
Resumable, time-budgeted. usage: DRONE_ROOT=.. NPROC=.. TBUDGET=.. python3 manifold_vs_unc.py OUT SEED_LO SEED_HI [arms]
"""
import os, sys, time, types, warnings
warnings.filterwarnings("ignore")
ROOT = os.environ.get("DRONE_ROOT", "/tmp/work"); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, multiprocessing as mp
from scipy.stats.qmc import LatinHypercube

OUT = sys.argv[1]; SEEDS = range(int(sys.argv[2]), int(sys.argv[3]))
ARMS = (sys.argv[4] if len(sys.argv) > 4 else "clamp,noclamp").split(",")
N_EXTRA, GEN, E_SCALE = 287, 48, 160.0 / 208.25
TBUDGET = float(os.environ.get("TBUDGET", "140"))
KEYS = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2",
        "tip_twist", "winglet_sweep_deg", "winglet_twist_deg", "battery_mass", "cruise_speed_ms"]
from flying_wing_optimizer.config import BOUNDS
import flying_wing_optimizer.main as fmain
import flying_wing_optimizer.geometry as geom

LOW = np.array([BOUNDS.sweep_inner[0], BOUNDS.sweep_outer[0], BOUNDS.y_blend[0], BOUNDS.c_root[0], 0.15, BOUNDS.taper_ratio[0],
                BOUNDS.theta0[0], BOUNDS.theta1[0], BOUNDS.theta2[0], BOUNDS.tip_twist[0], BOUNDS.winglet_sweep[0],
                BOUNDS.winglet_twist[0], BOUNDS.battery_mass[0], BOUNDS.cruise_speed_ms[0]])
HIGH = np.array([BOUNDS.sweep_inner[1], BOUNDS.sweep_outer[1], BOUNDS.y_blend[1], BOUNDS.c_root[1], 0.65, BOUNDS.taper_ratio[1],
                 BOUNDS.theta0[1], BOUNDS.theta1[1], BOUNDS.theta2[1], BOUNDS.tip_twist[1], BOUNDS.winglet_sweep[1],
                 BOUNDS.winglet_twist[1], BOUNDS.battery_mass[1], BOUNDS.cruise_speed_ms[1]])


# Bound-sensitivity arms (clamp builder): 'paper' = the paper's own bounds; 'wide' = the champion's active bounds relaxed
# (battery 3.5->4.5 kg, taper floor 0.55->0.35, winglet twist cap 0->+5 deg, c_blend cap 0.35->0.45). The cruise-speed floor is a
# mission requirement (130 km/h) and is not relaxed.
P_LOW = np.array([BOUNDS.sweep_inner[0], BOUNDS.sweep_outer[0], BOUNDS.y_blend[0], BOUNDS.c_root[0], BOUNDS.c_blend[0], BOUNDS.taper_ratio[0],
                  BOUNDS.theta0[0], BOUNDS.theta1[0], BOUNDS.theta2[0], BOUNDS.tip_twist[0], BOUNDS.winglet_sweep[0],
                  BOUNDS.winglet_twist[0], BOUNDS.battery_mass[0], BOUNDS.cruise_speed_ms[0]])
P_HIGH = np.array([BOUNDS.sweep_inner[1], BOUNDS.sweep_outer[1], BOUNDS.y_blend[1], BOUNDS.c_root[1], BOUNDS.c_blend[1], BOUNDS.taper_ratio[1],
                   BOUNDS.theta0[1], BOUNDS.theta1[1], BOUNDS.theta2[1], BOUNDS.tip_twist[1], BOUNDS.winglet_sweep[1],
                   BOUNDS.winglet_twist[1], BOUNDS.battery_mass[1], BOUNDS.cruise_speed_ms[1]])
W_LOW, W_HIGH = P_LOW.copy(), P_HIGH.copy()
W_LOW[5] = 0.35; W_HIGH[11] = 5.0; W_HIGH[12] = 4.5; W_HIGH[4] = 0.45


def bnds(arm):
    if arm == "paper": return P_LOW, P_HIGH
    if arm == "wide": return W_LOW, W_HIGH
    return LOW, HIGH


def make_noclamp_builder():
    src = open(geom.__file__).read()
    a = "    c_blend = float(np.clip(c_blend, 0.25, c_root - 0.05))\n    c_tip = float(np.clip(c_tip, 0.05, c_blend - 0.05))\n"
    assert a in src, "clamp lines not found"
    mod = types.ModuleType("geom_noclamp"); mod.__file__ = geom.__file__
    exec(compile(src.replace(a, "    pass  # clamp removed\n"), geom.__file__, "exec"), mod.__dict__)
    return mod.build_wing_geometry


def fix(x):
    x = x.copy(); x[6:10] = np.sort(x[6:10])[::-1]; return x


def init_worker(arm):
    if arm == "noclamp":
        import flying_wing_optimizer.aerodynamics as faero
        nc = make_noclamp_builder()
        geom.build_wing_geometry = nc      # main.evaluate_design_vector imports it from here at call time
        faero.build_wing_geometry = nc     # aerodynamics.evaluate_aerodynamics uses its own module-level name


def work(a):
    vec, k, arm, seed = a
    clamp_active = bool(vec[4] > vec[3] - 0.05 or vec[5] * vec[4] > vec[4] - 0.05 or vec[4] < 0.25)
    base = dict(arm=arm, seed=seed, k=k, clamp_active=clamp_active, inverted=bool(vec[4] > vec[3]),
                **{n: float(vec[i]) for i, n in enumerate(KEYS)})
    try:
        r = fmain.evaluate_design_vector(list(vec), 970000 + k, f"mvu_{arm}_{seed}")
        return dict(base, is_valid=bool(r["is_valid"]), range_km=float(r["range_calm_km"]) * E_SCALE, L_D=float(r["L_D"]),
                    SM=float(r["SM"]), Cm=float(r["Cm_trim"]), AoA=float(r["AoA_trim"]), CL=float(r["CL"]))
    except Exception as e:
        return dict(base, is_valid=False, range_km=0.0, L_D=0.0, SM=0.0, Cm=0.0, AoA=0.0, CL=0.0)


def propose(arm, seed, g, done):
    LOW, HIGH = bnds(arm)
    if g == 0:
        X = LOW + LatinHypercube(d=14, seed=20000 + seed).random(GEN) * (HIGH - LOW)
    else:
        rng = np.random.default_rng(30000 + 100 * seed + g)
        v = done[done.is_valid]
        pool = (v if len(v) >= 5 else done).nlargest(10, "range_km")[KEYS].values
        X = np.clip(pool[rng.integers(0, len(pool), GEN)] + rng.normal(0, 0.10, (GEN, 14)) * (HIGH - LOW), LOW, HIGH)
    return np.array([fix(x) for x in X])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); last = 50.0
    n_gen = -(-N_EXTRA // GEN)
    for seed in SEEDS:
        for arm in ARMS:
            fn = f"{OUT}/{arm}_seed{seed}.csv"
            if os.path.exists(fn): continue
            with mp.Pool(int(os.environ.get("NPROC", "2")), initializer=init_worker, initargs=(arm,)) as pool:
                for g in range(n_gen):
                    pf = f"{OUT}/{arm}_seed{seed}.gen{g}.csv"
                    if os.path.exists(pf): continue
                    if time.time() - t0 + last > TBUDGET:
                        print("budget reached; rerun to continue", flush=True); sys.exit(0)
                    tc = time.time()
                    done = pd.concat([pd.read_csv(f"{OUT}/{arm}_seed{seed}.gen{h}.csv") for h in range(g)], ignore_index=True) if g else None
                    X = propose(arm, seed, g, done)
                    ids = range(g * GEN, min((g + 1) * GEN, N_EXTRA))
                    rows = list(pool.imap(work, [(X[i - g * GEN], i, arm, seed) for i in ids], chunksize=2))
                    pd.DataFrame(rows).to_csv(pf, index=False)
                    last = (time.time() - tc) * 1.1
            d = pd.concat([pd.read_csv(f"{OUT}/{arm}_seed{seed}.gen{h}.csv") for h in range(n_gen)], ignore_index=True)
            d.to_csv(fn, index=False)
            for h in range(n_gen):
                try: os.remove(f"{OUT}/{arm}_seed{seed}.gen{h}.csv")
                except OSError: pass
            print(f"{arm:8s} seed {seed}: valid {d.is_valid.mean():.2f} best {d.range_km.max():6.1f} km  clamp_active {d.clamp_active.mean():.2f}  inverted {d.inverted.mean():.2f}", flush=True)
