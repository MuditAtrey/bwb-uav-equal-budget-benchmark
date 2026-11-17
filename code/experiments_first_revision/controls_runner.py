"""Equal-budget control arms for Reviewer 1 (item: +7.5% gain not attributable to active learning).

Setting mirrors the paper: a 9,800-sample Phase 1 LHS pool already exists (database.csv), then a further
budget of N_EXTRA = 287 evaluations is spent by each arm. Arms that do NOT use the surrogate:
  lhs       fresh Latin Hypercube samples in the full bounds
  uniform   i.i.d. uniform samples in the full bounds
  localSS   Gaussian perturbation (SS percent of each bound range) of a random member of the Phase 1 top-20
            (the same seed pool the active-learning loop starts from), no surrogate
Output: one CSV per (arm, seed) with every evaluation in order, so running-best curves can be rebuilt.
Range is rescaled to the database's 160 Wh/kg basis (current mission.py uses 208.25 Wh/kg).
"""
import os, sys, time, warnings, multiprocessing as mp
ROOT = os.environ.get("DRONE_ROOT", "/tmp/work"); sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats.qmc import LatinHypercube

OUT = sys.argv[1]
SEEDS = range(int(sys.argv[2]), int(sys.argv[3]))
ARMS = sys.argv[4].split(",")
N_EXTRA = 287
E_SCALE = 160.0 / 208.25
KEYS = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2",
        "tip_twist", "winglet_sweep_deg", "winglet_twist_deg", "battery_mass", "cruise_speed_ms"]

from flying_wing_optimizer.config import BOUNDS
from flying_wing_optimizer.main import evaluate_design_vector

LOW = np.array([BOUNDS.sweep_inner[0], BOUNDS.sweep_outer[0], BOUNDS.y_blend[0], BOUNDS.c_root[0], BOUNDS.c_blend[0],
                BOUNDS.taper_ratio[0], BOUNDS.theta0[0], BOUNDS.theta1[0], BOUNDS.theta2[0], BOUNDS.tip_twist[0],
                BOUNDS.winglet_sweep[0], BOUNDS.winglet_twist[0], BOUNDS.battery_mass[0], BOUNDS.cruise_speed_ms[0]])
HIGH = np.array([BOUNDS.sweep_inner[1], BOUNDS.sweep_outer[1], BOUNDS.y_blend[1], BOUNDS.c_root[1], BOUNDS.c_blend[1],
                 BOUNDS.taper_ratio[1], BOUNDS.theta0[1], BOUNDS.theta1[1], BOUNDS.theta2[1], BOUNDS.tip_twist[1],
                 BOUNDS.winglet_sweep[1], BOUNDS.winglet_twist[1], BOUNDS.battery_mass[1], BOUNDS.cruise_speed_ms[1]])

db = pd.read_csv(ROOT + "/flying_wing_optimizer/results/database.csv")
db["is_valid"] = db["is_valid"].astype(bool)
p1 = db[db.evaluation_type.str.startswith("Phase1") & db.is_valid]
TOP20 = p1.nlargest(20, "range_calm_km")[KEYS].values


def fix(x):
    x = x.copy()
    x[6:10] = np.sort(x[6:10])[::-1]  # washout monotone, as in generate_lhs_samples
    return x


def gen(arm, seed, n):
    rng = np.random.default_rng(10_000 + seed)
    if arm == "lhs":
        X = LOW + LatinHypercube(d=14, seed=10_000 + seed).random(n) * (HIGH - LOW)
    elif arm == "uniform":
        X = LOW + rng.random((n, 14)) * (HIGH - LOW)
    elif arm.startswith("local"):
        sig = float(arm[5:]) / 100.0
        base = TOP20[rng.integers(0, len(TOP20), n)]
        X = np.clip(base + rng.normal(0, sig, (n, 14)) * (HIGH - LOW), LOW, HIGH)
    else:
        raise ValueError(arm)
    return np.array([fix(x) for x in X])


def work(a):
    vec, k, arm, seed = a
    try:
        r = evaluate_design_vector(list(vec), 900000 + k, f"ctl_{arm}_{seed}")
        return dict(arm=arm, seed=seed, k=k, is_valid=bool(r["is_valid"]), range_km=float(r["range_calm_km"]) * E_SCALE,
                    L_D=float(r["L_D"]), SM=float(r["SM"]), Cm=float(r["Cm_trim"]), **{n: float(vec[i]) for i, n in enumerate(KEYS)})
    except Exception as e:
        return dict(arm=arm, seed=seed, k=k, is_valid=False, range_km=0.0, L_D=0.0, SM=0.0, Cm=0.0,
                    **{n: float(vec[i]) for i, n in enumerate(KEYS)})


if __name__ == "__main__":
    # Chunked and resumable: the Mac shell kills background jobs when a call ends, so each call does at most TBUDGET seconds.
    os.makedirs(OUT, exist_ok=True)
    CH = 96; TBUDGET = float(os.environ.get("TBUDGET", "140")); t0 = time.time(); last = 60.0
    with mp.Pool(int(os.environ.get("NPROC", "2"))) as pool:
        for seed in SEEDS:
            for arm in ARMS:
                fn = f"{OUT}/{arm}_seed{seed}.csv"
                if os.path.exists(fn):
                    continue
                X = gen(arm, seed, N_EXTRA)
                for c, lo in enumerate(range(0, N_EXTRA, CH)):
                    pf = f"{OUT}/{arm}_seed{seed}.part{c}.csv"
                    if os.path.exists(pf):
                        continue
                    if time.time() - t0 + last > TBUDGET:
                        print("budget reached; rerun to continue", flush=True); sys.exit(0)
                    tc = time.time()
                    idx = range(lo, min(lo + CH, N_EXTRA))
                    rows = list(pool.imap(work, [(X[i], i, arm, seed) for i in idx], chunksize=2))
                    pd.DataFrame(rows).to_csv(pf, index=False)
                    last = (time.time() - tc) * 1.1
                parts = [pd.read_csv(f"{OUT}/{arm}_seed{seed}.part{c}.csv") for c in range(-(-N_EXTRA // CH))]
                d = pd.concat(parts, ignore_index=True); d.to_csv(fn, index=False)
                for c in range(-(-N_EXTRA // CH)):
                    os.remove(f"{OUT}/{arm}_seed{seed}.part{c}.csv")
                print(f"{arm:9s} seed {seed}: valid {d.is_valid.mean():.2f}  best {d.range_km.max():6.1f} km", flush=True)
