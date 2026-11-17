"""B14 (2026-11-16). Independent initial pools: a fresh Latin hypercube of N designs per pool seed, generated like the original pool
(scipy LatinHypercube over config.BOUNDS, twist variables sorted into washout order) and evaluated with the full pipeline.
Ranges are stored on the database basis (160 Wh/kg) in column range_calm_km, so the files can replace the Phase 1 pool via POOL_FILE.
usage: NPROC=.. python3 indep_pool.py OUTDIR N SEED_LO SEED_HI     (LHS seed = 9000 + seed)"""
import os, sys, warnings; warnings.filterwarnings("ignore")
ROOT = os.environ.get("DRONE_ROOT", "/tmp/work"); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, multiprocessing as mp
from scipy.stats.qmc import LatinHypercube
from flying_wing_optimizer.config import BOUNDS
from flying_wing_optimizer.main import evaluate_design_vector
OUT, N = sys.argv[1], int(sys.argv[2]); SEEDS = range(int(sys.argv[3]), int(sys.argv[4])); E_SCALE = 160.0 / 208.25
KEYS = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2",
        "tip_twist", "winglet_sweep_deg", "winglet_twist_deg", "battery_mass", "cruise_speed_ms"]
nm = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper_ratio", "theta0", "theta1", "theta2", "tip_twist",
      "winglet_sweep", "winglet_twist", "battery_mass", "cruise_speed_ms"]
LOW = np.array([getattr(BOUNDS, n)[0] for n in nm]); HIGH = np.array([getattr(BOUNDS, n)[1] for n in nm])
def work(a):
    vec, k, seed = a
    try:
        r = evaluate_design_vector(list(vec), 990000 + k, f"indep_{seed}")
        return dict(k=k, is_valid=bool(r["is_valid"]), range_calm_km=float(r["range_calm_km"]) * E_SCALE, L_D=float(r["L_D"]), SM=float(r["SM"]), Cm=float(r["Cm_trim"]),
                    **{n: float(vec[i]) for i, n in enumerate(KEYS)})
    except Exception:
        return dict(k=k, is_valid=False, range_calm_km=0.0, L_D=0.0, SM=0.0, Cm=0.0, **{n: float(vec[i]) for i, n in enumerate(KEYS)})
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with mp.Pool(int(os.environ.get("NPROC", "2"))) as pool:
        for seed in SEEDS:
            fn = f"{OUT}/pool_seed{seed}.csv"
            if os.path.exists(fn) and len(pd.read_csv(fn)) >= N: continue
            X = LOW + LatinHypercube(d=14, seed=9000 + seed).random(n=N) * (HIGH - LOW); X[:, 6:10] = -np.sort(-X[:, 6:10], axis=1)
            rows = pool.map(work, [(x, i, seed) for i, x in enumerate(X)], chunksize=4)
            d = pd.DataFrame(rows); d.to_csv(fn, index=False)
            v = d[d.is_valid]; print(f"pool seed {seed}: valid {len(v)/N:.3f} best {v.range_calm_km.max():.1f}", flush=True)
