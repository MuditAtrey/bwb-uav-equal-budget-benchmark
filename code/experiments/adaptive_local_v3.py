"""Adaptive surrogate-free local search (control for the active-learning loop).
Same structure as the AL loop (cycles of 6 evaluations, seeds drawn from the current top-20 of the whole database, so the seed set
moves as better designs are found), but the proposals are Gaussian perturbations instead of surrogate-optimized candidates.
This removes the confound that the AL loop adapts its seed set while the fixed-seed local searches do not.
usage: DRONE_ROOT=.. NPROC=.. TBUDGET=.. python3 adaptive_local.py OUTDIR SIGMA_PERCENT SEED_LO SEED_HI"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = os.environ.get("DRONE_ROOT", "/tmp/work"); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, multiprocessing as mp
OUT, SIG = sys.argv[1], float(sys.argv[2]) / 100.0
SEEDS = range(int(sys.argv[3]), int(sys.argv[4]))
N_EXTRA, E_SCALE, BATCH = int(os.environ.get("N_EXTRA", "287")), 160.0 / 208.25, 6   # B15: N_EXTRA env (default unchanged)
TBUDGET = float(os.environ.get("TBUDGET", "140"))
KEYS = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2",
        "tip_twist", "winglet_sweep_deg", "winglet_twist_deg", "battery_mass", "cruise_speed_ms"]
from flying_wing_optimizer.config import BOUNDS
from flying_wing_optimizer.main import evaluate_design_vector
nm = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper_ratio", "theta0", "theta1", "theta2", "tip_twist",
      "winglet_sweep", "winglet_twist", "battery_mass", "cruise_speed_ms"]
LOW = np.array([getattr(BOUNDS, n)[0] for n in nm]); HIGH = np.array([getattr(BOUNDS, n)[1] for n in nm])


def fix(x):
    x = x.copy(); x[6:10] = np.sort(x[6:10])[::-1]; return x


def work(a):
    vec, k, seed = a
    try:
        r = evaluate_design_vector(list(vec), 960000 + k, f"adapt_{seed}")
        return dict(arm=f"adapt{int(SIG*100)}", seed=seed, k=k, is_valid=bool(r["is_valid"]), range_km=float(r["range_calm_km"]) * E_SCALE,
                    L_D=float(r["L_D"]), SM=float(r["SM"]), Cm=float(r["Cm_trim"]), **{n: float(vec[i]) for i, n in enumerate(KEYS)})
    except Exception:
        return dict(arm=f"adapt{int(SIG*100)}", seed=seed, k=k, is_valid=False, range_km=0.0, L_D=0.0, SM=0.0, Cm=0.0, **{n: float(vec[i]) for i, n in enumerate(KEYS)})


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    db = pd.read_csv(ROOT + "/flying_wing_optimizer/results/database.csv"); db["is_valid"] = db["is_valid"].astype(bool)
    p1 = db[db.evaluation_type.str.startswith("Phase1")]
    _pn = int(os.environ.get("POOL_N", "0"))   # B12: random subsample of the Phase 1 pool (no-op when unset)
    if _pn: p1 = p1.sample(n=_pn, random_state=7000 + SEEDS[0])
    _pf = os.environ.get("POOL_FILE")   # B14: independent pool file, '{seed}' is replaced by the run seed (no-op when unset)
    if _pf: p1 = pd.read_csv(_pf.format(seed=SEEDS[0])); p1["is_valid"] = p1.is_valid.astype(bool)
    p1 = p1[p1.is_valid]
    pool_X = p1[KEYS].values; pool_r = p1.range_calm_km.values
    with mp.Pool(int(os.environ.get("NPROC", "2"))) as pool:
        for seed in SEEDS:
            fn = f"{OUT}/adapt{int(SIG*100)}_seed{seed}.csv"
            new = pd.read_csv(fn) if os.path.exists(fn) else pd.DataFrame()
            rng = np.random.default_rng(30_000 + seed + len(new))
            while len(new) < N_EXTRA:
                if time.time() - t0 > float(os.environ.get("TBUDGET", "140")):
                    print("time budget reached; rerun to continue", flush=True); sys.exit(0)
                nv = new[new.is_valid] if len(new) else new
                X = np.vstack([pool_X] + ([nv[KEYS].values] if len(nv) else []))
                r = np.concatenate([pool_r] + ([nv.range_km.values] if len(nv) else []))   # v3: same basis as the pool; the original divided by E_SCALE (208.25 Wh/kg basis), inflating new designs by 1.30 in the top-20 ranking
                top = X[np.argsort(-r)[:20]]
                have = np.vstack([db[KEYS].values] + ([new[KEYS].values] if len(new) else []))
                batch = []
                while len(batch) < min(BATCH, N_EXTRA - len(new)):
                    x = fix(np.clip(top[rng.integers(0, 20)] + rng.normal(0, SIG, 14) * (HIGH - LOW), LOW, HIGH))
                    if not (np.abs(have - x) <= 1e-4).all(axis=1).any():
                        batch.append(x)
                rows = list(pool.imap(work, [(x, len(new) + i, seed) for i, x in enumerate(batch)]))
                new = pd.concat([new, pd.DataFrame(rows)], ignore_index=True); new.to_csv(fn, index=False)
            print(f"seed {seed} done: best {new[new.is_valid].range_km.max():.1f} valid {new.is_valid.mean():.2f}", flush=True)
