"""Active-learning arm under the same equal budget as the control arms (Reviewer 1, item 2).
Starts from the 9,800-design Phase 1 LHS pool, then runs the paper's AL loop (5-MLP ensemble, RF feasibility, UCB beta=0.3,
DE popsize=150 maxiter=300, top-20 seeding, 5 acquisition + 1 uncertainty candidates) until N_EXTRA=287 evaluations are spent.
Resumable: state is the CSV of evaluations so far; each call stops after TBUDGET seconds.
usage: DRONE_ROOT=... NPROC=.. TBUDGET=.. python3 al_arm.py OUTDIR SEED [DE_POPSIZE DE_MAXITER]
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = os.environ.get("DRONE_ROOT", "/tmp/work"); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, multiprocessing as mp
from sklearn.ensemble import RandomForestClassifier

OUT, SEED = sys.argv[1], int(sys.argv[2])
POP = int(sys.argv[3]) if len(sys.argv) > 3 else 150
MAXIT = int(sys.argv[4]) if len(sys.argv) > 4 else 300
N_EXTRA, E_SCALE = 287, 160.0 / 208.25
TBUDGET = float(os.environ.get("TBUDGET", "140"))
KEYS = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2",
        "tip_twist", "winglet_sweep_deg", "winglet_twist_deg", "battery_mass", "cruise_speed_ms"]
TARGETS = ["CL", "CD", "L_D", "range_calm_km", "penalty_score"]

from flying_wing_optimizer.main import evaluate_design_vector
from flying_wing_optimizer.surrogate import SurrogateEnsemble
from flying_wing_optimizer.optimizer import SurrogateOptimizer


def work(a):
    vec, k = a
    try:
        r = evaluate_design_vector(list(vec), 950000 + k, f"al_{SEED}")
        r = dict(r); r["range_calm_km"] = float(r["range_calm_km"]) * E_SCALE
        return {**{n: float(vec[i]) for i, n in enumerate(KEYS)}, **{t: float(r[t]) for t in TARGETS},
                "is_valid": bool(r["is_valid"]), "SM": float(r["SM"]), "Cm": float(r["Cm_trim"])}
    except Exception:
        return {**{n: float(vec[i]) for i, n in enumerate(KEYS)}, **{t: 0.0 for t in TARGETS}, "is_valid": False, "SM": 0.0, "Cm": 0.0}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fn = f"{OUT}/al_seed{SEED}.csv"
    t0 = time.time()
    db = pd.read_csv(ROOT + "/flying_wing_optimizer/results/database.csv"); db["is_valid"] = db["is_valid"].astype(bool)
    p1 = db[db.evaluation_type.str.startswith("Phase1")][KEYS + TARGETS + ["is_valid"]].copy()
    new = pd.read_csv(fn) if os.path.exists(fn) else pd.DataFrame(columns=KEYS + TARGETS + ["is_valid", "SM", "Cm"])
    last = 120.0; stall = 0; stalls_total = 0
    with mp.Pool(int(os.environ.get("NPROC", "2"))) as pool:
        while len(new) < N_EXTRA:
            if time.time() - t0 + last > TBUDGET:
                print(f"budget reached at {len(new)}/{N_EXTRA}; rerun to continue", flush=True); sys.exit(0)
            ti = time.time()
            cur = pd.concat([p1, new[p1.columns]], ignore_index=True); cur["is_valid"] = cur["is_valid"].astype(bool); cur[KEYS + TARGETS] = cur[KEYS + TARGETS].astype(float)
            v = cur[cur.is_valid]
            it = len(new) // 6
            seeds = [42 + SEED * 7 + i * 100 for i in range(5)]
            surr = SurrogateEnsemble(random_seeds=seeds)
            surr.feasibility_clf = RandomForestClassifier(n_estimators=100, random_state=42 + SEED)
            surr.fit(v[KEYS].values, v[TARGETS].values)
            surr.fit_feasibility(cur[KEYS].values, cur.is_valid.values)
            top = v.nlargest(20, "range_calm_km")[KEYS].values
            cands = SurrogateOptimizer(popsize=POP, maxiter=MAXIT, tol=0.001, workers=1, beta=0.3, seed=1000 * SEED + it + 7919 * stall).search(surr, seed_designs=top)
            have = cur[KEYS].values; keep = []
            for c in cands:
                x = np.array([c[k] for k in KEYS])
                if not (np.abs(have - x) <= 1e-4).all(axis=1).any():
                    keep.append(x)
            keep = keep[: N_EXTRA - len(new)]
            stall = stall + 1 if not keep else 0; stalls_total += (not keep)
            if stall >= 40: print('AL exhausted: 40 consecutive iterations without a new candidate', flush=True); break
            rows = list(pool.imap(work, [(x, len(new) + i) for i, x in enumerate(keep)]))
            new = pd.concat([new, pd.DataFrame(rows)], ignore_index=True); new.to_csv(fn, index=False)
            last = (time.time() - ti) * 1.15
            print(f"iter {it:3d} stall={stall} n={len(new):3d} best={new[new.is_valid].range_calm_km.max():.1f} valid={new.is_valid.mean():.2f} ({time.time()-ti:.0f}s)", flush=True)
    new["arm"] = "al"; new["seed"] = SEED
    print("done", new[new.is_valid].range_calm_km.max())
