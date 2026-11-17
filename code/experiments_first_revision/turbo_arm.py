"""Trust-region Bayesian optimization baseline (TuRBO-1 style, Eriksson et al. 2019), the standard strong baseline for
high-dimensional BO. Same pool, same evaluator, same budget (287 evaluations), batches of 6.
Model: Gaussian process (Matern 5/2, ARD, scikit-learn) fit to the designs inside the current trust region (at most 400, best first);
infeasible designs enter with the worst feasible value in the region (a standard penalty treatment).
Acquisition: Thompson sampling over 2,000 Sobol candidates in the trust region, with the TuRBO perturbation mask.
Trust region: side length L in normalized space, start 0.4, doubled after 3 consecutive successful batches, halved after
5 consecutive failures, restarted around the best design when L < 0.01 (as in TuRBO).
usage: NPROC=.. TBUDGET=.. python3 turbo_arm.py OUTDIR SEED_LO SEED_HI"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
ROOT = os.environ.get("DRONE_ROOT", "/tmp/work"); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, multiprocessing as mp
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
OUT = sys.argv[1]; SEEDS = range(int(sys.argv[2]), int(sys.argv[3]))
N_EXTRA, E_SCALE, BATCH = 287, 160.0 / 208.25, 6
KEYS = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2",
        "tip_twist", "winglet_sweep_deg", "winglet_twist_deg", "battery_mass", "cruise_speed_ms"]
from flying_wing_optimizer.config import BOUNDS
from flying_wing_optimizer.main import evaluate_design_vector
nm = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper_ratio", "theta0", "theta1", "theta2", "tip_twist",
      "winglet_sweep", "winglet_twist", "battery_mass", "cruise_speed_ms"]
LOW = np.array([getattr(BOUNDS, n)[0] for n in nm]); HIGH = np.array([getattr(BOUNDS, n)[1] for n in nm]); D = 14

def fix(x):
    x = x.copy(); x[6:10] = np.sort(x[6:10])[::-1]; return x

def work(a):
    vec, k, seed = a
    try:
        r = evaluate_design_vector(list(vec), 970000 + k, f"turbo_{seed}")
        return dict(arm="turbo", seed=seed, k=k, is_valid=bool(r["is_valid"]), range_km=float(r["range_calm_km"]) * E_SCALE,
                    L_D=float(r["L_D"]), SM=float(r["SM"]), Cm=float(r["Cm_trim"]), **{n: float(vec[i]) for i, n in enumerate(KEYS)})
    except Exception:
        return dict(arm="turbo", seed=seed, k=k, is_valid=False, range_km=0.0, L_D=0.0, SM=0.0, Cm=0.0, **{n: float(vec[i]) for i, n in enumerate(KEYS)})

def propose(Xn, y, ok, center, L, rng, nb):
    """Xn normalized designs, y values (valid range), ok mask. Returns nb normalized candidates."""
    d = np.abs(Xn - center).max(axis=1); inn = d <= L / 2 + 1e-12
    idx = np.where(inn)[0]
    if len(idx) < 20:
        idx = np.argsort(d)[:200]
    idx = idx[np.argsort(-np.where(ok[idx], y[idx], -np.inf))][:400]
    Xt = Xn[idx]; yt = y[idx].copy(); okt = ok[idx]
    worst = yt[okt].min() if okt.any() else 0.0; yt[~okt] = worst
    mu, sd = yt.mean(), yt.std() + 1e-9; yz = (yt - mu) / sd
    k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.full(D, 0.5), length_scale_bounds=(5e-3, 2.0), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1))
    gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=0, random_state=int(rng.integers(1e9))).fit(Xt, yz)
    ls = gp.kernel_.k1.k2.length_scale; w = ls / np.exp(np.mean(np.log(ls)))
    lo = np.clip(center - w * L / 2, 0, 1); hi = np.clip(center + w * L / 2, 0, 1)
    sob = qmc.Sobol(D, scramble=True, seed=int(rng.integers(1e9))).random(2000)
    pert = lo + (hi - lo) * sob
    prob = min(20.0 / D, 1.0); mask = rng.random((2000, D)) <= prob
    mask[np.where(mask.sum(1) == 0)[0], rng.integers(0, D, (mask.sum(1) == 0).sum())] = True
    cand = np.where(mask, pert, center)
    out = []
    for _ in range(nb):
        f = gp.sample_y(cand, 1, random_state=int(rng.integers(1e9))).ravel()
        j = int(np.argmax(f)); out.append(cand[j]); cand = np.delete(cand, j, axis=0)
    return np.array(out)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); TB = float(os.environ.get("TBUDGET", "140"))
    db = pd.read_csv(ROOT + "/flying_wing_optimizer/results/database.csv"); db["is_valid"] = db["is_valid"].astype(bool)
    p1 = db[db.evaluation_type.str.startswith("Phase1")]
    PX = (p1[KEYS].values - LOW) / (HIGH - LOW); Pok = p1.is_valid.values; Py = np.where(Pok, p1.range_calm_km.values, 0.0)
    with mp.Pool(int(os.environ.get("NPROC", "2"))) as pool:
        for seed in SEEDS:
            fn = f"{OUT}/turbo_seed{seed}.csv"; st = f"{OUT}/turbo_seed{seed}.state.npy"
            new = pd.read_csv(fn) if os.path.exists(fn) else pd.DataFrame()
            L, succ, fail = (np.load(st).tolist() if os.path.exists(st) else [0.4, 0, 0])
            rng = np.random.default_rng(40_000 + seed + len(new))
            while len(new) < N_EXTRA:
                if time.time() - t0 > TB:
                    np.save(st, np.array([L, succ, fail])); print("time budget reached; rerun to continue", flush=True); sys.exit(0)
                if len(new):
                    NX = (new[KEYS].values - LOW) / (HIGH - LOW); Nok = new.is_valid.values.astype(bool); Ny = np.where(Nok, new.range_km.values / E_SCALE, 0.0)
                    Xn = np.vstack([PX, NX]); ok = np.r_[Pok, Nok]; y = np.r_[Py, Ny]
                else:
                    Xn, ok, y = PX, Pok, Py
                best_before = y[ok].max(); center = Xn[np.argmax(np.where(ok, y, -np.inf))]
                nb = min(BATCH, N_EXTRA - len(new))
                C = propose(Xn, y, ok, center, L, rng, nb)
                batch = [fix(LOW + c * (HIGH - LOW)) for c in C]
                rows = list(pool.imap(work, [(x, len(new) + i, seed) for i, x in enumerate(batch)]))
                new = pd.concat([new, pd.DataFrame(rows)], ignore_index=True); new.to_csv(fn, index=False)
                bv = new[new.is_valid].range_km.max() / E_SCALE if new.is_valid.any() else -1
                if bv > best_before + 1e-3 * abs(best_before): succ, fail = succ + 1, 0
                else: succ, fail = 0, fail + 1
                if succ >= 3: L, succ = min(2 * L, 1.6), 0
                if fail >= 5: L, fail = L / 2, 0
                if L < 0.01: L, succ, fail = 0.4, 0, 0
                np.save(st, np.array([L, succ, fail]))
            print(f"seed {seed} done: best {new[new.is_valid].range_km.max():.1f} valid {new.is_valid.mean():.2f}", flush=True)
