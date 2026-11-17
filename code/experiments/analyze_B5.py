"""B5: range cost of each bound ("shadow prices"). Each arm = TuRBO (turbo_arm_v3.py) with ONE bound widened by 10% of its range, seeds 0-4, 287 evaluations.
Reference = plain TuRBO, same code path, seeds 0-9 with the original bounds (n = 10). Gain = mean best-of-run (203.3 Wh/kg basis) minus the reference mean;
two-sided Welch and Mann-Whitney p, bootstrap 95% CI (20,000 resamples, seed 0). 'at new bound' = share of runs whose best valid design lies within 2% of the
bound's range from the widened limit, i.e. the optimiser actually used the extra room. usage: python3 analyze_B5.py [OUTDIR]"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
R, V3 = f"{ROOT}/revision_runs/results_revision", f"{ROOT}/revision_runs/results_v3"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{V3}/bound_prices"; N = 287
EU = 48 * 18 * 0.8 / 3.4; F = EU / 160.0
ARMS = {"taper_lo": ("taper", 0.525, 0.55, 0.8, "lo", "taper ratio lower bound 0.55 -> 0.525"),
        "wtwist_hi": ("winglet_twist_deg", 1.0, 0.0, None, "hi", "winglet twist upper bound 0 -> +1 deg"),
        "bmass_hi": ("battery_mass", 3.7, 3.5, None, "hi", "battery mass upper bound 3.5 -> 3.7 kg"),
        "speed_lo": ("cruise_speed_ms", 33.61, 36.11, 61.11, "lo", "cruise speed lower bound 36.11 -> 33.61 m/s (mission requirement)")}
RANGES = {"taper": 0.25, "winglet_twist_deg": 10.0, "battery_mass": 2.0, "cruise_speed_ms": 25.0}   # full width of each variable's original range (10% = widening)
def best(f):
    d = pd.read_csv(f); d = d[d.is_valid.astype(bool)]
    return (d.range_km.max(), d.loc[d.range_km.idxmax()]) if len(d) else (np.nan, None)
ref = [best(f)[0] for f in sorted(glob.glob(f"{R}/turbo/turbo_seed?.csv")) + sorted(glob.glob(f"{V3}/turbo_extra/turbo_seed?.csv"))]
ref = np.array(ref) * F; rng = np.random.default_rng(0); rows = []
rows.append(dict(arm="TuRBO, original bounds (reference)", n=len(ref), mean=ref.mean(), sd=ref.std(ddof=1), gain=0.0))
for name, (var, newb, oldb, other, side, desc) in ARMS.items():
    fs = [f for f in sorted(glob.glob(f"{V3}/bound_prices/{name}/turbo_seed?.csv")) if len(pd.read_csv(f)) >= N]
    res = [best(f) for f in fs]; b = np.array([r[0] for r in res]) * F
    tol = 0.02 * RANGES[var]
    at = np.mean([abs(r[1][var] - newb) <= tol for r in res]) if len(res) else np.nan
    inside = np.mean([(r[1][var] < oldb) if side == "lo" else (r[1][var] > oldb) for r in res]) if len(res) else np.nan   # best design beyond the OLD bound
    diff = b.mean() - ref.mean()
    bs = [rng.choice(b, len(b)).mean() - rng.choice(ref, len(ref)).mean() for _ in range(20000)]
    rows.append(dict(arm=name, description=desc, n=len(b), mean=b.mean(), sd=b.std(ddof=1), gain=diff, welch_p=stats.ttest_ind(b, ref, equal_var=False).pvalue,
                     mwu_p=stats.mannwhitneyu(b, ref, alternative="two-sided").pvalue, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                     share_best_beyond_old_bound=inside, share_best_at_new_bound=at, mean_valid=np.mean([pd.read_csv(f).is_valid.mean() for f in fs])))
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/bound_prices_summary.csv", index=False); print(D.round(3).to_string())
