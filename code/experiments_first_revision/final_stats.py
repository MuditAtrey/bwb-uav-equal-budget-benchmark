"""Summary statistics for every search arm (best-of-run range on the 203.3 Wh/kg basis, valid share, evaluations to beat the pool,
anytime best at fixed evaluation counts), plus pairwise tests. Prints a report and writes final_stats.json."""
import glob, json, os, numpy as np, pandas as pd
from scipy import stats
R = os.environ.get("RESULTS_DIR", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone/revision_runs/results_revision"); F = 203.3 / 160.0; POOL = 277.75  # pool best, 160 basis (352.9 km at 203.3)
def load(pattern, col, valid="is_valid", min_evals=287, neval=None):
    out = {}
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f); ok = d[valid].astype(str).str.lower().eq("true") if d[valid].dtype == object else d[valid].astype(bool)
        n = d[neval].cumsum().values if neval else np.arange(1, len(d) + 1)
        if n[-1] < min_evals and "al_seed2" not in f: continue
        r = np.where(ok, d[col].values, 0.0) * F; best = np.maximum.accumulate(r)
        out[f.split("/")[-1]] = dict(n=n, best=best, valid=ok.mean())
    return out
def at(run, k):
    i = np.searchsorted(run["n"], k, side="right") - 1
    return run["best"][i] if i >= 0 else 0.0
ARMS = {"AL": load(f"{R}/al/al_seed*.csv", "range_calm_km"), "adapt10": load(f"{R}/controls/adapt10_seed*.csv", "range_km"),
        "local10": load(f"{R}/controls/local10_seed*.csv", "range_km"), "turbo": load(f"{R}/turbo/turbo_seed*.csv", "range_km"),
        "bal10": load(f"{R}/balance/bal10_seed*.csv", "range_km", neval="n_eval"),
        "clamp_wide": load(f"{R}/mvu/clamp_seed*.csv", "range_km"), "noclamp_wide": load(f"{R}/mvu/noclamp_seed*.csv", "range_km")}
ORIG = 379.6
rep = {}
for a, runs in ARMS.items():
    if not runs: continue
    b = np.array([at(r, 287) for r in runs.values()])
    if a == "AL": b = np.r_[b, ORIG]
    first = [int(r["n"][np.argmax(r["best"] > POOL * F)]) if (r["best"] > POOL * F).any() else None for r in runs.values()]
    rep[a] = dict(n=len(b), mean=b.mean(), sd=b.std(ddof=1) if len(b) > 1 else 0, runs=np.round(b, 1).tolist(),
                  valid=float(np.mean([r["valid"] for r in runs.values()])), first_beat=first,
                  any_k={k: float(np.mean([at(r, k) for r in runs.values()])) for k in (60, 100, 150, 200, 287)})
def test(a, b):
    x, y = np.array(rep[a]["runs"]), np.array(rep[b]["runs"]); w = stats.ttest_ind(x, y, equal_var=False)
    rng = np.random.default_rng(0); bs = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(20000)]
    return dict(diff=x.mean() - y.mean(), p_welch_2s=float(w.pvalue), p_mwu_2s=float(stats.mannwhitneyu(x, y).pvalue), ci=np.percentile(bs, [2.5, 97.5]).round(1).tolist())
pairs = [("AL", "adapt10"), ("AL", "local10"), ("adapt10", "local10"), ("turbo", "AL"), ("turbo", "adapt10"), ("bal10", "adapt10"), ("bal10", "AL"), ("clamp_wide", "noclamp_wide")]
T = {f"{a} vs {b}": test(a, b) for a, b in pairs if a in rep and b in rep and rep[a]["n"] > 1 and rep[b]["n"] > 1}
for a, v in rep.items(): print(f"{a:13s} n={v['n']:2d} mean={v['mean']:.1f} sd={v['sd']:.1f} valid={v['valid']:.2f} anytime={ {k: round(x,1) for k,x in v['any_k'].items()} } runs={v['runs']}")
for k, v in T.items(): print(f"{k:24s} diff={v['diff']:+.1f} km  Welch p={v['p_welch_2s']:.3f}  MWU p={v['p_mwu_2s']:.3f}  CI={v['ci']}")
json.dump(dict(arms={a: {k: x for k, x in v.items()} for a, v in rep.items()}, tests=T), open(f"{R}/final_stats.json", "w"), default=float, indent=1)
