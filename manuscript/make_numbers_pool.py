"""Pool-size sensitivity (B12 and B12b, 2026-11-16; runs filed by the pool size they really used, see results_v3/pool_size_v2/README.txt). Each run starts from a random subsample of POOL_N designs of the 9,800-design pool
(subsample seed 7000 + run seed, the same subsample for every strategy with that seed) and spends the same 287 evaluations.
The outcome of a run is the best feasible range among its subsample and its 287 evaluations. The full-pool rows reuse the runs of the main
benchmark with the same seeds. Writes numbers_pool.tex, tables/tab_pool.tex, figures/fig_pool_size.pdf, results_v3/pool_size_v2/pool_size_summary.json.
usage: python3 make_numbers_pool.py"""
import glob, json, os, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(f"{HERE}/../../../..")
V3, DB = f"{REPO}/revision_runs/results_v3", f"{REPO}/flying_wing_optimizer/results/database.csv"
F = (48 * 5.0 * 3.6 * 0.80 / 3.400) / 160.0; N = 287; FULL = 9800
db = pd.read_csv(DB, low_memory=False); db["is_valid"] = db.is_valid.astype(bool)
p1 = db[db.evaluation_type.str.startswith("Phase1")]; assert len(p1) == FULL
def pool_best(n, seed):
    s = p1 if n == FULL else p1.sample(n=n, random_state=7000 + seed)
    v = s[s.is_valid]; return (v.range_calm_km.max() * F if len(v) else np.nan), len(v)
SRC = {"Turbo": ("TuRBO", "turbo_{n}/turbo_seed{s}.csv", "turbo_fix/turbo_seed{s}.csv", "range_km", range(10)),
       "Adapt": ("adapt10", "adapt_{n}/adapt10_seed{s}.csv", "adapt_fix/adapt10_seed{s}.csv", "range_km", range(10)),
       "ALR": ("AL-R", "alr_{n}/al_seed{s}.csv", "al_repair/al_seed{s}.csv", "range_calm_km", range(5))}
rows = []
for arm, (lab, pat, full, rc, seeds) in SRC.items():
    for n in (288, 1000, 3000, FULL):
        for s in seeds:
            f = f"{V3}/" + (full.format(s=s) if n == FULL else "pool_size_v2/" + pat.format(n=n, s=s))
            d = pd.read_csv(f); assert len(d) >= N, f
            d = d.iloc[:N]; ok = d.is_valid.astype(bool); rb = d.loc[ok, rc].max() * F if ok.any() else np.nan
            pb, nv = pool_best(n, s)
            fin = np.nanmax([rb, pb]); cur = np.maximum(np.nan_to_num(pb), (d[rc].where(ok, 0.0) * F).cummax().values)
            rows.append(dict(arm=arm, pool=n, seed=s, pool_best=pb, pool_valid=nv, run_best=rb, final=fin, gain=fin - pb, valid=ok.mean(), at100=cur[99]))
T = pd.DataFrame(rows); T.to_csv(f"{V3}/pool_size_v2/pool_size_runs.csv", index=False)
G = T.groupby(["arm", "pool"]).agg(n=("final", "count"), pool_best=("pool_best", "mean"), mean=("final", "mean"), sd=("final", "std"), mn=("final", "min"), mx=("final", "max"),
                                   gain=("gain", "mean"), valid=("valid", "mean"), at100=("at100", "mean")).reset_index()
print(G.round(2).to_string()); print(T.pivot_table(index=["arm","seed"], columns="pool", values="final").round(1).to_string())
def vals(arm, n, seeds=None):
    t = T[(T.arm == arm) & (T.pool == n)]
    if seeds is not None: t = t[t.seed.isin(seeds)]
    return t.sort_values("seed").final.values
def welch(a, b): return float(stats.ttest_ind(a, b, equal_var=False).pvalue)
rng = np.random.default_rng(0)
def boot(a, b):
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(20000)]; return np.percentile(bs, [2.5, 97.5])
def pf(p): return "<0.001" if p < 0.001 else f"{p:.3f}"
M = []; mac = lambda k, v: M.append(f"\\newcommand{{\\Np{k}}}{{{v}}}")
W = {288: "A", 1000: "B", 3000: "C", FULL: "F"}
J = dict(groups=G.to_dict("records"), tests={})
for n in (288, 1000, 3000, FULL):
    a, b = vals("Turbo", n), vals("Adapt", n); ci = boot(a, b); p = welch(a, b)
    J["tests"][f"Turbo_minus_Adapt_{n}"] = dict(d=float(a.mean() - b.mean()), p=p, lo=float(ci[0]), hi=float(ci[1]), paired_p=float(stats.ttest_rel(a, b).pvalue))
    mac(f"TvA{W[n]}D", f"{a.mean()-b.mean():+.1f}".replace("+", "")); mac(f"TvA{W[n]}P", pf(p)); mac(f"TvA{W[n]}CI", f"[{ci[0]:.1f}, {ci[1]:.1f}]")
    for arm in SRC:
        g = G[(G.arm == arm) & (G.pool == n)]
        if len(g):
            g = g.iloc[0]; mac(f"{arm}{W[n]}Mean", f"{g['mean']:.1f}"); mac(f"{arm}{W[n]}Sd", f"{g['sd']:.1f}"); mac(f"{arm}{W[n]}Gain", f"{g['gain']:.1f}"); mac(f"{arm}{W[n]}N", int(g['n']))
    mac(f"Pool{W[n]}Best", f"{T[(T.arm=='Turbo')&(T.pool==n)].pool_best.mean():.1f}")
    if True:
        a5, r5 = vals("Turbo", n, range(5)), vals("ALR", n); p = welch(a5, r5)
        J["tests"][f"Turbo_minus_ALR_{n}_seeds0to4"] = dict(d=float(a5.mean() - r5.mean()), p=p)
        mac(f"TvR{W[n]}D", f"{a5.mean()-r5.mean():.1f}".replace("-", "$-$")); mac(f"TvR{W[n]}P", pf(p))
for arm in ("Turbo", "Adapt", "ALR"):   # effect of the pool itself: full pool minus small pool, same seeds
    for n in (288, 1000, 3000):
        a, b = vals(arm, FULL), vals(arm, n); p = welch(a, b); ci = boot(a, b); J['tests'][f'{arm}_full_minus_{n}_paired_p'] = float(stats.ttest_rel(a, b).pvalue)
        J["tests"][f"{arm}_full_minus_{n}"] = dict(d=float(a.mean() - b.mean()), p=p, lo=float(ci[0]), hi=float(ci[1]))
        mac(f"{arm}Loss{W[n]}", f"{a.mean()-b.mean():.1f}"); mac(f"{arm}Loss{W[n]}P", pf(p)); mac(f"{arm}Loss{W[n]}CI", f"[{ci[0]:.1f}, {ci[1]:.1f}]")
        mac(f"{arm}Loss{W[n]}Pct", f"{100*(a.mean()-b.mean())/a.mean():.1f}")
mac("TvAMaxP", pf(max(v["p"] for k, v in J["tests"].items() if k.startswith("Turbo_minus_Adapt")))); mac("TvRMinP", f'{min(v["p"] for k, v in J["tests"].items() if k.startswith("Turbo_minus_ALR")):.2f}')
POOLFULL = pool_best(FULL, 0)[0]; mac("FullPoolBest", f"{POOLFULL:.1f}")
t3 = T[(T.arm == "Turbo") & (T.pool == 288)]; mac("TurboABeatFull", f"{int((t3.final > POOLFULL).sum())}"); mac("TurboAMin", f"{t3.final.min():.1f}")
a3 = T[(T.arm == "Adapt") & (T.pool == 288)]; mac("AdaptABeatFull", f"{int((a3.final > POOLFULL).sum())}")
mac("EvalsA", f"{288+N}"); mac("EvalsFull", f"{FULL+N:,}".replace(",", "{,}")); mac("EvalRatio", f"{(FULL+N)/(288+N):.0f}")
J["full_pool_best"] = POOLFULL
# ---- B14: ten independent 1,000-design Latin hypercube pools (LHS seed 9000 + run seed), one per run
IND = f"{V3}/indep_pool"
if len(glob.glob(f"{IND}/turbo/turbo_seed?.csv")) == 10 and len(glob.glob(f"{IND}/adapt/adapt10_seed?.csv")) == 10:
    ir = []
    for arm, pat in [("Turbo", "turbo/turbo_seed{s}.csv"), ("Adapt", "adapt/adapt10_seed{s}.csv")]:
        for sd_ in range(10):
            d = pd.read_csv(f"{IND}/" + pat.format(s=sd_)); assert len(d) >= N; d = d.iloc[:N]; ok = d.is_valid.astype(bool)
            pl = pd.read_csv(f"{IND}/pools/pool_seed{sd_}.csv"); assert len(pl) == 1000; pv = pl[pl.is_valid.astype(bool)]; pb = pv.range_calm_km.max() * F
            rb = d.loc[ok, "range_km"].max() * F if ok.any() else np.nan
            ir.append(dict(arm=arm, seed=sd_, pool_best=pb, pool_valid=len(pv) / 1000, run_best=rb, final=np.nanmax([rb, pb]), valid=ok.mean()))
    E = pd.DataFrame(ir); E.to_csv(f"{IND}/indep_runs.csv", index=False)
    it, ia = E[E.arm == "Turbo"].sort_values("seed").final.values, E[E.arm == "Adapt"].sort_values("seed").final.values
    mac("IndTurboMean", f"{it.mean():.1f}"); mac("IndTurboSd", f"{it.std(ddof=1):.1f}"); mac("IndAdaptMean", f"{ia.mean():.1f}"); mac("IndAdaptSd", f"{ia.std(ddof=1):.1f}")
    mac("IndPoolBest", f"{E[E.arm == 'Turbo'].pool_best.mean():.1f}"); mac("IndPoolValid", f"{100 * E[E.arm == 'Turbo'].pool_valid.mean():.0f}")
    p = welch(it, ia); ci = boot(it, ia); mac("IndTvAD", f"{it.mean() - ia.mean():.1f}"); mac("IndTvAP", pf(p)); mac("IndTvACI", f"[{ci[0]:.1f}, {ci[1]:.1f}]")
    st, sa = vals("Turbo", 1000), vals("Adapt", 1000)
    for nm_, x_, y_ in [("Turbo", it, st), ("Adapt", ia, sa)]:
        mac(f"Ind{nm_}VsSub", f"{x_.mean() - y_.mean():+.1f}".replace("-", "$-$").replace("+", "$+$")); mac(f"Ind{nm_}VsSubP", pf(welch(x_, y_)))
    mac("IndTurboVsSubAbs", f"{abs(it.mean() - st.mean()):.1f}"); mac("IndTurboVsFull", f"{vals('Turbo', FULL).mean() - it.mean():.1f}"); mac("IndTurboVsFullPct", f"{100 * (vals('Turbo', FULL).mean() - it.mean()) / vals('Turbo', FULL).mean():.1f}"); mac("IndTurboVsFullP", pf(welch(vals("Turbo", FULL), it)))
    mac("IndTurboBeatFull", int((it > pool_best(FULL, 0)[0]).sum())); mac("IndAdaptBeatFull", int((ia > pool_best(FULL, 0)[0]).sum())); mac("IndEvals", f"{1000 + N:,}".replace(",", "{,}"))
    J["independent_pools"] = dict(turbo=it.tolist(), adapt=ia.tolist(), turbo_minus_adapt=float(it.mean() - ia.mean()), p=p, ci=[float(ci[0]), float(ci[1])],
                                  turbo_vs_subsample_p=welch(it, st), adapt_vs_subsample_p=welch(ia, sa), pool_best_mean=float(E[E.arm == "Turbo"].pool_best.mean()))
    print(E.groupby("arm").agg(final=("final", "mean"), sd=("final", "std"), mn=("final", "min"), pool=("pool_best", "mean"), valid=("valid", "mean")).round(2)); print(J["independent_pools"])
json.dump(J, open(f"{V3}/pool_size_v2/pool_size_summary.json", "w"), indent=1, default=float)
open(f"{HERE}/numbers_pool.tex", "w").write("% generated by make_numbers_pool.py; do not edit\n" + "\n".join(M) + "\n")
# table
L = ["% generated by make_numbers_pool.py; do not edit", "\\begin{tabular}{@{}rrrrrrrr@{}}", "\\toprule",
     "Pool size & Total evaluations & Pool best (km) & TuRBO (km) & adapt10 (km) & AL-R (km) & TuRBO minus adapt10 (km) & Welch $p$ \\\\", "\\midrule"]
for n in (288, 1000, 3000, FULL):
    c = lambda arm: (lambda g: f"{g.iloc[0]['mean']:.1f} $\\pm$ {g.iloc[0]['sd']:.1f}" if len(g) else "not run")(G[(G.arm == arm) & (G.pool == n)])
    t = J["tests"][f"Turbo_minus_Adapt_{n}"]
    L.append(f"{n:,} & {n+N:,} & {T[(T.arm=='Turbo')&(T.pool==n)].pool_best.mean():.1f} & {c('Turbo')} & {c('Adapt')} & {c('ALR')} & ${t['d']:+.1f}$ & {pf(t['p'])} \\\\".replace(",", "{,}"))
if "independent_pools" in J:
    ip = J["independent_pools"]; it_, ia_ = np.array(ip["turbo"]), np.array(ip["adapt"])
    L.append("\\midrule"); L.append(f"1{{,}}000, independent & 1{{,}}287 & {ip['pool_best_mean']:.1f} & {it_.mean():.1f} $\\pm$ {it_.std(ddof=1):.1f} & {ia_.mean():.1f} $\\pm$ {ia_.std(ddof=1):.1f} & not run & ${ip['turbo_minus_adapt']:+.1f}$ & {pf(ip['p'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]; open(f"{HERE}/tables/tab_pool.tex", "w").write("\n".join(L) + "\n")
# figure
fig, ax = plt.subplots(figsize=(6.2, 3.6)); COL = {"Turbo": "#1f4e9c", "Adapt": "#d1761a", "ALR": "#2a8a4a"}; OFF = {"Turbo": 0.94, "Adapt": 1.0, "ALR": 1.06}
for arm, (lab, *_r) in SRC.items():
    g = G[G.arm == arm]; t = T[T.arm == arm]
    ax.scatter(t.pool * OFF[arm], t.final, s=9, color=COL[arm], alpha=0.35, linewidths=0)
    ax.errorbar(g.pool * OFF[arm], g["mean"], yerr=g.sd, color=COL[arm], marker="o", ms=4.5, lw=1.4, capsize=2.5, label=f"{lab} (n = {int(g.n.iloc[0])})")
pb = T[T.arm == "Turbo"].groupby("pool").pool_best.mean(); ax.plot(pb.index, pb.values, color="0.45", ls="--", marker="s", ms=4, lw=1.2, label="Best design in the pool")
ax.set_xscale("log"); ax.set_xticks([288, 1000, 3000, FULL]); ax.set_xticklabels(["288", "1,000", "3,000", "9,800"]); ax.minorticks_off()
ax.set_xlabel("Initial pool size (designs)"); ax.set_ylabel("Best feasible range after 287 evaluations (km)"); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{HERE}/figures/fig_pool_size.pdf"); fig.savefig(f"{HERE}/figures/fig_pool_size.png", dpi=130)
print(json.dumps(J["tests"], indent=0, default=float)); print("\n".join(M[-8:]))
