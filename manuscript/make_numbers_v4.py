"""v4 numbers, tables and figures for the equal-budget benchmark (2026-11-15). Run AFTER make_numbers.py; writes
numbers_v4.tex, tables/tab_budget_v4.tex, tab_tests_v4.tex, tab_factorial.tex, tab_asrun.tex, figures/fig_equal_budget.pdf,
figures/fig_repair.pdf and results_v3/v4_summary/v4_summary.json. Nothing here is typed by hand: every value comes from the run CSVs.
What is new relative to make_numbers.py:
  * AL-R: active learning with the washout repair (results_v3/al_repair, al_repair_extra)
  * TuRBO and adaptive local search with the energy-basis fix (results_v3/turbo_fix, adapt_fix); the as-run versions are kept and tabulated
  * 2x2 design {classifier weight} x {washout repair} for the active-learning loop (al_nofeas, al_repair_nofeas, al_hard)
  * Holm correction over the headline family; analytic chord-monotonicity audit of the loft
Only complete runs (287 evaluations) are used, except AL replication 2 (stopped by exhaustion at 117) and AL-R seed 16 (stopped by exhaustion at 83), which are kept as run.
usage: python3 make_numbers_v4.py"""
import glob, json, os, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(f"{HERE}/../../../..")
R, V3, DB = f"{REPO}/revision_runs/results_revision", f"{REPO}/revision_runs/results_v3", f"{REPO}/flying_wing_optimizer/results/database.csv"
OUTJ = f"{V3}/v4_summary"; os.makedirs(OUTJ, exist_ok=True)
F = (48 * 5.0 * 3.6 * 0.80 / 3.400) / 160.0; N = 287
TW = ["theta0", "theta1", "theta2", "tip_twist"]
db = pd.read_csv(DB, low_memory=False); db["is_valid"] = db.is_valid.astype(bool)
p1 = db[db.evaluation_type.str.startswith("Phase1")]; POOL = p1[p1.is_valid].range_calm_km.max() * F
orig = db[~db.evaluation_type.str.startswith("Phase1")].rename(columns={"range_calm_km": "r"}).reset_index(drop=True)


def load(pats, rc, keep_short=()):
    runs = {}
    for pat in pats:
        for f in sorted(glob.glob(pat), key=lambda s: int(re.findall(r"seed(\d+)", s)[-1])):
            d = pd.read_csv(f); d["is_valid"] = d.is_valid.astype(bool); d = d.rename(columns={rc: "r"})
            tag = os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f)
            if len(d) >= N or any(k in tag for k in keep_short):
                runs[tag] = d.iloc[:N].reset_index(drop=True)
    return runs


ARMS = {  # key: (label, runs)
    "Turbo": ("TuRBO", load([f"{V3}/turbo_fix/turbo_seed*.csv", f"{V3}/turbo_fix_extra/turbo_seed*.csv"], "range_km")),   # seeds 0-9 and 10-19 (B15)
    "ALR": ("Active learning with washout repair (AL-R)", load([f"{V3}/al_repair/al_seed*.csv", f"{V3}/al_repair_extra/al_seed*.csv"], "range_calm_km", keep_short=("al_repair_extra/al_seed16",))),
    "AL": ("Active learning as published (AL)", {**{"orig": orig}, **load([f"{R}/al/al_seed?.csv"], "range_calm_km", keep_short=("al/al_seed2",))}),
    "Adapt": ("Adaptive local search, 10\\%", load([f"{V3}/adapt_fix/adapt10_seed*.csv"], "range_km")),
    "LocTen": ("Static local search, 10\\%", load([f"{R}/controls/local10_seed?.csv"], "range_km")),
    "LocFive": ("Static local search, 5\\%", load([f"{R}/controls/local5_seed?.csv", f"{V3}/controls_extra/local5_seed?.csv"], "range_km")),
    "Lhs": ("Fresh Latin hypercube", load([f"{R}/controls/lhs_seed?.csv", f"{V3}/controls_extra/lhs_seed?.csv"], "range_km")),
    "Uni": ("Uniform random", load([f"{R}/controls/uniform_seed?.csv", f"{V3}/controls_extra/uniform_seed?.csv"], "range_km")),
    "ALRc": ("AL-R, complete runs only", load([f"{V3}/al_repair/al_seed*.csv", f"{V3}/al_repair_extra/al_seed*.csv"], "range_calm_km")),
    # as run before the basis fix
    "TurboOld": ("TuRBO, first implementation", load([f"{R}/turbo/turbo_seed?.csv", f"{V3}/turbo_extra/turbo_seed?.csv"], "range_km")),
    "AdaptOld": ("Adaptive local search, first implementation", load([f"{R}/controls/adapt10_seed?.csv"], "range_km")),
    # classifier-weight ablation
    "ALrep": ("AL, sigmoid weight, no repair", load([f"{R}/al/al_seed?.csv"], "range_calm_km", keep_short=("al/al_seed2",))),
    "ALhard": ("AL, hard gate, no repair", load([f"{V3}/al_hard/al_seed?.csv"], "range_calm_km")),
    "ALnone": ("AL, no weight, no repair", load([f"{V3}/al_nofeas/al_seed?.csv"], "range_calm_km")),
    "ALRnone": ("AL, no weight, with repair", load([f"{V3}/al_repair_nofeas/al_seed?.csv"], "range_calm_km")),
    "ALRten": ("AL, sigmoid weight, with repair", load([f"{V3}/al_repair/al_seed?.csv"], "range_calm_km")),
    "TurboFeasFix": ("TuRBO, sigmoid weight", load([f"{V3}/turbo_feas_fix/turbo_seed?.csv"], "range_km")),
    "TurboFeas": ("TuRBO, first implementation, sigmoid weight", load([f"{V3}/turbo_feas/turbo_seed?.csv"], "range_km")),
}


def best(d):
    return d[d.is_valid].r.max() * F if d.is_valid.any() else np.nan


def curve(d):
    x = d.r.where(d.is_valid, 0.0).cummax().values * F
    return np.concatenate([x, np.full(max(0, N - len(x)), x[-1])])[:N]


def wash(d):
    return float((np.diff(d[TW].values.astype(float), axis=1) > 1e-12).any(axis=1).mean())


S = {}
for k, (lab, runs) in ARMS.items():
    if not runs: S[k] = dict(label=lab, n=0); continue
    b = np.array([best(d) for d in runs.values()]); cv = np.array([curve(d) for d in runs.values()])
    first = [int(np.argmax(c > POOL + 1e-9)) + 1 if (c > POOL + 1e-9).any() else None for c in cv]
    fb = [x for x in first if x is not None]
    S[k] = dict(label=lab, n=len(b), best=b.tolist(), mean=float(np.nanmean(b)) if np.isfinite(b).any() else float("nan"),
                sd=float(np.nanstd(b, ddof=1)) if np.isfinite(b).sum() > 1 else float("nan"), mn=float(np.nanmin(b)) if np.isfinite(b).any() else float("nan"),
                mx=float(np.nanmax(b)) if np.isfinite(b).any() else float("nan"), n_novalid=int((~np.isfinite(b)).sum()),
                at60=float(np.maximum(cv, POOL)[:, 59].mean()), at100=float(np.maximum(cv, POOL)[:, 99].mean()), at150=float(np.maximum(cv, POOL)[:, 149].mean()), at200=float(np.maximum(cv, POOL)[:, 199].mean()),
                valid=float(np.mean([d.is_valid.mean() for d in runs.values()])), first60=float(np.mean([d.is_valid.iloc[:60].mean() for d in runs.values()])),
                wash=float(np.mean([wash(d) for d in runs.values()])), beat=len(fb), evals=float(np.median(fb)) if fb else float("nan"), runs=list(runs))


def boot(a, b, n=20000):
    r = np.random.default_rng(0); d = r.choice(a, (n, len(a))).mean(1) - r.choice(b, (n, len(b))).mean(1); return np.percentile(d, [2.5, 97.5])


def pair(x, y):
    if S[x]["n"] < 2 or S[y]["n"] < 2: return None
    a, b = np.array(S[x]["best"]), np.array(S[y]["best"]); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2: return None
    t = stats.ttest_ind(a, b, equal_var=False); u = stats.mannwhitneyu(a, b, alternative="two-sided"); ci = boot(a, b)
    return dict(a=x, b=y, na=len(a), nb=len(b), d=float(a.mean() - b.mean()), pct=float(100 * (a.mean() - b.mean()) / b.mean()), p=float(t.pvalue),
                u=float(u.pvalue), lo=float(ci[0]), hi=float(ci[1]), ps=float(u.statistic / (len(a) * len(b))))


FAMILY = [("Turbo", "ALR"), ("ALR", "AL"), ("ALR", "Adapt"), ("Turbo", "Adapt"), ("AL", "Adapt"), ("Adapt", "LocTen"), ("AL", "LocTen")]
EXTRA = [("Turbo", "ALRc"), ("ALRc", "AL"), ("ALRc", "Adapt"), ("Turbo", "AL"), ("Turbo", "TurboOld"), ("Adapt", "AdaptOld"), ("ALhard", "ALrep"), ("ALRten", "ALrep"), ("ALRnone", "ALRten"),
         ("TurboFeas", "TurboOld"), ("TurboFeasFix", "Turbo"), ("ALR", "TurboOld"), ("ALhard", "ALRten")]
P = {f"{x}v{y}": pair(x, y) for x, y in FAMILY + EXTRA}
fam = [f"{x}v{y}" for x, y in FAMILY if P[f"{x}v{y}"]]
order = sorted(fam, key=lambda k: P[k]["p"]); m = len(order); run = 0.0
for i, k in enumerate(order):  # Holm step-down
    run = max(run, min(1.0, (m - i) * P[k]["p"])); P[k]["holm"] = run

# paired-by-seed check of the basis fix (same seeds, same random streams up to the first divergence)
def paired(x, y):
    sx = {int(re.findall(r"seed(\d+)", k)[-1]): v for k, v in zip(S[x].get("runs", []), S[x].get("best", []))}
    sy = {int(re.findall(r"seed(\d+)", k)[-1]): v for k, v in zip(S[y].get("runs", []), S[y].get("best", []))}
    c = sorted(set(sx) & set(sy))
    if len(c) < 3: return None
    d = np.array([sx[s] - sy[s] for s in c]); return dict(n=len(c), d=float(d.mean()), p=float(stats.ttest_1samp(d, 0).pvalue), w=float(stats.wilcoxon(d).pvalue))
PAIRED = {"Turbo": paired("Turbo", "TurboOld"), "Adapt": paired("Adapt", "AdaptOld")}

# analytic chord-monotonicity audit of the loft (Eq. mono): c(xi) - c_r = y_b [t xi (1-xi)^2 - delta xi]
v = db[db.is_valid]; cb = np.clip(v.c_blend, 0.25, v.c_root - 0.05).values; yb = np.clip(v.y_blend, 0.12, 0.60).values
t = np.tan(np.radians(v.sweep_inner.values)); dl = (v.c_root.values - cb) / yb; xi = np.linspace(0, 1, 4001)[:, None]
rise = (yb * (t * xi * (1 - xi) ** 2 - dl * xi)).max(axis=0); viol = t > dl
MONO = dict(n=int(len(v)), viol=float(viol.mean()), median_mm=float(np.median(rise[viol]) * 1e3), gt1=float((rise > 1e-3).mean()), max_mm=float(rise.max() * 1e3),
            max_pct_root=float((rise / v.c_root.values).max() * 100))
WASH = dict(orig=wash(orig), repl=float(np.mean([wash(d) for k, d in ARMS["ALrep"][1].items()])), repl_tot=int(sum((np.diff(d[TW].values.astype(float), axis=1) > 1e-12).any(axis=1).sum() for d in ARMS["ALrep"][1].values())),
            repl_n=int(sum(len(d) for d in ARMS["ALrep"][1].values())))
json.dump(dict(arms=S, pairs=P, paired=PAIRED, mono=MONO, wash=WASH, pool_best=POOL), open(f"{OUTJ}/v4_summary.json", "w"), indent=1, default=float)

# ---------- macros ----------
M = []
def mac(name, val):
    assert re.fullmatch(r"[A-Za-z]+", name), name; M.append(f"\\newcommand{{\\Nv{name}}}{{{val}}}")
def pf(p): return "< 0.001" if p < 0.001 else f"= {p:.3f}"
def sg(x, nd=1): return f"{x:+.{nd}f}"
for k, s in S.items():
    mac(f"N{k}", s["n"])
    if s["n"] and np.isfinite(s.get("mean", np.nan)):
        mac(f"Mean{k}", f"{s['mean']:.1f}"); mac(f"Sd{k}", f"{s['sd']:.1f}"); mac(f"Min{k}", f"{s['mn']:.1f}"); mac(f"Max{k}", f"{s['mx']:.1f}")
        mac(f"Valid{k}", f"{100*s['valid']:.0f}"); mac(f"First{k}", f"{100*s['first60']:.0f}"); mac(f"Wash{k}", f"{100*s['wash']:.0f}")
        mac(f"Gain{k}", f"{s['mean']-POOL:.1f}"); mac(f"GainPct{k}", f"{100*(s['mean']-POOL)/POOL:.1f}"); mac(f"AtHund{k}", f"{s['at100']:.1f}")
    elif s["n"]:
        mac(f"Valid{k}", f"{100*s['valid']:.0f}"); mac(f"Wash{k}", f"{100*s['wash']:.0f}")
for k, p in P.items():
    if not p: continue
    mac(f"D{k}", f"{abs(p['d']):.1f}"); mac(f"Ds{k}", sg(p["d"])); mac(f"Pct{k}", f"{abs(p['pct']):.1f}"); mac(f"P{k}", pf(p["p"])); mac(f"U{k}", pf(p["u"]))
    mac(f"CI{k}", f"[{p['lo']:.1f}, {p['hi']:.1f}]"); mac(f"Hi{k}", f"{max(abs(p['lo']), abs(p['hi'])):.1f}"); mac(f"HiPct{k}", f"{100*max(abs(p['lo']), abs(p['hi']))/S[p['b']]['mean']:.1f}")
    if "holm" in p: mac(f"H{k}", pf(p["holm"]))
for k, q in PAIRED.items():
    if q: mac(f"Paired{k}N", q["n"]); mac(f"Paired{k}D", sg(q["d"])); mac(f"Paired{k}P", pf(q["p"]))
mac("Nfamily", len(fam)); mac("Pool", f"{POOL:.1f}")
mac("MonoN", f"{MONO['n']:,}".replace(",", "{,}")); mac("MonoViolShare", f"{100*MONO['viol']:.0f}"); mac("MonoMedian", f"{MONO['median_mm']:.1f}")
mac("MonoGtOne", f"{100*MONO['gt1']:.0f}"); mac("MonoMax", f"{MONO['max_mm']:.1f}")
mac("WashOrig", f"{100*WASH['orig']:.1f}"); mac("WashRepl", f"{100*WASH['repl_tot']/WASH['repl_n']:.1f}"); mac("WashReplTot", f"{WASH['repl_tot']:,}".replace(",", "{,}")); mac("WashReplN", f"{WASH['repl_n']:,}".replace(",", "{,}"))
# attribution: share of the best strategy's gain over the pool that each simpler strategy reaches
top = max(S["Turbo"].get("mean", -1) if S["Turbo"]["n"] else -1, S["ALR"].get("mean", -1) if S["ALR"]["n"] else -1) - POOL
for k in ["LocTen", "Adapt", "AL", "ALR", "Turbo"]:
    if S[k]["n"]: mac(f"Share{k}", f"{100*(S[k]['mean']-POOL)/top:.0f}")
open(f"{HERE}/numbers_v4.tex", "w").write("% generated by make_numbers_v4.py; do not edit\n" + "\n".join(M) + "\n")

# ---------- tables ----------
SHORT = {"Turbo": "TuRBO", "ALR": "AL-R", "AL": "AL", "Adapt": "adapt10", "LocTen": "local10", "LocFive": "local5", "Lhs": "LHS", "Uni": "Uniform", "TurboOld": "TuRBO (first)",
         "AdaptOld": "adapt10 (first)", "TurboFeasFix": "TuRBO+w", "TurboFeas": "TuRBO+w (first)", "ALrep": "AL, sigmoid", "ALRten": "AL, sigmoid + R",
         "ALnone": "AL, none", "ALRnone": "AL, none + R", "ALhard": "AL, hard gate"}   # short row names (2026-11-16) so the tables are set at text size without scaling
def row(k, lab=None):
    s = S[k]; lab = lab or SHORT.get(k)
    if not s["n"]: return None
    if not np.isfinite(s.get("mean", np.nan)):
        return f"{lab or s['label']} & {s['n']} & n/a & none feasible & n/a & {100*s['valid']:.0f} & {100*s['first60']:.0f} & {100*s['wash']:.0f} & 0/{s['n']} & n/a \\\\"
    ev = "n/a" if not np.isfinite(s["evals"]) else f"{s['evals']:.0f}"
    return (f"{lab or s['label']} & {s['n']} & {(format(s['at100'], '.1f') if s['at100'] > 1 else 'none')} & {s['mean']:.1f} $\\pm$ {s['sd']:.1f} & {s['mn']:.1f} to {s['mx']:.1f} & {100*s['valid']:.0f} & "
            f"{100*s['first60']:.0f} & {100*s['wash']:.0f} & {s['beat']}/{s['n']} & {ev} \\\\")
HEAD = ("\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrrrrrrr@{}}\n\\toprule\n & & \\multicolumn{3}{c}{Best feasible range (km)} & \\multicolumn{3}{c}{Share of evaluations (\\%)} & \\multicolumn{2}{c}{Beating the pool} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(l){9-10}\nStrategy & Runs & at 100 & at 287 & min to max & valid & valid$_{60}$ & fail $g_8$ & runs & evals \\\\\n\\midrule\n")
def table(fn, keys, foot=""):
    rows = [r for r in (row(k) for k in keys) if r]
    open(f"{HERE}/tables/{fn}", "w").write(HEAD + "\n".join(rows) + foot + "\n\\bottomrule\n\\end{tabular*}\n")
table("tab_budget_v4.tex", ["Turbo", "ALR", "AL", "Adapt", "LocTen", "LocFive", "Lhs", "Uni"],
      f"\n\\midrule\nPool only & n/a & n/a & {POOL:.1f} & n/a & 28 & n/a & 0 & n/a & n/a \\\\")
table("tab_factorial.tex", ["ALrep", "ALRten", "ALnone", "ALRnone", "ALhard", "Turbo", "TurboFeasFix"])
table("tab_asrun.tex", ["Turbo", "TurboOld", "Adapt", "AdaptOld", "TurboFeasFix", "TurboFeas"])
NAME = {"Turbo": "TuRBO", "ALR": "AL-R", "AL": "AL", "Adapt": "adapt10", "LocTen": "local10"}
mth = lambda v, fmt: "$" + format(v, fmt).replace("-", "{-}") + "$"   # proper minus signs
T = ["\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lcrrrrrcr@{}}\n\\toprule\n & & \\multicolumn{2}{c}{Difference} & \\multicolumn{3}{c}{Two-sided $p$} & & \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\n"
     "$A$ minus $B$ & $n_A$, $n_B$ & km & \\% & Welch & Holm & M--W & 95\\% CI (km) & $P(A{>}B)$ \\\\\n\\midrule"]
for x, y in FAMILY:
    p = P[f"{x}v{y}"]
    if p: T.append(f"{NAME[x]} $-$ {NAME[y]} & {p['na']}, {p['nb']} & {mth(p['d'], '+.1f')} & {mth(p['pct'], '+.1f')} & {pf(p['p']).replace('= ','').replace('< ','$<$')} & "
                   f"{pf(p['holm']).replace('= ','').replace('< ','$<$')} & {pf(p['u']).replace('= ','').replace('< ','$<$')} & $[{p['lo']:.1f},\\ {p['hi']:.1f}]$ & {p['ps']:.2f} \\\\")
open(f"{HERE}/tables/tab_tests_v4.tex", "w").write("\n".join(T) + "\n\\bottomrule\n\\end{tabular*}\n")

# ---------- figures ----------
BL, OR, VI, GR, INK, MUT, TEAL, BR = "#2a78d6", "#eb6834", "#6250d6", "#9a9a9a", "#0b0b0b", "#52514e", "#1f9d8a", "#8a5a2b"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUT, "axes.labelcolor": INK, "xtick.color": MUT,
                     "ytick.color": MUT, "font.family": "serif", "axes.linewidth": 0.6, "lines.linewidth": 1.4})
SHOW = [("Turbo", "TuRBO", TEAL, "-"), ("ALR", "AL-R (with washout repair)", BL, "-"), ("AL", "AL (as published)", BL, "--"), ("Adapt", "Adaptive local, 10%", VI, "-"),
        ("LocTen", "Static local, 10%", OR, "-"), ("LocFive", "Static local, 5%", BR, "-"), ("Lhs", "Fresh LHS", GR, "-"), ("Uni", "Uniform random", GR, "--")]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
x = np.arange(1, N + 1)
for k, lab, c, ls in SHOW:
    runs = ARMS[k][1]
    if not runs: continue
    cv = np.array([curve(d) for d in runs.values()]); cv = np.maximum(cv, POOL)
    ax[0].plot(x, np.nanmean(cv, axis=0), color=c, ls=ls, label=f"{lab} (n={len(runs)})")
    if k in ("Turbo", "ALR", "AL", "Adapt"): ax[0].fill_between(x, np.nanmin(cv, axis=0), np.nanmax(cv, axis=0), color=c, alpha=0.10, lw=0)
ax[0].axhline(POOL, color=INK, lw=0.6, ls=":"); ax[0].text(285, POOL + 0.6, "pool best", ha="right", va="bottom", fontsize=7, color=MUT)
ax[0].set_ylim(350, 406); ax[0].set_xlim(1, N); ax[0].set_xlabel("Evaluations after the pool"); ax[0].set_ylabel("Best feasible range so far (km)")
ax[0].legend(frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); ax[0].set_title("(a)", loc="left", fontsize=8)
rng = np.random.default_rng(1)
for i, (k, lab, c, ls) in enumerate(SHOW):
    if not S[k]["n"]: continue
    b = np.array(S[k]["best"]); ax[1].scatter(i + rng.uniform(-0.16, 0.16, len(b)), b, s=9, color=c, alpha=0.75, lw=0, marker="o" if ls == "-" else "D")
    ax[1].plot([i - 0.28, i + 0.28], [b.mean()] * 2, color=INK, lw=1.2)
ax[1].axhline(POOL, color=INK, lw=0.6, ls=":"); ax[1].set_xticks(range(len(SHOW)))
ax[1].set_xticklabels(["TuRBO", "AL-R", "AL", "adapt10", "local10", "local5", "LHS", "uniform"], rotation=40, ha="right", fontsize=7)
ax[1].set_ylabel("Best feasible range of each run (km)"); ax[1].set_title("(b)", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig(f"{HERE}/figures/fig_equal_budget.pdf"); plt.close(fig)

FA = [("ALrep", "sigmoid"), ("ALRten", "sigmoid + repair"), ("ALhard", "hard gate"), ("ALnone", "no weight"), ("ALRnone", "no weight + repair")]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.0))
for i, (k, lab) in enumerate(FA):
    if not S[k]["n"]: continue
    ax[0].bar(i, 100 * S[k]["wash"], color=OR, width=0.6); ax[1].bar(i, 100 * S[k]["valid"], color=BL, width=0.6)
    b = np.array(S[k]["best"]); b = b[np.isfinite(b)]
    if len(b):
        ax[2].scatter(i + rng.uniform(-0.15, 0.15, len(b)), b, s=9, color=BL, alpha=0.75, lw=0); ax[2].plot([i - 0.28, i + 0.28], [b.mean()] * 2, color=INK, lw=1.2)
    else:
        ax[2].text(i, 367, "no feasible design", ha="center", va="bottom", fontsize=6.5, color=MUT, rotation=90)
for a, yl, tt in zip(ax, ["Evaluations failing the washout rule (%)", "Feasible evaluations (%)", "Best feasible range of each run (km)"], "abc"):
    a.set_xticks(range(len(FA))); a.set_xticklabels([l for _, l in FA], fontsize=6.5, rotation=35, ha="right"); a.set_ylabel(yl); a.set_title(f"({tt})", loc="left", fontsize=8)
ax[2].set_ylim(365, 405)
fig.tight_layout(); fig.savefig(f"{HERE}/figures/fig_repair.pdf"); plt.close(fig)

for k in ["Turbo", "ALR", "AL", "Adapt", "TurboOld", "AdaptOld", "ALRnone", "ALnone", "ALhard", "LocTen"]:
    s = S[k]; print(f"{k:9s} n={s['n']:2d}", "" if not s["n"] else f"mean {s.get('mean', float('nan')):.1f} sd {s.get('sd', float('nan')):.1f} valid {s['valid']:.2f} wash {s['wash']:.2f} novalid {s['n_novalid']}")
for k, p in P.items():
    if p: print(f"{k:18s} d {p['d']:+6.1f} p {p['p']:.4f} holm {p.get('holm', float('nan')):.4f} mwu {p['u']:.4f} CI [{p['lo']:.1f},{p['hi']:.1f}]")
print("paired", PAIRED); print("mono", MONO); print("wash", WASH)

# ---- budget sensitivity (anytime performance): running best (pool included) at intermediate budgets -------------------
from scipy import stats as _st
BUD = [50, 100, 150, 200, 287]
def _at(k):
    cv = np.maximum(np.array([curve(d) for d in ARMS[k][1].values()]), POOL); return cv[:, [b - 1 for b in BUD]]
AT = {k: _at(k) for k in ["Turbo", "ALR", "AL", "Adapt", "LocTen"]}
T = ["\\begin{tabular}{@{}l" + "r" * len(BUD) + "@{}}\n\\toprule\nEvaluations beyond the pool & " + " & ".join(str(b) for b in BUD) + " \\\\\n\\midrule"]
for k in AT: T.append(f"{NAME[k]} ($n={len(AT[k])}$) & " + " & ".join(f"{m:.1f} $\\pm$ {s:.1f}" for m, s in zip(AT[k].mean(0), AT[k].std(0, ddof=1))) + " \\\\")
T.append("\\midrule")
for x, y in [("Turbo", "ALR"), ("ALR", "Adapt"), ("Turbo", "Adapt")]:
    cells = []
    for j in range(len(BUD)):
        a, b = AT[x][:, j], AT[y][:, j]; p = _st.ttest_ind(a, b, equal_var=False).pvalue
        cells.append(f"{a.mean()-b.mean():+.1f} ({pf(p).replace('= ','').replace('< ','$<$')})")
    T.append(f"{NAME[x]} minus {NAME[y]}, km (Welch $p$) & " + " & ".join(cells) + " \\\\")
open(f"{HERE}/tables/tab_anytime.tex", "w").write("\n".join(T) + "\n\\bottomrule\n\\end{tabular}\n")
