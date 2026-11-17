"""Benchmark figures in the style of figstyle.py (2026-11-16, second version after author review). Run LAST (after make_numbers.py,
make_paper_figures.py, make_numbers_v4.py, make_numbers_pool.py, make_figures_v5.py): it overwrites fig_equal_budget, fig_repair, fig_pool_size,
fig_forest, fig_champions, fig_protocol, fig_manifold_bounds and fig_bound_robust, and writes an editable SVG of each to figures/svg/.
No number is typed by hand: data come from the run files, results_v3/v4_summary/{v4_summary.json, champions.csv},
results_v3/pool_size_v2/pool_size_runs.csv, results_v3/bound_prices_fix, results_v3/robustness, results_revision/{mvu,bnd}.
usage: python3 make_figures_bench.py"""
import glob, json, os, re, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(f"{HERE}/../../../.."); sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import figstyle as fs; from figstyle import K, NAVY, CRIMSON, GREEN, GREY, LGREY, LBLUE, LGREEN, LRED, AMBER, LAMBER, LGREYF
fs.apply(9)
R, V3, OUT = f"{REPO}/revision_runs/results_revision", f"{REPO}/revision_runs/results_v3", f"{HERE}/figures"
F = (48 * 5.0 * 3.6 * 0.80 / 3.400) / 160.0; N = 287
J = json.load(open(f"{V3}/v4_summary/v4_summary.json")); POOL = J["pool_best"]; A = J["arms"]
db = pd.read_csv(f"{REPO}/flying_wing_optimizer/results/database.csv", low_memory=False)
orig = db[~db.evaluation_type.str.startswith("Phase1")].rename(columns={"range_calm_km": "r"}).reset_index(drop=True)
def load(pats, rc, keep=()):
    out = []
    for p in pats:
        for f in sorted(glob.glob(p), key=lambda s: int(re.findall(r"seed(\d+)", s)[-1])):
            d = pd.read_csv(f).rename(columns={rc: "r"})
            if len(d) >= N or any(k in f for k in keep): out.append(d.iloc[:N])
    return out
def curve(d, floor=None):
    x = d.r.where(d.is_valid.astype(bool), 0.0).cummax().values * F
    if floor is not None: x = np.maximum(x, floor)
    return np.concatenate([x, np.full(max(0, N - len(x)), x[-1])])[:N]
def sub(ax, s, y=-0.27): ax.text(0.5, y, s, transform=ax.transAxes, ha="center", va="top")
RUNS = {"Turbo": load([f"{V3}/turbo_fix/turbo_seed*.csv", f"{V3}/turbo_fix_extra/turbo_seed*.csv"], "range_km"), "ALR": load([f"{V3}/al_repair/al_seed*.csv", f"{V3}/al_repair_extra/al_seed*.csv"], "range_calm_km", ("al_repair_extra/al_seed16",)),
        "AL": [orig] + load([f"{R}/al/al_seed?.csv"], "range_calm_km", ("al/al_seed2",)), "Adapt": load([f"{V3}/adapt_fix/adapt10_seed*.csv"], "range_km"),
        "LocTen": load([f"{R}/controls/local10_seed?.csv"], "range_km")}
for k in RUNS: assert len(RUNS[k]) == A[k]["n"], (k, len(RUNS[k]), A[k]["n"])

# ------------------------------------------------------------ fig_equal_budget
SER = [("Turbo", "TuRBO", NAVY, "-", "o"), ("ALR", "AL-R", CRIMSON, "--", "s"), ("AL", "AL", AMBER, ":", "v"), ("Adapt", "adapt10", GREEN, "-.", "^"), ("LocTen", "local10", GREY, "-", "D")]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.3, 1]}); x = np.arange(1, N + 1)
for j, (k, lab, c, ls, mk) in enumerate(SER):
    cv = np.array([curve(d, POOL) for d in RUNS[k]]); m = cv.mean(0); se = cv.std(0, ddof=1) / np.sqrt(len(cv))
    ax[0].fill_between(x, m - se, m + se, color=c, alpha=0.12, lw=0)
    ax[0].plot(x, m, color=c, ls=ls, marker=mk, markevery=(25 + 7 * j, 50), mfc="white", mec=c, label=f"{lab} ($n={len(cv)}$)")
ax[0].axhline(POOL, color=K, lw=0.8, ls=":"); ax[0].text(282, POOL + 0.9, "best design in the pool", ha="right", va="bottom", fontsize=8)
ax[0].set_xlim(0, N); ax[0].set_ylim(350, 415); ax[0].set_xlabel("Evaluations after the pool"); ax[0].set_ylabel("Best feasible range, km")
ax[0].legend(loc="upper left", ncol=2, columnspacing=1.0, handlelength=2.6)
ORD = [("Turbo", "TuRBO"), ("ALR", "AL-R"), ("AL", "AL"), ("Adapt", "adapt10"), ("LocTen", "local10"), ("LocFive", "local5")]
COLS = {"Turbo": (NAVY, LBLUE), "ALR": (CRIMSON, LRED), "AL": (AMBER, LAMBER), "Adapt": (GREEN, LGREEN), "LocTen": (GREY, LGREYF), "LocFive": (GREY, LGREYF)}
pv = lambda q: "$p<0.001$" if q < 0.001 else f"$p={q:.3f}$"
for i, (k, lab) in enumerate(ORD):
    b = np.array(A[k]["best"], float); fs.sbox(ax[1], b[np.isfinite(b)], i, *COLS[k], seed=i)
fs.bracket(ax[1], 0, 1, 405.0, pv(J["pairs"]["TurbovALR"]["p"])); fs.bracket(ax[1], 1, 3, 409.5, pv(J["pairs"]["ALRvAdapt"]["p"])); fs.bracket(ax[1], 2, 3, 400.0, pv(J["pairs"]["ALvAdapt"]["p"]))
ax[1].axhline(POOL, color=K, lw=0.8, ls=":"); ax[1].set_xticks(range(len(ORD))); ax[1].set_xticklabels([f"{l}\n$n={A[k]['n']}$" for k, l in ORD], fontsize=8)
ax[1].set_ylim(348, 415); ax[1].set_xlim(-0.6, len(ORD) - 0.4); ax[1].set_ylabel("Best feasible range of a run, km"); ax[1].grid(axis="x", visible=False)
sub(ax[0], "(a) Mean running best $\\pm$ one standard error", -0.2); sub(ax[1], "(b) Distribution over runs (Welch $p$)", -0.2)
fig.tight_layout(); fig.subplots_adjust(bottom=0.2); fs.save(fig, f"{OUT}/fig_equal_budget.pdf"); plt.close(fig)

# ------------------------------------------------------------ fig_repair
FA = [("ALrep", "S"), ("ALRten", "S+R"), ("ALhard", "H"), ("ALnone", "N"), ("ALRnone", "N+R")]
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.9))
for i, (k, lab) in enumerate(FA):
    ax[0].bar(i, 100 * A[k]["wash"], width=0.6, color=LRED, edgecolor=CRIMSON); ax[1].bar(i, 100 * A[k]["valid"], width=0.6, color=LBLUE, edgecolor=NAVY)
    ax[0].text(i, 100 * A[k]["wash"] + 1.5, f"{100 * A[k]['wash']:.0f}", ha="center", va="bottom", fontsize=8); ax[1].text(i, 100 * A[k]["valid"] + 1.0, f"{100 * A[k]['valid']:.0f}", ha="center", va="bottom", fontsize=8)
    b = np.array(A[k]["best"], float); b = b[np.isfinite(b)]
    if len(b): fs.sbox(ax[2], b, i, *[(AMBER, LAMBER), (CRIMSON, LRED), (NAVY, LBLUE)][i], width=0.55, seed=i)
ok = [f"{int(np.isfinite(np.array(A[k]['best'], float)).sum())}/{A[k]['n']}" for k, _ in FA]
for a_, yl, t in zip(ax, ["Evaluations failing $g_8$, %", "Feasible evaluations, %", "Best feasible range of a run, km"], ["(a) Washout violations", "(b) Feasible fraction", "(c) Outcome; runs with a feasible design"]):
    a_.set_xticks(range(len(FA))); a_.set_xticklabels([l for _, l in FA]); a_.set_ylabel(yl); a_.set_xlim(-0.6, len(FA) - 0.4); a_.grid(axis="x", visible=False); sub(a_, t, -0.17)
ax[0].set_ylim(0, 112); ax[1].set_ylim(0, 74); ax[2].set_ylim(361, 410); fs.bracket(ax[2], 0, 1, 401.0, pv(J["pairs"]["ALRtenvALrep"]["p"])); fs.bracket(ax[2], 1, 2, 405.3, pv(J["pairs"]["ALhardvALRten"]["p"]))
for i, s in enumerate(ok): ax[2].text(i, 362.0, s, ha="center", va="bottom", fontsize=8, color=K if not s.startswith("0/") else CRIMSON)
fig.tight_layout(); fig.subplots_adjust(bottom=0.2); fs.save(fig, f"{OUT}/fig_repair.pdf"); plt.close(fig)

# ------------------------------------------------------------ fig_forest
NAME = {"Turbo": "TuRBO", "ALR": "AL-R", "AL": "AL", "Adapt": "adapt10", "LocTen": "local10"}
FAM = ["TurbovALR", "TurbovAdapt", "ALRvAdapt", "ALRvAL", "ALvAdapt", "AdaptvLocTen", "ALvLocTen"]
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for i, k in enumerate(FAM):
    p = J["pairs"][k]; sig = p["holm"] < 0.05; c = NAVY if sig else CRIMSON
    ax.errorbar(p["d"], i, xerr=[[p["d"] - p["lo"]], [p["hi"] - p["d"]]], fmt="s" if sig else "o", color=c, mfc=c if sig else "white", mec=c, ms=5.5, capsize=3, lw=1.3)
ax.axvline(0, color=K, lw=0.8, ls="--"); ax.set_yticks(range(len(FAM))); ax.set_yticklabels([f"{NAME[J['pairs'][k]['a']]} $-$ {NAME[J['pairs'][k]['b']]}" for k in FAM])
ax.set_ylim(len(FAM) - 0.4, -0.6); ax.set_xlim(-10, 30); ax.set_xlabel("Difference in mean best feasible range, km (bootstrap 95% interval)"); ax.grid(axis="y", visible=False)
h = [plt.Line2D([], [], ls="", marker="s", color=NAVY, mfc=NAVY, label="Holm $p<0.05$"), plt.Line2D([], [], ls="", marker="o", color=CRIMSON, mfc="white", label="Holm $p\\geq 0.05$")]
ax.legend(handles=h, loc="upper right"); fig.tight_layout(); fs.save(fig, f"{OUT}/fig_forest.pdf"); plt.close(fig)

# ------------------------------------------------------------ fig_pool_size
T = pd.read_csv(f"{V3}/pool_size_v2/pool_size_runs.csv"); SZ = [288, 1000, 3000, 9800]
EXTRA = f"{V3}/indep_pool/indep_runs.csv"   # written by make_numbers_pool.py when the independent-pool runs exist
fig, ax = plt.subplots(figsize=(5.6, 3.5))
for arm, lab, c, mk, ls, off in [("Turbo", "TuRBO", NAVY, "o", "-", 0.93), ("ALR", "AL-R", CRIMSON, "s", "--", 1.0), ("Adapt", "adapt10", GREEN, "^", "-.", 1.075)]:
    g = T[T.arm == arm].groupby("pool").final.agg(["mean", "std", "count"]).loc[SZ]
    ax.errorbar(np.array(SZ) * off, g["mean"], yerr=g["std"], color=c, marker=mk, mfc="white", mec=c, ls=ls, label=f"{lab} ($n={int(g['count'].iloc[0])}$)")
pb = T[T.arm == "Turbo"].groupby("pool").pool_best.mean().loc[SZ]; ax.plot(SZ, pb.values, color=GREY, ls=":", marker="x", label="best design in the pool")
if os.path.exists(EXTRA):
    E = pd.read_csv(EXTRA)
    for arm, c, mk, off in [("Turbo", NAVY, "o", 0.93), ("Adapt", GREEN, "^", 1.075)]:
        e = E[E.arm == arm].final; ax.errorbar([1000 * off * 1.25], [e.mean()], yerr=[e.std(ddof=1)], color=c, marker=mk, mfc=c, mec=c, ls="")
    ax.plot([1250], [E[E.arm == "Turbo"].pool_best.mean()], color=GREY, ls="", marker="x")
    ax.plot([], [], ls="", marker="o", color=K, mfc=K, label="independent pools (filled)")
ax.set_xscale("log"); ax.set_xticks(SZ); ax.set_xticklabels(["288", "1,000", "3,000", "9,800"]); ax.minorticks_off(); ax.set_xlim(220, 13000); ax.set_ylim(228, 408)
ax.set_xlabel("Number of designs in the initial pool"); ax.set_ylabel("Best feasible range after 287 evaluations, km"); ax.legend(loc="lower right")
fig.tight_layout(); fs.save(fig, f"{OUT}/fig_pool_size.pdf"); plt.close(fig)

# ------------------------------------------------------------ fig_champions
from flying_wing_optimizer.config import BOUNDS
from flying_wing_optimizer.geometry import build_wing_geometry
KEYS = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2", "tip_twist", "winglet_sweep_deg", "winglet_twist_deg", "battery_mass", "cruise_speed_ms"]
nm = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper_ratio", "theta0", "theta1", "theta2", "tip_twist", "winglet_sweep", "winglet_twist", "battery_mass", "cruise_speed_ms"]
LAB = ["Inner sweep $\\Lambda_\\mathrm{in}$", "Outer sweep $\\Lambda_\\mathrm{out}$", "Blend station $y_b$", "Root chord $c_r$", "Blend chord $c_b$", "Taper ratio $\\lambda$", "Twist $\\theta_0$", "Twist $\\theta_1$",
       "Twist $\\theta_2$", "Tip twist $\\theta_t$", "Winglet sweep $\\Lambda_w$", "Winglet twist $\\theta_w$", "Battery mass $m_b$", "Cruise speed $V$"]
LO = np.array([getattr(BOUNDS, n)[0] for n in nm]); HI = np.array([getattr(BOUNDS, n)[1] for n in nm])
C = pd.read_csv(f"{V3}/v4_summary/champions.csv"); Z = (C[KEYS].values - LO) / (HI - LO); order = [12, 13, 4, 5, 3, 2, 0, 1, 6, 7, 8, 9, 10, 11]
fig = plt.figure(figsize=(7.0, 4.4)); gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1]); ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
ax.axhspan(-0.5, 3.5, color=LBLUE, lw=0, zorder=0)
for (arm, c, mk, off) in [("TuRBO", NAVY, "o", -0.25), ("AL-R", CRIMSON, "s", 0.0), ("adapt10", GREEN, "^", 0.25)]:
    z = Z[(C.arm == arm).values][:, order]; q1, med, q3 = np.percentile(z, [25, 50, 75], axis=0)
    ax.errorbar(med, np.arange(14) + off, xerr=[med - q1, q3 - med], ls="", marker=mk, color=c, mfc="white", mec=c, capsize=2, lw=1.1, ms=4.5, label=f"{arm} ($n={len(z)}$)")
for yy in np.arange(0.5, 13.5): ax.axhline(yy, color=LGREY, lw=0.5)
ax.set_yticks(range(14)); ax.set_yticklabels([LAB[i] for i in order], fontsize=8.3); ax.set_ylim(13.5, -0.5); ax.set_xlim(-0.04, 1.04); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.grid(axis="y", visible=False)
ax.set_xticklabels(["lower\nbound", "0.25", "0.5", "0.75", "upper\nbound"]); ax.set_xlabel("Position between the bounds"); ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.42, 1.0), columnspacing=0.9, handletextpad=0.2)
GK = KEYS[:10]
def planform(r):
    ap, _ = build_wing_geometry(*[float(r[k]) for k in GK], battery_mass=float(r.battery_mass), winglet_sweep_deg=float(r.winglet_sweep_deg), winglet_twist_deg=float(r.winglet_twist_deg))
    xs = ap.wings[0].xsecs; y = np.array([float(s.xyz_le[1]) for s in xs]); le = np.array([float(s.xyz_le[0]) for s in xs]); ch = np.array([float(s.chord) for s in xs]); m = y <= 0.70 + 1e-9
    return y[m], le[m], le[m] + ch[m]
for arm, c, ls, sgn in [("TuRBO", NAVY, "-", -1), ("AL-R", CRIMSON, "--", 1)]:
    for _, r in C[C.arm == arm].sort_values("range", ascending=False).head(5).iterrows():
        y, le, te = planform(r); bx.plot(np.r_[sgn * y, sgn * y[::-1]], np.r_[le, te[::-1]], color=c, ls=ls, lw=0.9)
bx.axvline(0, color=K, lw=0.7, ls="-."); bx.set_aspect("equal"); bx.set_ylim(1.0, -0.1); bx.set_xlim(-0.8, 0.8); bx.set_xlabel("Spanwise coordinate, m"); bx.set_ylabel("Chordwise coordinate, m")
bx.text(-0.4, 0.9, "TuRBO", ha="center", color=NAVY); bx.text(0.4, 0.9, "AL-R", ha="center", color=CRIMSON)
sub(ax, "(a) Median and interquartile range over runs", -0.2); sub(bx, "(b) Planforms of the five best runs", -0.3)
fig.tight_layout(); fig.subplots_adjust(bottom=0.23); fs.save(fig, f"{OUT}/fig_champions.pdf"); plt.close(fig)

# ------------------------------------------------------------ fig_protocol: plain flow chart, same line weight everywhere, nothing clipped
fig, ax = plt.subplots(figsize=(7.0, 3.6)); ax.set_xlim(-3, 143); ax.set_ylim(-3, 71); ax.axis("off")
def blk(x, y, w, h, lines, bold=0, fc="white", ec=NAVY):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.0, clip_on=False)); n = len(lines)
    for i, l in enumerate(lines): ax.text(x + w / 2, y + h / 2 + (n - 1) * 2.3 - 4.6 * i, l, ha="center", va="center", fontsize=9, fontweight="bold" if i < bold else "normal")
def arr(x0, y0, x1, y1): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=NAVY, shrinkA=0, shrinkB=0, clip_on=False))
blk(0, 44, 30, 22, ["Shared pool", "9,800 designs", "28% feasible", f"best {POOL:.1f} km"], 1, LBLUE)
ax.add_patch(Rectangle((38, 2), 34, 64, fc="white", ec=NAVY, lw=1.0, ls=(0, (5, 3)), clip_on=False)); ax.text(55, 62.3, "Search strategy", ha="center", va="center", fontsize=9, fontweight="bold")
for i, s in enumerate(["TuRBO", "AL-R", "AL", "adapt10", "local10, local5", "LHS, uniform"]): blk(41, 50 - 9.2 * i, 28, 7.2, [s])
blk(80, 44, 26, 22, ["Equal budget", "287 evaluations", "per run,", "10 to 20 seeds"], 1, LRED, CRIMSON)
blk(114, 44, 26, 22, ["Outcome", "best feasible", "range of each run", "and its statistics"], 1)
blk(80, 4, 60, 28, ["Audits", "panel refinement, second solver,", "Euler calibration of the static margin,", "bound prices, weight ablation, pool size"], 1, LGREEN, GREEN)
arr(30, 55, 38, 55); arr(72, 55, 80, 55); arr(106, 55, 114, 55); arr(127, 44, 127, 32)
fs.save(fig, f"{OUT}/fig_protocol.pdf", bbox_inches="tight", pad_inches=0.08); plt.close(fig)

# ------------------------------------------------------------ fig_manifold_bounds (Fig. 4): same layout as the earlier figure, labels computed from the runs
def bl(p): return np.array([(lambda d: d[d.is_valid.astype(bool)].range_km.max())(pd.read_csv(f)) for f in sorted(glob.glob(p))]) * F
def cv2(p): return np.array([(lambda d: d.range_km.where(d.is_valid.astype(bool), 0.0).cummax().values[:N] * F)(pd.read_csv(f)) for f in sorted(glob.glob(p))])
cl, nc = bl(f"{R}/mvu/clamp_seed?.csv"), bl(f"{R}/mvu/noclamp_seed?.csv"); pp, ww = cv2(f"{R}/bnd/paper_seed?.csv"), cv2(f"{R}/bnd/wide_seed?.csv")
assert len(cl) == len(nc) == 10 and len(pp) == len(ww) == 10
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2))
for s in range(len(cl)): ax[0].plot([0, 1], [cl[s], nc[s]], color=GREY, lw=0.8, alpha=0.7)
ax[0].plot(np.zeros(len(cl)), cl, ls="", marker="o", color=NAVY, mfc="white", mec=NAVY); ax[0].plot(np.ones(len(nc)), nc, ls="", marker="s", color=CRIMSON, mfc="white", mec=CRIMSON)
ax[0].hlines(cl.mean(), -0.2, 0.2, color=NAVY, lw=2.2); ax[0].hlines(nc.mean(), 0.8, 1.2, color=CRIMSON, lw=2.2)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels([f"Clamp\n(mean {cl.mean():.1f} km)", f"No clamp\n(mean {nc.mean():.1f} km)"]); ax[0].set_xlim(-0.4, 1.4); ax[0].set_ylabel("Best feasible range, km"); ax[0].grid(axis="x", visible=False)
x = np.arange(1, N + 1)
for cvv, c, ls, lab in [(pp, NAVY, "-", "Paper bounds"), (ww, CRIMSON, "--", "Widened bounds")]:
    ax[1].fill_between(x, cvv.min(0), cvv.max(0), color=c, alpha=0.12, lw=0); ax[1].plot(x, cvv.mean(0), color=c, ls=ls, label=f"{lab} (mean {cvv[:, -1].mean():.1f} km)")
ax[1].set_xlim(0, N); ax[1].set_xlabel("Evaluations"); ax[1].set_ylabel("Best feasible range, km"); ax[1].legend(loc="lower right")
pw = stats.ttest_ind(ww[:, -1], pp[:, -1], equal_var=False).pvalue; pc = stats.ttest_ind(cl, nc, equal_var=False).pvalue
sub(ax[0], f"(a) Clamp against no clamp, paired by seed (Welch $p = {pc:.2f}$)", -0.3); sub(ax[1], f"(b) Bound sensitivity ($+{ww[:, -1].mean() - pp[:, -1].mean():.1f}$ km, Welch $p = {pw:.3f}$)", -0.3)
fig.tight_layout(); fig.subplots_adjust(bottom=0.27); fs.save(fig, f"{OUT}/fig_manifold_bounds.pdf"); plt.close(fig)

# ------------------------------------------------------------ fig_bound_robust (Fig. 16)
bp = pd.read_csv(f"{V3}/bound_prices_fix/bound_prices_summary.csv"); rb = pd.read_csv(f"{V3}/robustness/robustness.csv")
nm_ = {"taper_lo": "Taper ratio, lower bound\n0.55 to 0.525", "wtwist_hi": "Winglet twist, upper bound\n0 to +1 deg", "bmass_hi": "Battery mass, upper bound\n3.5 to 3.7 kg", "speed_lo": "Cruise speed, lower bound\n36.1 to 33.6 m/s"}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
for i, (_, r) in enumerate(bp.iloc[1:].iterrows()):
    sig = r.welch_p < 0.05; c = NAVY if sig else CRIMSON
    ax[0].errorbar(r.gain, i, xerr=[[r.gain - r.ci_lo], [r.ci_hi - r.gain]], fmt="s" if sig else "o", color=c, mfc=c if sig else "white", mec=c, ms=5.5, capsize=3, lw=1.3)
    ax[0].text(r.ci_hi + 1.2, i, f"{r.gain:+.1f} km", va="center", fontsize=8.5)
ax[0].axvline(0, color=K, lw=0.8, ls="--"); ax[0].set_yticks(range(4)); ax[0].set_yticklabels([nm_[k] for k in bp.arm.iloc[1:]], fontsize=8.5); ax[0].set_ylim(3.5, -0.5); ax[0].set_xlim(-8, 36)
ax[0].set_xlabel("Range gain from widening one bound, km"); ax[0].grid(axis="y", visible=False)
fb = rb[rb.kind == "fb"]; rel = {}
for d_, g in fb.groupby("design"):
    g = g.sort_values("value"); base = g.loc[(g.value - 0.82).abs().idxmin()]
    if base.range_km <= 0 or not bool(base.is_valid): continue
    for _, q in g[g.is_valid.astype(bool)].iterrows(): rel.setdefault(round(float(q.value), 3), []).append(100 * (q.range_km / base.range_km - 1))
xs_ = sorted(rel); med = [np.median(rel[v]) for v in xs_]; lo = [np.min(rel[v]) for v in xs_]; hi = [np.max(rel[v]) for v in xs_]; q1 = [np.percentile(rel[v], 25) for v in xs_]; q3 = [np.percentile(rel[v], 75) for v in xs_]
ax[1].fill_between(xs_, lo, hi, color=LBLUE, lw=0, label="all valid designs"); ax[1].fill_between(xs_, q1, q3, color="#aebbe6", lw=0, label="middle half")
ax[1].plot(xs_, med, color=NAVY, marker="o", mfc="white", mec=NAVY, label="median"); ax[1].axvline(0.82, color=GREEN, lw=1.2, ls="-.", label="value used (0.82)"); ax[1].axhline(0, color=K, lw=0.6)
ax[1].set_xticks(xs_); ax[1].set_xticklabels([f"{v:.2f}\n({len(rel[v])})" for v in xs_]); ax[1].set_ylim(min(lo) - 0.1, max(hi) + 0.1); ax[1].set_xlabel("Battery centroid, fraction of root chord"); ax[1].set_ylabel("Change of range relative to 0.82, %"); ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=7.5, columnspacing=1.0, borderaxespad=0.2)
sub(ax[0], "(a) Range price of single bounds (95% interval)", -0.4); sub(ax[1], "(b) Sensitivity to the battery centroid", -0.4)
fig.tight_layout(); fig.subplots_adjust(bottom=0.33); fs.save(fig, f"{OUT}/fig_bound_robust.pdf"); plt.close(fig); print("ok", len(rel), {v: len(rel[v]) for v in xs_})
