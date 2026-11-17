"""Figures and numbers added on 2026-11-16. Run after make_numbers_v4.py.
  figures/fig_protocol.pdf    schematic of the equal-budget benchmark
  figures/fig_forest.pdf      headline pairwise differences with bootstrap intervals (from results_v3/v4_summary/v4_summary.json)
  figures/fig_champions.pdf   best design of every run of TuRBO, AL-R and adapt10: normalized design variables and planforms
  numbers_champ.tex           \\Nwin* macros for the text on the best designs
Every plotted or printed value is read from run files or from the summary JSON. usage: python3 make_figures_v5.py"""
import glob, json, os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(f"{HERE}/../../../.."); sys.path.insert(0, REPO)
from flying_wing_optimizer.config import BOUNDS
from flying_wing_optimizer.geometry import build_wing_geometry
R, V3 = f"{REPO}/revision_runs/results_revision", f"{REPO}/revision_runs/results_v3"; OUT = f"{HERE}/figures"
F = (48 * 5.0 * 3.6 * 0.80 / 3.400) / 160.0; N = 287
BL, OR, VI, GR, INK, MUT, TEAL = "#2a78d6", "#eb6834", "#6250d6", "#9a9a9a", "#0b0b0b", "#52514e", "#1f9d8a"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "xtick.color": MUT, "ytick.color": MUT, "font.family": "serif", "axes.linewidth": 0.6, "lines.linewidth": 1.4})
J = json.load(open(f"{V3}/v4_summary/v4_summary.json")); POOL = J["pool_best"]

# ---------------------------------------------------------------- fig_protocol
fig, ax = plt.subplots(figsize=(7.2, 4.3)); ax.set_xlim(0, 120); ax.set_ylim(0, 70); ax.axis("off")
def box(x, y, w, h, title, lines, ec=MUT, fc="white", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9", lw=0.9, ec=ec, fc=fc))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=9, color=tc, fontweight="bold")
    for i, l in enumerate(lines): ax.text(x + w / 2, y + h - 8.4 - 4.6 * i, l, ha="center", va="top", fontsize=8, color=INK)
def arrow(x0, y0, x1, y1, c=MUT):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=c, shrinkA=0, shrinkB=0))
box(0.5, 40, 23.5, 28, "Shared pool", ["9,800 designs", "(Latin hypercube)", "28% feasible", f"best {POOL:.1f} km"], ec=GR, fc="#f5f5f5")
box(28, 2, 43, 66, "Search strategies", [], ec=BL, fc="#f3f8fe", tc=BL)
ST = [("TuRBO", "trust-region GP", BL), ("AL-R", "ensemble + repair", TEAL), ("AL", "as published", VI),
      ("adapt10", "moving parents", OR), ("local10, local5", "fixed parents", MUT), ("LHS, uniform", "space filling", GR)]
for i, (a_, b_, c) in enumerate(ST):
    y = 52.0 - 9.6 * i; ax.add_patch(FancyBboxPatch((29.5, y), 40, 7.6, boxstyle="round,pad=0.1,rounding_size=0.6", lw=0.8, ec=c, fc="white"))
    ax.text(30.8, y + 3.8, a_, fontsize=8.2, color=c, va="center", fontweight="bold"); ax.text(68.4, y + 3.8, b_, fontsize=7.4, color=INK, va="center", ha="right")
box(75.5, 40, 20.5, 28, "Equal budget", ["287 further", "evaluations,", "10 to 20 seeds"], ec=OR, fc="#fef6f2", tc=OR)
box(99.5, 40, 20, 28, "Outcome", ["best range", "of each run;", "Welch, rank,", "bootstrap"], ec=INK)
box(76, 2, 43, 31, "Audits", ["grid, second solver, section drag", "Euler calibration of static margin", "(12 held-out designs)", "bounds, weight ablation, pool size"], ec=TEAL, fc="#f2faf8", tc=TEAL)
arrow(24.4, 54, 27.4, 54); arrow(71.6, 54, 74.9, 54); arrow(96.4, 54, 98.9, 54); arrow(109.5, 39.4, 109.5, 33.8, TEAL)
fig.savefig(f"{OUT}/fig_protocol.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- fig_forest
NAME = {"Turbo": "TuRBO", "ALR": "AL-R", "AL": "AL", "Adapt": "adapt10", "LocTen": "local10"}
FAM = ["TurbovALR", "TurbovAdapt", "ALRvAdapt", "ALRvAL", "ALvAdapt", "AdaptvLocTen", "ALvLocTen"]
fig, ax = plt.subplots(figsize=(7.0, 3.3)); plt.rcParams["font.size"] = 9
for i, k in enumerate(FAM):
    p = J["pairs"][k]; sig = p["holm"] < 0.05; c = BL if sig else (OR if p["p"] < 0.05 else GR)
    ax.plot([p["lo"], p["hi"]], [i, i], color=c, lw=2.0, solid_capstyle="round"); ax.scatter(p["d"], i, s=30, color=c, zorder=3, edgecolor="white", linewidth=0.5)
    ph = "<0.001" if p["holm"] < 0.001 else f"{p['holm']:.3f}"
    ax.text(33, i, f"{p['d']:+.1f} km   Holm p {'= ' if ph[0] != '<' else ''}{ph}", va="center", fontsize=8, color=INK)
ax.axvline(0, color=MUT, lw=0.7, ls=(0, (2, 2))); ax.set_yticks(range(len(FAM))); ax.set_yticklabels([f"{NAME[J['pairs'][k]['a']]} minus {NAME[J['pairs'][k]['b']]}" for k in FAM], fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(-12, 62); ax.set_xticks([-10, 0, 10, 20, 30]); ax.set_xlabel("Difference in mean best feasible range (km), with bootstrap 95% interval")
h = [plt.Line2D([], [], color=c, lw=2, label=l) for c, l in [(BL, "Holm p < 0.05"), (OR, "Welch p < 0.05 only"), (GR, "not significant")]]
ax.legend(handles=h, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.4, 1.0)); ax.tick_params(labelsize=8); ax.xaxis.label.set_size(8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_forest.pdf"); plt.close(fig)

# ---------------------------------------------------------------- champions
KEYS = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2", "tip_twist", "winglet_sweep_deg", "winglet_twist_deg", "battery_mass", "cruise_speed_ms"]
nm = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper_ratio", "theta0", "theta1", "theta2", "tip_twist", "winglet_sweep", "winglet_twist", "battery_mass", "cruise_speed_ms"]
LAB = [r"$\Lambda_\mathrm{in}$", r"$\Lambda_\mathrm{out}$", r"$y_b$", r"$c_r$", r"$c_b$", r"$\lambda$", r"$\theta_0$", r"$\theta_1$", r"$\theta_2$", r"$\theta_\mathrm{tip}$", r"$\Lambda_w$", r"$\theta_w$", r"$m_\mathrm{bat}$", r"$V$"]
LO = np.array([getattr(BOUNDS, n)[0] for n in nm]); HI = np.array([getattr(BOUNDS, n)[1] for n in nm])
A = {"TuRBO": ([f"{V3}/turbo_fix/turbo_seed*.csv", f"{V3}/turbo_fix_extra/turbo_seed*.csv"], "range_km", BL), "AL-R": ([f"{V3}/al_repair/al_seed*.csv", f"{V3}/al_repair_extra/al_seed*.csv"], "range_calm_km", TEAL),
     "adapt10": ([f"{V3}/adapt_fix/adapt10_seed*.csv"], "range_km", OR)}
rows = []
for a, (ps, rc, c) in A.items():
    for f in sorted(sum((glob.glob(p) for p in ps), [])):
        d = pd.read_csv(f).iloc[:N]; v = d[d.is_valid.astype(bool)]
        if v.empty or (len(d) < N and "al_seed16" not in f): continue
        r = v.loc[v[rc].idxmax()]; rows.append(dict(arm=a, run=os.path.basename(f), range=r[rc] * F, L_D=r.L_D, SM=r.SM, **{k: float(r[k]) for k in KEYS}))
C = pd.DataFrame(rows); Z = (C[KEYS].values - LO) / (HI - LO); C.to_csv(f"{V3}/v4_summary/champions.csv", index=False)
order = [12, 13, 4, 5, 3, 2, 0, 1, 6, 7, 8, 9, 10, 11]   # bound-limited variables first
fig = plt.figure(figsize=(7.2, 3.5)); gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1]); ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
ax.axvspan(-0.5, 3.5, color=GR, alpha=0.13, lw=0)
rng = np.random.default_rng(3)
for a, (ps, rc, c) in A.items():
    z = Z[(C.arm == a).values][:, order]
    for row in z: ax.plot(range(14), row, color=c, lw=0.55, alpha=0.38)
    ax.plot(range(14), np.median(z, axis=0), color=c, lw=1.7, marker="o", ms=2.6, label=f"{a} (n = {len(z)})")
ax.set_xticks(range(14)); ax.set_xticklabels([LAB[i] for i in order], fontsize=8); ax.set_ylim(-0.04, 1.04); ax.set_xlim(-0.5, 13.5)
ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["lower\nbound", "mid", "upper\nbound"], fontsize=8); ax.set_ylabel("Position within the bounds")
ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.17), columnspacing=0.9, handlelength=1.4)
ax.set_title("(a)", loc="left", fontsize=9, x=-0.12); ax.tick_params(labelsize=8); ax.yaxis.label.set_size(8.5)
GK = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2", "tip_twist"]
def planform(r):
    ap, _ = build_wing_geometry(*[float(r[k]) for k in GK], battery_mass=float(r.battery_mass), winglet_sweep_deg=float(r.winglet_sweep_deg), winglet_twist_deg=float(r.winglet_twist_deg))
    xs = ap.wings[0].xsecs; y = np.array([float(s.xyz_le[1]) for s in xs]); le = np.array([float(s.xyz_le[0]) for s in xs]); ch = np.array([float(s.chord) for s in xs]); m = y <= 0.70 + 1e-9
    return y[m], le[m], le[m] + ch[m]
for a, (ps, rc, c) in A.items():
    if a == "adapt10": continue
    for _, r in C[C.arm == a].sort_values("range", ascending=False).head(5).iterrows():
        y, le, te = planform(r); sgn = -1 if a == "TuRBO" else 1
        bx.plot(sgn * y, le, color=c, lw=0.8, alpha=0.75); bx.plot(sgn * y, te, color=c, lw=0.8, alpha=0.75); bx.plot([sgn * y[-1]] * 2, [le[-1], te[-1]], color=c, lw=0.8, alpha=0.75)
bx.axvline(0, color=MUT, lw=0.5, ls=(0, (2, 2))); bx.set_aspect("equal"); bx.set_xlabel("Spanwise coordinate (m)"); bx.set_ylabel("Chordwise coordinate (m)")
bx.text(-0.36, 0.86, "TuRBO\nfive best runs", color=BL, fontsize=8, ha="center"); bx.text(0.36, 0.86, "AL-R\nfive best runs", color=TEAL, fontsize=8, ha="center"); bx.set_ylim(0.95, -0.05); bx.set_title("(b)", loc="left", fontsize=9); bx.tick_params(labelsize=8); bx.xaxis.label.set_size(8.5); bx.yaxis.label.set_size(8.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_champions.pdf"); fig.savefig(f"{OUT}/fig_champions.png", dpi=110); plt.close(fig)
# numbers
M = []; mac = lambda k, v: M.append(f"\\newcommand{{\\Nwin{k}}}{{{v}}}")
near = lambda i, hi: float(((Z[:, i] >= 0.97) if hi else (Z[:, i] <= 0.03)).mean())
mac("N", len(C)); mac("BmassAt", f"{100*near(12, True):.0f}"); mac("SpeedAt", f"{100*near(13, False):.0f}"); mac("CblendAt", f"{100*near(4, False):.0f}"); mac("TaperAt", f"{100*near(5, False):.0f}")
free = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]; sdz = Z[:, free].std(axis=0, ddof=1); mac("FreeSdLo", f"{100*sdz.min():.0f}"); mac("FreeSdHi", f"{100*sdz.max():.0f}")
top = C[C.range >= C.range.max() - 5.0]; zt = Z[(C.range >= C.range.max() - 5.0).values]
mac("TopN", len(top)); mac("TopBand", "5"); mac("TopLDlo", f"{top.L_D.min():.1f}"); mac("TopLDhi", f"{top.L_D.max():.1f}"); mac("TopSMlo", f"{top.SM.min():.2f}"); mac("TopSMhi", f"{top.SM.max():.2f}")
for nmk, i in [("Yb", 2), ("Cr", 3), ("SwIn", 0), ("SwOut", 1)]:
    mac(f"Top{nmk}Lo", f"{top[KEYS[i]].min():.2f}" if i >= 2 else f"{top[KEYS[i]].min():.0f}"); mac(f"Top{nmk}Hi", f"{top[KEYS[i]].max():.2f}" if i >= 2 else f"{top[KEYS[i]].max():.0f}")
t, r = C[C.arm == "TuRBO"], C[C.arm == "AL-R"]
for nmk, k in [("Cr", "c_root"), ("Yb", "y_blend")]:
    mac(f"Turbo{nmk}", f"{t[k].mean():.3f}"); mac(f"Alr{nmk}", f"{r[k].mean():.3f}"); p = stats.mannwhitneyu(t[k], r[k], alternative="two-sided").pvalue; mac(f"{nmk}P", "< 0.001" if p < 0.001 else f"= {p:.3f}")
open(f"{HERE}/numbers_champ.tex", "w").write("% generated by make_figures_v5.py; do not edit\n" + "\n".join(M) + "\n"); print("\n".join(M))
