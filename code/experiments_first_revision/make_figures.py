"""Revision figures. Range on the database 160 Wh/kg basis internally, rescaled by E_NEW/160 at plot time."""
import glob, sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.environ.get("RESULTS_DIR", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone/revision_runs/results_revision"); OUTF = sys.argv[1] if len(sys.argv) > 1 else R + "/figs"
E_NEW = float(sys.argv[2]) if len(sys.argv) > 2 else 203.3; F = E_NEW / 160.0
import os; os.makedirs(OUTF, exist_ok=True)
BL, OR, VI, GR, INK, MUT = "#2a78d6", "#eb6834", "#6250d6", "#7a7a7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "xtick.color": MUT, "ytick.color": MUT, "font.family": "serif", "axes.linewidth": 0.6, "lines.linewidth": 1.4})
def curve(df, col=None, n=287):
    col = col or ("range_km" if "range_km" in df else "range_calm_km")
    v = df[col].where(df.is_valid.astype(bool), 0.0); return v.cummax().values[:n] * F
db = pd.read_csv("/tmp/work/flying_wing_optimizer/results/database.csv"); db["v"] = db.is_valid.astype(bool)
p1b = db[db.evaluation_type.str.startswith("Phase1") & db.v].range_calm_km.max()
orig = db[~db.evaluation_type.str.startswith("Phase1")].sort_values("eval_id")
oc = orig.range_calm_km.where(orig.v, 0).cummax().values * F
# ---- Figure A: equal-budget running best
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
a = ax[0]; x = np.arange(1, 288)
for arm, col, lab in [("lhs", VI, "fresh LHS"), ("uniform", GR, "uniform random"), ("local10", OR, "local search (10%)"), ("local5", "#b8501f", "local search (5%)")]:
    cs = np.array([curve(pd.read_csv(f)) for f in sorted(glob.glob(f"{R}/controls/{arm}_seed?.csv"))])
    a.plot(x, cs.mean(0), color=col, lw=1.4, ls="-" if arm != "local5" else "--"); a.fill_between(x, cs.min(0), cs.max(0), color=col, alpha=0.15, lw=0)
    a.text(289, cs.mean(0)[-1], lab, color=MUT, va="center", fontsize=7)
for i, f in enumerate(sorted(glob.glob(f"{R}/al/al_seed?.csv"))):
    d = pd.read_csv(f); c = curve(d); a.plot(np.arange(1, len(c) + 1), c, color=BL, lw=1.0, alpha=0.9, ls=":" if len(c) < 287 else "-")
a.plot(np.arange(1, len(oc) + 1), oc, color=INK, lw=1.2, ls="-."); a.text(289, oc[-1] + 3, "AL (original run)", fontsize=7, color=MUT, va="center")
a.text(289, curve(pd.read_csv(sorted(glob.glob(f'{R}/al/al_seed?.csv'))[0]))[-1] + 8, "AL replications", fontsize=7, color=BL, va="center")
a.axhline(p1b * F, color=MUT, lw=0.6, ls=(0, (2, 2))); a.text(3, p1b * F + 3, "best of 9,800-sample Phase 1 pool", fontsize=6.5, color=MUT)
a.set_xlim(0, 287); a.set_ylim(150 * F, 330 * F); a.set_xlabel("Evaluations after Phase 1"); a.set_ylabel("Best feasible range [km]"); a.set_title("(a) running best, equal budget", fontsize=8, loc="left")
a.set_xlim(0, 287 + 75); a.spines["bottom"].set_bounds(0, 287)
# ---- panel b: best per seed
b = ax[1]; items = []
def best_list(pattern, col="range_km"): return [pd.read_csv(f).pipe(lambda d: d[col].where(d.is_valid.astype(bool), 0).max()) * F for f in sorted(glob.glob(pattern))]
items = [("fresh LHS", VI, best_list(f"{R}/controls/lhs_seed?.csv")), ("uniform", GR, best_list(f"{R}/controls/uniform_seed?.csv")),
         ("local 5%", "#b8501f", best_list(f"{R}/controls/local5_seed?.csv")), ("local 10%", OR, best_list(f"{R}/controls/local10_seed?.csv")),
         ("AL", BL, best_list(f"{R}/al/al_seed?.csv", "range_calm_km"))]
for k, (lab, col, vals) in enumerate(items):
    yy = np.arange(len(vals)); b.scatter(np.full(len(vals), k) + (yy - (len(vals) - 1) / 2) * 0.08, vals, s=14, color=col, zorder=3, edgecolor="white", linewidth=0.5)
    b.hlines(np.mean(vals), k - 0.28, k + 0.28, color=col, lw=1.6)
b.axhline(p1b * F, color=MUT, lw=0.6, ls=(0, (2, 2))); b.set_xticks(range(len(items))); b.set_xticklabels([i[0] for i in items], rotation=30, ha="right")
b.set_ylabel("Best feasible range [km]"); b.set_title("(b) best per seed (bar: mean)", fontsize=8, loc="left"); b.set_ylim(150 * F, 330 * F)
plt.tight_layout(); plt.savefig(f"{OUTF}/fig_equal_budget.pdf"); plt.close()
print("A done", {i[0]: np.round(i[2], 1).tolist() for i in items})
