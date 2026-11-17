"""Figures of the manuscript. Ranges: database basis x F (F = e_u/160). Run after make_numbers.py.
usage: python3 make_paper_figures.py"""
import glob, os, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/figures"; os.makedirs(OUT, exist_ok=True)
RES = os.environ.get("RES", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone/revision_runs/results_revision")
DB = os.environ.get("DB", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone/flying_wing_optimizer/results/database.csv")
V3 = os.environ.get("V3", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone/revision_runs/results_v3")
F = (48 * 5.0 * 3.6 * 0.80 / 3.400) / 160.0
import sys; sys.path.insert(0, HERE); import figstyle as _fs; _fs.apply(8.5)   # journal style, 2026-11-16 (was a custom palette with open axes)
BL, OR, VI, GR, INK, MUT, TEAL = _fs.BLUE, _fs.RED, _fs.GREY, "#9a9a9a", _fs.K, _fs.K, _fs.GREEN
N = 287
db = pd.read_csv(DB); db["v"] = db.is_valid.astype(bool)
p1 = db[db.evaluation_type.str.startswith("Phase1") & db.v]; p2 = db[~db.evaluation_type.str.startswith("Phase1")]
P1B = p1.range_calm_km.max() * F


def curve(d, col):
    v = d[col].where(d.is_valid.astype(bool), 0.0).cummax().values * F
    v = np.where(v > 0, v, np.nan)
    return v[:N]


def ctrl(arm):
    out = []
    for f in sorted(glob.glob(f"{RES}/controls/{arm}_seed?.csv") + glob.glob(f"{V3}/controls_extra/{arm}_seed?.csv")):
        d = pd.read_csv(f)
        if len(d) >= N:
            out.append(d.iloc[:N])
    return out


def al_runs():
    runs = []
    for f in sorted(glob.glob(f"{RES}/al/al_seed?.csv")):
        d = pd.read_csv(f); s = os.path.basename(f)[7]
        log = f"/tmp/cfd/al_seed{s}.log"
        exhausted = os.path.exists(log) and "AL exhausted" in open(log).read()
        if len(d) >= N or exhausted or s == "2":
            runs.append(d)
    return runs


def turbo_runs():
    runs = []
    for f in sorted(glob.glob(f"{RES}/turbo/turbo_seed*.csv") + glob.glob(f"{V3}/turbo_extra/turbo_seed*.csv")):
        d = pd.read_csv(f)
        if len(d) >= N:
            runs.append(d.iloc[:N])
        elif len(d) > 0:
            runs.append(d)
    return runs


# ------------------------------------------------------------------ Fig 1: equal budget
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.55, 1]})
a = ax[0]; x = np.arange(1, N + 1)
ARMS = [("lhs", GR, "-", "fresh LHS"), ("uniform", "#6b6b6b", "--", "uniform"), ("local5", "#b8501f", "--", "local 5%, fixed"),
        ("local10", OR, "-", "local 10%, fixed"), ("adapt10", VI, "-", "local 10%, adaptive")]
best = {}; LAB = []
for arm, col, ls, lab in ARMS:
    rs = ctrl(arm)
    if not rs:
        continue
    cs = np.array([curve(d, "range_km") for d in rs]); best[arm] = (lab, col, cs[:, -1])
    a.plot(x, cs.mean(0), color=col, ls=ls, lw=1.4); a.fill_between(x, cs.min(0), cs.max(0), color=col, alpha=0.13, lw=0)
    LAB.append([cs.mean(0)[-1], lab, MUT])
als = al_runs()
for d in als:
    c = curve(d, "range_calm_km"); a.plot(np.arange(1, len(c) + 1), c, color=BL, lw=0.9, alpha=0.85, ls=":" if len(c) < N else "-")
oc = curve(p2.sort_values("eval_id"), "range_calm_km")
a.plot(np.arange(1, len(oc) + 1), oc, color=INK, lw=1.1, ls="-.")
LAB.append([oc[-1], "AL, original run", MUT])
mx = max(curve(d, "range_calm_km")[-1] for d in als)
LAB.append([mx, "AL replications", BL])

tr_runs = turbo_runs()
if tr_runs:
    t_cs = np.array([curve(d, "range_km") for d in tr_runs])
    best["turbo"] = ("TuRBO", TEAL, t_cs[:, -1])
    a.plot(x, t_cs.mean(0), color=TEAL, ls="-", lw=1.6)
    a.fill_between(x, t_cs.min(0), t_cs.max(0), color=TEAL, alpha=0.15, lw=0)
    LAB.append([t_cs.mean(0)[-1], "TuRBO (BO)", TEAL])

LAB.sort(key=lambda t: t[0]); ys = [t[0] for t in LAB]
for i in range(1, len(ys)):
    ys[i] = max(ys[i], ys[i - 1] + 9.5)
for (y0, lab, col), y1 in zip(LAB, ys):
    a.text(N + 4, y1, lab, color=col, va="center", fontsize=6.8)
best["AL"] = ("AL", BL, np.array([curve(d, "range_calm_km")[-1] for d in [p2.sort_values("eval_id")] + als]))
a.axhline(P1B, color=MUT, lw=0.6, ls=(0, (2, 2))); a.text(120, P1B - 9, "best of the 9,800-design pool", fontsize=6.5, color=MUT)
a.set_xlim(0, N + 95); a.set_ylim(P1B - 100, P1B + 60); a.spines["bottom"].set_bounds(0, N)
a.set_xlabel("Evaluations after the pool"); a.set_ylabel("Best feasible range [km]"); a.set_title("(a) running best", fontsize=8, loc="left")
b = ax[1]
order = ["lhs", "uniform", "local5", "local10", "adapt10", "AL", "turbo"]
labs = {"lhs": "fresh\nLHS", "uniform": "uniform", "local5": "local 5%\nfixed", "local10": "local 10%\nfixed", "adapt10": "local 10%\nadaptive", "AL": "active\nlearning", "turbo": "TuRBO\n(BO)"}
k = 0; ticks = []
for key in order:
    if key not in best:
        continue
    lab, col, vals = best[key]
    xs = k + (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.09
    b.scatter(xs, vals, s=14, color=col, zorder=3, edgecolor="white", linewidth=0.5)
    b.hlines(np.mean(vals), k - 0.3, k + 0.3, color=col, lw=1.6)
    ticks.append((k, labs[key])); k += 1
b.axhline(P1B, color=MUT, lw=0.6, ls=(0, (2, 2)))
b.set_xticks([t[0] for t in ticks]); b.set_xticklabels([t[1] for t in ticks], rotation=45, ha="right", fontsize=6.5)
b.set_ylim(P1B - 100, P1B + 60); b.set_ylabel("Best feasible range [km]"); b.set_title("(b) best per run (bar: mean)", fontsize=8, loc="left")
plt.tight_layout(); _fs.save(plt.gcf(), f"{OUT}/fig_equal_budget.pdf"); plt.close()

# ------------------------------------------------------------------ Fig 2: fidelity
x_ = pd.read_csv(f"{RES}/mvu/xsolver.csv"); x_ = x_[x_.ab_LD.notna()]
pv = pd.read_csv(f"{RES}/viscous/profile_drag_check.csv").groupby("design", sort=False).head(1)
cf = []
for f in sorted(glob.glob(f"{RES}/cfd/*_h7_result.csv")):
    d = pd.read_csv(f).sort_values("alpha")
    if len(d) < 2:
        continue
    a0, a1 = d.iloc[0], d.iloc[1]
    cf.append((a0.design, a0.CL, a0.VLM_CL_cruise, -(a1.CMy - a0.CMy) / (a1.CL - a0.CL), a0.VLM_SM))
cf = pd.DataFrame(cf, columns=["design", "CL", "CLv", "SM", "SMv"])
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
a = ax[0]
top = x_[x_.sel == "top"]; rnd = x_[x_.sel != "top"]
a.scatter(rnd.L_D, rnd.ab_LD, s=10, color=GR, label="random valid", zorder=2, edgecolor="white", linewidth=0.3)
a.scatter(top.L_D, top.ab_LD, s=12, color=BL, label="best of each search", zorder=3, edgecolor="white", linewidth=0.3)
lim = [0, max(x_.ab_LD.max(), x_.L_D.max()) * 1.05]; a.plot(lim, lim, color=MUT, lw=0.6, ls=(0, (2, 2)))
a.set_xlabel("Pipeline $L/D$"); a.set_ylabel("AeroBuildup $L/D$"); a.set_title("(a) second 3D solver", fontsize=8, loc="left"); a.legend(fontsize=7, loc="lower right", borderpad=0.3, handletextpad=0.2)
b = ax[1]
nm = {"PAPER_champion": "best design", "AL_champion": "AL rep. 0", "P1_best": "pool best", "P1_median": "pool median", "P1_p10": "pool 10th pct"}
pcol = {"PAPER_champion": INK, "AL_champion": BL, "P1_best": OR, "P1_median": VI, "P1_p10": GR}
for _, r in pv.iterrows():
    b.scatter(r.CD_raymer * 1e4, r.CD_neuralfoil * 1e4, s=16, color=pcol.get(r.design, GR), zorder=3, edgecolor="black", linewidth=0.4, label=nm.get(r.design, r.design))
l = [50, 110]; b.plot(l, l, color=MUT, lw=0.6, ls=(0, (2, 2))); b.set_xlim(l); b.set_ylim(40, 110)
b.set_xlabel("Pipeline $C_{D,\\mathrm{profile}}$ [counts]"); b.set_ylabel("NeuralFoil [counts]"); b.set_title("(b) section drag", fontsize=8, loc="left")
c = ax[2]
for _, r in cf.iterrows():
    col = pcol.get(r.design, GR)
    c.scatter(r.SMv, r.SM, s=16, marker="o", color=col, zorder=3, edgecolor="black", linewidth=0.4)
    c.scatter(r.CLv, r.CL, s=16, marker="s", color=col, zorder=3, edgecolor="black", linewidth=0.4)
l = [0, 0.3]; c.plot(l, l, color=MUT, lw=0.6, ls=(0, (2, 2))); c.axhline(0.25, color=OR, lw=0.6); c.text(0.002, 0.2535, "$SM$ limit", fontsize=6.3, color=OR)
c.set_xlim(l); c.set_ylim(l); c.set_xlabel("VLM"); c.set_ylabel("Euler CFD"); c.set_title("(c) $C_L$ (squares), $SM$ (circles)", fontsize=8, loc="left")
h = [plt.Line2D([], [], marker="o", ls="", color=pcol[k_], markersize=4, label=nm[k_]) for k_ in pcol if k_ in set(pv.design) | set(cf.design)]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=6.5)
plt.tight_layout(rect=(0, 0.07, 1, 1)); _fs.save(plt.gcf(), f"{OUT}/fig_fidelity.pdf"); plt.close()

# ------------------------------------------------------------------ Fig 3: bay fraction
sr = pd.read_csv(f"{RES}/sensitivity_runs.csv"); bf = sr[sr.tag == "bay_fraction"].sort_values("frac")
fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.4))
for k_, (col, lab) in enumerate([("SM", "Static margin $SM$"), ("range_km_160", "Range [km]")]):
    a = ax[k_]; y = bf[col] * (F if col == "range_km_160" else 1)
    ok = bf.valid.astype(bool)
    if col == "range_km_160":
        a.plot(bf.frac[ok], y[ok], color=BL, marker="o", ms=3.5)
    else:
        a.plot(bf.frac, y, color=BL, lw=1.2)
        a.scatter(bf.frac[ok], y[ok], s=14, color=BL, zorder=3); a.scatter(bf.frac[~ok], y[~ok], s=14, facecolor="white", edgecolor=BL, zorder=3)
        a.axhline(0.25, color=OR, lw=0.6); a.text(0.5, 0.257, "upper limit 0.25", fontsize=6.5, color=OR)
    a.axvline(0.82, color=MUT, lw=0.6, ls=(0, (2, 2))); a.set_xlabel("Battery position $f_b$ [fraction of $c_r$]"); a.set_ylabel(lab)
    a.set_title("(a)" if k_ == 0 else "(b)", fontsize=8, loc="left")
plt.tight_layout(); _fs.save(plt.gcf(), f"{OUT}/fig_bay.pdf"); plt.close()

# ------------------------------------------------------------------ Fig 4: ablation area confound
ROOT = os.environ.get("DRONE_ROOT", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone")
ab = pd.read_csv(f"{ROOT}/research/paper_b_flagship/results/ablation_database.csv")
ab["inv"] = ab.c_blend > ab.c_root; v_ = ab[ab.is_valid.astype(bool)]
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
a = ax[0]
a.scatter(v_[~v_.inv].S_ref, v_[~v_.inv].L_D, s=7, color=BL, alpha=0.65, edgecolor="none", label="monotone chord")
a.scatter(v_[v_.inv].S_ref, v_[v_.inv].L_D, s=7, color=OR, alpha=0.65, edgecolor="none", label="chord-inverted")
a.set_xlabel("Reference area $S_\\mathrm{ref}$ [m$^2$]"); a.set_ylabel("$L/D$"); a.legend(frameon=False, fontsize=6.5); a.set_title("(a) all valid designs", fontsize=8, loc="left")
b = ax[1]
lo = max(v_[v_.inv].S_ref.min(), v_[~v_.inv].S_ref.min()); hi = min(v_[v_.inv].S_ref.max(), v_[~v_.inv].S_ref.max())
bins = np.linspace(lo, hi, 5); xs = []
for k_, (g, col, lab) in enumerate([(v_[~v_.inv], BL, "monotone"), (v_[v_.inv], OR, "inverted")]):
    m = g[(g.S_ref >= lo) & (g.S_ref <= hi)]
    idx = np.digitize(m.S_ref, bins[1:-1]); cen = (bins[:-1] + bins[1:]) / 2
    mu = [m.L_D[idx == i].mean() if (idx == i).sum() >= 5 else np.nan for i in range(4)]
    b.plot(cen + (k_ - 0.5) * 0.006, mu, color=col, marker="o", ms=3.5, label=lab)
b.set_xlabel("$S_\\mathrm{ref}$ bin centre [m$^2$]"); b.set_ylabel("Mean $L/D$ in bin"); b.legend(frameon=False, fontsize=6.5); b.set_title("(b) area-matched", fontsize=8, loc="left")
plt.tight_layout(); _fs.save(plt.gcf(), f"{OUT}/fig_ablation.pdf"); plt.close()
# ------------------------------------------------------------------ Fig 5: surrogate parity (random split and shifted test)
pr = pd.read_csv(f"{RES}/surrogate_parity.csv")
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
for a, (panel, ttl, col) in zip(ax, [("random", "(a) random 80/20 split", BL), ("shift", "(b) trained on the pool, tested on later designs", OR)]):
    d = pr[pr.panel == panel]; y, mu, sd = d.y * F, d.mu * F, d.sd * F
    r2 = 1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    if panel == "shift":
        a.errorbar(y, mu, yerr=sd, fmt="none", ecolor=col, alpha=0.25, lw=0.6, zorder=1)
    a.scatter(y, mu, s=7, color=col, alpha=0.7, edgecolor="none", zorder=2)
    lo_, hi_ = (40, 420) if panel == "random" else (float(min(y.min(), mu.min())) - 8, float(max(y.max(), mu.max())) + 8)
    a.plot([lo_, hi_], [lo_, hi_], color=MUT, lw=0.6, ls=(0, (2, 2))); a.set_xlim(lo_, hi_); a.set_ylim(lo_, hi_)
    a.text(0.04, 0.93, f"$R^2$ = {r2:.3f}\nn = {len(d)}", transform=a.transAxes, fontsize=7, va="top")
    a.set_title(ttl, fontsize=7.6, loc="left"); a.set_xlabel("Computed range [km]")
ax[0].set_ylabel("Surrogate range [km]"); ax[1].set_ylabel("Surrogate range [km]")
plt.tight_layout(); _fs.save(plt.gcf(), f"{OUT}/fig_parity.pdf"); plt.close()
print("figures written", os.listdir(OUT))


# ------------------------------------------------------------------ Phase B figures (results_v3)
def runs_of(pats, col):
    out = []
    for pt in pats:
        for f in sorted(glob.glob(pt)):
            d = pd.read_csv(f)
            if len(d) >= N:
                out.append(d.iloc[:N])
    return out


# Fig: feasibility weighting ablation (B1, B2)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
FA = [("AL, sigmoid weight", BL, "-", [f"{RES}/al/al_seed?.csv"], "range_calm_km"),
      ("AL, hard 0/1 gate", OR, "-", [f"{V3}/al_hard/al_seed?.csv"], "range_calm_km"),
      ("AL, no weight", "#b3261e", "-", [f"{V3}/al_nofeas/al_seed?.csv"], "range_calm_km"),
      ("TuRBO", TEAL, "-", [f"{RES}/turbo/turbo_seed?.csv", f"{V3}/turbo_extra/turbo_seed?.csv"], "range_km"),
      ("TuRBO, sigmoid weight", VI, "--", [f"{V3}/turbo_feas/turbo_seed?.csv"], "range_km")]
a = ax[0]; x = np.arange(1, N + 1); strips = []
for lab, col, ls, pats, rc in FA:
    rs = []
    for pt in pats:
        for f in sorted(glob.glob(pt)):
            d = pd.read_csv(f)
            if len(d) >= N or "seed2" in f and "/al/" in f:
                rs.append(d.iloc[:N])
    vals = np.array([d[d.is_valid.astype(bool)][rc].max() * F if d.is_valid.astype(bool).any() else np.nan for d in rs])
    strips.append((lab, col, vals))
    cs = np.array([(lambda c: np.concatenate([c, np.full(max(0, N - len(c)), c[-1])])[:N])(np.nan_to_num(curve(d, rc), nan=0.0)) for d in rs])   # a run that stopped early keeps its last best
    a.plot(x, cs.mean(0), color=col, ls=ls, lw=1.5); a.fill_between(x, cs.min(0), cs.max(0), color=col, alpha=0.12, lw=0)
a.axhline(P1B, color=MUT, lw=0.6, ls=(0, (2, 2))); a.text(150, P1B - 22, "best of the 9,800-design pool", fontsize=6.3, color=MUT)
a.set_ylim(0, 420); a.set_xlim(0, N); a.set_xlabel("Evaluations after the pool"); a.set_ylabel("Best feasible range so far [km]"); a.set_title("(a) running best (mean, min to max)", fontsize=8, loc="left")
h_ = [plt.Line2D([], [], color=c_, ls=l_, lw=1.5, label=lb_) for lb_, c_, l_, _, _ in FA]
fig.legend(handles=h_, loc="lower center", ncol=5, frameon=False, fontsize=6.4)
b = ax[1]
for k, (lab, col, vals) in enumerate(strips):
    v = vals[~np.isnan(vals)]
    if len(v):
        b.scatter(k + (np.arange(len(v)) - (len(v) - 1) / 2) * 0.07, v, s=12, color=col, zorder=3, edgecolor="black", linewidth=0.4); b.hlines(v.mean(), k - 0.3, k + 0.3, color=col, lw=1.6)
    else:
        b.text(k, P1B + 6, "no valid\ndesign\nin 10 runs", ha="center", va="bottom", fontsize=6.2, color=col)
b.axhline(P1B, color=MUT, lw=0.6, ls=(0, (2, 2)))
b.set_xticks(range(len(strips))); b.set_xticklabels(["AL\nsigmoid", "AL\nhard", "AL\nnone", "TuRBO", "TuRBO\nsigmoid"], fontsize=6.5)
b.set_ylim(P1B - 40, 420); b.set_ylabel("Best feasible range [km]"); b.set_title("(b) best per run (bar: mean)", fontsize=8, loc="left")
plt.tight_layout(rect=(0, 0.07, 1, 1)); _fs.save(plt.gcf(), f"{OUT}/fig_feas_ablation.pdf"); plt.close()

# Fig: calibration of the static-margin constraint (B4)
cj = json.load(open(f"{V3}/cfd_calibration/calibration.json")); cp = pd.read_csv(f"{V3}/cfd_calibration/cfd_points.csv"); rr = pd.read_csv(f"{V3}/cfd_calibration/rerank_per_run_fix.csv"); oos = pd.read_csv(f"{V3}/cfd_oos/oos_validation.csv")   # corrected runs (b4b_rerank_fix.py) and held-out check (b13_oos_validate.py)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 1.15]})
a = ax[0]; xs_ = np.linspace(0.04, 0.27, 50)
a.axvspan(0.05, 0.25, color=GR, alpha=0.10, lw=0); a.axhspan(0.05, 0.25, color=GR, alpha=0.10, lw=0)
a.plot(xs_, xs_, color=MUT, lw=0.6, ls=(0, (2, 2)))
a.plot(xs_, cj["a"] + cj["b"] * xs_, color=BL, lw=1.3)
new = cp[cp.src == "results_v3"]; old = cp[cp.src != "results_v3"]
a.scatter(new.SM_vlm, new.SM_euler, s=14, color=BL, zorder=3, edgecolor="black", linewidth=0.4, label=f"new designs (n={len(new)})")
a.scatter(old.SM_vlm, old.SM_euler, s=16, marker="s", color=OR, zorder=3, edgecolor="black", linewidth=0.4, label=f"earlier designs (n={len(old)})")
a.scatter(oos.SM_vlm, oos.SM_euler, s=20, marker="^", color=TEAL, zorder=4, edgecolor="black", linewidth=0.4, label=f"held-out champions (n={len(oos)})")
a.axhline(0.25, color=OR, lw=0.7); a.axvline(cj["SM_vlm_cap_equiv"], color=BL, lw=0.7, ls=":")
a.text(0.052, 0.2535, "$SM \\leq 0.25$", fontsize=7.6, color=OR); a.text(cj["SM_vlm_cap_equiv"] - 0.003, 0.285, f"VLM equivalent {cj['SM_vlm_cap_equiv']:.3f}", fontsize=7.6, color=BL, ha="right")
a.set_xlim(0.04, 0.27); a.set_ylim(0.04, 0.30); a.set_xlabel("Static margin, VLM pipeline"); a.set_ylabel("Static margin, Euler CFD")
a.set_title(f"(a) $SM_\\mathrm{{Euler}} = {cj['a']:.3f} + {cj['b']:.3f}\\,SM_\\mathrm{{VLM}}$, $R^2 = {cj['r2']:.3f}$", fontsize=7.6, loc="left"); a.legend(fontsize=7, loc="lower right", borderpad=0.3, labelspacing=0.25, handletextpad=0.2)
b = ax[1]
grp = {"AL (soft)": "AL (published)", "AL-R": "AL-R", "AL hard weight": "AL hard gate", "TuRBO": "TuRBO", "TuRBO extra": "TuRBO", "TuRBO + feas weight": "TuRBO + weight", "adapt10": "adapt10", "local10": "local 10%",
       "local5": "local 5%", "local5 (v3 extra)": "local 5%", "lhs": "fresh LHS", "lhs (v3 extra)": "fresh LHS", "uniform": "uniform", "uniform (v3 extra)": "uniform"}
r_ = rr[rr.arm.isin(grp)].copy(); r_["g"] = r_.arm.map(grp)
order_ = ["fresh LHS", "uniform", "local 5%", "local 10%", "AL (published)", "adapt10", "AL-R", "AL hard gate", "TuRBO", "TuRBO + weight"]
for i, g in enumerate(order_):
    x_ = r_[r_.g == g]
    b.plot([x_.best_vlm.mean() * F, x_.best_cal.mean() * F], [i, i], color=MUT, lw=1.2, zorder=2)
    b.scatter(x_.best_vlm.mean() * F, i, s=22, color=GR, zorder=3, edgecolor="black", linewidth=0.4)
    b.scatter(x_.best_cal.mean() * F, i, s=22, color=BL, zorder=4, edgecolor="black", linewidth=0.4)
    b.text(max(x_.best_vlm.mean(), x_.best_cal.mean()) * F + 4, i, f"{int(x_.champ_infeasible.sum())}/{len(x_)}", va="center", fontsize=7.6, color=OR)
b.set_yticks(range(len(order_))); b.set_yticklabels(order_, fontsize=7.6); b.set_xlim(280, 440)
b.set_xlabel("Mean best-of-run range [km]"); b.set_title("(b) VLM margin (gray) vs calibrated margin (blue)", fontsize=7.6, loc="left")
b.text(0.99, 0.02, "labels: runs whose best design\nleaves the calibrated band", transform=b.transAxes, ha="right", va="bottom", fontsize=7.6, color=OR)
plt.tight_layout(); _fs.save(plt.gcf(), f"{OUT}/fig_sm_calibration.pdf"); plt.close()

# Fig: bound shadow prices (B5) and robustness (B6)
bp = pd.read_csv(f"{V3}/bound_prices_fix/bound_prices_summary.csv"); rb = pd.read_csv(f"{V3}/robustness/robustness.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
a = ax[0]
nm_ = {"taper_lo": "taper ratio, lower\n0.55 to 0.525", "wtwist_hi": "winglet twist, upper\n0 to +1 deg", "bmass_hi": "battery mass, upper\n3.5 to 3.7 kg", "speed_lo": "cruise speed, lower\n36.1 to 33.6 m/s"}
for i, (_, r) in enumerate(bp.iloc[1:].iterrows()):
    sig = r.welch_p < 0.05; col = BL if sig else GR
    a.hlines(i, r.ci_lo, r.ci_hi, color=col, lw=1.6); a.scatter(r.gain, i, s=24, color=col, zorder=3, edgecolor="black", linewidth=0.4)
    a.text(max(r.ci_hi, 0) + 1.2, i, f"{r.gain:+.1f} km", va="center", fontsize=7.6, color=INK)
a.axvline(0, color=MUT, lw=0.6); a.set_yticks(range(4)); a.set_yticklabels([nm_[k] for k in bp.arm.iloc[1:]], fontsize=7.6); a.invert_yaxis()
a.set_xlim(-10, 34); a.set_xlabel("Range gain from widening one bound by 10% [km]"); a.set_title("(a) range price of single bounds (95% CI)", fontsize=8, loc="left")
b = ax[1]; fb = rb[rb.kind == "fb"]
cols_ = plt.cm.Blues(np.linspace(0.35, 0.95, fb.design.nunique())) if False else None
for d_, g in fb.groupby("design"):
    g = g.sort_values("value"); base = g.loc[(g.value - 0.82).abs().idxmin()]
    gv = g[g.is_valid]
    if len(gv) < 2 or base.range_km <= 0: continue
    b.plot(gv.value, 100 * (gv.range_km / base.range_km - 1), color=BL, alpha=0.35, lw=0.8)
b.axvline(0.82, color=MUT, lw=0.6, ls=(0, (2, 2))); b.set_xlabel("Battery centroid, fraction of root chord"); b.set_ylabel("Range change vs 0.82 [%]")
b.set_title("(b) battery-centroid sensitivity (valid designs)", fontsize=8, loc="left")
plt.tight_layout(); _fs.save(plt.gcf(), f"{OUT}/fig_bound_robust.pdf"); plt.close()
