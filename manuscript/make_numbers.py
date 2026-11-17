"""Generates numbers.tex (macros) and tables/*.tex from the raw result files, so no number in the manuscript is typed by hand.
usage: RES=<results_revision> DB=<database.csv> python3 make_numbers.py   (writes into the directory of this file; defaults are the Mac repo paths)
Ranges are stored on the 160 Wh/kg database basis; every range printed in the paper is multiplied by F = e_u/160."""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RES = os.environ.get("RES", f"{REPO}/revision_runs/results_revision")
DB = os.environ.get("DB", f"{REPO}/flying_wing_optimizer/results/database.csv")
V3 = os.environ.get("V3", f"{REPO}/revision_runs/results_v3")
os.makedirs(f"{HERE}/tables", exist_ok=True)
M = {}          # macro name -> text
REPORT = []     # human readable log


def say(s=""):
    REPORT.append(s); print(s)


def mac(name, val):
    M[name] = val


def mac_get(name):
    return M[name]


DISP = {"AL_champion": "AL replication 0, best", "P1_best": "Pool best", "P1_median": "Pool median", "P1_p10": "Pool 10th percentile", "PAPER_champion": "Original campaign, best"}
# ------------------------------------------------------------------ energy basis
CELL_WH = 5.0 * 3.6
NCELL = 48
PACK_KG = 3.400
EU = NCELL * CELL_WH * 0.80 / PACK_KG            # 203.29 Wh/kg
F = EU / 160.0
mac("NeU", f"{EU:.1f}")
mac("NeUkJ", f"{EU*3.6:.0f}")
mac("NeF", f"{F:.3f}")
PACK_NOM = NCELL * CELL_WH
mac("NpackNom", f"{PACK_NOM:.0f}")


def km(x):               # database km -> paper km
    return x * F


def fm(x, d=0):
    return f"{km(x):.{d}f}"


# ------------------------------------------------------------------ database
db = pd.read_csv(DB)
db["is_valid"] = db["is_valid"].astype(bool)
p1 = db[db.evaluation_type.str.startswith("Phase1")]
p2 = db[~db.evaluation_type.str.startswith("Phase1")]
v = db[db.is_valid]
champ = v.loc[v.range_calm_km.idxmax()]
P1_BEST = p1[p1.is_valid].range_calm_km.max()
ORIG_AL_BEST = p2[p2.is_valid].range_calm_km.max()
mac("NdbN", f"{len(db):,}".replace(",", "{,}"))
mac("NdbValid", f"{len(v):,}".replace(",", "{,}"))
mac("NpOneN", f"{len(p1):,}".replace(",", "{,}"))
mac("NpOneValid", f"{int(p1.is_valid.sum()):,}".replace(",", "{,}"))
mac("NpOneValidPct", f"{100*p1.is_valid.mean():.1f}")
mac("NpTwoN", f"{len(p2)}")
mac("NpTwoValid", f"{int(p2.is_valid.sum())}")
mac("NpTwoValidPct", f"{100*p2.is_valid.mean():.0f}")
mac("NpOneBest", fm(P1_BEST, 1))
mac("NalOrigBest", fm(ORIG_AL_BEST, 1))
mac("NalOrigUplift", f"{100*(ORIG_AL_BEST/P1_BEST-1):.1f}")
mac("NchampRange", fm(champ.range_calm_km, 1))
mac("NchampRangeRound", fm(champ.range_calm_km, 0))
assert abs(champ.range_calm_km - ORIG_AL_BEST) < 1e-6, "champion is not the Phase 2 best"
VR = json.load(open(f"{RES}/cfd/PAPER_champion_vlm_ref.json"))
assert abs(VR["L_D"] - champ.L_D) < 1e-3, "vlm_ref.json is not the champion"
for k, val in [("LD", champ.L_D), ("CL", champ.CL), ("SM", champ.SM), ("Cm", champ.Cm_trim), ("Sref", VR["S_ref"]), ("mtot", champ.m_total_kg),
               ("V", champ.cruise_speed_ms), ("mbat", champ.m_battery_kg), ("CD", champ.CD)]:
    mac("Nch" + k, f"{val:.4f}" if k in ("CD",) else (f"{val:.3f}" if k in ("CL", "SM", "Sref", "Cm") else f"{val:.2f}"))
mac("NchSMpct", f"{100*champ.SM:.1f}")
mac("NchAR", f"{1.5**2/VR['S_ref']:.2f}")
mac("NchVkmh", f"{champ.cruise_speed_ms*3.6:.0f}")

# phase 2 composition
acq = p2[p2.evaluation_type.str.contains("acq|active", case=False)]
unc = p2[p2.evaluation_type.str.contains("unc", case=False)]
mac("NacqN", f"{len(acq)}"); mac("NuncN", f"{len(unc)}")
mac("NacqValid", f"{int(acq.is_valid.sum())}"); mac("NuncValid", f"{int(unc.is_valid.sum())}")

# ------------------------------------------------------------------ equal-budget arms
KEYS = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2", "tip_twist",
        "winglet_sweep_deg", "winglet_twist_deg", "battery_mass", "cruise_speed_ms"]
N_EXTRA = 287


def runs_ctrl(arm):
    out = []
    pats = [f"{RES}/turbo/turbo_seed?.csv", f"{V3}/turbo_extra/turbo_seed?.csv"] if arm == "turbo" else \
           [f"{RES}/controls/{arm}_seed?.csv", f"{V3}/controls_extra/{arm}_seed?.csv"]     # seeds 0-2/0-6 from the revision, remaining seeds from results_v3
    for f in sorted(g for pt in pats for g in glob.glob(pt)):
        d = pd.read_csv(f); d["is_valid"] = d.is_valid.astype(bool)
        if len(d) < N_EXTRA:
            continue            # unfinished run, not used
        out.append(d.rename(columns={"range_km": "r"})[["r", "is_valid"]].iloc[:N_EXTRA])
    return out


def runs_al():
    out = []
    o = p2.copy()
    o = o.rename(columns={"range_calm_km": "r"})[["r", "is_valid"]].iloc[:N_EXTRA]
    out.append(("orig", o.reset_index(drop=True), False))
    for f in sorted(glob.glob(f"{RES}/al/al_seed?.csv")):
        d = pd.read_csv(f); d["is_valid"] = d.is_valid.astype(bool)
        stopped = len(d) < N_EXTRA
        # an AL run still in progress is not used; a run that stopped by exhaustion is flagged in the log file
        logf = f"/tmp/cfd/al_seed{os.path.basename(f)[7]}.log"
        done = (not stopped) or (os.path.exists(logf) and "AL exhausted" in open(logf).read()) or \
               os.path.basename(f) in ("al_seed2.csv",)
        if not done:
            continue
        out.append((os.path.basename(f)[:-4], d.rename(columns={"range_calm_km": "r"})[["r", "is_valid"]].reset_index(drop=True), stopped))
    return out


def curve(d, n=N_EXTRA):
    x = d.r.where(d.is_valid, 0.0).cummax().values
    if len(x) < n:
        x = np.concatenate([x, np.full(n - len(x), x[-1])])
    return x[:n]


def first_beat(d, thr):
    x = d.r.where(d.is_valid, 0.0).values
    i = np.where(x > thr)[0]
    return int(i[0]) + 1 if len(i) else None


arms = {}
arms["AL"] = [d for _, d, _ in runs_al()]
al_flags = [(n, s) for n, _, s in runs_al()]
for a in ["turbo", "adapt10", "local5", "local10", "lhs", "uniform"]:
    arms[a] = runs_ctrl(a)
mac("NnAL", str(len(arms["AL"])))
mac("NnLoc", str(len(arms["local10"]))); mac("NnAdapt", str(len(arms["adapt10"])))
mac("NnALrep", str(len(arms["AL"]) - 1))
stopped_runs = [n for n, s in al_flags if s]
mac("NnALstopped", str(len(stopped_runs)))
if stopped_runs:
    for n, s in al_flags:
        if s:
            d = [dd for nn, dd, _ in runs_al() if nn == n][0]
            mac("NalStopN", str(len(d))); mac("NalStopBest", fm(d[d.is_valid].r.max(), 1))

LABEL = {"turbo": "TuRBO (BO)", "AL": "Active learning", "adapt10": "Adaptive local search, 10\\%", "local10": "Local search, 10\\%", "local5": "Local search, 5\\%", "lhs": "Fresh LHS", "uniform": "Uniform random"}
rows = []
bests = {}
for a in ["turbo", "AL", "adapt10", "local10", "local5", "lhs", "uniform"]:
    rs = arms[a]
    if not rs:
        continue
    b = np.array([d[d.is_valid].r.max() for d in rs]); bests[a] = b
    c100 = np.array([curve(d)[99] for d in rs]); c200 = np.array([curve(d)[199] for d in rs])
    vr = np.mean([d.is_valid.mean() for d in rs])
    vr60 = np.mean([d.is_valid.iloc[:60].mean() for d in rs])
    med = np.mean([d[d.is_valid].r.median() for d in rs])
    fb = [first_beat(d, P1_BEST) for d in rs]
    nb = sum(x is not None for x in fb)
    medfb = np.median([x for x in fb if x is not None]) if nb else np.nan
    sd = b.std(ddof=1) if len(b) > 1 else np.nan
    rows.append(dict(arm=a, n=len(rs), c100=c100.mean(), c200=c200.mean(), mean=b.mean(), sd=sd, lo=b.min(), hi=b.max(), vr=vr, vr60=vr60,
                     med=med, nb=nb, medfb=medfb))
    key = {"turbo": "Turbo", "AL": "AL", "adapt10": "AdaptTen", "local10": "LocTen", "local5": "LocFive", "lhs": "Lhs", "uniform": "Uni"}[a]
    mac(f"N{key}mean", fm(b.mean(), 1)); mac(f"N{key}sd", fm(sd, 1) if len(b) > 1 else "n/a")
    mac(f"N{key}lo", fm(b.min(), 1)); mac(f"N{key}hi", fm(b.max(), 1)); mac(f"N{key}n", str(len(rs)))
    mac(f"N{key}vr", f"{100*vr:.0f}"); mac(f"N{key}vrsixty", f"{100*vr60:.0f}")
    mac(f"N{key}upl", f"{100*(b.mean()/P1_BEST-1):.1f}")
    mac(f"N{key}firstN", f"{nb}/{len(rs)}"); mac(f"N{key}firstMed", ("%.0f" % medfb) if nb else "n/a")
    say(f"{a}: n={len(rs)} best mean={b.mean()*F:.1f} sd={sd*F:.1f} range=[{b.min()*F:.1f},{b.max()*F:.1f}] valid={vr:.2f} first60={vr60:.2f} "
        f"beat pool best in {nb}/{len(rs)} median evals={medfb}")
tb = ["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
      "Strategy & Runs & Best at 100 & Best at 287, mean $\\pm$ sd & Min-max & Valid (\\%) & Valid, first 60 (\\%) & Beats pool best & Evals to do so \\\\", "\\midrule"]
for r in rows:
    sd = f"{r['sd']*F:.1f}" if r["n"] > 1 else "n/a"
    nbs = f"{r['nb']}/{r['n']}"
    mf = f"{r['medfb']:.0f}" if r["nb"] else "n/a"
    tb.append(f"{LABEL[r['arm']]} & {r['n']} & {r['c100']*F:.1f} & {r['mean']*F:.1f} $\\pm$ {sd} & {r["lo"]*F:.1f} to {r["hi"]*F:.1f} & "
              f"{100*r['vr']:.0f} & {100*r['vr60']:.0f} & {nbs} & {mf} \\\\")
tb += ["\\midrule", f"Pool only (9,800 LHS) & n/a & n/a & {P1_BEST*F:.1f} & n/a & {100*p1.is_valid.mean():.0f} & n/a & n/a & n/a \\\\", "\\bottomrule", "\\end{tabular}"]
open(f"{HERE}/tables/tab_budget.tex", "w").write("\n".join(tb) + "\n")

# per-run list for AL
al_names = [n for n, _, _ in runs_al()]
tr = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "Run & Evaluations & Best (km) & Valid (\\%) & Note \\\\", "\\midrule"]
for (n, d, s) in runs_al():
    note = "original campaign, wall-clock stop" if n == "orig" else ("stopped: no new candidate for 40 cycles" if s else "")
    nm = "Original" if n == "orig" else "Replication " + n[-1]
    tr.append(f"{nm} & {len(d)} & {d[d.is_valid].r.max()*F:.1f} & {100*d.is_valid.mean():.0f} & {note} \\\\")
tr += ["\\bottomrule", "\\end{tabular}"]
open(f"{HERE}/tables/tab_replic.tex", "w").write("\n".join(tr) + "\n")

# tests AL vs each control
def boot_diff(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    da = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return np.percentile(da, [2.5, 97.5])


tests = []
for c in ["turbo", "adapt10", "local10", "local5", "lhs", "uniform"]:
    if c not in bests:
        continue
    a, b = bests["AL"], bests[c]
    if len(b) < 2:
        continue
    t = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
    u = stats.mannwhitneyu(a, b, alternative="two-sided", method="exact" if min(len(a), len(b)) < 8 else "auto")
    ci = boot_diff(a, b)
    diff = a.mean() - b.mean()
    ps = u.statistic / (len(a) * len(b))
    tests.append((c, diff, 100 * diff / b.mean(), t.pvalue, u.pvalue, ci, ps))
    key = {"turbo": "Turbo", "adapt10": "AdaptTen", "local10": "LocTen", "local5": "LocFive", "lhs": "Lhs", "uniform": "Uni"}[c]
    mac(f"Nd{key}", f"{km(diff):.1f}"); mac(f"Nd{key}pct", f"{100*diff/b.mean():.1f}")
    mac(f"Nt{key}p", f"{t.pvalue:.3f}" if t.pvalue >= 0.001 else "<0.001")
    mac(f"Nu{key}p", f"{u.pvalue:.3f}" if u.pvalue >= 0.001 else "<0.001")
    mac(f"Nci{key}", f"[{km(ci[0]):.1f}, {km(ci[1]):.1f}]")
    mac(f"Nps{key}", f"{ps:.2f}")
    say(f"AL vs {c}: diff {km(diff):.1f} km ({100*diff/b.mean():.1f}%), Welch p={t.pvalue:.3f}, MWU p={u.pvalue:.3f}, boot95 [{km(ci[0]):.1f},{km(ci[1]):.1f}], P(sup)={ps:.2f}")
tt = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
      "AL minus control & $\\Delta$ (km) & $\\Delta$ (\\%) & Welch $p$ (two-sided) & Mann-Whitney $p$ (two-sided) & Bootstrap 95\\% CI (km) & $P$(AL $>$ control) \\\\", "\\midrule"]
for (c, d, dp, pt, pu, ci, ps) in tests:
    tt.append(f"{LABEL[c]} & {km(d):.1f} & {dp:.1f} & {'<0.001' if pt < 0.001 else f'{pt:.3f}'} & {'<0.001' if pu < 0.001 else f'{pu:.3f}'} & [{km(ci[0]):.1f}, {km(ci[1]):.1f}] & {ps:.2f} \\\\")
tt += ["\\bottomrule", "\\end{tabular}"]
open(f"{HERE}/tables/tab_tests.tex", "w").write("\n".join(tt) + "\n")

# TuRBO minus adaptive local search (two-sided), for the results text
if "turbo" in bests and "adapt10" in bests:
    a_, b_ = bests["turbo"], bests["adapt10"]
    t_ = stats.ttest_ind(a_, b_, equal_var=False, alternative="two-sided"); ci_ = boot_diff(a_, b_)
    mac("NdTurboAdapt", f"{km(a_.mean()-b_.mean()):.1f}"); mac("NtTurboAdaptp", f"{t_.pvalue:.3f}" if t_.pvalue >= 0.001 else "<0.001")
    mac("NciTurboAdapt", f"[{km(ci_[0]):.1f}, {km(ci_[1]):.1f}]")
    say(f"TuRBO minus adapt10: {km(a_.mean()-b_.mean()):.1f} km, Welch p={t_.pvalue:.4f}, boot95 [{km(ci_[0]):.1f},{km(ci_[1]):.1f}]")
if "turbo" in bests:
    mac("NciTurboOverAL", f"[{-km(tests[0][5][1]):.1f}, {-km(tests[0][5][0]):.1f}]")
    mac("NdTurboOverAL", f"{-km(tests[0][1]):.1f}"); mac("NdTurboOverALpct", f"{100*(bests['turbo'].mean()/bests['AL'].mean()-1):.1f}")

# attribution of the original +7.5 %
loc = bests["adapt10"].mean() if "adapt10" in bests else (bests["local10"].mean())
mac("NattrLocPct", f"{100*(loc/P1_BEST-1):.1f}")
mac("NattrShare", f"{100*(loc/P1_BEST-1)/(bests['AL'].mean()/P1_BEST-1):.0f}")
mac("NattrAlMeanPct", f"{100*(bests['AL'].mean()/P1_BEST-1):.1f}")
mac("NattrAlOverLocPct", f"{100*(bests['AL'].mean()/loc-1):.1f}")
mac("NattrOrigOverLocPct", f"{100*(ORIG_AL_BEST/loc-1):.1f}")
# AL excluding stopped run
b_ex = np.array([d[d.is_valid].r.max() for (n, d, s) in runs_al() if not s])
mac("NALexMean", fm(b_ex.mean(), 1)); mac("NALexN", str(len(b_ex)))
# matched evaluation count for the stopped run: best of every arm at its evaluation count
if stopped_runs:
    nst = int([len(d) for n, d, s in runs_al() if s][0])
    for a in ["AL", "adapt10", "local10", "local5"]:
        if not arms[a]:
            continue
        cs = np.array([curve(d)[nst - 1] for d in arms[a]])
        say(f"best after {nst} evaluations, {a}: mean {cs.mean()*F:.1f}")
        mac("NatStop" + {"AL": "AL", "adapt10": "AdaptTen", "local10": "LocTen", "local5": "LocFive"}[a], fm(cs.mean(), 1))
    mac("NstopAt", str(nst))

# ------------------------------------------------------------------ loft: mvu, bounds
def load_set(d, arm):
    out = []
    for f in sorted(glob.glob(f"{RES}/{d}/{arm}_seed?.csv")):
        x = pd.read_csv(f); x["is_valid"] = x.is_valid.astype(bool); out.append(x)
    return out

mv = {a: load_set("mvu", a) for a in ["clamp", "noclamp"]}
bn = {a: load_set("bnd", a) for a in ["paper", "wide"]}
lr = []
for lab, d, extra in [("Clamp (widened $c_b\\in[0.15,0.65]$)", mv["clamp"], "inverted"), ("No clamp (same bounds)", mv["noclamp"], "inverted"),
                      ("Paper bounds, clamp", bn["paper"], "clamp_active"), ("Relaxed bounds, clamp", bn["wide"], "clamp_active")]:
    b = np.array([x[x.is_valid].range_km.max() for x in d])
    vr = np.mean([x.is_valid.mean() for x in d])
    ex = np.mean([x[extra].mean() for x in d])
    lr.append((lab, len(d), b, vr, ex, extra))
tl = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule", "Search & Seeds & Best per seed (km) & Mean (km) & Valid (\\%) & Inverted / clamp-active samples (\\%) \\\\", "\\midrule"]
for lab, n, b, vr, ex, extra in lr:
    bl = ", ".join(f"{x*F:.0f}" for x in b)
    tl.append(f"{lab} & {n} & {bl} & {b.mean()*F:.1f} & {100*vr:.0f} & {100*ex:.0f} \\\\")
tl += ["\\bottomrule", "\\end{tabular}"]
open(f"{HERE}/tables/tab_loft.tex", "w").write("\n".join(tl) + "\n")
for lab, n, b, vr, ex, extra in lr:
    say(f"{lab}: n={n} best={[round(x*F,1) for x in b]} mean={b.mean()*F:.1f} valid={vr:.2f} {extra}={ex:.2f}")
cm, nc = lr[0][2], lr[1][2]
mac("NclampMean", fm(cm.mean(), 1)); mac("NnoclampMean", fm(nc.mean(), 1))
mac("NclampMinusPct", f"{100*(cm.mean()/nc.mean()-1):.0f}")
mac("NclampSeeds", str(len(cm)))
tcl = stats.ttest_ind(cm, nc, equal_var=False)
mac("NclampP", f"{tcl.pvalue:.2f}")
mac("NclampValid", f"{100*lr[0][3]:.0f}"); mac("NnoclampValid", f"{100*lr[1][3]:.0f}")
mac("NinvShare", f"{100*lr[1][4]:.0f}")
pm, wm = lr[2][2], lr[3][2]
mac("NpaperBndMean", fm(pm.mean(), 1)); mac("NwideBndMean", fm(wm.mean(), 1))
mac("NwideGainPct", f"{100*(wm.mean()/pm.mean()-1):.0f}")
mac("NclampActWide", f"{100*lr[3][4]:.0f}")
mac("NwideBndMax", fm(wm.max(), 0))
say(f"clamp vs noclamp Welch p={tcl.pvalue:.3f}")

# ------------------------------------------------------------------ ablation re-analysis and other json
ex = json.load(open(f"{RES}/extra_analyses.json"))
A = ex["ablation"]
for k, name in [("n", "AbN"), ("n_inv", "AbNinv"), ("valid_inv", "AbVinv"), ("valid_mono", "AbVmono")]:
    mac("N" + name, str(int(A[k])))
mac("NAbVRinv", f"{100*A['vr_inv']:.0f}"); mac("NAbVRmono", f"{100*A['vr_mono']:.0f}")
mac("NAbLDinv", f"{A['mean_inv']:.2f}"); mac("NAbLDmono", f"{A['mean_mono']:.2f}")
mac("NAbMedInv", f"{A['med_inv']:.2f}"); mac("NAbMedMono", f"{A['med_mono']:.2f}")
mac("NAbSinv", f"{A['Sref_inv']:.2f}"); mac("NAbSmono", f"{A['Sref_mono']:.2f}")
mac("NAbARinv", f"{A['AR_inv']:.2f}"); mac("NAbARmono", f"{A['AR_mono']:.2f}")
mac("NAbPraw", f"{A['p']:.1e}".replace("e-", "\\times10^{-") + "}") if False else None
mac("NAbRho", f"{A['rho_LD_Sref']:.2f}")
mac("NAbMatchN", f"{A['n_match_inv']}+{A['n_match_mono']}")
mac("NAbMatchInv", f"{A['mean_match_inv']:.2f}"); mac("NAbMatchMono", f"{A['mean_match_mono']:.2f}")
mac("NAbMatchP", f"{A['p_match']:.2f}")
mac("NAbOLS", f"{A['ols_inv']:.2f}"); mac("NAbOLSse", f"{A['ols_inv_se']:.2f}")
mac("NAbSlo", f"{A['sref_lo']:.2f}"); mac("NAbShi", f"{A['sref_hi']:.2f}")
mac("NAbMaxRinv", fm(A["maxR_inv"], 0)); mac("NAbMaxRmono", fm(A["maxR_mono"], 0))
tab = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule", " & Chord-inverted & Monotone \\\\", "\\midrule",
       f"Designs (valid) & {A['n_inv']} ({A['valid_inv']}) & {A['n']-A['n_inv']} ({A['valid_mono']}) \\\\",
       f"Mean $S_\\mathrm{{ref}}$ (m$^2$) & {A['Sref_inv']:.2f} & {A['Sref_mono']:.2f} \\\\",
       f"Mean aspect ratio & {A['AR_inv']:.2f} & {A['AR_mono']:.2f} \\\\",
       f"Mean $L/D$, all valid & {A['mean_inv']:.2f} & {A['mean_mono']:.2f} \\\\",
       f"Mean $L/D$, $S_\\mathrm{{ref}}\\in[{A['sref_lo']:.2f},{A['sref_hi']:.2f}]$ (n={A['n_match_inv']}, {A['n_match_mono']}) & {A['mean_match_inv']:.2f} & {A['mean_match_mono']:.2f} \\\\",
       "\\bottomrule", "\\end{tabular}"]
open(f"{HERE}/tables/tab_ablation.tex", "w").write("\n".join(tab) + "\n")
Mo = ex["mono"]
mac("NmonoN", f"{Mo['n']:,}".replace(",", "{,}")); mac("NmonoShare", f"{100*Mo['share_nonmono']:.0f}\\%")
mac("NmonoMed", f"{Mo['median_mm_nonmono']:.1f}"); mac("NmonoMax", f"{Mo['max_mm']:.1f}")
mac("NmonoGone", f"{100*Mo['share_gt1mm']:.0f}\\%" if False else f"{100*Mo['share_gt1mm']:.0f}\\% of all valid designs")
# classifier audit: single scaled-RF variant from results_v3/classifier_audit (audit_classifier_v3.py)
CA = json.load(open(f"{V3}/classifier_audit/classifier_audit.json"))
cm_, cs_ = CA["mean"], CA["sd"]
mac("NfnRecall", f"{100*cm_['recall']:.1f}"); mac("NfnRecallSd", f"{100*cs_['recall']:.1f}")
mac("NfnPrec", f"{100*cm_['precision']:.1f}"); mac("NfnPrecSd", f"{100*cs_['precision']:.1f}")
mac("NfnTenRecall", f"{100*cm_['recall_top10']:.1f}"); mac("NfnTenRecallSd", f"{100*cs_['recall_top10']:.1f}")
mac("NfnTwoRecall", f"{100*cm_['recall_top2']:.1f}"); mac("NfnTwoRecallSd", f"{100*cs_['recall_top2']:.1f}")
mac("NfnMeanP", f"{cm_['mean_p_top10']:.3f}"); mac("NfnMeanPSd", f"{cs_['mean_p_top10']:.3f}")
mac("NfnTenN", f"{min(r['n_top10'] for r in CA['per_seed'])}--{max(r['n_top10'] for r in CA['per_seed'])}")
mac("NfnTwoN", f"{min(r['n_top2'] for r in CA['per_seed'])}--{max(r['n_top2'] for r in CA['per_seed'])}")
mac("NfnTestValid", f"{cm_['n_test_valid']:.0f}")
mac("NfnWithin", f"{100*cm_['fn_within10']:.0f}"); mac("NtpWithin", f"{100*cm_['tp_within10']:.0f}")
mac("NfnMedian", fm(cm_["fn_median_range"], 0)); mac("NtpMedian", fm(cm_["tp_median_range"], 0))
mac("NfnMax", fm(CA["fn_best_range_mean"], 0)); mac("NfnMaxSplit", fm(CA["fn_best_range_max"], 0))
mac("NclfInfeasShare", f"{100*CA['share_infeasible']:.1f}"); mac("NclfNrows", f"{CA['n_rows']:,}".replace(",", "{,}")); mac("NclfNvalid", f"{CA['n_valid']:,}".replace(",", "{,}"))
mac("NpoolInfeasShare", f"{100*(1-p1.is_valid.mean()):.1f}"); mac("NpoolValidN", f"{int(p1.is_valid.sum()):,}".replace(",", "{,}"))
mac("NclfQten", fm(CA["q90"], 1)); mac("NclfQtwo", fm(CA["q98"], 1))
mac("NclfQtenRaw", f"{CA['q90']:.1f}"); mac("NclfQtwoRaw", f"{CA['q98']:.1f}")
tcl = ["\\begin{tabular}{@{}llrr@{}}", "\\toprule", "Metric & Evaluated subset & Mean $\\pm$ SD & Sample size per split ($n$) \\\\", "\\midrule",
       f"Global precision & All designs ($p \\ge 0.5$) & {100*cm_['precision']:.1f}\\% $\\pm$ {100*cs_['precision']:.1f}\\% & $\\approx {cm_['n_test']:.0f}$ test designs \\\\",
       f"Global recall & All designs ($p \\ge 0.5$) & {100*cm_['recall']:.1f}\\% $\\pm$ {100*cs_['recall']:.1f}\\% & $\\approx {cm_['n_test_valid']:.0f}$ valid test designs \\\\",
       f"Top 10\\% frontier recall & Valid designs ($R \\ge {km(CA['q90']):.1f}\\,\\mathrm{{km}}$) & {100*cm_['recall_top10']:.1f}\\% $\\pm$ {100*cs_['recall_top10']:.1f}\\% & {min(r['n_top10'] for r in CA['per_seed'])}--{max(r['n_top10'] for r in CA['per_seed'])} designs \\\\",
       f"Top 2\\% frontier recall & Valid designs ($R \\ge {km(CA['q98']):.1f}\\,\\mathrm{{km}}$) & {100*cm_['recall_top2']:.1f}\\% $\\pm$ {100*cs_['recall_top2']:.1f}\\% & {min(r['n_top2'] for r in CA['per_seed'])}--{max(r['n_top2'] for r in CA['per_seed'])} designs$^*$ \\\\",
       f"Mean $\\mathcal{{P}}_{{\\mathrm{{feas}}}}$ & Top 10\\% valid designs & {cm_['mean_p_top10']:.3f} $\\pm$ {cs_['mean_p_top10']:.3f} & {min(r['n_top10'] for r in CA['per_seed'])}--{max(r['n_top10'] for r in CA['per_seed'])} designs \\\\",
       f"Valid designs within 10\\% of a constraint boundary & Rejected vs.\\ accepted & {100*cm_['fn_within10']:.0f}\\% vs.\\ {100*cm_['tp_within10']:.0f}\\% & $\\approx {cm_['n_fn']:.0f}$ vs.\\ ${cm_['n_tp']:.0f}$ \\\\",
       f"Median range of valid designs ($\\mathrm{{km}}$) & Rejected vs.\\ accepted & {km(cm_['fn_median_range']):.0f} vs.\\ {km(cm_['tp_median_range']):.0f} & same splits \\\\",
       f"Best rejected valid design ($\\mathrm{{km}}$) & Largest range missed, mean over splits & {km(CA['fn_best_range_mean']):.1f} (max {km(CA['fn_best_range_max']):.1f}) & 5 splits \\\\",
       "\\bottomrule",
       "\\multicolumn{4}{@{}l@{}}{\\footnotesize Scaled-feature random forest (100 trees); five repeated stratified 80/20 random splits (seeds 0 to 4). Ranges on the " + f"{EU:.1f}" + "\\,Wh/kg basis.}\\\\",
       "\\multicolumn{4}{@{}l@{}}{\\footnotesize $^*$Small samples; quoted with the indicated sample-size caveat.}", "\\end{tabular}"]
open(f"{HERE}/tables/tab_classifier.tex", "w").write("\n".join(tcl) + "\n")
mac("NfnRecallUnscaled", f"{100*ex['fn']['recall']:.1f}"); mac("NfnPrecUnscaled", f"{100*ex['fn']['prec']:.1f}")
cvr = ex["cv"]
tcv = ["\\begin{tabular}{@{}lrrr@{}}", "\\toprule", "Target & $R^2$ (5-fold, mean $\\pm$ sd) & MAPE (\\%) \\\\", "\\midrule"]
names = {"CL": "$C_L$", "CD": "$C_D$", "L_D": "$L/D$", "range": "Range", "SM": "Static margin"}
tcv = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule", "Target & $R^2$ (5-fold, mean $\\pm$ sd) & MAPE (\\%) \\\\", "\\midrule"]
for k in ["CL", "CD", "L_D", "range", "SM"]:
    tcv.append(f"{names[k]} & {cvr[k]['r2']:.4f} $\\pm$ {cvr[k]['r2_sd']:.4f} & {cvr[k]['mape']:.2f} \\\\")
tcv += ["\\bottomrule", "\\end{tabular}"]
open(f"{HERE}/tables/tab_cv.tex", "w").write("\n".join(tcv) + "\n")
mac("NcvRange", f"{cvr['range']['r2']:.3f}"); mac("NcvRangeMape", f"{cvr['range']['mape']:.1f}")

sh = pd.read_csv(f"{RES}/surrogate_shift.csv")
tsh = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule", "Test & Target & $n_{\\rm test}$ & $R^2$ & MAE & Bias & Spearman & 1-sd coverage \\\\", "\\midrule"]
for _, r in sh.iterrows():
    t = r["test"].replace("->", "$\\to$").replace(">=", "$\\geq$").replace("<", "$<$")
    tgt = {"range_calm_km": "range (km)", "L_D": "$L/D$", "SM": "static margin"}[r["target"]]
    sc = F if r["target"] == "range_calm_km" else 1.0
    tsh.append(f"{t} & {tgt} & {int(r.n_test)} & {r.R2:.2f} & {r.MAE*sc:.2f} & {r.bias*sc:+.2f} & {r.spearman:.2f} & {r.cover_1sd:.2f} \\\\")
tsh += ["\\bottomrule", "\\end{tabular}"]
open(f"{HERE}/tables/tab_shift.tex", "w").write("\n".join(tsh) + "\n")
r0, r1 = sh.iloc[0], sh.iloc[1]
mac("NshiftRtwo", f"{r0.R2:.2f}"); mac("NshiftMae", f"{r0.MAE*F:.1f}"); mac("NshiftBias", f"{abs(r0.bias)*F:.1f}")
mac("NshiftTopRho", f"{r0.spearman_top10:.2f}")
mac("NextrapRtwo", f"{r1.R2:.2f}"); mac("NextrapBias", f"{abs(r1.bias)*F:.1f}"); mac("NextrapCover", f"{r1.cover_1sd:.2f}")
mac("NshiftCover", f"{r0.cover_1sd:.2f}")

# ------------------------------------------------------------------ fidelity: AeroBuildup, grid, CFD, NeuralFoil
x = pd.read_csv(f"{RES}/mvu/xsolver.csv"); x = x[x.ab_LD.notna()].copy(); x["ratio"] = x.ab_LD / x.L_D
mac("NxN", str(len(x)))
rho_all = stats.spearmanr(x.L_D, x.ab_LD)[0]; rho_top = stats.spearmanr(x[x.sel == "top"].L_D, x[x.sel == "top"].ab_LD)[0]
rho_rand = stats.spearmanr(x[x.sel != "top"].L_D, x[x.sel != "top"].ab_LD)[0] if (x.sel != "top").sum() > 3 else np.nan
mac("NxRho", f"{rho_all:.2f}"); mac("NxRhoTop", f"{rho_top:.2f}"); mac("NxRhoRand", f"{rho_rand:.2f}")
mac("NxRatioMed", f"{x.ratio.median():.2f}")
mac("NxRatioLo", f"{x.ratio.quantile(0.05):.2f}"); mac("NxRatioHi", f"{x.ratio.quantile(0.95):.2f}")
mac("NxNtop", str(int((x.sel == "top").sum())))
say(f"AeroBuildup: n={len(x)} rho all {rho_all:.3f} top {rho_top:.3f} rand {rho_rand:.3f}, ratio median {x.ratio.median():.2f} (5-95%: {x.ratio.quantile(.05):.2f}-{x.ratio.quantile(.95):.2f})")
mp = pd.read_csv("/mnt/user-data/uploads/drone/research/paper_b_flagship/results/multipoint_validation.csv") if os.path.exists("/mnt/user-data/uploads/drone/research/paper_b_flagship/results/multipoint_validation.csv") else None
if mp is not None:
    mac("NmpN", str(len(mp)))
    rho_mp = stats.spearmanr(mp.vlm_LD, mp.ab_LD)[0]
    mac("NmpRho", f"{rho_mp:.2f}")
    mac("NmpDeltaMed", f"{mp.delta_LD_pct.median():.0f}")
    say(f"earlier multipoint n={len(mp)} rho={rho_mp:.2f}, median delta L/D {mp.delta_LD_pct.median():.0f}%")

g = pd.read_csv(f"{RES}/grid/grid_convergence.csv")
tg = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule", "Design & Panels (span$\\times$chord) & $C_L$ & $C_{D}$ & $L/D$ & $SM$ & Time (s) \\\\", "\\midrule"]
gr = g.sort_values(["design", "panels"])
for _, r in gr.iterrows():
    tg.append(f"{DISP.get(r.design, r.design)} & {int(r.panels)} ({int(r.span_res)}$\\times${int(r.chord_res)}) & {r.CL:.4f} & {r.CD:.5f} & {r.L_D:.2f} & {r.SM:.3f} & {r.secs:.1f} \\\\")
tg += ["\\bottomrule", "\\end{tabular}"]
open(f"{HERE}/tables/tab_grid.tex", "w").write("\n".join(tg) + "\n")
ref = g[(g.span_res == 2) & (g.chord_res == 12)].set_index("design")
fine = g[g.panels == g.panels.max()]
dev = []
for _, r in g.iterrows():
    pass
# deviation from the production grid, per design: worst case over all other grids
dev = {}
for dsg, grp in g.groupby("design"):
    base = grp[(grp.span_res == 2) & (grp.chord_res == 12)].iloc[0]
    oth = grp[~((grp.span_res == 2) & (grp.chord_res == 12))]
    dev[dsg] = dict(ld=(oth.L_D / base.L_D - 1).abs().max(), sm=(oth.SM - base.SM).abs().max(), cl=(oth.CL / base.CL - 1).abs().max(),
                    ld_x=(oth[oth.panels <= 2688].L_D / base.L_D - 1).abs().max())
top2 = [d for d in dev if d in ("AL_champion", "P1_best")]
mx_ld = max(dev[d]["ld"] for d in top2); mx_sm = max(dev[d]["sm"] for d in dev)
mac("NgridTopLD", f"{100*mx_ld:.1f}"); mac("NgridSM", f"{mx_sm:.3f}")
mac("NgridTopCL", f"{100*max(dev[d]['cl'] for d in top2):.2f}")
mac("NgridMedLD", f"{100*dev['P1_median']['ld']:.0f}"); mac("NgridPtenLD", f"{100*dev['P1_p10']['ld']:.0f}")
mac("NgridPtenLDx", f"{100*dev['P1_p10']['ld_x']:.0f}"); mac("NgridMedLDx", f"{100*dev['P1_median']['ld_x']:.0f}")
mac("NgridMaxPanels", f"{int(g.panels.max()):,}".replace(",", "{,}"))
say(f"grid: top designs max L/D change {100*mx_ld:.2f}% ; SM {mx_sm:.4f}; per design {dev}")

# CFD
cf = []
for f in sorted(glob.glob(f"{RES}/cfd/*_h7_result.csv")):
    d = pd.read_csv(f)
    if len(d) < 2:
        continue
    d = d.sort_values("alpha")
    a0, a1 = d.iloc[0], d.iloc[1]
    slope = (a1.CL - a0.CL) / (a1.alpha - a0.alpha)
    # neutral point from Cm slope: SM = -dCm/dCL (moment about the CG used in the CFD reference)
    sm = -(a1.CMy - a0.CMy) / (a1.CL - a0.CL)
    cf.append(dict(design=a0.design, CL=a0.CL, CLv=a0.VLM_CL_cruise, SM=sm, SMv=a0.VLM_SM, slope=slope, nodes=int(a0.nodes)))
cf = pd.DataFrame(cf)
DISP = {"AL_champion": "AL replication 0, best", "P1_best": "Pool best", "P1_median": "Pool median", "P1_p10": "Pool 10th percentile", "PAPER_champion": "Original campaign, best"}
tc = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule", "Design & $C_L$ Euler & $C_L$ VLM & $C_L$ ratio & $SM$ Euler & $SM$ VLM & $SM$ ratio \\\\", "\\midrule"]
for _, r in cf.iterrows():
    tc.append(f"{DISP.get(r.design, r.design)} & {r.CL:.4f} & {r.CLv:.4f} & {r.CL/r.CLv:.2f} & {r.SM:.3f} & {r.SMv:.3f} & {r.SM/r.SMv:.2f} \\\\")
tc += ["\\bottomrule", "\\end{tabular}"]
open(f"{HERE}/tables/tab_cfd.tex", "w").write("\n".join(tc) + "\n")
mac("NcfdN", str(len(cf)))
if len(cf):
    mac("NcfdSMlo", f"{(cf.SM/cf.SMv).min():.2f}"); mac("NcfdSMhi", f"{(cf.SM/cf.SMv).max():.2f}")
    ratio_cl = cf.CL / cf.CLv
    big = cf[cf.CLv > 0.15]
    mac("NcfdCLlo", f"{(big.CL/big.CLv).min():.2f}"); mac("NcfdCLhi", f"{(big.CL/big.CLv).max():.2f}")
    mac("NcfdSlope", f"{cf.slope.mean():.3f}")
    say(cf.assign(clr=cf.CL / cf.CLv, smr=cf.SM / cf.SMv).to_string())
    if "PAPER_champion" in set(cf.design):
        pc = cf[cf.design == "PAPER_champion"].iloc[0]
        mac("NcfdChampSM", f"{pc.SM:.3f}"); mac("NcfdChampSMratio", f"{pc.SM/pc.SMv:.2f}")
        mac("NcfdChampCL", f"{pc.CL/pc.CLv:.2f}")
        mac("NcfdChampHave", "1")

# NeuralFoil profile drag
pdv = pd.read_csv(f"{RES}/viscous/profile_drag_check.csv")
mid = pdv.groupby("design", sort=False).head(1).reset_index(drop=True)
tv = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "Design & Raymer, $x_{tr}=0.40$ & NeuralFoil, free transition & NeuralFoil, fully turbulent & Ratio NF/Raymer \\\\", "\\midrule"]
for _, r in mid.iterrows():
    tv.append(f"{DISP.get(r.design, r.design)} & {r.CD_raymer*1e4:.0f} & {r.CD_neuralfoil*1e4:.0f} & {r.CD_nf_forced_turb*1e4:.0f} & {r.ratio_nf_over_raymer:.2f} \\\\")
tv += ["\\bottomrule", "\\end{tabular}"]
open(f"{HERE}/tables/tab_visc.tex", "w").write("\n".join(tv) + "\n")
mac("NnfLo", f"{pdv.ratio_nf_over_raymer.min():.2f}"); mac("NnfHi", f"{pdv.ratio_nf_over_raymer.max():.2f}")
mac("NnfTurbRatio", f"{(mid.CD_nf_forced_turb/mid.CD_raymer).mean():.2f}")
say(mid[["design", "CD_raymer", "CD_neuralfoil", "CD_nf_forced_turb", "ratio_nf_over_raymer"]].to_string())
if "AL_champion" in set(mid.design):
    r = mid[mid.design == "AL_champion"].iloc[0]
    cdtot = r.CD_total; cd_nf = cdtot - r.CD_raymer + r.CD_neuralfoil
    mac("NnfChampLDgain", f"{cdtot/cd_nf:.2f}")

# ------------------------------------------------------------------ sensitivities
BD = json.load(open(f"{RES}/best_designs.json"))
A_, B_ = BD["original"], BD["allruns"]
mac("NbBsource", B_["source"].replace("_", "\\_"))
for tag, d in (("A", A_), ("B", B_)):
    mac(f"Nb{tag}Range", fm(d["range_160"], 1)); mac(f"Nb{tag}RangeSnap", fm(d["snapped_range_160"], 1)); mac(f"Nb{tag}RangeSnapRound", fm(d["snapped_range_160"], 0))
    mac(f"Nb{tag}LD", f"{d['L_D']:.2f}"); mac(f"Nb{tag}CL", f"{d['CL']:.3f}"); mac(f"Nb{tag}SM", f"{d['SM']:.3f}"); mac(f"Nb{tag}Cm", f"{d['Cm_cruise']:.3f}")
    mac(f"Nb{tag}Sref", f"{d['S_ref']:.3f}"); mac(f"Nb{tag}AR", f"{1.5**2/d['S_ref']:.2f}"); mac(f"Nb{tag}mtot", f"{d['m_total']:.2f}")
    mac(f"Nb{tag}Vkmh", f"{d['cruise_speed_ms']*3.6:.0f}"); mac(f"Nb{tag}mb", f"{d['battery_mass']:.2f}"); mac(f"Nb{tag}SMsnap", f"{d['snapped_SM']:.3f}")
    mac(f"Nb{tag}Trim", f"{100*d['CD_trim']/d['CD']:.1f}")
mac("NbSnapLossPct", f"{100*(1-B_['snapped_range_160']/B_['range_160']):.1f}")
mac("NbBoverAPct", f"{100*(B_['snapped_range_160']/A_['snapped_range_160']-1):.1f}")
def cfd_sm(name):
    r = cf[cf.design == name]
    return None if r.empty else (float(r.SM.iloc[0]), float(r.CL.iloc[0] / r.CLv.iloc[0]))
for tag, nm in (("A", "PAPER_champion"), ("B", "AL_champion")):
    x_ = cfd_sm(nm)
    if x_ and (tag == "A" or B_["source"] == "al_seed0.csv:271"):
        mac(f"Nb{tag}SMeuler", f"{x_[0]:.3f}"); mac(f"Nb{tag}CLratio", f"{x_[1]:.2f}")
    else:
        mac(f"Nb{tag}SMeuler", "n/a"); mac(f"Nb{tag}CLratio", "n/a")
bt = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule", " & Original campaign & AL replication 0, best \\\\", "\\midrule",
      f"Range, as optimized (km) & {fm(A_['range_160'],1)} & {fm(B_['range_160'],1)} \\\\",
      f"Range with battery mass set to 3.400\\,kg (km) & {fm(A_['snapped_range_160'],1)} & {fm(B_['snapped_range_160'],1)} \\\\",
      f"Battery mass variable (kg) & {A_['battery_mass']:.3f} & {B_['battery_mass']:.3f} \\\\",
      f"$L/D$ & {A_['L_D']:.2f} & {B_['L_D']:.2f} \\\\", f"$C_L$ & {A_['CL']:.3f} & {B_['CL']:.3f} \\\\",
      f"$S_\\mathrm{{ref}}$ (m$^2$), aspect ratio & {A_['S_ref']:.3f}, {1.5**2/A_['S_ref']:.2f} & {B_['S_ref']:.3f}, {1.5**2/B_['S_ref']:.2f} \\\\",
      f"All-up mass (kg) & {A_['m_total']:.2f} & {B_['m_total']:.2f} \\\\", f"Cruise speed (km/h) & {A_['cruise_speed_ms']*3.6:.0f} & {B_['cruise_speed_ms']*3.6:.0f} \\\\",
      f"$SM$, lattice & {A_['SM']:.3f} & {B_['SM']:.3f} \\\\",
      f"$SM$, Euler CFD & {mac_get('NbASMeuler')} & {mac_get('NbBSMeuler')} \\\\",
      f"$C_m$ at cruise & {A_['Cm_cruise']:.3f} & {B_['Cm_cruise']:.3f} \\\\",
      f"Trim drag (\\% of $C_D$) & {100*A_['CD_trim']/A_['CD']:.1f} & {100*B_['CD_trim']/B_['CD']:.1f} \\\\", "\\bottomrule", "\\end{tabular}"]
open(f"{HERE}/tables/tab_best.tex", "w").write("\n".join(bt) + "\n")
# airframe mass and bay position
def _sens(kind, name):
    f = f"{RES}/sens_{kind}_{name}.csv"
    if not os.path.exists(f):
        print(f"WARNING: {f} missing, using the original-campaign design as a stand-in (draft only)"); f = f"{RES}/sens_{kind}_orig.csv"
    return pd.read_csv(f)
def mass_tab(name):
    return _sens("mass", name)
tm = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule", "Design & Airframe mass factor & Range at $f_b=0.82$ & $SM$ & Best $f_b$ & Range at best $f_b$ & $SM$ at best $f_b$ \\\\", "\\midrule"]
for nm, lab in (("orig", "Original campaign"), ("allruns", "AL replication 0, best")):
    for _, r in mass_tab(nm).iterrows():
        if r.mult == 1.5: continue
        tm.append(f"{lab if r.mult == 0.7 else ''} & {r.mult:.1f} & {r.R082*F:.0f} & {r.SM082:.3f} & {r.frac_best:.2f} & {r.R_best*F:.0f} & {r.SM_best:.3f} \\\\")
    tm.append("\\midrule")
tm[-1] = "\\bottomrule"; tm.append("\\end{tabular}")
open(f"{HERE}/tables/tab_mass.tex", "w").write("\n".join(tm) + "\n")
for nm, key in (("orig", "A"), ("allruns", "B")):
    t_ = mass_tab(nm); b1 = t_[t_.mult == 1.0].iloc[0]
    for m_, k_ in ((0.7, "Lo"), (1.3, "Hi")):
        r_ = t_[t_.mult == m_].iloc[0]; mac(f"Nmass{key}{k_}", f"{100*(r_.R082/b1.R082-1):+.0f}"); mac(f"Nmass{key}{k_}Range", fm(r_.R082, 0))
    bb = _sens("bay", nm); okb = bb[bb.valid]
    mac(f"Nbay{key}Gain", f"{100*(okb.range_km_160.max()/bb[bb.frac == 0.82].range_km_160.iloc[0]-1):.1f}")
    mac(f"Nbay{key}SMlo", f"{okb.SM.min():.2f}"); mac(f"Nbay{key}SMbase", f"{bb[bb.frac == 0.82].SM.iloc[0]:.3f}")
    mac(f"Nbay{key}FwdLimit", f"{okb.frac.min():.2f}")
# SM constraint values from config
sys.path.insert(0, REPO)
try:
    from flying_wing_optimizer.config import BOUNDS, CONSTRAINTS
    cons = CONSTRAINTS
except Exception:
    cons = None

# top-design bound analysis
try:
    from flying_wing_optimizer.config import BOUNDS
    lo = np.array([getattr(BOUNDS, n)[0] for n in ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper_ratio", "theta0", "theta1", "theta2", "tip_twist", "winglet_sweep", "winglet_twist", "battery_mass", "cruise_speed_ms"]])
    hi = np.array([getattr(BOUNDS, n)[1] for n in ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper_ratio", "theta0", "theta1", "theta2", "tip_twist", "winglet_sweep", "winglet_twist", "battery_mass", "cruise_speed_ms"]])
    names = ["$\\Lambda_{in}$", "$\\Lambda_{out}$", "$y_b$", "$c_r$", "$c_b$", "$\\lambda$", "$\\theta_0$", "$\\theta_1$", "$\\theta_2$", "$\\theta_t$", "$\\Lambda_w$", "$\\theta_w$", "$m_b$", "$V$"]
    top = v.nlargest(20, "range_calm_km")[KEYS].values
    span = hi - lo
    near_lo = ((top - lo) / span < 0.03).mean(0); near_hi = ((hi - top) / span < 0.03).mean(0)
    say("top-20 share within 3% of lower/upper bound:")
    tb2 = ["\\begin{tabular}{@{}lrr@{}}", "\\toprule", "Variable & Top 20 at lower bound (\\%) & Top 20 at upper bound (\\%) \\\\", "\\midrule"]
    for n_, a_, b_ in zip(names, near_lo, near_hi):
        tb2.append(f"{n_} & {100*a_:.0f} & {100*b_:.0f} \\\\")
        say(f"  {n_}: {100*a_:.0f} {100*b_:.0f}")
    tb2 += ["\\bottomrule", "\\end{tabular}"]
    open(f"{HERE}/tables/tab_bound.tex", "w").write("\n".join(tb2) + "\n")
    mac("NtopTaperLo", f"{100*near_lo[5]:.0f}"); mac("NtopWtHi", f"{100*near_hi[11]:.0f}"); mac("NtopVLo", f"{100*near_lo[13]:.0f}")
    cbest = champ[KEYS].values.astype(float)
    at = ((cbest - lo) / span < 0.03) | ((hi - cbest) / span < 0.03)
    mac("NchAtBound", str(int(at.sum())))
    mac("NtopAtBoundN", str(int(((near_lo + near_hi) > 0.5).sum())))
except Exception as e:
    say(f"bound analysis failed: {e}")

# volume proxy vs lofted chord at the best design
yb, cr, cb, th = champ.y_blend, champ.c_root, champ.c_blend, np.tan(np.radians(champ.sweep_inner))
yy = np.linspace(0, yb, 21); xi = yy / yb
tf = 0.150 + (0.101 - 0.150) * xi
def vol(cfun):
    tot = 0
    for y_, x_, t_ in zip(yy, xi, tf):
        c = cfun(x_); h = t_ * c - 0.003
        if h > 0: tot += 0.75 * 0.45 * c * h * (yb / 20)
    return tot * 2000
cH = lambda x_: cr + (cb - cr) * (3 * x_**2 - 2 * x_**3)
cL = lambda x_: cr + (th * yb + cb - cr) * x_ - th * yb * (2 * x_**2 - x_**3)
vp, vl = vol(cH), vol(cL)
mac("NvolProxy", f"{vp:.2f}"); mac("NvolLoft", f"{vl:.2f}"); mac("NvolDiff", f"{100*(1-vp/vl):.0f}\\%")
say(f"volume proxy {vp:.2f} L vs lofted {vl:.2f} L")
# pack cell volume
cell_vol = 48 * np.pi / 4 * 0.0217**2 * 0.0702 * 1000
mac("NcellVol", f"{cell_vol:.2f}")

mac("NclampActPaper", f"{100*lr[2][4]:.0f}")
mac("NchTrimDrag", f"{100*VR['CD_trim']/VR['CD']:.1f}")
# energy basis sensitivity (48 cells, 18 Wh each)
cell_kg = NCELL * 0.069
def eu(u, oh): return PACK_NOM * u / (cell_kg + oh)
lo_, hi_ = eu(0.80, 0.003 * NCELL + 0.080), eu(0.85, 0.088)
mac("NeuLo", f"{lo_:.0f}"); mac("NeuHi", f"{hi_:.0f}")
mac("NeuLoPct", f"{100*(1-lo_/EU):.0f}"); mac("NeuHiPct", f"{100*(hi_/EU-1):.0f}")
mac("NcellKg", f"{cell_kg:.3f}"); mac("NoverheadG", f"{(PACK_KG-cell_kg)*1000:.0f}")


# ------------------------------------------------------------------ margin recalibration (post-hoc filter on all in-bounds evaluations)
pool_rows = [("db", v[["range_calm_km", "SM"]].rename(columns={"range_calm_km": "r"}))]
for f in sorted(glob.glob(f"{RES}/al/al_seed?.csv")):
    d = pd.read_csv(f); d = d[d.is_valid.astype(bool)]; pool_rows.append(("run", d[["range_calm_km", "SM"]].rename(columns={"range_calm_km": "r"})))
for f in sorted(glob.glob(f"{RES}/controls/*.csv")):
    d = pd.read_csv(f); d = d[d.is_valid.astype(bool)]; pool_rows.append(("run", d[["range_km", "SM"]].rename(columns={"range_km": "r"})))
allp = pd.concat([r for _, r in pool_rows]); dbp = pool_rows[0][1]
for r_, key in ((1.14, "Lo"), (1.21, "Mid"), (1.28, "Hi")):
    lim = 0.25 / r_
    mac(f"NsmLim{key}", f"{lim:.2f}")
    mac(f"NsmDb{key}", fm(dbp[dbp.SM <= lim].r.max(), 0)); mac(f"NsmAll{key}", fm(allp[allp.SM <= lim].r.max(), 0))
    mac(f"NsmExcl{key}", f"{100*(1-(allp.SM <= lim).mean()):.0f}")
mac("NsmAllNone", fm(allp.r.max(), 0))
sms = [p2[p2.is_valid].loc[p2[p2.is_valid].range_calm_km.idxmax(), "SM"]]
for n_, d_, s_ in runs_al():
    if n_ != "orig":
        dd_ = pd.read_csv(f"{RES}/al/{n_}.csv"); dd_ = dd_[dd_.is_valid.astype(bool)]; sms.append(dd_.loc[dd_.range_calm_km.idxmax(), "SM"])
mac("NalSMlo", f"{min(sms):.2f}"); mac("NalSMhi", f"{max(sms):.2f}")


# ------------------------------------------------------------------ Phase B (results_v3): B1/B2 weighting ablation, B4 stability calibration, B5 bound prices
def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"

B123 = f"{V3}/B123_summary"
if os.path.exists(f"{B123}/arm_summary.csv"):
    asum = pd.read_csv(f"{B123}/arm_summary.csv").set_index("arm"); ptests = pd.read_csv(f"{B123}/pairwise_tests.csv")
    def pt(a_, b_):
        r_ = ptests[(ptests.a == a_) & (ptests.b == b_)].iloc[0]; return r_
    for lab, key in [("AL soft (weighted)", "AlSoft"), ("AL hard weight", "AlHard"), ("TuRBO", "Turbo"), ("TuRBO + feasibility weight", "TurboFeas")]:
        r_ = asum.loc[lab]; mac(f"Nb{key}Mean", f"{r_['mean']:.1f}"); mac(f"Nb{key}Sd", f"{r_['sd']:.1f}"); mac(f"Nb{key}N", str(int(r_["n"])))
        mac(f"Nb{key}Valid", f"{100*r_['valid_share']:.0f}"); mac(f"Nb{key}VSixty", f"{100*r_['valid_first60']:.0f}")
        mac(f"Nb{key}Min", f"{r_['min']:.1f}"); mac(f"Nb{key}Max", f"{r_['max']:.1f}")
    r_ = asum.loc["AL no weight"]; mac("NbAlNoneN", str(int(r_["n"]))); mac("NbAlNoneValid", f"{100*r_['valid_share']:.0f}")
    mac("NbAlHardAtHundred", f"{asum.loc['AL hard weight','at100']:.1f}"); mac("NbAlHardAtOneFifty", f"{asum.loc['AL hard weight','at150']:.1f}")
    for key, (a_, b_) in {"AlHardOverSoft": ("AL hard weight", "AL soft (weighted)"), "TurboFeasOverTurbo": ("TuRBO + feasibility weight", "TuRBO"),
                          "TurboOverAlRep": ("TuRBO", "AL soft (weighted)"), "TurboFeasOverAlRep": ("TuRBO + feasibility weight", "AL soft (weighted)"),
                          "AlRepOverAdapt": ("AL soft (weighted)", "Adaptive local, 10%"), "AlRepOverLocTen": ("AL soft (weighted)", "Local, 10%"),
                          "TurboOverAdapt": ("TuRBO", "Adaptive local, 10%")}.items():
        r_ = pt(a_, b_); mac(f"Nb{key}D", f"{r_.diff_km:.1f}"); mac(f"Nb{key}Welch", pfmt(r_.welch_p)); mac(f"Nb{key}Mwu", pfmt(r_.mwu_p))
        mac(f"Nb{key}CI", f"[{r_.ci_lo:.1f}, {r_.ci_hi:.1f}]")
        say(f"B: {a_} minus {b_}: {r_.diff_km:.1f} km, Welch p={r_.welch_p:.3f}, MWU p={r_.mwu_p:.3f}, CI [{r_.ci_lo:.1f},{r_.ci_hi:.1f}]")
    tf = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule", "Search & Runs & Best at 287, mean $\\pm$ sd & Min-max & Valid (\\%) & Valid, first 60 (\\%) & Best at 150 \\\\", "\\midrule"]
    for lab, nm in [("AL soft (weighted)", "AL, sigmoid weight (as published)"), ("AL hard weight", "AL, hard 0/1 gate"), ("AL no weight", "AL, no classifier weight"),
                    ("TuRBO", "TuRBO"), ("TuRBO + feasibility weight", "TuRBO, with sigmoid weight")]:
        r_ = asum.loc[lab]
        if np.isnan(r_["mean"]):
            tf.append(f"{nm} & {int(r_['n'])} & no valid design in any run & n/a & {100*r_['valid_share']:.0f} & {100*r_['valid_first60']:.0f} & 0.0 \\\\")
        else:
            tf.append(f"{nm} & {int(r_['n'])} & {r_['mean']:.1f} $\\pm$ {r_['sd']:.1f} & {r_['min']:.1f} to {r_['max']:.1f} & {100*r_['valid_share']:.0f} & {100*r_['valid_first60']:.0f} & {r_['at150']:.1f} \\\\")
    tf += ["\\bottomrule", "\\end{tabular}"]
    open(f"{HERE}/tables/tab_feas_ablation.tex", "w").write("\n".join(tf) + "\n")
    # n = 10 control arms from the same table, for the text
    for lab, key in [("Adaptive local, 10%", "Adapt"), ("Local, 10%", "LocTen"), ("Local, 5%", "LocFive"), ("Fresh LHS", "Lhs"), ("Uniform random", "Uni")]:
        mac(f"Nb{key}Mean", f"{asum.loc[lab,'mean']:.1f}"); mac(f"Nb{key}N", str(int(asum.loc[lab, "n"])))

CAL = f"{V3}/cfd_calibration"
if os.path.exists(f"{CAL}/calibration.json"):
    cj = json.load(open(f"{CAL}/calibration.json")); cp = pd.read_csv(f"{CAL}/cfd_points.csv")
    mac("NcalN", str(cj["n_designs"])); mac("NcalA", f"{cj['a']:.3f}"); mac("NcalB", f"{cj['b']:.3f}"); mac("NcalAci", f"[{cj['a_ci'][0]:.3f}, {cj['a_ci'][1]:.3f}]")
    mac("NcalBci", f"[{cj['b_ci'][0]:.3f}, {cj['b_ci'][1]:.3f}]"); mac("NcalRsq", f"{cj['r2']:.3f}"); mac("NcalSd", f"{cj['resid_sd']:.3f}")
    mac("NcalCapEquiv", f"{cj['SM_vlm_cap_equiv']:.3f}"); mac("NcalFloorEquiv", f"{cj['SM_vlm_floor_equiv']:.3f}")
    mac("NcalCLmed", f"{cj['CL_ratio_median']:.3f}"); mac("NcalCLlo", f"{cj['CL_ratio_min']:.3f}"); mac("NcalCLhi", f"{cj['CL_ratio_max']:.3f}")
    mac("NcalOffsetMean", f"{(cp.SM_euler-cp.SM_vlm).mean():.3f}"); mac("NcalOffsetMin", f"{(cp.SM_euler-cp.SM_vlm).min():.3f}"); mac("NcalOffsetMax", f"{(cp.SM_euler-cp.SM_vlm).max():.3f}")
    mac("NcalNnew", str(int((cp.src == "results_v3").sum())))
    say(f"B4: SM_Euler = {cj['a']:.3f} + {cj['b']:.3f} SM_VLM, n={cj['n_designs']}, R2={cj['r2']:.3f}, resid sd={cj['resid_sd']:.4f}")
    # AL champion, finer wall spacing (5 mm)
    h5f = f"{V3}/cfd/top_AL_h5/result.csv"; h7f = f"{V3}/cfd/top_AL_h7/result.csv"
    if os.path.exists(h5f) and os.path.exists(h7f):
        def sm_of(f_):
            d_ = pd.read_csv(f_).sort_values("alpha"); a0_, a1_ = d_.iloc[0], d_.iloc[1]; return -(a1_.CMy - a0_.CMy) / (a1_.CL - a0_.CL), int(a0_.nodes), a0_.CL
        s5, n5, cl5 = sm_of(h5f); s7, n7, cl7 = sm_of(h7f)
        mac("NcalAlHfiveSM", f"{s5:.3f}"); mac("NcalAlHsevenSM", f"{s7:.3f}"); mac("NcalAlHdelta", f"{100*(s5/s7-1):.1f}"); mac("NcalAlHfiveNodes", f"{n5/1e3:.0f}")
        mac("NcalAlHfiveCL", f"{cl5:.3f}"); mac("NcalAlHsevenCL", f"{cl7:.3f}")
        say(f"B4 grid check, top_AL: SM_Euler h=5 {s5:.4f} ({n5} nodes) vs h=7 {s7:.4f} ({n7} nodes)")
    # re-ranking
    pr = pd.read_csv(f"{CAL}/rerank_per_run_fix.csv" if os.path.exists(f"{CAL}/rerank_per_run_fix.csv") else f"{CAL}/rerank_per_run.csv")   # _fix: corrected runs (b4b_rerank_fix.py)
    grp = {"AL (soft)": "AL, sigmoid weight", "AL-R": "AL-R", "TuRBO": "TuRBO", "TuRBO extra": "TuRBO", "TuRBO + feas weight": "TuRBO, with sigmoid weight", "AL hard weight": "AL, hard 0/1 gate",
           "adapt10": "Adaptive local, 10\\%", "local10": "Local, 10\\%", "local5": "Local, 5\\%", "local5 (v3 extra)": "Local, 5\\%", "lhs": "Fresh LHS", "lhs (v3 extra)": "Fresh LHS",
           "uniform": "Uniform random", "uniform (v3 extra)": "Uniform random"}
    pr = pr[pr.arm.isin(grp)].copy(); pr["g"] = pr.arm.map(grp)
    tr_ = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule", "Search & Runs & Best, VLM margin (km) & Best, calibrated margin (km) & Mean loss (km) & Runs whose best design leaves the band \\\\", "\\midrule"]
    for g, x in pr.groupby("g", sort=False):
        tr_.append(f"{g} & {len(x)} & {x.best_vlm.mean()*F:.1f} & {x.best_cal.mean()*F:.1f} & {x.loss.mean()*F:.1f} & {int(x.champ_infeasible.sum())} of {len(x)} \\\\")
        k_ = {"AL, sigmoid weight": "AlSoft", "AL-R": "AlR", "AL, hard 0/1 gate": "AlHard", "TuRBO": "Turbo", "TuRBO, with sigmoid weight": "TurboFeas", "Adaptive local, 10\\%": "Adapt",
              "Local, 10\\%": "LocTen", "Local, 5\\%": "LocFive", "Fresh LHS": "Lhs", "Uniform random": "Uni"}[g]
        mac(f"NcalOut{k_}", str(int(x.champ_infeasible.sum()))); mac(f"NcalOutN{k_}", str(len(x))); mac(f"NcalLoss{k_}", f"{x.loss.mean()*F:.1f}")
    tr_ += ["\\bottomrule", "\\end{tabular}"]
    open(f"{HERE}/tables/tab_sm_rerank.tex", "w").write("\n".join(tr_) + "\n")
    mac("NcalLossArmMax", f"{pr.groupby('g').loss.mean().max()*F:.1f}")

BP = f"{V3}/bound_prices_fix/bound_prices_summary.csv"   # B11: corrected TuRBO (analyze_B11.py); the first-implementation runs stay in bound_prices/
if not os.path.exists(BP): BP = f"{V3}/bound_prices/bound_prices_summary.csv"
if os.path.exists(BP):
    bp = pd.read_csv(BP); keys_ = {"taper_lo": "Taper", "wtwist_hi": "Wtwist", "bmass_hi": "Bmass", "speed_lo": "Speed"}
    names_ = {"taper_lo": "Taper ratio, lower & $0.55 \\to 0.525$", "wtwist_hi": "Winglet twist, upper & $0 \\to +1^\\circ$",
              "bmass_hi": "Battery mass, upper & $3.5 \\to 3.7$\\,kg", "speed_lo": "Cruise speed, lower & $36.1 \\to 33.6$\\,m/s"}   # name & change (two columns, 2026-11-16)
    mac("NbpBmassMassPct", f"{100*0.2/3.5:.1f}")
    ref_ = bp.iloc[0]; mac("NbpRefMean", f"{ref_['mean']:.1f}"); mac("NbpRefSd", f"{ref_['sd']:.1f}")
    tb_ = ["\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}llrrrcrr@{}}", "\\toprule", " & & & & \\multicolumn{3}{c}{Gain over the original bounds} & \\\\", "\\cmidrule(lr){5-7}",
           "Bound & Change & Runs & Best (km) & km & 95\\% CI (km) & Welch $p$ & At new bound \\\\", "\\midrule",
           f"None & original bounds & {int(ref_['n'])} & {ref_['mean']:.1f} $\\pm$ {ref_['sd']:.1f} & n/a & n/a & n/a & n/a \\\\"]
    for _, r_ in bp.iloc[1:].iterrows():
        k_ = keys_[r_["arm"]]
        mac(f"Nbp{k_}N", str(int(r_["n"]))); mac(f"Nbp{k_}Mean", f"{r_['mean']:.1f}"); mac(f"Nbp{k_}Gain", f"{r_['gain']:.1f}"); mac(f"Nbp{k_}GainPct", f"{100*r_['gain']/ref_['mean']:.1f}")
        mac(f"Nbp{k_}CI", f"[{r_['ci_lo']:.1f}, {r_['ci_hi']:.1f}]"); mac(f"Nbp{k_}P", pfmt(r_["welch_p"])); mac(f"Nbp{k_}AtBound", f"{100*r_['share_best_at_new_bound']:.0f}")
        tb_.append(f"{names_[r_['arm']]} & {int(r_['n'])} & {r_['mean']:.1f} $\\pm$ {r_['sd']:.1f} & ${r_['gain']:+.1f}$ & $[{r_['ci_lo']:.1f},\\ {r_['ci_hi']:.1f}]$ & {pfmt(r_['welch_p']).replace('<', '$<$')} & {100*r_['share_best_at_new_bound']:.0f}\\% \\\\")
        say(f"B5 {r_['arm']}: n={int(r_['n'])} mean {r_['mean']:.1f} gain {r_['gain']:.1f} p={r_['welch_p']:.3f}")
    tb_ += ["\\bottomrule", "\\end{tabular*}"]
    open(f"{HERE}/tables/tab_bound_prices.tex", "w").write("\n".join(tb_) + "\n")

# robustness of the champions to the battery centroid and battery mass (B6)
RB = f"{V3}/robustness"
if os.path.isdir(RB):
    rj = json.load(open(f"{RB}/robustness_summary.json")); rs_ = pd.read_csv(f"{RB}/robustness_summary.csv")
    mac("NrbN", str(rj["n_designs"])); mac("NrbAllValid", str(rj["n_all_valid"])); mac("NrbBandValid", str(rj["n_centroid_band_valid"]))
    mac("NrbPfiveInvalid", str(int((~rs_.mass_p5_valid).sum()))); mac("NrbOswald", str(sum(rj["mass_p5_failure_reasons"].values())))
    b_ = rj["best_all_valid_by_baseline"]; mac("NrbAllBest", b_["design"]); mac("NrbAllBestRange", f"{b_['base_range']*F:.1f}")
    mac("NrbAllBestBandLoss", f"{b_['centroid_band_loss_pct']:.1f}"); mac("NrbAllBestMfiveLoss", f"{100*(1-b_['mass_m5_range']/b_['base_range']):.1f}")
    mac("NrbAllBestPfiveGain", f"{100*(b_['mass_p5_range']/b_['base_range']-1):.1f}")
    c_ = rj["best_centroid_band_valid_by_baseline"]; mac("NrbBandBest", c_["design"]); mac("NrbBandBestRange", f"{c_['base_range']*F:.1f}"); mac("NrbBandBestLoss", f"{c_['centroid_band_loss_pct']:.1f}")
    mac("NrbBandLossMax", f"{rs_[rs_.centroid_band_valid].centroid_band_loss_pct.max():.1f}"); mac("NrbMfiveLossMean", f"{100*(1-(rs_[rs_.mass_m5_valid].mass_m5_range/rs_[rs_.mass_m5_valid].base_range)).mean():.1f}")
    say(f"B6: {rj['n_designs']} designs, {rj['n_all_valid']} valid in all perturbations, {rj['n_centroid_band_valid']} in the centroid band; best all-valid {b_['design']} {b_['base_range']*F:.1f} km")

# ------------------------------------------------------------------ write numbers.tex
with open(f"{HERE}/numbers.tex", "w") as fh:
    fh.write("% generated by make_numbers.py; do not edit\n")
    for k, val in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
open(f"{HERE}/numbers_report.txt", "w").write("\n".join(REPORT))
print("wrote", len(M), "macros")
