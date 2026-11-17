"""Builds RESULTS.md numbers from the revision experiment CSVs. usage: python3 analyze_revision.py RESULTS_DIR"""
import sys, glob, os, pandas as pd, numpy as np
from scipy.stats import spearmanr
R = sys.argv[1]; P1_BEST, PAPER_AL = 277.8, 298.7
def load(d, arm):
    return [pd.read_csv(f) for f in sorted(glob.glob(f"{R}/{d}/{arm}_seed?.csv"))]
def best_curve(df, col="range_km", n=287):
    v = df[col].where(df.is_valid, 0.0); return v.cummax().values[:n]
print("## Equal-budget controls (287 evaluations after the 9,800-sample pool; range on 160 Wh/kg basis)")
rows = []
for arm in ["lhs", "uniform", "local5", "local10"]:
    ds = load("controls", arm); b = [d.range_km.max() for d in ds]
    rows.append((arm, len(ds), np.mean(b), np.std(b, ddof=1) if len(b) > 1 else np.nan, np.mean([d.is_valid.mean() for d in ds]),
                 np.mean([d[d.is_valid].range_km.median() for d in ds])))
print("| arm | seeds | best (mean) | sd | valid rate | median valid |\n|---|---|---|---|---|---|")
for r in rows: print(f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.2f} | {r[5]:.1f} |")
print(f"Reference: Phase 1 best {P1_BEST} km; paper AL {PAPER_AL} km")
als = sorted(glob.glob(f"{R}/al/al_seed?.csv"))
for f in als:
    d = pd.read_csv(f); d["is_valid"] = d.is_valid.astype(bool)
    print(f"AL replication {os.path.basename(f)}: n={len(d)} best={d[d.is_valid].range_calm_km.max():.1f} valid={d.is_valid.mean():.2f}; first 60 evals valid={d.is_valid[:60].mean():.2f}")
print("\n## Manifold vs no clamp (widened c_blend 0.15-0.65)")
for arm in ["clamp", "noclamp"]:
    ds = load("mvu", arm); print(arm, "best per seed", [round(d.range_km.max(), 1) for d in ds], "mean", round(np.mean([d.range_km.max() for d in ds]), 1),
                                 "valid", round(np.mean([d.is_valid.mean() for d in ds]), 2), "inverted share of samples", round(np.mean([d.inverted.mean() for d in ds]), 2))
print("\n## Paper bounds vs relaxed champion-active bounds")
for arm in ["paper", "wide"]:
    ds = load("bnd", arm); print(arm, "best per seed", [round(d.range_km.max(), 1) for d in ds], "mean", round(np.mean([d.range_km.max() for d in ds]), 1),
                                 "clamp active share", round(np.mean([d.clamp_active.mean() for d in ds]), 3))
x = pd.read_csv(f"{R}/mvu/xsolver.csv"); x["ratio"] = x.ab_LD / x.L_D; x = x[x.ab_LD.notna()]
print("\n## Cross-solver (AeroBuildup vs pipeline), n =", len(x), " Spearman pooled =", round(spearmanr(x.L_D, x.ab_LD)[0], 3),
      " top-only =", round(spearmanr(x[x.sel == 'top'].L_D, x[x.sel == 'top'].ab_LD)[0], 3), " median AB/pipeline =", round(x.ratio.median(), 2))
