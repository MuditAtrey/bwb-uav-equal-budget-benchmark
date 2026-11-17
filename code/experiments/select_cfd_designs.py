"""B4 step 3: choose designs for the extra CFD runs from the existing arms (AL, TuRBO, controls) in results_revision.
 - arm champions: the best valid design of each arm (5 arms) -> 'top' group
 - SM sweep: for 10 VLM-SM targets between 0.05 and 0.25 (inclusive), the valid design nearest each target, among designs with range >= pool median
Excludes designs already simulated (PAPER_champion, AL replication 0 best, P1 best/median/p10). Writes designs.json and cfd_selection.csv,
with a ranked backup list in case a mesh fails."""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
R = f"{ROOT}/revision_runs/results_revision"; OUT = f"{ROOT}/revision_runs/results_v3/cfd"; os.makedirs(OUT, exist_ok=True)
KEYS = ["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist","winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"]
frames = []
for arm, pat, rc in [("AL", f"{R}/al/al_seed?.csv", "range_calm_km"), ("turbo", f"{R}/turbo/turbo_seed?.csv", "range_km")] + \
        [(a, f"{R}/controls/{a}_seed?.csv", "range_km") for a in ("adapt10", "local10", "local5")]:
    for f in sorted(glob.glob(pat)):
        d = pd.read_csv(f); d = d[d.is_valid.astype(bool)].copy()
        d["range"] = d[rc]; d["arm"] = arm; d["src"] = os.path.basename(f); d["row"] = d.index; frames.append(d[KEYS + ["SM", "range", "arm", "src", "row"]])
A = pd.concat(frames, ignore_index=True)
done = pd.read_csv(f"{R}/cfd/AL_champion_h7_result.csv")        # only to know the AL replication 0 champion is excluded
alr0 = A[(A.arm == "AL") & (A.src == "al_seed0.csv")].nlargest(1, "range").iloc[0]
def same(x, y): return np.allclose(x[KEYS].astype(float).values, y[KEYS].astype(float).values, atol=1e-9)
sel, backup = [], []
for arm in ["turbo", "AL", "adapt10", "local10", "local5"]:
    cand = A[A.arm == arm].sort_values("range", ascending=False)
    cand = cand[[not same(r, alr0) for _, r in cand.iterrows()]] if arm == "AL" else cand
    r = cand.iloc[0]; sel.append(dict(group="top", target=np.nan, name=f"top_{arm}", **r.to_dict()))
    for j in (1, 2): r = cand.iloc[j]; backup.append(dict(group="top", target=np.nan, name=f"topB{j}_{arm}", **r.to_dict()))
med = A.range.median()
Am = A[A.range >= med]
for t in np.linspace(0.05, 0.25, 10):
    c = Am.iloc[(Am.SM - t).abs().argsort()[:3]]
    for j, (_, r) in enumerate(c.iterrows()):
        (sel if j == 0 else backup).append(dict(group="sm", target=t, name=f"sm_{t:.3f}" + ("" if j == 0 else f"_b{j}"), **r.to_dict()))
S = pd.DataFrame(sel); B = pd.DataFrame(backup)
pd.concat([S, B]).to_csv(f"{OUT}/cfd_selection.csv", index=False)
json.dump({r["name"]: {k: float(r[k]) for k in KEYS} for r in sel + backup}, open(f"{OUT}/designs.json", "w"), indent=1)
print(S[["name", "arm", "src", "range", "SM", "target"]].round(3).to_string()); print(f"{len(B)} backups; pool median range {med:.1f}")
