"""B6: robustness of the best designs to battery-centroid and battery-mass error.
Designs: top 20 valid designs over the whole database and all arms in results_revision, plus every arm's best and the original campaign
champion (deduplicated). Each is re-evaluated with the full pipeline at battery centroid fraction 0.78, 0.80, 0.82, 0.84, 0.86 c_root and at
battery mass x0.95 and x1.05 (centroid 0.82). The centroid is set in-process (object.__setattr__ on the frozen config, as in balance.py), so
config.py is untouched and the default 0.82 is unchanged for every other run.
Ranges are on the 160 Wh/kg database basis (x 160/208.25 as in the arm scripts).
usage: DRONE_ROOT=.. NPROC=2 python3 robustness_b6.py OUTDIR"""
import glob, json, os, sys, warnings
ROOT = os.environ.get("DRONE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, ROOT); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, multiprocessing as mp
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
E_SCALE = 160.0 / 208.25
KEYS = ["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist","winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"]
R = f"{ROOT}/revision_runs/results_revision"
PERT = [("fb", 0.78), ("fb", 0.80), ("fb", 0.82), ("fb", 0.84), ("fb", 0.86), ("mass", 0.95), ("mass", 1.05)]

def work(a):
    import flying_wing_optimizer.config as C, flying_wing_optimizer.geometry as G
    from flying_wing_optimizer.main import evaluate_design_vector
    name, vec, kind, val = a
    fb = val if kind == "fb" else 0.82
    for m in (C.MASS_PLACEMENT, G.MASS_PLACEMENT):
        object.__setattr__(m, "battery_centroid_fraction", float(fb))
    v = list(vec)
    if kind == "mass": v[12] = v[12] * val
    try:
        r = evaluate_design_vector(v, 990000, f"b6_{name}")
        return dict(design=name, kind=kind, value=val, is_valid=bool(r["is_valid"]), range_km=float(r["range_calm_km"]) * E_SCALE,
                    SM=float(r["SM"]), Cm=float(r["Cm_trim"]), L_D=float(r["L_D"]), reasons="; ".join(map(str, r.get("reasons") or [])))
    except Exception as e:
        return dict(design=name, kind=kind, value=val, is_valid=False, range_km=0.0, SM=0.0, Cm=0.0, L_D=0.0, reasons="exception")

if __name__ == "__main__":
    fr = []
    db = pd.read_csv(f"{ROOT}/flying_wing_optimizer/results/database.csv"); db = db[db.is_valid.astype(bool)]
    d = db.copy(); d["range"] = d.range_calm_km; d["arm"] = np.where(d.evaluation_type.str.startswith("Phase1"), "pool", "campaign"); fr.append(d[KEYS + ["range", "arm"]])
    for arm, pat, rc in [("AL", f"{R}/al/al_seed?.csv", "range_calm_km"), ("turbo", f"{R}/turbo/turbo_seed?.csv", "range_km")] + \
            [(a, f"{R}/controls/{a}_seed?.csv", "range_km") for a in ("adapt10", "local10", "local5", "lhs", "uniform")]:
        for f in sorted(glob.glob(pat)):
            x = pd.read_csv(f); x = x[x.is_valid.astype(bool)].copy(); x["range"] = x[rc]; x["arm"] = arm; fr.append(x[KEYS + ["range", "arm"]])
    A = pd.concat(fr, ignore_index=True); A["key"] = A[KEYS].round(9).astype(str).agg("|".join, axis=1)
    A = A.sort_values("range", ascending=False).drop_duplicates("key")
    pick = [("top%02d" % (i + 1), r) for i, (_, r) in enumerate(A.head(20).iterrows())]
    seen = set(A.head(20).key)
    for arm in ["campaign", "turbo", "AL", "adapt10", "local10", "local5", "lhs", "uniform"]:
        r = A[A.arm == arm].iloc[0]
        if r.key not in seen: pick.append((f"best_{arm}", r)); seen.add(r.key)
        else: pick.append((f"best_{arm}_dup", r))      # already in the top 20: keep the label, do not re-run
    run = [(n, r) for n, r in pick if not n.endswith("_dup")]
    meta = pd.DataFrame([dict(design=n, arm=r.arm, range_160=r.range, **{k: r[k] for k in KEYS}) for n, r in pick]); meta.to_csv(f"{OUT}/designs.csv", index=False)
    tasks = [(n, [r[k] for k in KEYS], kind, val) for n, r in run for kind, val in PERT]
    print(f"{len(run)} designs x {len(PERT)} perturbations = {len(tasks)} evaluations", flush=True)
    with mp.Pool(int(os.environ.get("NPROC", "2"))) as pool:
        rows = list(pool.imap(work, tasks, chunksize=2))
    res = pd.DataFrame(rows); res.to_csv(f"{OUT}/robustness.csv", index=False)
    base = res[(res.kind == "fb") & (res.value == 0.82)].set_index("design")
    out = []
    for n, g in res.groupby("design", sort=False):
        b = base.loc[n]; fbg = g[g.kind == "fb"]; mg = g[g.kind == "mass"]; v = g[g.is_valid]; vf = fbg[fbg.is_valid]
        m95 = mg[mg.value == 0.95].iloc[0]; m105 = mg[mg.value == 1.05].iloc[0]
        out.append(dict(design=n, base_range=b.range_km, base_valid=bool(b.is_valid), n_valid=int(g.is_valid.sum()), n_pert=len(g), all_valid=bool(g.is_valid.all()),
                        centroid_band_valid=bool(fbg.is_valid.all()), centroid_band_range_min=vf.range_km.min() if len(vf) else 0.0,
                        centroid_band_loss_pct=100 * (1 - vf.range_km.min() / b.range_km) if len(vf) and b.range_km else 100.0,
                        mass_m5_valid=bool(m95.is_valid), mass_m5_range=m95.range_km, mass_p5_valid=bool(m105.is_valid), mass_p5_range=m105.range_km,
                        mass_p5_reason=m105.reasons, min_range=v.range_km.min() if len(v) else 0.0, min_SM=g.SM.min(), max_SM=g.SM.max()))
    S = pd.DataFrame(out); S.to_csv(f"{OUT}/robustness_summary.csv", index=False)
    ok = S[S.all_valid].sort_values("min_range", ascending=False); okc = S[S.centroid_band_valid].sort_values("base_range", ascending=False)
    json.dump(dict(n_designs=len(S), n_all_valid=int(S.all_valid.sum()), n_centroid_band_valid=int(S.centroid_band_valid.sum()),
                   best_all_valid_by_baseline=(S[S.all_valid].sort_values("base_range", ascending=False).iloc[0].to_dict() if len(ok) else None),
                   best_centroid_band_valid_by_baseline=(okc.iloc[0].to_dict() if len(okc) else None),
                   mass_p5_failure_reasons=res[(res.kind == "mass") & (res.value == 1.05) & (~res.is_valid)].reasons.value_counts().to_dict()), open(f"{OUT}/robustness_summary.json", "w"), indent=1, default=float)
    print(S.round(2).to_string())
