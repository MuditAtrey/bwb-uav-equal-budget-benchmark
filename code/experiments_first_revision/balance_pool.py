"""E1: apply the balance solver post hoc to pool designs (first evaluation taken from the database, one new evaluation each).
usage: python3 balance_pool.py N_RANDOM N_TOP"""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, multiprocessing as mp
KEYS = ["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist","winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"]
OUT = "/mnt/user-data/outputs/results_revision/balance/pool_balanced.csv"; E = 160.0 / 208.25

def work(row):
    import balance as B
    vec = np.array([row[k] for k in KEYS], float)
    mt = row["m_total_kg"]
    if not mt > 0:   # the database does not store the mass of invalid designs: rebuild it with one nominal evaluation
        mt = B.total_mass(vec) or 0.0
    t = B.target_fb(vec[3], vec[12], mt, row["SM"], row["Cm_trim"], row["CL"], row["x_np"], row["x_cg"])
    base = dict(eval_id=int(row["eval_id"]), group=row["group"], valid0=bool(row["is_valid"]), R0=float(row["range_calm_km"]),  # database ranges are already on the 160 Wh/kg basis
                SM0=float(row["SM"]), Cm0=float(row["Cm_trim"]), LD0=float(row["L_D"]))
    if t is None:
        return dict(base, fb=np.nan, SMt=np.nan, valid1=False, R1=0.0, SM1=np.nan, Cm1=np.nan, LD1=np.nan, reached=False)
    try:
        r = B.evaluate_at(vec, t[0])
        return dict(base, fb=t[0], SMt=t[1], valid1=bool(r["is_valid"]), R1=float(r["range_calm_km"]) * E if r["is_valid"] else 0.0,
                    SM1=float(r["SM"]), Cm1=float(r["Cm_trim"]), LD1=float(r["L_D"]), reached=True)
    except Exception:
        return dict(base, fb=t[0], SMt=t[1], valid1=False, R1=0.0, SM1=np.nan, Cm1=np.nan, LD1=np.nan, reached=True)

if __name__ == "__main__":
    nr, nt = int(sys.argv[1]), int(sys.argv[2])
    db = pd.read_csv("/tmp/work/flying_wing_optimizer/results/database.csv"); db["is_valid"] = db.is_valid.astype(bool)
    p1 = db[db.evaluation_type.str.startswith("Phase1") & (db.x_np > 0)]
    rnd = p1.sample(nr, random_state=7).assign(group="random")
    top = db[db.is_valid].nlargest(nt, "range_calm_km").assign(group="top")
    rows = pd.concat([rnd, top]).to_dict("records")
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["eval_id", "group"])
    key = set(zip(done.eval_id, done.group)); todo = [r for r in rows if (int(r["eval_id"]), r["group"]) not in key]
    print("reached-stability share of Phase 1:", len(p1) / db.evaluation_type.str.startswith("Phase1").sum(), "todo", len(todo), flush=True)
    with mp.Pool(int(os.environ.get("NPROC", "2"))) as pool:
        for i, res in enumerate(pool.imap(work, todo)):
            done = pd.concat([done, pd.DataFrame([res])], ignore_index=True)
            if i % 10 == 0: done.to_csv(OUT, index=False)
    done.to_csv(OUT, index=False); print("done", len(done))
