"""Predictions behind the parity figure: (a) random 80/20 split of the valid database, (b) train on Phase 1, predict the Phase 2 designs.
Same single-output networks as surrogate_shift_test.py. Writes surrogate_parity.csv (database range basis 160 Wh/kg)."""
import warnings, os, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
ROOT = os.environ.get("DRONE_ROOT", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone")
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "revision_runs/results_revision"))
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
KEYS = ["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist","winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"]
db = pd.read_csv(f"{ROOT}/flying_wing_optimizer/results/database.csv"); v = db[db.is_valid.astype(bool)].reset_index(drop=True)
p1 = v[v.evaluation_type.str.startswith("Phase1")]; p2 = v[~v.evaluation_type.str.startswith("Phase1")]
def fp(tr, te, target="range_calm_km", seeds=(42,101,2024,777,999)):
    sx, sy = StandardScaler().fit(tr[KEYS]), StandardScaler().fit(tr[[target]]); P = []
    for s in seeds:
        m = MLPRegressor(hidden_layer_sizes=(64,64), activation="relu", solver="adam", early_stopping=True, max_iter=800, random_state=s)
        m.fit(sx.transform(tr[KEYS]), sy.transform(tr[[target]]).ravel()); P.append(sy.inverse_transform(m.predict(sx.transform(te[KEYS])).reshape(-1,1)).ravel())
    P = np.array(P); return P.mean(0), P.std(0, ddof=1)
tr, te = train_test_split(v, test_size=0.2, random_state=0)
mu, sd = fp(tr, te); a = pd.DataFrame(dict(panel="random", y=te.range_calm_km.values, mu=mu, sd=sd))
mu, sd = fp(p1, p2); b = pd.DataFrame(dict(panel="shift", y=p2.range_calm_km.values, mu=mu, sd=sd))
pd.concat([a, b]).to_csv(f"{RESULTS_DIR}/surrogate_parity.csv", index=False); print("done", len(a), len(b))
