"""Distribution-shift test of the 5-MLP ensemble: train on Phase 1 (space-filling) valid rows only, test on Phase 2
(acquisition) valid rows of the original campaign, and a leave-top-out extrapolation test (train range < 250 km, test >= 250 km).
Same architecture as the paper: 5 x MLP(64,64), ReLU, Adam, early stopping, standardised inputs and outputs."""
import warnings, os, numpy as np, pandas as pd
ROOT = os.environ.get("DRONE_ROOT", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone")
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "revision_runs/results_revision"))
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
KEYS = ["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist","winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"]
db = pd.read_csv(f"{ROOT}/flying_wing_optimizer/results/database.csv"); v = db[db.is_valid.astype(bool)].copy()
print(v.evaluation_type.value_counts().to_dict())
p1 = v[v.evaluation_type.str.startswith("Phase1")]; p2 = v[~v.evaluation_type.str.startswith("Phase1")]
def fit_predict(tr, te, target, seeds=(42,101,2024,777,999)):
    sx, sy = StandardScaler().fit(tr[KEYS]), StandardScaler().fit(tr[[target]])
    P = []
    for s in seeds:
        m = MLPRegressor(hidden_layer_sizes=(64,64), activation="relu", solver="adam", early_stopping=True, max_iter=800, random_state=s)
        m.fit(sx.transform(tr[KEYS]), sy.transform(tr[[target]]).ravel()); P.append(sy.inverse_transform(m.predict(sx.transform(te[KEYS])).reshape(-1,1)).ravel())
    P = np.array(P); return P.mean(0), P.std(0, ddof=1)
def report(name, tr, te, target):
    mu, sd = fit_predict(tr, te, target); y = te[target].values
    r2 = 1 - ((y-mu)**2).sum()/((y-y.mean())**2).sum(); mae = np.abs(y-mu).mean(); bias = (mu-y).mean()
    top = y >= np.quantile(y, 0.9); z = np.abs(y-mu)/np.maximum(sd,1e-9)
    out = dict(test=name, target=target, n_train=len(tr), n_test=len(te), R2=round(r2,4), MAE=round(mae,3), bias=round(bias,3), spearman=round(spearmanr(y,mu)[0],3),
               MAE_top10=round(np.abs(y[top]-mu[top]).mean(),3), bias_top10=round((mu[top]-y[top]).mean(),3), spearman_top10=round(spearmanr(y[top],mu[top])[0],3),
               mean_sd=round(sd.mean(),3), cover_1sd=round((z<=1).mean(),3), cover_2sd=round((z<=2).mean(),3)); print(out); return out
rows = []
for t in ["range_calm_km", "L_D", "SM"]:
    rows.append(report("train Phase1 -> test Phase2", p1, p2, t))
    rows.append(report("train range<250 -> test range>=250", v[v.range_calm_km < 250], v[v.range_calm_km >= 250], t) if t == "range_calm_km" else report("train range<250 -> test range>=250", v[v.range_calm_km < 250], v[v.range_calm_km >= 250], t))
pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/surrogate_shift.csv", index=False)
