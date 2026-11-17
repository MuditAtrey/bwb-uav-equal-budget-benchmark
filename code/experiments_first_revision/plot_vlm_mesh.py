"""Plot publication-quality 3D and 2D VLM panel mesh discretization for Reviewer 2, Point 1.
Exports to revision_runs/results_revision/figs/fig_vlm_discretization.pdf and .png."""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.environ.get("DRONE_ROOT", "/Volumes/ExternalOS/Users/muditatrey/Documents/Codes/drone")
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "revision_runs/results_revision"))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import aerosandbox as asb
from flying_wing_optimizer.geometry import build_wing_geometry

# Load champion design
al = pd.read_csv(f"{RESULTS_DIR}/al/al_seed0.csv")
KEYS = ["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist","winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"]
row = al[al.is_valid].nlargest(1, "range_calm_km").iloc[0]
d = dict(zip(KEYS, [float(row[k]) for k in KEYS]))

airplane, _ = build_wing_geometry(
    sweep_inner=d["sweep_inner"], sweep_outer=d["sweep_outer"], y_blend=d["y_blend"],
    c_root=d["c_root"], c_blend=d["c_blend"], taper=d["taper"],
    theta0=d["theta0"], theta1=d["theta1"], theta2=d["theta2"], tip_twist=d["tip_twist"],
    battery_mass=d["battery_mass"], winglet_sweep_deg=d["winglet_sweep_deg"], winglet_twist_deg=d["winglet_twist_deg"]
)

op_point = asb.OperatingPoint(velocity=d["cruise_speed_ms"], alpha=2.0)
vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=op_point, spanwise_resolution=2, chordwise_resolution=12)
sol = vlm.run()

fl = vlm.front_left_vertices
bl = vlm.back_left_vertices
br = vlm.back_right_vertices
fr = vlm.front_right_vertices

N = len(fl)
quads = [np.array([fl[i], bl[i], br[i], fr[i]]) for i in range(N)]

# Plotting
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6
})

fig = plt.figure(figsize=(7.5, 3.4), dpi=300)

# (a) 3D Isometric View
ax0 = fig.add_subplot(1, 2, 1, projection="3d")
poly = Poly3DCollection(quads, facecolors="#2a78d6", edgecolors="#0b0b0b", linewidths=0.25, alpha=0.35)
ax0.add_collection3d(poly)

# Scale limits
all_pts = np.vstack([fl, bl, br, fr])
xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
zmin, zmax = all_pts[:, 2].min(), all_pts[:, 2].max()

max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max() / 2.0
mid_x = (xmax+xmin) * 0.5
mid_y = (ymax+ymin) * 0.5
mid_z = (zmax+zmin) * 0.5

ax0.set_xlim(mid_x - max_range, mid_x + max_range)
ax0.set_ylim(mid_y - max_range, mid_y + max_range)
ax0.set_zlim(mid_z - max_range * 0.35, mid_z + max_range * 0.35)

ax0.view_init(elev=28, azim=-62)
ax0.set_xlabel("X (streamwise) [m]", labelpad=-2, fontsize=7)
ax0.set_ylabel("Y (spanwise) [m]", labelpad=-2, fontsize=7)
ax0.set_zlabel("Z [m]", labelpad=-5, fontsize=7)
ax0.tick_params(labelsize=6.5, pad=0)
ax0.set_title(f"(a) 3D Vortex Lattice Mesh ({N} panels)", fontsize=8, loc="left", pad=10)

# (b) Planform 2D Projection (Top View)
ax1 = fig.add_subplot(1, 2, 2)
for q in quads:
    # plot boundary of quad in x-y plane
    closed = np.vstack([q[:, [1, 0]], q[0:1, [1, 0]]])  # Y on horizontal, X inverted on vertical
    ax1.plot(closed[:, 0], closed[:, 1], color="#2a78d6", lw=0.35, alpha=0.85)

ax1.set_xlabel("Spanwise Y [m]", fontsize=7)
ax1.set_ylabel("Streamwise X [m]", fontsize=7)
ax1.invert_yaxis()
ax1.set_aspect("equal", "datalim")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(labelsize=6.5)
ax1.set_title(f"(b) Surface Planform Discretization (2 spanwise $\\times$ 12 chordwise)", fontsize=8, loc="left")

plt.tight_layout()
out_dir = f"{RESULTS_DIR}/figs"
os.makedirs(out_dir, exist_ok=True)
pdf_path = f"{out_dir}/fig_vlm_discretization.pdf"
png_path = f"{out_dir}/fig_vlm_discretization.png"
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, bbox_inches="tight", dpi=300)
plt.close()

# Also copy to analysis/figs
os.makedirs(f"{RESULTS_DIR}/analysis/figs", exist_ok=True)
import shutil
shutil.copy(pdf_path, f"{RESULTS_DIR}/analysis/figs/fig_vlm_discretization.pdf")
shutil.copy(png_path, f"{RESULTS_DIR}/analysis/figs/fig_vlm_discretization.png")

print(f"Generated {pdf_path} and {png_path} successfully!")

