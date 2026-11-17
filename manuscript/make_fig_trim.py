#!/usr/bin/env python3
"""
research/paper_b_flagship/scripts/trim_sensitivity.py

Trim Loop Sensitivity Analysis (Test 3) for Flagship BWB UAV Paper.

This script evaluates longitudinal static stability (Static Margin) and cruise pitch trim (Cm)
as a function of the longitudinal battery centroid position (battery_centroid_fraction) across
the Pareto champion geometry.

Outputs:
  1. Feasibility window identification (SM in [5%, 25%], |Cm| <= 0.10)
  2. Publication-Quality Figure 4 (Dual-Axis Trim Sensitivity Plot):
     - research/paper_b_flagship/figures/figure4_trim_sensitivity.pdf
     - research/paper_b_flagship/figures/figure4_trim_sensitivity.png
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import aerosandbox as asb

# ── Paths & Setup ────────────────────────────────────────────────────────────
# Copy of research/paper_b_flagship/scripts/trim_sensitivity.py for the manuscript (2026-11-17): title removed from the figure (CJA guide),
# SVG text kept editable. Reads the cached sweep from research/paper_b_flagship/figures/trim_sweep_cache.csv.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
FIGURES_DIR = SCRIPT_DIR / "figures"
CACHE_DIR = SCRIPT_DIR.parents[1] / "figures"
OPTIMAL_JSON_PATH = PROJECT_ROOT / "flying_wing_optimizer" / "results" / "optimal_design.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from flying_wing_optimizer.config import (
    ATMOSPHERE,
    CONSTRAINTS,
    MASS,
    MASS_PLACEMENT,
    MOTOR_MOUNT,
)
from flying_wing_optimizer.geometry import build_wing_geometry
from flying_wing_optimizer.aerodynamics import _find_cruise_aoa, _run_vlm

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def load_champion_parameters() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Loads optimal champion design parameters and aerodynamic metrics."""
    if OPTIMAL_JSON_PATH.exists():
        with open(OPTIMAL_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        p = data["design_parameters"]
        aero = data["aerodynamic_performance"]
    else:
        p = {
            "sweep_inner": 38.9934,
            "sweep_outer": 39.21096,
            "y_blend": 0.25417,
            "c_root": 0.45496,
            "c_blend": 0.25002,
            "taper": 0.55628,
            "theta0": 0.77232,
            "theta1": -0.85423,
            "theta2": -1.20609,
            "tip_twist": -6.33226,
            "winglet_sweep_deg": 19.3674,
            "winglet_twist_deg": -0.08997,
            "battery_mass": 3.40165,
            "cruise_speed_ms": 36.67294,
        }
        aero = {
            "CL": 0.24711,
            "CD": 0.01443,
            "L_D": 17.1237,
            "AoA_trim": 4.43398,
            "Cm_trim": -0.01925,
            "SM": 0.2254,
            "x_np": 0.33779,
            "x_cg": 0.27534,
        }
    return p, aero


def run_battery_cg_sweep(
    p: Dict[str, float],
    frac_min: float = 0.15,
    frac_max: float = 0.95,
    n_steps: int = 50,
) -> pd.DataFrame:
    """Sweeps battery_centroid_fraction and computes x_cg, x_np, SM, and Cm.

    Args:
        p: Champion design parameter dictionary.
        frac_min: Minimum fraction of c_root (default: 0.15).
        frac_max: Maximum fraction of c_root (default: 0.95).
        n_steps: Number of evaluation steps (default: 50).

    Returns:
        pd.DataFrame containing sensitivity sweep results.
    """
    c_root: float = p["c_root"]
    v_cruise: float = p["cruise_speed_ms"]
    battery_mass: float = p.get("battery_mass", 3.40165)

    # Build reference wing geometry once
    airplane_base, _ = build_wing_geometry(
        sweep_inner=p["sweep_inner"],
        sweep_outer=p["sweep_outer"],
        y_blend=p["y_blend"],
        c_root=p["c_root"],
        c_blend=p["c_blend"],
        taper=p["taper"],
        theta0=p["theta0"],
        theta1=p["theta1"],
        theta2=p["theta2"],
        tip_twist=p["tip_twist"],
        battery_mass=battery_mass,
        winglet_sweep_deg=p.get("winglet_sweep_deg", 19.3674),
        winglet_twist_deg=p.get("winglet_twist_deg", -0.08997),
    )

    mac: float = airplane_base.mac
    S_ref: float = airplane_base.S_ref
    x_le_mac: float = airplane_base.x_le_mac

    fractions = np.linspace(frac_min, frac_max, n_steps)
    results = []

    atmos = asb.Atmosphere(altitude=ATMOSPHERE.cruise_altitude)
    rho: float = float(atmos.density())
    g: float = float(ATMOSPHERE.gravity)

    for frac in fractions:
        # Dynamic mass and CG model
        x_payload: float = MASS_PLACEMENT.payload_centroid_fraction * c_root
        x_battery: float = frac * c_root
        x_motor: float = c_root + MOTOR_MOUNT.motor_cg_beyond_te_m
        x_avionics: float = MASS_PLACEMENT.avionics_fraction * c_root
        x_airframe: float = x_le_mac + 0.40 * mac

        airframe_mass: float = 1.0 + 1.5 * S_ref
        fixed_mass: float = MASS.payload_mass + airframe_mass + MASS.propulsion_mass + MASS.avionics_mass
        total_mass: float = fixed_mass + battery_mass

        num_cg: float = (
            MASS.payload_mass * x_payload
            + battery_mass * x_battery
            + airframe_mass * x_airframe
            + MASS.propulsion_mass * x_motor
            + MASS.avionics_mass * x_avionics
        )
        x_cg: float = num_cg / total_mass

        # Re-reference airplane to the dynamic x_cg
        airplane = asb.Airplane(
            name="TrimSensitivityAirplane",
            xyz_ref=[x_cg, 0.0, 0.0],
            wings=airplane_base.wings,
        )

        CL_required: float = (2.0 * total_mass * g) / (rho * (v_cruise**2) * S_ref)

        alpha_cruise, _, _ = _find_cruise_aoa(airplane, v_cruise, CL_required)
        if alpha_cruise is None:
            continue

        op_cruise = asb.OperatingPoint(atmosphere=atmos, velocity=v_cruise, alpha=alpha_cruise)
        vlm_res = _run_vlm(airplane, op_cruise, run_stability=True)
        if vlm_res is None:
            continue

        x_np: float = float(vlm_res["x_np"])
        Cm: float = float(vlm_res["Cm"])
        CL: float = float(vlm_res["CL"])
        CD_vlm: float = float(vlm_res["CD"])
        SM: float = float((x_np - x_cg) / mac) if mac > 0 else 0.0

        is_feasible = bool(
            CONSTRAINTS.min_static_margin <= SM <= CONSTRAINTS.max_static_margin
            and abs(Cm) <= CONSTRAINTS.max_abs_cm_at_cruise
        )

        results.append({
            "battery_centroid_fraction": frac,
            "x_cg_m": x_cg,
            "x_np_m": x_np,
            "StaticMargin": SM,
            "StaticMargin_pct": SM * 100.0,
            "Cm_cruise": Cm,
            "CL": CL,
            "CD_vlm": CD_vlm,
            "alpha_cruise_deg": alpha_cruise,
            "is_feasible": is_feasible,
        })

    return pd.DataFrame(results)


def plot_trim_sensitivity_figure(
    df: pd.DataFrame,
    champion_fraction: float,
    champ_sm: float,
    champ_cm: float,
    save_pdf_path: Path,
    save_png_path: Path,
):
    """Generates Publication-Quality Figure 6: Dual-Axis Trim Sensitivity."""
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))

    fractions = df["battery_centroid_fraction"]
    sm_pct = df["StaticMargin_pct"]
    cm_vals = df["Cm_cruise"]

    # Left Y-Axis: Static Margin (%)
    color_sm = "navy"
    ax1.set_xlabel(r"Battery Centroid Position ($x_{\mathrm{batt}} / c_{\mathrm{root}}$)")
    ax1.set_ylabel(r"Static Margin $SM$ [$\%$ MAC]", color=color_sm, fontweight="bold")
    line1 = ax1.plot(
        fractions,
        sm_pct,
        color=color_sm,
        linewidth=2.2,
        label=r"Static Margin $SM$ [$\%$]",
    )
    ax1.tick_params(axis="y", labelcolor=color_sm)

    # Feasibility band for Static Margin [5%, 25%]
    ax1.axhspan(
        CONSTRAINTS.min_static_margin * 100.0,
        CONSTRAINTS.max_static_margin * 100.0,
        color="royalblue",
        alpha=0.12,
        label=r"Feasible Stability Window ($SM \in [5\%, 25\%]$)",
    )
    ax1.set_ylim(-5.0, 70.0)
    ax1.grid(True, which="major", linestyle="--", color="#0f3057", alpha=0.12, zorder=0)

    # Right Y-Axis: Pitching Moment Coefficient Cm
    ax2 = ax1.twinx()
    color_cm = "crimson"
    ax2.set_ylabel(r"Cruise Pitching Moment $C_m$", color=color_cm, fontweight="bold")
    line2 = ax2.plot(
        fractions,
        cm_vals,
        color=color_cm,
        linestyle="--",
        linewidth=2.0,
        label=r"Pitching Moment $C_m$",
    )
    ax2.tick_params(axis="y", labelcolor=color_cm)
    # Color-coded Crimson dotted line matching the right y-axis for Ideal Neutral Trim
    ax2.axhline(
        0.0,
        color="crimson",
        linestyle=":",
        linewidth=1.4,
        alpha=0.85,
        zorder=1,
        label=r"Ideal Neutral Trim ($C_m = 0$)",
    )
    ax2.set_ylim(-0.15, 0.15)

    # Mark Champion Position (strictly clipped to axis box)
    ax1.axvline(
        champion_fraction,
        color="darkgreen",
        linestyle="-.",
        linewidth=1.6,
        clip_on=True,
        zorder=3,
        label=f"Champion (${champion_fraction:.2f}\\cdot c_{{\\mathrm{{root}}}}$)",
    )

    # Feasible window vertical shading (strictly clipped to axis box)
    feasible_df = df[df["is_feasible"] == True]
    if len(feasible_df) > 0:
        f_min = feasible_df["battery_centroid_fraction"].min()
        f_max = feasible_df["battery_centroid_fraction"].max()
        window_width = f_max - f_min
        ax1.axvspan(
            f_min, f_max,
            color="forestgreen",
            alpha=0.10,
            clip_on=True,
            zorder=2,
            label=f"Feasible Packaging Window",
        )

    # Points on curves at champion placement
    ax1.scatter([champion_fraction], [champ_sm * 100.0], color=color_sm, s=55, zorder=5, edgecolors="black")
    ax2.scatter([champion_fraction], [champ_cm], color=color_cm, s=55, zorder=5, edgecolors="black")

    # Layout 1: Horizontal Top Banner Legend (3 columns x 2 rows, positioned directly above plot)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower center",
        bbox_to_anchor=(0.50, 1.02),
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.96,
        fontsize=7.8,
        edgecolor="#b0bec5",
    )

    # Place Title cleanly above the top legend with tight elegant spacing
    pass  # title removed: the caption carries it

    # Ensure spines sit on top so borders are razor clean
    for spine in ax1.spines.values():
        spine.set_zorder(10)
    for spine in ax2.spines.values():
        spine.set_zorder(10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_pdf_path, bbox_inches="tight")
    plt.savefig(save_png_path, bbox_inches="tight")
    save_svg_path = save_pdf_path.parent / "svg" / (save_pdf_path.stem + ".svg"); plt.rcParams["svg.fonttype"] = "none"
    plt.savefig(save_svg_path, bbox_inches="tight", format="svg")
    plt.close()
    print(f"\nSaved Figure to:")
    print(f"  - {save_pdf_path}")
    print(f"  - {save_png_path}")
    print(f"  - {save_svg_path}")


def main():
    p, aero = load_champion_parameters()
    champion_fraction = float(MASS_PLACEMENT.battery_centroid_fraction)

    print("=" * 80)
    print("      DISPATCH #4 — TEST 3: TRIM LOOP SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Champion Geometry: c_root = {p['c_root']:.4f} m, sweep_inner = {p['sweep_inner']:.2f} deg")
    print(f"Operational Cruise Point: V = {p['cruise_speed_ms']:.2f} m/s, Altitude = 500 m ASL")
    print(f"Nominal Battery Mass: {p.get('battery_mass', 3.40165):.3f} kg")

    csv_cache = CACHE_DIR / "trim_sweep_cache.csv"
    if csv_cache.exists():
        df_sweep = pd.read_csv(csv_cache)
    else:
        print("\nSweeping battery_centroid_fraction from 0.15 to 0.95 across 50 steps...")
        df_sweep = run_battery_cg_sweep(p, frac_min=0.15, frac_max=0.95, n_steps=50)
        df_sweep.to_csv(csv_cache, index=False)

    # Feasibility analysis
    feasible_df = df_sweep[df_sweep["is_feasible"] == True]
    if len(feasible_df) > 0:
        f_min = float(feasible_df["battery_centroid_fraction"].min())
        f_max = float(feasible_df["battery_centroid_fraction"].max())
        window_width = f_max - f_min
    else:
        f_min, f_max, window_width = 0.0, 0.0, 0.0

    # Nearest evaluation to nominal champion
    champ_idx = (df_sweep["battery_centroid_fraction"] - champion_fraction).abs().argmin()
    champ_eval = df_sweep.iloc[champ_idx]

    pdf_out = FIGURES_DIR / "figure4_trim_sensitivity.pdf"
    png_out = FIGURES_DIR / "svg" / "_figure4_preview.png"
    plot_trim_sensitivity_figure(
        df=df_sweep,
        champion_fraction=champion_fraction,
        champ_sm=float(champ_eval["StaticMargin"]),
        champ_cm=float(champ_eval["Cm_cruise"]),
        save_pdf_path=pdf_out,
        save_png_path=png_out,
    )

    print("\n" + "=" * 80)
    print("Battery CG Sensitivity Analysis:")
    print(f"  Feasibility window: battery_centroid_fraction ∈ [{f_min:.2f}, {f_max:.2f}]")
    print(f"  Window width: {window_width:.2f} (out of 0.80 total range, {window_width/0.80*100:.1f}% usable space)")
    print(f"  Champion placement: {champion_fraction:.2f} (SM = {champ_eval['StaticMargin_pct']:.1f}%, Cm = {champ_eval['Cm_cruise']:.4f})")
    print("  Designs outside window: SM and/or Cm constraints violated")
    print("=" * 80)


if __name__ == "__main__":
    main()
