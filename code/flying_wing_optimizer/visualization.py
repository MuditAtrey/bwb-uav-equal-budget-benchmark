"""
flying_wing_optimizer/visualization.py

Visualization and reporting module for exporting optimal design results to JSON
and rendering publication-quality multi-panel convergence plots.
"""

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flying_wing_optimizer.config import PATHS, TARGETS
from flying_wing_optimizer.database import load_database, get_valid_evaluations


def _safe_atomic_replace(src: Path, dst: Path) -> None:
    """
    Atomically replace `dst` with `src`.

    Uses Path.replace() first (atomic on POSIX within the same filesystem).
    Falls back to shutil.move() which handles cross-device moves gracefully
    (e.g., tmp on internal SSD → results on an ExternalOS drive).

    This fixes the 'Invalid cross-device link' OSError that was causing crash logs.
    """
    try:
        src.replace(dst)
    except OSError:
        shutil.move(str(src), str(dst))

def export_optimal_design(database_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Identifies the valid design record with the maximum calm-air range (range_calm_km).
    Formats a clean JSON structure containing parameter values, aerodynamic performance,
    mass breakdown, and range results.
    Saves atomically to flying_wing_optimizer/results/optimal_design.json.

    Args:
        database_df: Pandas DataFrame containing evaluation records.

    Returns:
        Optional[Dict[str, Any]]: Clean dictionary of optimal design if found, else None.
    """
    PATHS.results_dir.mkdir(parents=True, exist_ok=True)

    if database_df.empty or "is_valid" not in database_df.columns or "range_calm_km" not in database_df.columns:
        return None

    # Filter for valid design records
    valid_mask = database_df["is_valid"].astype(str).str.lower().isin(["true", "1", "1.0"])
    valid_df = database_df[valid_mask].copy()

    if valid_df.empty:
        return None

    # NOTE: soft penalties (tip thickness, CG, Cm) are informational — they
    # don't disqualify a design. Use all valid designs when selecting the best.

    # Convert range_calm_km to float numeric safely and locate maximum
    valid_df["_range_numeric"] = pd.to_numeric(valid_df["range_calm_km"], errors="coerce").fillna(-1.0)
    best_idx = valid_df["_range_numeric"].idxmax()
    best_row = valid_df.loc[best_idx]

    # Helper extraction routines for scalar conversions
    def _to_float(val: Any, default: float = 0.0) -> float:
        try:
            f_val = float(val)
            return default if math.isnan(f_val) or math.isinf(f_val) else f_val
        except (ValueError, TypeError):
            return default

    def _to_int(val: Any, default: int = 0) -> int:
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    optimal_data: Dict[str, Any] = {
        "eval_id": _to_int(best_row.get("eval_id", 0)),
        "timestamp": str(best_row.get("timestamp", "")),
        "evaluation_type": str(best_row.get("evaluation_type", "")),
        "is_valid": True,
        "penalty_score": _to_float(best_row.get("penalty_score", 0.0)),
        "design_parameters": {
            "sweep_inner": _to_float(best_row.get("sweep_inner")),
            "sweep_outer": _to_float(best_row.get("sweep_outer")),
            "y_blend": _to_float(best_row.get("y_blend")),
            "c_root": _to_float(best_row.get("c_root")),
            "c_blend": _to_float(best_row.get("c_blend")),
            "taper": _to_float(best_row.get("taper")),
            "theta0": _to_float(best_row.get("theta0")),
            "theta1": _to_float(best_row.get("theta1")),
            "theta2": _to_float(best_row.get("theta2")),
            "tip_twist": _to_float(best_row.get("tip_twist")),
            "winglet_sweep_deg": _to_float(best_row.get("winglet_sweep_deg", 30.0)),
            "winglet_twist_deg": _to_float(best_row.get("winglet_twist_deg", -5.0)),
            "battery_mass": _to_float(best_row.get("battery_mass", 2.0)),
            "cruise_speed_ms": _to_float(best_row.get("cruise_speed_ms", 36.11)),
        },
        "aerodynamic_performance": {
            "CL": _to_float(best_row.get("CL")),
            "CD": _to_float(best_row.get("CD")),
            "CD_vlm": _to_float(best_row.get("CD_vlm")),
            "CD_profile": _to_float(best_row.get("CD_profile")),
            "CD_nacelle": _to_float(best_row.get("CD_nacelle")),
            "L_D": _to_float(best_row.get("L_D")),
            "AoA_trim": _to_float(best_row.get("AoA_trim")),
            "Cm_trim": _to_float(best_row.get("Cm_trim")),
            "SM": _to_float(best_row.get("SM")),
            "x_np": _to_float(best_row.get("x_np")),
            "x_cg": _to_float(best_row.get("x_cg")),
        },
        "mass_breakdown": {
            "m_battery_kg": _to_float(best_row.get("m_battery_kg")),
            "m_total_kg": _to_float(best_row.get("m_total_kg")),
        },
        "range_results": {
            "range_calm_km": _to_float(best_row.get("range_calm_km")),
            "range_headwind_km": _to_float(best_row.get("range_headwind_km")),
        },
        "penalty_score": _to_float(best_row.get("penalty_score", 0.0)),
        "CL": _to_float(best_row.get("CL")),
        "CD": _to_float(best_row.get("CD")),
        "L_D": _to_float(best_row.get("L_D")),
        "AoA_trim": _to_float(best_row.get("AoA_trim")),
        "Cm_trim": _to_float(best_row.get("Cm_trim")),
        "SM": _to_float(best_row.get("SM")),
        "m_battery_kg": _to_float(best_row.get("m_battery_kg")),
        "m_total_kg": _to_float(best_row.get("m_total_kg")),
        "range_calm_km": _to_float(best_row.get("range_calm_km")),
        "range_headwind_km": _to_float(best_row.get("range_headwind_km")),
    }

    # Save atomically via temporary file swap (cross-device safe)
    temp_path = PATHS.optimal_design_json.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(optimal_data, f, indent=2)

    _safe_atomic_replace(temp_path, PATHS.optimal_design_json)
    return optimal_data


def generate_convergence_plot(database_df: pd.DataFrame) -> None:
    """
    Generates a multi-panel publication-quality plot using matplotlib:
      - Panel 1: Calm-Air Range (km) vs Evaluation Iteration (valid/invalid & running max).
      - Panel 2: L/D Ratio and Static Margin (%) progression over iterations.
      - Panel 3: Battery Mass (kg) vs Calm Range (km) scatter plot.

    Saves plot atomically to flying_wing_optimizer/results/convergence.png.
    Handles empty or single-row database DataFrames without crashing.

    Args:
        database_df: Pandas DataFrame containing evaluation records.
    """
    PATHS.results_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.suptitle("Flying Wing Design Optimization - Convergence & Performance Analysis", fontsize=14, fontweight="bold", y=0.98)

    # Handle empty DataFrame gracefully
    if database_df.empty or "range_calm_km" not in database_df.columns:
        for ax in (ax1, ax2, ax3):
            ax.text(0.5, 0.5, "No Evaluation Data Available", ha="center", va="center", fontsize=11, color="gray")
            ax.set_axis_off()
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        temp_path = PATHS.convergence_plot.with_suffix(".png.tmp")
        plt.savefig(temp_path, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        _safe_atomic_replace(temp_path, PATHS.convergence_plot)
        return

    df = database_df.copy()

    # Pre-process iteration vector
    if "eval_id" in df.columns and not df["eval_id"].empty:
        eval_ids = pd.to_numeric(df["eval_id"], errors="coerce").values
        if np.isnan(eval_ids).any():
            iterations = np.arange(1, len(df) + 1)
        else:
            iterations = eval_ids
    else:
        iterations = np.arange(1, len(df) + 1)

    # Pre-process numeric vectors
    range_calm = pd.to_numeric(df["range_calm_km"], errors="coerce").fillna(0.0).values
    if "is_valid" in df.columns:
        is_valid = df["is_valid"].astype(str).str.lower().isin(["true", "1", "1.0"]).values
    else:
        is_valid = np.ones(len(df), dtype=bool)

    valid_mask = is_valid
    invalid_mask = ~is_valid

    # Compute running maximum curve for valid design range
    running_max = np.zeros(len(df))
    curr_max = 0.0
    for i in range(len(df)):
        if is_valid[i] and range_calm[i] > curr_max:
            curr_max = range_calm[i]
        running_max[i] = curr_max

    # -------------------------------------------------------------------------
    # Panel 1: Calm-Air Range (km) vs Evaluation Iteration
    # -------------------------------------------------------------------------
    if np.any(invalid_mask):
        ax1.scatter(
            iterations[invalid_mask],
            range_calm[invalid_mask],
            color="#e74c3c",
            marker="x",
            s=35,
            alpha=0.6,
            label="Invalid Evaluation",
            zorder=2,
        )

    if np.any(valid_mask):
        ax1.scatter(
            iterations[valid_mask],
            range_calm[valid_mask],
            color="#2ecc71",
            marker="o",
            s=35,
            alpha=0.8,
            edgecolors="none",
            label="Valid Evaluation",
            zorder=3,
        )

    ax1.plot(
        iterations,
        running_max,
        color="#1b4f72",
        linewidth=2.0,
        linestyle="-",
        label="Running Max Range",
        zorder=4,
    )

    # No fixed target range — the objective is to maximize range

    ax1.set_title("Calm-Air Range Progression", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Evaluation Iteration", fontsize=10)
    ax1.set_ylabel("Calm-Air Range (km)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best", fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel 2: L/D Ratio & Static Margin (%) Progression
    # -------------------------------------------------------------------------
    ld_ratio = pd.to_numeric(df.get("L_D", 0.0), errors="coerce").fillna(0.0).values
    static_margin_pct = pd.to_numeric(df.get("SM", 0.0), errors="coerce").fillna(0.0).values * 100.0

    color_ld = "#2980b9"
    color_sm = "#d35400"

    ax2.set_title("L/D & Static Margin Progression", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Evaluation Iteration", fontsize=10)
    ax2.set_ylabel("L/D Ratio", fontsize=10, color=color_ld)

    if np.any(valid_mask):
        ax2.scatter(
            iterations[valid_mask],
            ld_ratio[valid_mask],
            color=color_ld,
            marker="o",
            s=30,
            alpha=0.75,
            label="L/D Ratio",
        )
    else:
        ax2.scatter(iterations, ld_ratio, color=color_ld, marker="o", s=30, alpha=0.5, label="L/D Ratio")

    ax2.tick_params(axis="y", labelcolor=color_ld)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel("Static Margin (% MAC)", fontsize=10, color=color_sm)
    if np.any(valid_mask):
        ax2_twin.scatter(
            iterations[valid_mask],
            static_margin_pct[valid_mask],
            color=color_sm,
            marker="^",
            s=30,
            alpha=0.75,
            label="Static Margin (%)",
        )
    else:
        ax2_twin.scatter(iterations, static_margin_pct, color=color_sm, marker="^", s=30, alpha=0.5, label="Static Margin (%)")

    ax2_twin.tick_params(axis="y", labelcolor=color_sm)

    # Combined legend for twin axes
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel 3: Battery Mass (kg) vs Calm Range (km) Scatter Plot
    # -------------------------------------------------------------------------
    m_battery = pd.to_numeric(df.get("m_battery_kg", 0.0), errors="coerce").fillna(0.0).values

    ax3.set_title("Battery Mass vs Calm Range", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Battery Mass (kg)", fontsize=10)
    ax3.set_ylabel("Calm Range (km)", fontsize=10)

    if np.any(invalid_mask):
        ax3.scatter(
            m_battery[invalid_mask],
            range_calm[invalid_mask],
            color="#e74c3c",
            marker="x",
            s=35,
            alpha=0.6,
            label="Invalid Design",
        )

    if np.any(valid_mask):
        ld_valid = ld_ratio[valid_mask]
        min_ld, max_ld = np.min(ld_valid), np.max(ld_valid)
        vmin = min_ld if min_ld != max_ld else min_ld - 1.0
        vmax = max_ld if min_ld != max_ld else max_ld + 1.0

        sc = ax3.scatter(
            m_battery[valid_mask],
            range_calm[valid_mask],
            c=ld_valid,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            marker="o",
            s=45,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.5,
            label="Valid Design",
        )
        cbar = fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label("L/D Ratio", fontsize=9)

    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="best", fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    temp_path = PATHS.convergence_plot.with_suffix(".png.tmp")
    plt.savefig(temp_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    _safe_atomic_replace(temp_path, PATHS.convergence_plot)
