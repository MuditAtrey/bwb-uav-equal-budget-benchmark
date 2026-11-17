"""
flying_wing_optimizer/database.py

Database and checkpoint management module for recording evaluation results,
checking for duplicate evaluations, filtering valid designs, and managing
atomic checkpoint state persistence.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from flying_wing_optimizer.config import PATHS

# Schema field definitions
SCHEMA_COLUMNS = [
    "eval_id",
    "timestamp",
    "evaluation_type",
    # 14-parameter Dual-Sweep BWB design parameters
    "sweep_inner",
    "sweep_outer",
    "y_blend",
    "c_root",
    "c_blend",
    "taper",
    "theta0",
    "theta1",
    "theta2",
    "tip_twist",
    "winglet_sweep_deg",
    "winglet_twist_deg",
    "battery_mass",
    "cruise_speed_ms",
    # Feasibility
    "is_valid",
    "penalty_score",
    # Aerodynamic outputs
    "CL",
    "CD",
    "CD_vlm",
    "CD_profile",
    "CD_trim",
    "L_D",
    "AoA_trim",
    "Cm_trim",
    "SM",
    "x_np",
    "x_cg",
    # Mission outputs
    "m_battery_kg",
    "m_total_kg",
    "range_calm_km",
    "range_headwind_km",
    "eta_prop",
    "eta_total",
]

# 14-parameter Dual-Sweep BWB design keys
DESIGN_PARAM_KEYS = [
    "sweep_inner",
    "sweep_outer",
    "y_blend",
    "c_root",
    "c_blend",
    "taper",
    "theta0",
    "theta1",
    "theta2",
    "tip_twist",
    "winglet_sweep_deg",
    "winglet_twist_deg",
    "battery_mass",
    "cruise_speed_ms",
]


def init_database() -> None:
    """
    Ensures the results directory exists and creates database.csv with header
    if it does not already exist.
    """
    PATHS.results_dir.mkdir(parents=True, exist_ok=True)
    if not PATHS.database_file.exists() or PATHS.database_file.stat().st_size == 0:
        df_empty = pd.DataFrame(columns=SCHEMA_COLUMNS)
        df_empty.to_csv(PATHS.database_file, index=False)


def append_evaluation(eval_dict: Dict[str, Any], round_decimals: int = 5) -> None:
    """
    Appends an evaluation record to database.csv safely.
    Applies floating-point rounding to handle duplicate detection and formatting.

    Args:
        eval_dict: Dictionary containing field names and values matching database schema.
        round_decimals: Decimal places to round floating point numbers (default: 5).
    """
    init_database()

    # Format record according to schema and round float fields
    row_data: Dict[str, Any] = {}
    for col in SCHEMA_COLUMNS:
        val = eval_dict.get(col, None)
        if isinstance(val, (float, np.floating)):
            if math.isnan(val) or math.isinf(val):
                row_data[col] = val
            else:
                row_data[col] = round(float(val), round_decimals)
        else:
            row_data[col] = val

    df_row = pd.DataFrame([row_data], columns=SCHEMA_COLUMNS)
    df_row.to_csv(PATHS.database_file, mode="a", header=False, index=False)


def is_duplicate(
    design_params_tuple: Union[Tuple[float, ...], Sequence[float], np.ndarray],
    tolerance: float = 1e-4,
) -> bool:
    """
    Checks if a 7-DoF design vector has already been evaluated in the database
    within a specified tolerance.

    Args:
        design_params_tuple: Tuple/list/array of 7 DoFs
            (sweep_deg, taper, twist_root_deg, twist_mid_deg, twist_tip_deg,
             winglet_sweep_deg, winglet_twist_deg).
            5-DoF legacy vectors are also accepted; the check compares only the
            columns present in both the input and DESIGN_PARAM_KEYS.
        tolerance: Absolute tolerance for duplicate matching (default: 1e-4).

    Returns:
        True if design vector matches an existing evaluation within tolerance, False otherwise.
    """
    df = load_database()
    if df.empty:
        return False

    # Compare only the DoFs that are present in both the input vector and the database
    input_arr = np.array(design_params_tuple, dtype=float)
    n_dofs = min(len(input_arr), len(DESIGN_PARAM_KEYS))
    keys_to_check = DESIGN_PARAM_KEYS[:n_dofs]

    if not all(col in df.columns for col in keys_to_check):
        return False

    existing_params = df[keys_to_check].to_numpy(dtype=float)
    target_params = input_arr[:n_dofs]

    if existing_params.shape[0] == 0:
        return False

    diffs = np.abs(existing_params - target_params)
    match_mask = np.all(diffs <= tolerance, axis=1)

    return bool(np.any(match_mask))


def load_database() -> pd.DataFrame:
    """
    Loads all records from database.csv into a pandas DataFrame.

    Returns:
        pd.DataFrame containing all database records, or an empty DataFrame if file missing.
    """
    if not PATHS.database_file.exists():
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    try:
        df = pd.read_csv(PATHS.database_file)
        if df.empty:
            return pd.DataFrame(columns=SCHEMA_COLUMNS)
        return df
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(columns=SCHEMA_COLUMNS)


def get_valid_evaluations() -> pd.DataFrame:
    """
    Filters database for records where is_valid == True.

    Returns:
        pd.DataFrame containing only valid evaluations.
    """
    df = load_database()
    if df.empty or "is_valid" not in df.columns:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    # Convert to boolean handling string/numeric representations safely
    valid_mask = df["is_valid"].astype(str).str.lower().isin(["true", "1", "1.0"])
    valid_df = df[valid_mask].reset_index(drop=True)

    return valid_df


def save_checkpoint(checkpoint_data: Dict[str, Any]) -> None:
    """
    Writes checkpoint.json atomically via a temporary file swap to prevent
    corruption during unexpected crashes or power loss.

    Args:
        checkpoint_data: Dictionary representing current optimizer state / checkpoint.
    """
    PATHS.results_dir.mkdir(parents=True, exist_ok=True)
    temp_path = PATHS.checkpoint_json.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    # Atomic swap
    temp_path.replace(PATHS.checkpoint_json)


def load_checkpoint() -> Dict[str, Any]:
    """
    Reads checkpoint.json safely.

    Returns:
        Dictionary containing checkpoint data, or empty dict if file missing or corrupted.
    """
    if not PATHS.checkpoint_json.exists():
        return {}

    try:
        with open(PATHS.checkpoint_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
