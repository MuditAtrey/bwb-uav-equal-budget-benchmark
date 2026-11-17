"""
flying_wing_optimizer/main.py

Main orchestration for the flying wing optimization pipeline.

Phase 1: Latin Hypercube Sampling (parallel)
Phase 2: Surrogate Active Learning (parallel)

Rewrite 2026-08-02:
  - Fixed Phase 2 key name mismatch (was using old 5-DoF names)
  - Builds 16-element design vectors matching database schema
  - Removed SIGALRM timeout (doesn't work in multiprocessing workers)
  - S_ref passed through to mission solver
  - Aero result keys updated (AoA_cruise, Cm_cruise, CD_profile)
"""

import logging
import math
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

from flying_wing_optimizer.config import (
    BOUNDS,
    MASS,
    N_CORES,
    PATHS,
    PHASE1_DESIGNS,
    TARGETS,
)
from flying_wing_optimizer.database import (
    DESIGN_PARAM_KEYS,
    SCHEMA_COLUMNS,
    append_evaluation,
    get_valid_evaluations,
    init_database,
    is_duplicate,
    load_database,
    save_checkpoint,
    load_checkpoint,
)
from flying_wing_optimizer.aerodynamics import evaluate_aerodynamics
from flying_wing_optimizer.mission import calculate_mission_performance
from flying_wing_optimizer.surrogate import TARGET_COLUMNS, SurrogateEnsemble
from flying_wing_optimizer.optimizer import SurrogateOptimizer
from flying_wing_optimizer.visualization import (
    export_optimal_design,
    generate_convergence_plot,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("flying_wing_optimizer.main")

TOTAL_RUN_TIME_LIMIT_SECONDS: float = 36000.0
PHASE_2_TIME_LIMIT_SECONDS: float = 7200.0
PHASE_1_SAMPLE_COUNT: int = 5000


def evaluate_design_vector(
    design_vector: Union[Tuple[float, ...], List[float], np.ndarray],
    eval_id: int,
    evaluation_type: str = "LHS",
) -> Dict[str, Any]:
    """Evaluate a single 18-DoF design vector.

    Design vector indices (matching DESIGN_PARAM_KEYS):
        0: sweep_deg        8: c3
        1: taper             9: theta0
        2: c_root           10: theta1
        3: x1               11: theta2
        4: x2               12: theta3
        5: x3               13: tip_twist
        6: c1               14: winglet_sweep_deg
        7: c2               15: winglet_twist_deg
                            16: battery_mass [kg]
                            17: cruise_speed_ms [m/s]

    Returns a dict matching SCHEMA_COLUMNS.
    """
    dv = [float(v) for v in design_vector]

    sweep_inner = dv[0]
    sweep_outer = dv[1]
    y_blend = dv[2]
    c_root = dv[3]
    c_blend = dv[4]
    taper = dv[5]
    theta0 = dv[6]
    theta1 = dv[7]
    theta2 = dv[8]
    tip_twist_val = dv[9]
    winglet_sweep_deg = dv[10]
    winglet_twist_deg = dv[11]
    battery_mass = dv[12]
    v_cruise = dv[13]

    timestamp_str = datetime.now(timezone.utc).isoformat()

    def build_result(
        aero: Dict[str, Any],
        is_val: bool,
        m_bat: float,
        m_tot: float,
        r_calm: float,
        r_hw: float,
        eta_p: float = 0.0,
        eta_t: float = 0.0,
    ) -> Dict[str, Any]:
        return {
            "eval_id": eval_id,
            "timestamp": timestamp_str,
            "evaluation_type": evaluation_type,
            "sweep_inner": aero.get("sweep_inner", sweep_inner),
            "sweep_outer": aero.get("sweep_outer", sweep_outer),
            "y_blend": aero.get("y_blend", y_blend),
            "c_root": aero.get("c_root", c_root),
            "c_blend": aero.get("c_blend", c_blend),
            "taper": aero.get("taper", taper),
            "theta0": aero.get("theta0", theta0),
            "theta1": aero.get("theta1", theta1),
            "theta2": aero.get("theta2", theta2),
            "tip_twist": aero.get("tip_twist", tip_twist_val),
            "winglet_sweep_deg": aero.get("winglet_sweep_deg", winglet_sweep_deg),
            "winglet_twist_deg": aero.get("winglet_twist_deg", winglet_twist_deg),
            "battery_mass": aero.get("battery_mass", battery_mass),
            "cruise_speed_ms": aero.get("cruise_speed_ms", v_cruise),
            "is_valid": is_val,
            "penalty_score": float(aero.get("penalty_score", 0.0)),
            "CL": float(aero.get("CL", 0.0)),
            "CD": float(aero.get("CD", 0.0)),
            "CD_vlm": float(aero.get("CD_vlm", 0.0)),
            "CD_profile": float(aero.get("CD_profile", 0.0)),
            "CD_trim": float(aero.get("CD_trim", 0.0)),
            "L_D": float(aero.get("L_D", 0.0)),
            "AoA_trim": float(aero.get("AoA_cruise", 0.0)),
            "Cm_trim": float(aero.get("Cm_cruise", 0.0)),
            "SM": float(aero.get("SM", 0.0)),
            "x_np": float(aero.get("x_np", 0.0)),
            "x_cg": float(aero.get("x_cg", 0.0)),
            "m_battery_kg": m_bat,
            "m_total_kg": m_tot,
            "range_calm_km": r_calm,
            "range_headwind_km": r_hw,
            "eta_prop": eta_p,
            "eta_total": eta_t,
            "reasons": aero.get("reasons", []),
        }

    from flying_wing_optimizer.geometry import build_wing_geometry
    
    geom_check = build_wing_geometry(
        sweep_inner=sweep_inner,
        sweep_outer=sweep_outer,
        y_blend=y_blend,
        c_root=c_root,
        c_blend=c_blend,
        taper=taper,
        theta0=theta0,
        theta1=theta1,
        theta2=theta2,
        tip_twist=tip_twist_val,
        battery_mass=battery_mass,
        winglet_sweep_deg=winglet_sweep_deg,
        winglet_twist_deg=winglet_twist_deg,
    )
    if geom_check is None:
        aero = {"is_valid": False, "penalty_score": 1e6, "reasons": ["Invalid geometry planform"]}
        return build_result(aero, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Aerodynamic evaluation
    aero = evaluate_aerodynamics(
        sweep_inner=sweep_inner,
        sweep_outer=sweep_outer,
        y_blend=y_blend,
        c_root=c_root,
        c_blend=c_blend,
        taper=taper,
        theta0=theta0,
        theta1=theta1,
        theta2=theta2,
        tip_twist=tip_twist_val,
        v_cruise=v_cruise,
        battery_mass=battery_mass,
        winglet_sweep_deg=winglet_sweep_deg,
        winglet_twist_deg=winglet_twist_deg,
    )

    if not aero.get("is_valid", False):
        return build_result(aero, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    CL = float(aero.get("CL", 0.0))
    L_D = float(aero.get("L_D", 0.0))
    S_ref = float(aero.get("S_ref", 0.0))

    if L_D <= 0 or S_ref <= 0:
        return build_result(aero, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Mission (range) evaluation — uses actual S_ref from geometry
    mission = calculate_mission_performance(
        L_D=L_D,
        CL=CL,
        S_ref=S_ref,
        battery_mass=battery_mass,
        total_mass=float(aero.get("m_total", 0.0)),
    )

    return build_result(
        aero,
        True,
        float(mission["m_battery_kg"]),
        float(mission["m_total_kg"]),
        float(mission["range_calm_km"]),
        float(mission["range_headwind_km"]),
        float(mission.get("eta_prop", 0.0)),
        float(mission.get("eta_total", 0.0)),
    )


def _evaluate_worker(
    args: Tuple[Union[Tuple[float, ...], List[float], np.ndarray], int, str],
) -> Dict[str, Any]:
    """Worker for parallel evaluation. No SIGALRM (doesn't work in pool workers)."""
    design_vector, eval_id, eval_type = args
    try:
        return evaluate_design_vector(
            design_vector=design_vector,
            eval_id=eval_id,
            evaluation_type=eval_type,
        )
    except Exception as e:
        logger.error(f"Worker crashed on eval {eval_id}: {e}")
        dv = [float(v) for v in design_vector]
        return {
            "eval_id": eval_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluation_type": eval_type,
            "sweep_inner": dv[0], "sweep_outer": dv[1], "y_blend": dv[2],
            "c_root": dv[3], "c_blend": dv[4], "taper": dv[5],
            "theta0": dv[6], "theta1": dv[7], "theta2": dv[8], "tip_twist": dv[9],
            "winglet_sweep_deg": dv[10], "winglet_twist_deg": dv[11],
            "battery_mass": dv[12], "cruise_speed_ms": dv[13],
            "is_valid": False,
            "penalty_score": 1e6,
            "CL": 0.0, "CD": 0.0, "CD_vlm": 0.0, "CD_profile": 0.0, "CD_trim": 0.0, "L_D": 0.0,
            "AoA_trim": 0.0, "Cm_trim": 0.0, "SM": 0.0,
            "x_np": 0.0, "x_cg": 0.0,
            "m_battery_kg": 0.0, "m_total_kg": 0.0,
            "range_calm_km": 0.0, "range_headwind_km": 0.0,
            "eta_prop": 0.0, "eta_total": 0.0,
            "reasons": [f"Worker exception: {e}"],
        }


def generate_lhs_samples(
    n_samples: int = PHASE_1_SAMPLE_COUNT, seed: int = 42
) -> np.ndarray:
    """Generate 14-DoF LHS samples for True Dual-Sweep BWB geometry."""
    lower = np.array([
        BOUNDS.sweep_inner[0], BOUNDS.sweep_outer[0], BOUNDS.y_blend[0],
        BOUNDS.c_root[0], BOUNDS.c_blend[0], BOUNDS.taper_ratio[0],
        BOUNDS.theta0[0], BOUNDS.theta1[0], BOUNDS.theta2[0], BOUNDS.tip_twist[0],
        BOUNDS.winglet_sweep[0], BOUNDS.winglet_twist[0],
        BOUNDS.battery_mass[0], BOUNDS.cruise_speed_ms[0],
    ])
    upper = np.array([
        BOUNDS.sweep_inner[1], BOUNDS.sweep_outer[1], BOUNDS.y_blend[1],
        BOUNDS.c_root[1], BOUNDS.c_blend[1], BOUNDS.taper_ratio[1],
        BOUNDS.theta0[1], BOUNDS.theta1[1], BOUNDS.theta2[1], BOUNDS.tip_twist[1],
        BOUNDS.winglet_sweep[1], BOUNDS.winglet_twist[1],
        BOUNDS.battery_mass[1], BOUNDS.cruise_speed_ms[1],
    ])

    sampler = LatinHypercube(d=14, seed=seed)
    unit = sampler.random(n=n_samples)
    samples = lower + unit * (upper - lower)

    for i in range(len(samples)):
        # Ensure washouts monotonically decrease root -> tip
        samples[i, 6:10] = np.sort(samples[i, 6:10])[::-1]

    return samples


def main() -> None:
    """Main loop: Phase 1 (LHS) + Phase 2 (Active Learning)."""
    logger.info("=" * 72)
    logger.info("Starting Flying Wing Optimization Pipeline")
    logger.info("=" * 72)

    init_database()
    checkpoint = load_checkpoint()

    run_start = time.time()
    prior_elapsed = float(checkpoint.get("elapsed_time_seconds", 0.0))

    def elapsed():
        return (time.time() - run_start) + prior_elapsed

    df_db = load_database()
    total_evals = len(df_db)
    valid_df = get_valid_evaluations()
    logger.info(f"Database: {total_evals} evaluations ({len(valid_df)} valid). Prior: {prior_elapsed:.0f}s.")

    # ── Phase 1: LHS ─────────────────────────────────────────────────────
    if total_evals < 50:
        cores = os.cpu_count() or N_CORES
        logger.info(f"Phase 1: {PHASE_1_SAMPLE_COUNT} LHS samples on {cores} cores")

        lhs = generate_lhs_samples(n_samples=PHASE_1_SAMPLE_COUNT, seed=42)

        df_curr = load_database()
        start_id = len(df_curr) + 1
        tasks: List[Tuple[Any, int, str, float]] = []

        for sample in lhs:
            vec_key = tuple(np.round(sample, 5))
            if not is_duplicate(vec_key):
                tasks.append((sample, start_id + len(tasks), "LHS"))

        if tasks:
            logger.info(f"Evaluating {len(tasks)} LHS samples on {cores} cores...")
            with multiprocessing.Pool(processes=cores, maxtasksperchild=1) as pool:
                for rec in pool.imap_unordered(_evaluate_worker, tasks):
                    append_evaluation(rec)

            df_up = load_database()
            export_optimal_design(df_up)
            generate_convergence_plot(df_up)

            save_checkpoint({
                "phase": "phase_1_lhs",
                "total_evaluations": len(df_up),
                "valid_evaluations": len(get_valid_evaluations()),
                "elapsed_time_seconds": elapsed(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            })
            logger.info(f"Phase 1 complete. Valid: {len(get_valid_evaluations())}. Elapsed: {elapsed():.0f}s.")
            
            # Print diagnostic stats
            df_val = get_valid_evaluations()
            evals_complete = len(df_up)
            valid_so_far = len(df_val)
            rejection_rate = ((evals_complete - valid_so_far) / evals_complete) * 100 if evals_complete > 0 else 0.0
            
            best_range = 0.0
            median_ld = 0.0
            best_params = {}
            if valid_so_far > 0:
                best_idx = df_val["range_calm_km"].idxmax()
                best_design = df_val.loc[best_idx]
                best_range = best_design["range_calm_km"]
                median_ld = df_val["L_D"].median()
                best_params = best_design[DESIGN_PARAM_KEYS].to_dict()
                
            print("=" * 60)
            print(f"PHASE 1 STATS: {evals_complete} evaluations complete")
            print(f"Valid designs: {valid_so_far} ({rejection_rate:.1f}% rejection rate)")
            print(f"Best range_calm_km: {best_range:.2f} km")
            print(f"Median L/D (valid): {median_ld:.2f}")
            print(f"Best parameters: {best_params}")
            print("=" * 60)
    else:
        logger.info(f"Phase 1 already done ({total_evals} evals). Skipping.")

    # ── Phase 2: Active Learning ─────────────────────────────────────────
    logger.info("--- Phase 2: Active Learning ---")
    phase_2_iter = int(checkpoint.get("phase_2_iteration", 0))
    phase_2_start = elapsed()

    while (elapsed() - phase_2_start) < PHASE_2_TIME_LIMIT_SECONDS and elapsed() < TOTAL_RUN_TIME_LIMIT_SECONDS:
        phase_2_iter += 1
        logger.info(f"=== Iteration {phase_2_iter} | {elapsed():.0f}s / {TOTAL_RUN_TIME_LIMIT_SECONDS:.0f}s ===")

        valid_df = get_valid_evaluations()
        df_all = load_database()

        surr = SurrogateEnsemble()
        candidates: List[Dict[str, Any]] = []

        if len(valid_df) >= surr.min_samples:
            X = valid_df[DESIGN_PARAM_KEYS]
            Y = valid_df[TARGET_COLUMNS]
            fitted = surr.fit(X, Y)

            if fitted:
                surr.fit_feasibility(df_all[DESIGN_PARAM_KEYS], df_all["is_valid"].astype(bool))
                logger.info(f"Surrogate trained on {len(valid_df)} valid designs.")

                # Seed DE with top 20 known designs so it starts near the proven optimum
                top_designs = valid_df.nlargest(20, "range_calm_km")[DESIGN_PARAM_KEYS].values
                
                # beta=0.3: mostly exploit the surrogate's predictions (R²=0.99),
                # with only a small exploration bonus. High beta (1.5) caused DE
                # to chase high-uncertainty corners instead of the known optimum.
                opt = SurrogateOptimizer(popsize=150, maxiter=300, tol=0.001, workers=1, beta=0.3)
                candidates = opt.search(surr, seed_designs=top_designs)
            else:
                logger.warning("Surrogate fit failed.")
        else:
            logger.warning(f"Insufficient valid designs ({len(valid_df)} < {surr.min_samples}).")

        # Fallback
        if not candidates:
            rand = generate_lhs_samples(n_samples=6, seed=int(time.time() * 1000) % 100000)
            for row in rand:
                cand = {"candidate_type": "fallback_random"}
                for i, key in enumerate(DESIGN_PARAM_KEYS):
                    cand[key] = float(row[i])
                candidates.append(cand)

        # Dedup against database
        filtered: List[Dict[str, Any]] = []
        seen_batch = set()

        for cand in candidates:
            vec = tuple(cand.get(k, 0.0) for k in DESIGN_PARAM_KEYS)
            vec_key = tuple(np.round(vec, 5))
            if vec_key not in seen_batch and not is_duplicate(vec):
                seen_batch.add(vec_key)
                filtered.append(cand)

        logger.info(f"Selected {len(filtered)} new unique candidates.")

        if filtered:
            df_curr = load_database()
            start_id = len(df_curr) + 1
            cand_tasks: List[Tuple[Any, int, str, float]] = []

            for i, cand in enumerate(filtered):
                design_vec = [cand.get(k, 0.0) for k in DESIGN_PARAM_KEYS]
                cand_type = cand.get("candidate_type", "active_learning")
                cand_tasks.append((
                    design_vec,
                    start_id + i,
                    f"ActiveLearning_{cand_type}",
                ))

            if cand_tasks:
                cores = os.cpu_count() or N_CORES
                n_proc = max(1, min(len(cand_tasks), cores))
                with multiprocessing.Pool(processes=n_proc, maxtasksperchild=1) as pool:
                    for rec in pool.imap_unordered(_evaluate_worker, cand_tasks):
                        append_evaluation(rec)
            else:
                logger.info("No new unique candidates in this iteration.")

            df_up = load_database()
            export_optimal_design(df_up)
            generate_convergence_plot(df_up)

            save_checkpoint({
                "phase": "phase_2_active_learning",
                "phase_2_iteration": phase_2_iter,
                "total_evaluations": len(df_up),
                "valid_evaluations": len(get_valid_evaluations()),
                "elapsed_time_seconds": elapsed(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            })
            logger.info(f"Iteration {phase_2_iter} done. Valid: {len(get_valid_evaluations())}. Elapsed: {elapsed():.0f}s.")

    logger.info("=" * 72)
    logger.info(f"Optimization complete. Total: {elapsed():.0f}s.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
