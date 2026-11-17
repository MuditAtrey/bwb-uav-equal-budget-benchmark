"""
flying_wing_optimizer/optimizer.py

Surrogate optimization via Differential Evolution over the 18-DoF parameter space.
Specifically:
  - Takes a trained SurrogateEnsemble
  - Runs scipy.optimize.differential_evolution
  - 18-DoF bounds from config.BOUNDS
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import os
import numpy as np
from scipy.optimize import differential_evolution

from flying_wing_optimizer.config import BOUNDS
from flying_wing_optimizer.surrogate import SurrogateEnsemble


def _feasibility_weight(p: np.ndarray) -> np.ndarray:
    """Classifier weight on the acquisition. FEAS_MODE env var (default 'soft' = original behaviour):
    soft: 1/(1+exp(-10(p-0.5)))   none: 1 (classifier ignored)   hard: 1 if p >= 0.5 else 0."""
    mode = os.environ.get("FEAS_MODE", "soft")
    p = np.asarray(p, dtype=float)
    if mode == "none":
        return np.ones_like(p)
    if mode == "hard":
        return (p >= 0.5).astype(float)
    return 1.0 / (1.0 + np.exp(-10.0 * (p - 0.5)))


class _AcquisitionObjective:
    """Vectorized objective for batch DE evaluation.
    
    When DE calls with vectorized=True, it passes x as shape (D, N) where
    D=18 dimensions and N=population size. We transpose to (N, D) for
    batch prediction, which is 300x faster than single-sample calls.
    """

    def __init__(self, surrogate_ensemble: SurrogateEnsemble, beta: float = 1.5):
        self.surrogate_ensemble = surrogate_ensemble
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x is (D, N) when vectorized=True
        X = x.T  # -> (N, D)
        N = X.shape[0]

        # Batch predictions
        mu, sigma = self.surrogate_ensemble.predict(X)
        feas_probs = self.surrogate_ensemble.predict_feasibility_proba(X)

        # Acquisition: predicted_range_mean + beta * std_range (index 3 = range_calm_km)
        if mu.ndim > 1:
            mu_range = mu[:, 3]
            sigma_range = sigma[:, 3]
        else:
            mu_range = np.array([mu[3]])
            sigma_range = np.array([sigma[3]])

        raw_acq = mu_range + self.beta * sigma_range

        # Feasibility sigmoid weighting
        smooth_weights = _feasibility_weight(feas_probs)

        # Predicted penalty (index 4 = penalty_score)
        if mu.ndim > 1 and mu.shape[1] > 4:
            pred_penalties = np.maximum(0.0, mu[:, 4])
        else:
            pred_penalties = np.zeros(N)

        acquisitions = (raw_acq * smooth_weights) - pred_penalties

        # DE minimizes, we want to maximize acquisition
        return -acquisitions


# Keys matching the database schema (DESIGN_PARAM_KEYS in database.py)
_CANDIDATE_KEYS = [
    "sweep_inner", "sweep_outer", "y_blend",
    "c_root", "c_blend", "taper",
    "theta0", "theta1", "theta2", "tip_twist",
    "winglet_sweep_deg", "winglet_twist_deg",
    "battery_mass", "cruise_speed_ms"
]


def _vec_to_candidate(x_vec: np.ndarray, acquisition: float,
                       uncertainty: float, predicted_range: float,
                       cand_type: str) -> Dict[str, Any]:
    """Convert a 14-element design vector to a candidate dict with correct keys."""
    cand = {"design_vector": x_vec.tolist()}
    for i, key in enumerate(_CANDIDATE_KEYS):
        cand[key] = float(x_vec[i])
    cand["acquisition"] = acquisition
    cand["uncertainty"] = uncertainty
    cand["predicted_range_mean"] = predicted_range
    cand["candidate_type"] = cand_type
    return cand


class SurrogateOptimizer:
    """Differential Evolution optimizer over 18-DoF parameter space."""

    def __init__(
        self,
        popsize: int = 300,
        maxiter: int = 500,
        tol: float = 0.001,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        workers: int = 1,
        beta: float = 1.5,
        seed: Optional[int] = None,
    ):
        self.popsize = popsize
        self.maxiter = maxiter
        self.tol = tol
        self.mutation = mutation
        self.recombination = recombination
        self.workers = workers
        self.beta = beta
        self.seed = seed

        # 14-DoF bounds matching DesignBoundsConfig order
        self.bounds = [
            BOUNDS.sweep_inner,
            BOUNDS.sweep_outer,
            BOUNDS.y_blend,
            BOUNDS.c_root,
            BOUNDS.c_blend,
            BOUNDS.taper_ratio,
            BOUNDS.theta0, BOUNDS.theta1, BOUNDS.theta2,
            BOUNDS.tip_twist,
            BOUNDS.winglet_sweep,
            BOUNDS.winglet_twist,
            BOUNDS.battery_mass,
            BOUNDS.cruise_speed_ms,
        ]

    def search(self, surrogate_ensemble: SurrogateEnsemble,
               seed_designs: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Run DE optimization and return top candidates."""
        objective = _AcquisitionObjective(surrogate_ensemble, beta=self.beta)
        n_dims = len(self.bounds)

        # Build initial population in PHYSICAL SPACE (scipy clips init to bounds)
        init_pop = None
        if seed_designs is not None and len(seed_designs) > 0:
            if seed_designs.shape[1] == len(self.bounds):
                pop_size = self.popsize * n_dims
                n_seed = min(len(seed_designs), pop_size)
                
                # Keep seeds in physical space
                seeds_physical = np.clip(seed_designs[:n_seed],
                                          [b[0] for b in self.bounds],
                                          [b[1] for b in self.bounds])
            else:
                pop_size = self.popsize * n_dims
                n_seed = 0
                seeds_physical = np.empty((0, n_dims))
                
            # Fill remaining population with random samples IN PHYSICAL SPACE
            bounds_arr = np.array(self.bounds)
            lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
            n_random = pop_size - n_seed
            rng = np.random.default_rng(42)
            random_pop = lo + rng.random((n_random, n_dims)) * (hi - lo)
            
            init_pop = np.vstack([seeds_physical, random_pop])

        result = differential_evolution(
            func=objective,
            bounds=self.bounds,
            popsize=self.popsize,
            maxiter=self.maxiter,
            tol=self.tol,
            mutation=self.mutation,
            recombination=self.recombination,
            workers=self.workers,
            seed=self.seed,
            vectorized=True,
            updating="deferred",
            init=init_pop if init_pop is not None else "latinhypercube",
        )

        population = np.asarray(result.population, dtype=float)
        if hasattr(result, "x") and result.x is not None:
            population = np.vstack([result.x.reshape(1, -1), population])

        # Evaluate surrogate on entire population
        mu, sigma = surrogate_ensemble.predict(population)
        raw_acq = surrogate_ensemble.calculate_acquisition(population, beta=self.beta)
        feas_probs = surrogate_ensemble.predict_feasibility_proba(population)
        smooth_weights = _feasibility_weight(feas_probs)

        if mu.ndim > 1 and mu.shape[1] > 4:
            pred_penalties = np.maximum(0.0, mu[:, 4])
        else:
            pred_penalties = np.zeros(len(population))

        acquisitions = (raw_acq * smooth_weights) - pred_penalties

        # Range predictions (index 3 = range_calm_km)
        if mu.ndim > 1:
            mu_range = mu[:, 3]
            sigma_range = sigma[:, 3]
        else:
            mu_range = np.array([mu[3]])
            sigma_range = np.array([sigma[3]])
            acquisitions = np.array([acquisitions])
            feas_probs = np.array([feas_probs])

        # Top 5 by acquisition
        indices_by_acq = np.argsort(-acquisitions)
        candidates: List[Dict[str, Any]] = []
        seen = set()

        for idx in indices_by_acq:
            key = tuple(np.round(population[idx], 5))
            if key not in seen:
                seen.add(key)
                candidates.append(_vec_to_candidate(
                    population[idx],
                    float(acquisitions[idx]),
                    float(sigma_range[idx]),
                    float(mu_range[idx]),
                    "acquisition",
                ))
                if len(candidates) == 5:
                    break

        # 1 max-uncertainty candidate
        indices_by_unc = np.argsort(-(sigma_range * smooth_weights))
        for idx in indices_by_unc:
            key = tuple(np.round(population[idx], 5))
            if key not in seen:
                seen.add(key)
                candidates.append(_vec_to_candidate(
                    population[idx],
                    float(acquisitions[idx]),
                    float(sigma_range[idx]),
                    float(mu_range[idx]),
                    "uncertainty",
                ))
                break

        return candidates
