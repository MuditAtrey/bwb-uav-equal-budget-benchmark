"""
flying_wing_optimizer/surrogate.py

Surrogate model module implementing an ensemble of MLP regressors for multi-objective
aerodynamic and mission performance prediction and Upper Confidence Bound (UCB) acquisition.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from flying_wing_optimizer.config import BOUNDS, PATHS

# Target output variable names in required order
TARGET_COLUMNS = ["CL", "CD", "L_D", "range_calm_km", "penalty_score"]
MIN_TRAINING_SAMPLES = 30


class SurrogateEnsemble:
    """
    Ensemble of 5 distinct scikit-learn MLPRegressor models with different random seeds.
    Used for predicting CL, CD, L_D, range_calm_km, and penalty_score, estimating prediction uncertainty,
    and calculating Upper Confidence Bound (UCB) acquisition values.
    Also contains a RandomForestClassifier to predict design feasibility probability.
    """

    def __init__(
        self,
        n_models: int = 5,
        hidden_layer_sizes: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        solver: str = "adam",
        max_iter: int = 1000,
        early_stopping: bool = True,
        min_samples: int = MIN_TRAINING_SAMPLES,
        random_seeds: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes the Surrogate Ensemble.

        Args:
            n_models: Number of MLPRegressor models in the ensemble (default: 5).
            hidden_layer_sizes: Hidden layer architecture (default: (64, 64)).
            activation: Activation function (default: 'relu').
            solver: Solver algorithm (default: 'adam').
            max_iter: Maximum training iterations (default: 1000).
            early_stopping: Enable early stopping during training (default: True).
            min_samples: Minimum required valid evaluation samples to train (default: 10).
            random_seeds: List of random seeds for the ensemble models.
        """
        self.n_models = n_models
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.min_samples = min_samples

        if random_seeds is None:
            self.random_seeds = [42, 101, 2024, 777, 999][: self.n_models]
            if len(self.random_seeds) < self.n_models:
                self.random_seeds = [42 + i * 100 for i in range(self.n_models)]
        else:
            self.random_seeds = random_seeds[: self.n_models]

        self.scaler = MinMaxScaler()
        self.target_scaler = StandardScaler()
        self.feasibility_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models: List[MLPRegressor] = []
        self.is_fitted = False
        self.is_feasibility_fitted = False
        self.target_names: List[str] = list(TARGET_COLUMNS)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, List[List[float]], Dict[str, Any]],
        Y: Union[np.ndarray, pd.DataFrame, List[List[float]], Dict[str, Any]],
    ) -> bool:
        """
        Fits feature scaler on 18-DoF input vectors and trains all ensemble MLPRegressor models
        on target variables (CL, CD, L_D, range_calm_km, penalty_score).

        Args:
            X: Input feature dataset of shape (N, 16).
            Y: Target dataset of shape (N, 5) or DataFrame/dict with target columns.

        Returns:
            True if training succeeded, False if sample size is below min_samples threshold.
        """
        # Convert X to 2D numpy array
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        elif isinstance(X, dict):
            X_arr = np.column_stack([np.asarray(X[col], dtype=float) for col in BOUNDS.__dict__.keys()])
        else:
            X_arr = np.asarray(X, dtype=float)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Check sample size requirement
        if X_arr.shape[0] < self.min_samples:
            self.is_fitted = False
            return False

        # Convert Y to 2D numpy array with columns corresponding to TARGET_COLUMNS
        if isinstance(Y, pd.DataFrame):
            Y_arr = Y[self.target_names].to_numpy(dtype=float)
        elif isinstance(Y, dict):
            Y_arr = np.column_stack([np.asarray(Y[col], dtype=float) for col in self.target_names])
        else:
            Y_arr = np.asarray(Y, dtype=float)

        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(-1, 1)

        # Fit feature and target scalers
        X_scaled = self.scaler.fit_transform(X_arr)
        Y_scaled = self.target_scaler.fit_transform(Y_arr)

        # Train ensemble models
        self.models = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for seed in self.random_seeds:
                # Disable early stopping dynamically if there are fewer than 20 valid samples
                use_early_stopping = self.early_stopping if X_scaled.shape[0] >= 20 else False
                
                model = MLPRegressor(
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation=self.activation,
                    solver=self.solver,
                    max_iter=self.max_iter,
                    early_stopping=use_early_stopping,
                    random_state=seed,
                )
                model.fit(X_scaled, Y_scaled)
                self.models.append(model)

        self.is_fitted = True
        return True

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame, List[float], List[List[float]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean μ and standard deviation σ for all outputs across the ensemble.

        Args:
            X: Input 18-DoF parameter vector(s) of shape (N, 18) or (18,).

        Returns:
            Tuple of (mu, sigma):
                mu: Array of mean predictions of shape (N, 5) for (CL, CD, L_D, range_calm_km, penalty_score).
                sigma: Array of standard deviations of shape (N, 5) across ensemble models.
                (Returns zero vectors/uncertainty if model is not yet fit or sample size insufficient).
        """
        # Convert X to 2D numpy array
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        is_single_sample = False
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
            is_single_sample = True

        n_samples = X_arr.shape[0]
        n_targets = len(self.target_names)

        # Handle edge cases: model not fit or sample size insufficient
        if not self.is_fitted or len(self.models) == 0:
            mu = np.zeros((n_samples, n_targets), dtype=float)
            sigma = np.zeros((n_samples, n_targets), dtype=float)
            if is_single_sample:
                return mu[0], sigma[0]
            return mu, sigma

        # Transform inputs using fitted scaler
        X_scaled = self.scaler.transform(X_arr)

        # Predict across all ensemble models: shape (n_models, n_samples, n_targets)
        raw_preds = np.array([model.predict(X_scaled) for model in self.models])
        
        # Inverse transform predictions back to target scale
        predictions = np.array([self.target_scaler.inverse_transform(p) for p in raw_preds])

        # Compute mean μ and standard deviation σ across models (axis=0)
        mu = np.mean(predictions, axis=0)
        sigma = np.std(predictions, axis=0)

        if is_single_sample:
            return mu[0], sigma[0]
        return mu, sigma

    def calculate_acquisition(
        self,
        X: Union[np.ndarray, pd.DataFrame, List[float], List[List[float]]],
        beta: float = 1.5,
    ) -> Union[float, np.ndarray]:
        """
        Calculates Upper Confidence Bound (UCB) acquisition values for given candidate designs.
        acquisition = predicted_range_mean + beta * std_range

        Args:
            X: Input candidate 18-DoF vector(s) of shape (N, 18) or (18,).
            beta: Exploration trade-off parameter (default: 1.5).

        Returns:
            Acquisition values array of shape (N,) or float if single candidate given.
            (Returns zeros if model is not fit).
        """
        # Convert X to 2D array format for consistent processing
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        is_single_sample = (X_arr.ndim == 1)

        if not self.is_fitted:
            if is_single_sample:
                return 0.0
            return np.zeros(X_arr.shape[0], dtype=float)

        mu, sigma = self.predict(X_arr)

        if is_single_sample:
            range_mean = mu[3]
            range_std = sigma[3]
            return float(range_mean + beta * range_std)
        else:
            range_mean = mu[:, 3]
            range_std = sigma[:, 3]
            return range_mean + beta * range_std

    def fit_feasibility(
        self,
        X_all: Union[np.ndarray, pd.DataFrame, List[List[float]]],
        y_valid: Union[np.ndarray, pd.Series, List[bool]],
    ) -> bool:
        """
        Fits the feasibility RandomForestClassifier on all evaluation rows (valid and invalid).

        Args:
            X_all: Feature dataset of shape (N, 18) containing all evaluated 18-DoF parameters.
            y_valid: Target vector of shape (N,) containing boolean is_valid flags.

        Returns:
            True if classifier trained successfully, False otherwise.
        """
        if isinstance(X_all, pd.DataFrame):
            X_arr = X_all.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X_all, dtype=float)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        y_arr = np.asarray(y_valid, dtype=bool)

        if len(X_arr) == 0 or len(np.unique(y_arr)) < 2:
            self.is_feasibility_fitted = False
            return False

        X_scaled = self.scaler.transform(X_arr)
        self.feasibility_clf.fit(X_scaled, y_arr)
        self.is_feasibility_fitted = True
        return True

    def predict_feasibility_proba(
        self, X: Union[np.ndarray, pd.DataFrame, List[float], List[List[float]]]
    ) -> np.ndarray:
        """
        Predicts the probability of design feasibility (is_valid == True).

        Args:
            X: Input candidate 18-DoF vector(s) of shape (N, 18) or (18,).

        Returns:
            1D array of feasibility probabilities of shape (N,).
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        is_single_sample = (X_arr.ndim == 1)
        if is_single_sample:
            X_arr = X_arr.reshape(1, -1)

        if not getattr(self, "is_feasibility_fitted", False):
            probs = np.ones(X_arr.shape[0], dtype=float)
        else:
            X_scaled = self.scaler.transform(X_arr)
            probs = self.feasibility_clf.predict_proba(X_scaled)[:, 1]

        return probs
