"""Ensemble methods: CRPS-optimal weighted blend and multi-model stacking."""

from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Computes mean pinball (quantile) loss for quantile level alpha.

    rho_alpha(u) = u * (alpha - I(u < 0))
    """
    diff = y_true - y_pred
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


def quantile_crps(predicted_quantiles: np.ndarray, quantiles: list[float], y_true: np.ndarray) -> float:
    """Computes Continuous Ranked Probability Score (CRPS) approximation from discrete quantiles (Gneiting 2007).

    CRPS = (2 / K) * sum_{k=1}^K pinball_loss(y, q_hat_k, alpha_k)
    """
    k = len(quantiles)
    total_loss = 0.0
    for idx, q in enumerate(quantiles):
        total_loss += pinball_loss(y_true, predicted_quantiles[:, idx], q)
    return float((2.0 / k) * total_loss)


class CRPSOptimalEnsemble:
    """CRPS-optimal ensemble blending across multiple probabilistic models.

    Solves the constrained convex optimization problem:
        min_w CRPS( sum_m w_m * Q_m, y )
        subject to: w_m >= 0, sum_m w_m = 1
    using Sequential Least Squares Programming (SLSQP).
    """

    def __init__(self, models: list[Any], model_names: list[str] | None = None, quantiles: list[float] | None = None) -> None:
        self.models = models
        self.model_names = model_names or [f"model_{i}" for i in range(len(models))]
        self.quantiles = quantiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.weights: np.ndarray = np.ones(len(models)) / len(models)

    def fit_weights(self, val_quantile_preds: list[np.ndarray], y_val: np.ndarray) -> "CRPSOptimalEnsemble":
        """Optimizes blend weights directly on validation out-of-fold quantile predictions.

        Args:
            val_quantile_preds: List of arrays, each of shape (n_val, n_quantiles), one per model.
            y_val: Validation target array.
        """
        n_models = len(self.models)
        val_stack = np.stack(val_quantile_preds, axis=0)  # (n_models, n_val, n_quantiles)

        def loss_fn(w: np.ndarray) -> float:
            w_norm = w / (np.sum(w) + 1e-9)
            blended = np.tensordot(w_norm, val_stack, axes=(0, 0))
            return quantile_crps(blended, self.quantiles, y_val)

        init_w = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(loss_fn, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            self.weights = res.x / np.sum(res.x)
        else:
            self.weights = init_w

        return self

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Computes weighted blend of predicted quantiles and enforces monotonicity."""
        preds = []
        for model in self.models:
            preds.append(model.predict_quantiles(X))

        pred_stack = np.stack(preds, axis=0)
        blended = np.tensordot(self.weights, pred_stack, axes=(0, 0))
        # Ensure strict monotonicity
        return np.sort(blended, axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Median point prediction from blended distribution."""
        q_preds = self.predict_quantiles(X)
        mid_idx = len(self.quantiles) // 2
        return q_preds[:, mid_idx]
