"""Evaluation metrics for deterministic, quantile, and distributional probabilistic forecasts."""

import numpy as np
import pandas as pd


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: (1/N) * sum |y - y_hat|."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: sqrt( (1/N) * sum (y - y_hat)^2 )."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean pinball (asymmetric Laplace) loss for quantile alpha.

    rho_alpha(u) = u * (alpha - I(u < 0))
    """
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def compute_crps(predicted_quantiles: np.ndarray, quantiles: list[float], y_true: np.ndarray) -> float:
    """Quantile-based approximation of Continuous Ranked Probability Score (Gneiting & Raftery 2007).

    CRPS = (2 / K) * sum_{k=1}^K rho_{alpha_k}(y - q_hat_k)
    """
    k = len(quantiles)
    loss = 0.0
    for idx, q in enumerate(quantiles):
        loss += compute_pinball_loss(y_true, predicted_quantiles[:, idx], q)
    return float((2.0 / k) * loss)


def compute_winkler_score(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray, alpha: float = 0.10) -> float:
    """Winkler Interval Score for (1 - alpha) prediction interval [L, U] (Winkler 1972).

    W_alpha = (U - L) + (2 / alpha) * (L - y) * I(y < L) + (2 / alpha) * (y - U) * I(y > U)
    Penalizes interval width while strictly punishing non-coverage.
    """
    width = upper - lower
    under = np.maximum(lower - y_true, 0.0)
    over = np.maximum(y_true - upper, 0.0)
    score = width + (2.0 / alpha) * under + (2.0 / alpha) * over
    return float(np.mean(score))


def compute_calibration_ece(predicted_quantiles: np.ndarray, quantiles: list[float], y_true: np.ndarray) -> float:
    """Expected Calibration Error (ECE) across all nominal quantile levels:

    ECE = (1 / K) * sum_k | nominal_k - empirical_k |
    """
    k = len(quantiles)
    n = len(y_true)
    ece_sum = 0.0
    for idx, q in enumerate(quantiles):
        emp_rate = np.sum(y_true <= predicted_quantiles[:, idx]) / n
        ece_sum += abs(emp_rate - q)
    return float(ece_sum / k)
