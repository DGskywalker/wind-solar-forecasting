"""Probabilistic calibration diagnostics: Reliability diagrams, PIT histograms, and Expected Calibration Error."""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def compute_pit_values(predicted_quantiles: np.ndarray, nominal_quantiles: list[float], y_true: np.ndarray) -> np.ndarray:
    """Computes Probability Integral Transform (PIT) values by interpolating predicted CDF at y_true.

    Args:
        predicted_quantiles: Array of shape (n_samples, n_quantiles).
        nominal_quantiles: List of nominal quantile levels (e.g. [0.05, 0.1, ..., 0.95]).
        y_true: True realization array.

    Returns:
        Array of PIT values in [0.0, 1.0]. Under perfect calibration, PIT ~ Uniform(0, 1).
    """
    n = len(y_true)
    q_arr = np.asarray(nominal_quantiles)
    pit_vals = np.zeros(n)

    for i in range(n):
        pred_q = predicted_quantiles[i, :]
        val = y_true[i]
        # Linear interpolation across empirical CDF
        if val <= pred_q[0]:
            pit_vals[i] = (q_arr[0] / max(pred_q[0], 1e-4)) * val
        elif val >= pred_q[-1]:
            pit_vals[i] = q_arr[-1] + (1.0 - q_arr[-1]) * min((val - pred_q[-1]) / max(1.0 - pred_q[-1], 1e-4), 1.0)
        else:
            pit_vals[i] = np.interp(val, pred_q, q_arr)

    return np.clip(pit_vals, 0.0, 1.0)


def compute_reliability_table(
    predicted_quantiles: np.ndarray,
    nominal_quantiles: list[float],
    y_true: np.ndarray
) -> pd.DataFrame:
    """Computes empirical hit rates vs nominal quantile levels and empirical coverage error.

    Expected Calibration Error (ECE):
        ECE = (1/K) * sum_k | nominal_k - empirical_k |
    """
    records = []
    n = len(y_true)

    for idx, q_nom in enumerate(nominal_quantiles):
        pred_k = predicted_quantiles[:, idx]
        hits = np.sum(y_true <= pred_k)
        emp_rate = hits / n
        records.append({
            "nominal_quantile": q_nom,
            "empirical_coverage": np.round(emp_rate, 4),
            "calibration_error": np.round(emp_rate - q_nom, 4),
            "abs_error": np.round(abs(emp_rate - q_nom), 4)
        })

    df = pd.DataFrame(records)
    ece = float(df["abs_error"].mean())
    df["ece_mean"] = ece
    return df


class IsotonicCalibrator:
    """Isotonic regression recalibrator for non-parametric monotonic quantile adjustment."""

    def __init__(self) -> None:
        self.calibrators: list[IsotonicRegression] = []

    def fit(self, raw_quantiles: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        n_q = raw_quantiles.shape[1]
        self.calibrators = []
        for i in range(n_q):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw_quantiles[:, i], y_true)
            self.calibrators.append(iso)
        return self

    def transform(self, raw_quantiles: np.ndarray) -> np.ndarray:
        calib_preds = []
        for i, iso in enumerate(self.calibrators):
            calib_preds.append(iso.predict(raw_quantiles[:, i]))
        mat = np.column_stack(calib_preds)
        return np.sort(mat, axis=1)
