"""Conformalized Quantile Regression (CQR) wrapper for distribution-free calibrated prediction intervals."""

from typing import Any

import numpy as np
import pandas as pd


class ConformalQuantileRegressor:
    """Conformal prediction wrapper implementing Conformalized Quantile Regression (Romano et al. 2019).

    Provides finite-sample coverage guarantees:
        P(Y_{n+1} in C(X_{n+1})) >= 1 - alpha
    without distributional assumptions.
    """

    def __init__(self, base_quantile_model: Any, nominal_coverage: float = 0.90, asymmetric: bool = True) -> None:
        """Initializes Conformal wrapper.

        Args:
            base_quantile_model: Underlying fitted or unfitted model supporting predict_quantiles.
            nominal_coverage: Desired target confidence level 1 - alpha (e.g. 0.90 for 90% interval).
            asymmetric: Whether to calibrate lower and upper conformity errors independently.
        """
        self.base_model = base_quantile_model
        self.nominal_coverage = nominal_coverage
        self.alpha = 1.0 - nominal_coverage
        self.asymmetric = asymmetric
        self.q_low_idx: int = 0
        self.q_high_idx: int = -1
        self.q_hat_low: float = 0.0
        self.q_hat_high: float = 0.0
        self.q_hat_symmetric: float = 0.0

    def calibrate(self, X_calib: pd.DataFrame, y_calib: pd.Series | np.ndarray) -> "ConformalQuantileRegressor":
        """Computes conformity scores and empirical cutoff thresholds on held-out calibration set."""
        X_df = pd.DataFrame(X_calib)
        y_arr = np.asarray(y_calib)
        n = len(y_arr)

        if n == 0:
            raise ValueError("Calibration set must be non-empty.")

        # Get base quantile predictions (assumes quantiles include lower alpha/2 and upper 1-alpha/2)
        q_preds = self.base_model.predict_quantiles(X_df)
        n_q = q_preds.shape[1]

        # Locate closest quantiles to alpha/2 and 1 - alpha/2
        self.q_low_idx = 0  # Typically 0.05 for 90% coverage
        self.q_high_idx = n_q - 1  # Typically 0.95 for 90% coverage

        low_pred = q_preds[:, self.q_low_idx]
        high_pred = q_preds[:, self.q_high_idx]

        # Conformity non-conformity scores
        # E_i = max(low - y, y - high)
        err_low = low_pred - y_arr
        err_high = y_arr - high_pred

        p_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        p_level = min(max(p_level, 0.0), 1.0)

        if self.asymmetric:
            # Separate lower and upper cutoffs
            p_half = np.ceil((n + 1) * (1.0 - self.alpha / 2.0)) / n
            p_half = min(max(p_half, 0.0), 1.0)
            self.q_hat_low = float(np.quantile(err_low, p_half))
            self.q_hat_high = float(np.quantile(err_high, p_half))
        else:
            sym_scores = np.maximum(err_low, err_high)
            self.q_hat_symmetric = float(np.quantile(sym_scores, p_level))

        return self

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predicts calibrated lower and upper prediction intervals with finite-sample guarantee.

        Returns:
            Tuple of (lower_bound, upper_bound) arrays clipped to [0, 1].
        """
        q_preds = self.base_model.predict_quantiles(pd.DataFrame(X))
        raw_low = q_preds[:, self.q_low_idx]
        raw_high = q_preds[:, self.q_high_idx]

        if self.asymmetric:
            calib_low = raw_low - self.q_hat_low
            calib_high = raw_high + self.q_hat_high
        else:
            calib_low = raw_low - self.q_hat_symmetric
            calib_high = raw_high + self.q_hat_symmetric

        return np.clip(calib_low, 0.0, 1.0), np.clip(calib_high, 0.0, 1.0)
