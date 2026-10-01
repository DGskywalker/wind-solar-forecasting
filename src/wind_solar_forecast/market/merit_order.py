"""Non-linear merit-order stack spline modeling and forecast error propagation (Sensfuß et al. 2008)."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline


@dataclass(frozen=True)
class MeritOrderSensitivity:
    mean_marginal_slope_eur_gwh: float
    slope_low_residual: float
    slope_mid_residual: float
    slope_high_residual: float
    implied_price_error_rmse: float


class MeritOrderModel:
    """Fits non-linear empirical merit-order price curve P = f(Residual Load) using smoothing splines."""

    def __init__(self, smoothing_s: float | None = None) -> None:
        self.spline: UnivariateSpline | None = None
        self.smoothing_s = smoothing_s

    def fit(self, residual_load_mw: np.ndarray, price_eur_mwh: np.ndarray) -> "MeritOrderModel":
        """Fits cubic smoothing spline P = f(Residual_Load)."""
        idx_sort = np.argsort(residual_load_mw)
        x_sorted = residual_load_mw[idx_sort]
        y_sorted = price_eur_mwh[idx_sort]

        # Deduplicate strictly increasing x for UnivariateSpline
        x_unique, unique_indices = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[unique_indices]

        # Normalize to GW
        x_gw = x_unique / 1000.0
        self.spline = UnivariateSpline(x_gw, y_unique, s=self.smoothing_s or len(x_gw) * 25.0)
        return self

    def predict_price(self, residual_load_mw: np.ndarray) -> np.ndarray:
        """Evaluates merit-order clearing price at given residual load."""
        if self.spline is None:
            raise RuntimeError("MeritOrderModel must be fitted before predict_price.")
        x_gw = np.asarray(residual_load_mw) / 1000.0
        return self.spline(x_gw)

    def marginal_slope(self, residual_load_mw: np.ndarray) -> np.ndarray:
        """Evaluates derivative f'(R) = dP/dR in EUR/(MWh * GW)."""
        if self.spline is None:
            raise RuntimeError("MeritOrderModel must be fitted.")
        x_gw = np.asarray(residual_load_mw) / 1000.0
        # Derivative of spline
        deriv = self.spline.derivative(1)
        # Merit order slope must be positive (higher residual load -> higher price)
        return np.maximum(deriv(x_gw), 0.0)

    def propagate_forecast_error(
        self,
        base_residual_load_mw: np.ndarray,
        generation_surprise_mw: np.ndarray,
        actual_price_delta_eur: np.ndarray
    ) -> MeritOrderSensitivity:
        """Propagates renewable generation forecast error through merit-order curve to obtain implied price impact.

        Delta_P_implied = - f'(Residual_Load) * (Surprise_MW / 1000.0)
        """
        slope_eur_gwh = self.marginal_slope(base_residual_load_mw)
        surprise_gw = np.asarray(generation_surprise_mw) / 1000.0
        implied_delta_price = - slope_eur_gwh * surprise_gw

        rmse_implied = float(np.sqrt(np.mean((actual_price_delta_eur - implied_delta_price) ** 2)))

        # Evaluate regime slopes
        q25, q50, q75 = np.quantile(base_residual_load_mw, [0.25, 0.50, 0.75])
        slope_low = float(np.mean(self.marginal_slope(np.array([q25]))))
        slope_mid = float(np.mean(self.marginal_slope(np.array([q50]))))
        slope_high = float(np.mean(self.marginal_slope(np.array([q75]))))

        return MeritOrderSensitivity(
            mean_marginal_slope_eur_gwh=float(np.mean(slope_eur_gwh)),
            slope_low_residual=slope_low,
            slope_mid_residual=slope_mid,
            slope_high_residual=slope_high,
            implied_price_error_rmse=rmse_implied
        )
