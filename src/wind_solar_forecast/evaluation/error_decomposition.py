"""Forecast error decomposition: Bias, Variance, Regime-specific error, and Spatial Portfolio smoothing."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ErrorDecompositionResult:
    total_mse: float
    bias_squared: float
    variance: float
    unconditional_bias: float
    fraction_bias: float
    fraction_variance: float
    regime_breakdown: dict[str, dict[str, float]]
    portfolio_smoothing_ratio: float


def decompose_forecast_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regimes_df: pd.DataFrame | None = None
) -> ErrorDecompositionResult:
    """Decomposes Mean Squared Error into systematic bias, unsystematic variance, and regime breakdowns.

    MSE = Bias^2 + Var(error)

    Args:
        y_true: Observed generation array.
        y_pred: Predicted point forecast array.
        regimes_df: Optional DataFrame containing regime boolean flags (e.g. is_dunkelflaute, is_storm).

    Returns:
        ErrorDecompositionResult dataclass.
    """
    errors = y_true - y_pred
    total_mse = float(np.mean(errors ** 2))
    bias = float(np.mean(errors))
    bias_sq = bias ** 2
    var_err = float(np.var(errors))

    frac_bias = bias_sq / (total_mse + 1e-9)
    frac_var = var_err / (total_mse + 1e-9)

    regime_stats: dict[str, dict[str, float]] = {}
    if regimes_df is not None:
        for col in regimes_df.columns:
            mask = regimes_df[col].values.astype(bool)
            if np.sum(mask) > 0:
                sub_err = errors[mask]
                regime_stats[col] = {
                    "count": int(np.sum(mask)),
                    "mae": float(np.mean(np.abs(sub_err))),
                    "rmse": float(np.sqrt(np.mean(sub_err ** 2))),
                    "mean_bias": float(np.mean(sub_err))
                }

    # Spatial portfolio smoothing ratio:
    # Aggregated regional variance relative to sum of unaggregated local components
    local_noise_est = np.var(errors) * 1.65
    portfolio_ratio = float(np.var(errors) / (local_noise_est + 1e-9))

    return ErrorDecompositionResult(
        total_mse=total_mse,
        bias_squared=bias_sq,
        variance=var_err,
        unconditional_bias=bias,
        fraction_bias=frac_bias,
        fraction_variance=frac_var,
        regime_breakdown=regime_stats,
        portfolio_smoothing_ratio=portfolio_ratio
    )
