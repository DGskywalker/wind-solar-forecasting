"""Benchmark baseline models: Persistence, Climatology, and Physical NWP-direct."""

import numpy as np
import pandas as pd


class PersistenceForecaster:
    """Persistence baseline: predicts y_t = y_{t - lag_hours}."""

    def __init__(self, lag_hours: int = 24) -> None:
        self.lag_hours = lag_hours

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceForecaster":
        return self

    def predict(self, y_history: pd.Series, horizon_steps: int) -> np.ndarray:
        """Projects historical values forward by the persistence lag."""
        if len(y_history) < self.lag_hours:
            raise ValueError(f"History length {len(y_history)} is less than lag {self.lag_hours}")
        # Repeat the last lag_hours pattern
        vals = y_history.iloc[-self.lag_hours:].values
        reps = int(np.ceil(horizon_steps / self.lag_hours))
        preds = np.tile(vals, reps)[:horizon_steps]
        return np.clip(preds, 0.0, 1.0)


class ClimatologyForecaster:
    """Climatology baseline: predicts empirical quantile distributions grouped by (month, hour)."""

    def __init__(self, quantiles: list[float] | None = None) -> None:
        self.quantiles = quantiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.empirical_quantiles: dict[tuple[int, int], np.ndarray] = {}
        self.global_quantiles: np.ndarray = np.array([])

    def fit(self, timestamps: pd.DatetimeIndex, y: np.ndarray) -> "ClimatologyForecaster":
        """Computes empirical quantiles for each (month, hour) pair in historical data."""
        df = pd.DataFrame({
            "month": timestamps.month,
            "hour": timestamps.hour,
            "target": y
        })
        self.global_quantiles = np.quantile(y, self.quantiles)

        grouped = df.groupby(["month", "hour"])["target"]
        for (m, h), group in grouped:
            self.empirical_quantiles[(m, h)] = np.quantile(group.values, self.quantiles)

        return self

    def predict_quantiles(self, target_timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Predicts quantile matrix (n_samples, n_quantiles) based on historical calendar empiricals."""
        n = len(target_timestamps)
        preds = np.zeros((n, len(self.quantiles)))

        for i, dt in enumerate(target_timestamps):
            key = (dt.month, dt.hour)
            if key in self.empirical_quantiles:
                preds[i, :] = self.empirical_quantiles[key]
            else:
                preds[i, :] = self.global_quantiles

        return np.clip(preds, 0.0, 1.0)


class NWPDirectForecaster:
    """Physical baseline: directly outputs the physical capacity factor modeled from NWP."""

    def __init__(self, nwp_cf_column: str = "nwp_wind_onshore_cf") -> None:
        self.nwp_cf_column = nwp_cf_column
        self.bias_correction: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NWPDirectForecaster":
        """Fits an optional linear bias correction offset: bias = mean(actual - nwp)."""
        if self.nwp_cf_column in X.columns:
            self.bias_correction = float(np.mean(y.values - X[self.nwp_cf_column].values))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts capacity factor directly using NWP physical simulation."""
        if self.nwp_cf_column not in X.columns:
            raise KeyError(f"Column '{self.nwp_cf_column}' missing from feature matrix.")
        preds = X[self.nwp_cf_column].values + self.bias_correction
        return np.clip(preds, 0.0, 1.0)
