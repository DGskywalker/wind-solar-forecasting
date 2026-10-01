"""Linear benchmarks: Ridge regression and Linear Quantile Regression."""

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, Ridge


class LinearBenchmark:
    """Linear regression benchmark suite providing both point (Ridge) and quantile predictions."""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        ridge_alpha: float = 1.0,
        quantile_alpha: float = 0.01
    ) -> None:
        self.quantiles = quantiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.ridge_alpha = ridge_alpha
        self.quantile_alpha = quantile_alpha
        self.point_model = Ridge(alpha=ridge_alpha)
        self.quantile_models: dict[float, QuantileRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "LinearBenchmark":
        """Fits Ridge point model and pinball quantile models across all quantiles."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # Fit point Ridge
        self.point_model.fit(X_arr, y_arr)

        # Fit each quantile regressor
        for q in self.quantiles:
            qr = QuantileRegressor(quantile=q, alpha=self.quantile_alpha, solver="highs")
            qr.fit(X_arr, y_arr)
            self.quantile_models[q] = qr

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point forecast via Ridge."""
        X_arr = np.asarray(X)
        preds = self.point_model.predict(X_arr)
        return np.clip(preds, 0.0, 1.0)

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Multi-quantile probabilistic forecast with non-crossing monotonic sorting.

        Returns:
            Array of shape (n_samples, n_quantiles).
        """
        X_arr = np.asarray(X)
        q_preds = []
        for q in self.quantiles:
            pred_q = self.quantile_models[q].predict(X_arr)
            q_preds.append(np.clip(pred_q, 0.0, 1.0))

        # Matrix: (n_samples, n_quantiles)
        mat = np.column_stack(q_preds)
        # Enforce monotonicity: q_i <= q_{i+1}
        mat = np.sort(mat, axis=1)
        return mat
