"""LightGBM Quantile and Point Regressors, plus NGBoost Natural Gradient Boosting."""

from typing import Any
import lightgbm as lgb
from ngboost import NGBRegressor
from ngboost.distns import Normal
import numpy as np
import pandas as pd


class LGBMQuantileForecaster:
    """LightGBM model ladder with dedicated quantile regressors and quantile rearrangement."""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ) -> None:
        self.quantiles = quantiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.point_model = lgb.LGBMRegressor(objective="regression", **self.params)
        self.quantile_models: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "LGBMQuantileForecaster":
        """Fits L2 point model and individual pinball quantile regressors."""
        X_df = pd.DataFrame(X)
        y_arr = np.asarray(y)

        # 1. Fit Point Model
        self.point_model.fit(X_df, y_arr)

        # 2. Fit Quantile Models
        for q in self.quantiles:
            q_model = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params)
            q_model.fit(X_df, y_arr)
            self.quantile_models[q] = q_model

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates point predictions."""
        preds = self.point_model.predict(pd.DataFrame(X))
        return np.clip(preds, 0.0, 1.0)

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts multi-quantile matrix with Chernozhukov (2010) monotonic rearrangement."""
        X_df = pd.DataFrame(X)
        q_preds = []
        for q in self.quantiles:
            pred_q = self.quantile_models[q].predict(X_df)
            q_preds.append(np.clip(pred_q, 0.0, 1.0))

        mat = np.column_stack(q_preds)
        # Rearrangement to strictly prevent quantile crossing
        mat = np.sort(mat, axis=1)
        return mat


class NGBoostForecaster:
    """NGBoost probabilistic forecaster using Natural Gradient Boosting for predictive distributions."""

    def __init__(
        self,
        quantiles: list[float] | None = None,
        n_estimators: int = 150,
        learning_rate: float = 0.03,
        random_state: int = 42
    ) -> None:
        self.quantiles = quantiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.model = NGBRegressor(
            Dist=Normal,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=False
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "NGBoostForecaster":
        self.model.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(np.asarray(X))
        return np.clip(preds, 0.0, 1.0)

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Evaluates normal distribution quantiles from predicted (mu, sigma) parameters."""
        dist = self.model.pred_dist(np.asarray(X))
        q_preds = []
        for q in self.quantiles:
            q_val = dist.ppf(q)
            q_preds.append(np.clip(q_val, 0.0, 1.0))

        mat = np.column_stack(q_preds)
        mat = np.sort(mat, axis=1)
        return mat
