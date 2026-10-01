"""Double Machine Learning (DML) for causal price impact estimation (Chernozhukov et al. 2018)."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class DMLCausalResult:
    theta_effect: float
    std_error: float
    t_statistic: float
    p_value: float
    ci_95: tuple[float, float]
    n_obs: int
    n_folds: int
    controls: list[str]


def estimate_double_machine_learning(
    df: pd.DataFrame,
    outcome_col: str = "delta_price_eur_mwh",
    treatment_col: str = "surprise_mw",
    control_cols: list[str] | None = None,
    n_folds: int = 5,
    random_state: int = 42
) -> DMLCausalResult:
    """Estimates the causal treatment effect theta using Robinson (1988) / Chernozhukov (2018) DML.

    Partially Linear Model:
        Y = theta * D + g(X) + U
        D = m(X) + V

    Uses K-Fold cross-fitting with LightGBM regressors to partial out non-linear confounders X
    (load, gas prices, carbon allowances, cross-border flows).

    Args:
        df: Input DataFrame.
        outcome_col: Outcome Y (e.g. price change).
        treatment_col: Treatment D (generation surprise in MW or GW).
        control_cols: List of confounders X.
        n_folds: Number of cross-fitting folds.
        random_state: Random state for K-Fold splitting.

    Returns:
        DMLCausalResult with orthogonalized causal coefficient theta, SE, and 95% CI.
    """
    if control_cols is None:
        control_cols = ["total_load_mw", "gas_ttf_eur_mwh", "eua_carbon_eur_ton", "temperature_2m_c"]

    clean_df = df[[outcome_col, treatment_col] + control_cols].dropna().reset_index(drop=True)

    Y = clean_df[outcome_col].values
    # Rescale treatment D to Gigawatts for numerical conditioning
    D = clean_df[treatment_col].values / 1000.0
    X = clean_df[control_cols].values

    n = len(Y)
    y_res = np.zeros(n)
    d_res = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D[train_idx], D[test_idx]

        # ML model for E[Y|X]
        model_y = lgb.LGBMRegressor(n_estimators=40, max_depth=4, learning_rate=0.05, verbose=-1, random_state=random_state)
        model_y.fit(X_tr, Y_tr)
        y_hat = model_y.predict(X_te)
        y_res[test_idx] = Y_te - y_hat

        # ML model for E[D|X]
        model_d = lgb.LGBMRegressor(n_estimators=40, max_depth=4, learning_rate=0.05, verbose=-1, random_state=random_state)
        model_d.fit(X_tr, D_tr)
        d_hat = model_d.predict(X_te)
        d_res[test_idx] = D_te - d_hat

    # Orthogonalized IV estimate: theta = sum(d_res * y_res) / sum(d_res^2)
    denom = np.sum(d_res ** 2)
    if denom < 1e-9:
        theta = 0.0
        se = 1.0
    else:
        theta = float(np.sum(d_res * y_res) / denom)
        # Asymptotic variance of Neyman-orthogonal score
        residuals = y_res - theta * d_res
        score_var = float(np.mean((d_res * residuals) ** 2))
        var_theta = float(score_var / (((float(np.mean(d_res ** 2))) ** 2) * n))
        se = float(np.sqrt(max(var_theta, 1e-9)))

    t_stat = theta / (se + 1e-9)
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
    ci_95 = (theta - 1.96 * se, theta + 1.96 * se)

    return DMLCausalResult(
        theta_effect=theta,
        std_error=se,
        t_statistic=t_stat,
        p_value=p_val,
        ci_95=ci_95,
        n_obs=n,
        n_folds=n_folds,
        controls=control_cols
    )
