"""Unit tests for probabilistic model ladder, baseline forecasters, conformal prediction, and ensembling."""

import numpy as np
import pandas as pd
import pytest

from wind_solar_forecast.models.baselines import (
    ClimatologyForecaster,
    NWPDirectForecaster,
    PersistenceForecaster,
)
from wind_solar_forecast.models.conformal import ConformalQuantileRegressor
from wind_solar_forecast.models.deep import DeepProbabilisticForecaster
from wind_solar_forecast.models.ensemble import CRPSOptimalEnsemble
from wind_solar_forecast.models.gbdt import LGBMQuantileForecaster, NGBoostForecaster
from wind_solar_forecast.models.linear import LinearBenchmark


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generates synthetic dataset for model unit testing."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    ws = np.random.uniform(2.0, 15.0, n)
    cf = np.clip((ws - 3.0) / 9.0, 0.0, 1.0) ** 2.0 + np.random.normal(0, 0.05, n)
    cf = np.clip(cf, 0.0, 1.0)

    X = pd.DataFrame({
        "nwp_wind_onshore_cf": np.clip((ws - 3.0) / 9.0, 0.0, 1.0) ** 2.0,
        "forecast_wind_speed_100m": ws,
        "hour_sin": np.sin(2 * np.pi * dates.hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * dates.hour / 24.0),
    }, index=dates)

    y = pd.Series(cf, index=dates)
    return X, y


def test_baseline_models(synthetic_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Tests Persistence, Climatology, and NWPDirect baselines."""
    X, y = synthetic_data
    # 1. Persistence
    pers = PersistenceForecaster(lag_hours=24)
    pred_pers = pers.predict(y.iloc[:48], horizon_steps=24)
    assert len(pred_pers) == 24
    assert (pred_pers >= 0.0).all() and (pred_pers <= 1.0).all()

    # 2. Climatology
    clim = ClimatologyForecaster(quantiles=[0.1, 0.5, 0.9])
    clim.fit(X.index[:80], y.iloc[:80].values)
    pred_clim = clim.predict_quantiles(X.index[80:])
    assert pred_clim.shape == (40, 3)
    assert (pred_clim[:, 0] <= pred_clim[:, 1]).all()
    assert (pred_clim[:, 1] <= pred_clim[:, 2]).all()

    # 3. NWP Direct
    nwp_d = NWPDirectForecaster(nwp_cf_column="nwp_wind_onshore_cf")
    nwp_d.fit(X.iloc[:80], y.iloc[:80])
    pred_nwp = nwp_d.predict(X.iloc[80:])
    assert len(pred_nwp) == 40
    assert (pred_nwp >= 0.0).all() and (pred_nwp <= 1.0).all()


def test_linear_benchmark(synthetic_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Tests LinearBenchmark point and non-crossing quantile predictions."""
    X, y = synthetic_data
    linear = LinearBenchmark(quantiles=[0.1, 0.5, 0.9], ridge_alpha=1.0)
    linear.fit(X.iloc[:80], y.iloc[:80])

    p_pred = linear.predict(X.iloc[80:])
    q_pred = linear.predict_quantiles(X.iloc[80:])

    assert len(p_pred) == 40
    assert q_pred.shape == (40, 3)
    # Check monotonicity
    assert (q_pred[:, 0] <= q_pred[:, 1]).all()
    assert (q_pred[:, 1] <= q_pred[:, 2]).all()


def test_lgbm_quantile_forecaster(synthetic_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Tests LightGBM point and quantile regression."""
    X, y = synthetic_data
    lgbm = LGBMQuantileForecaster(quantiles=[0.1, 0.5, 0.9], n_estimators=20)
    lgbm.fit(X.iloc[:80], y.iloc[:80])

    p_pred = lgbm.predict(X.iloc[80:])
    q_pred = lgbm.predict_quantiles(X.iloc[80:])

    assert len(p_pred) == 40
    assert q_pred.shape == (40, 3)
    assert (q_pred[:, 0] <= q_pred[:, 1]).all()
    assert (q_pred[:, 1] <= q_pred[:, 2]).all()


def test_ngboost_forecaster(synthetic_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Tests NGBoost probabilistic forecasting."""
    X, y = synthetic_data
    ngb = NGBoostForecaster(quantiles=[0.1, 0.5, 0.9], n_estimators=15)
    ngb.fit(X.iloc[:80], y.iloc[:80])

    q_pred = ngb.predict_quantiles(X.iloc[80:])
    assert q_pred.shape == (40, 3)
    assert (q_pred[:, 0] <= q_pred[:, 1]).all()


def test_deep_probabilistic_forecaster(synthetic_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Tests Deep Temporal Fusion Net with seed averaging."""
    X, y = synthetic_data
    deep = DeepProbabilisticForecaster(
        quantiles=[0.1, 0.5, 0.9],
        hidden_dim=16,
        max_epochs=3,
        n_seeds=2
    )
    deep.fit(X.iloc[:80], y.iloc[:80])

    q_pred = deep.predict_quantiles(X.iloc[80:])
    assert q_pred.shape == (40, 3)
    assert (q_pred[:, 0] <= q_pred[:, 1]).all()


def test_conformal_quantile_regressor(synthetic_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Tests Conformalized Quantile Regression calibration and interval validity."""
    X, y = synthetic_data
    lgbm = LGBMQuantileForecaster(quantiles=[0.05, 0.50, 0.95], n_estimators=20)
    lgbm.fit(X.iloc[:60], y.iloc[:60])

    cqr = ConformalQuantileRegressor(base_quantile_model=lgbm, nominal_coverage=0.90)
    cqr.calibrate(X.iloc[60:90], y.iloc[60:90])

    low, high = cqr.predict_interval(X.iloc[90:])
    assert len(low) == 30
    assert len(high) == 30
    assert (low <= high).all()
    # Check empirical coverage on test set
    y_test = y.iloc[90:].values
    coverage = np.mean((y_test >= low) & (y_test <= high))
    # Empirical coverage should be reasonable (> 0.75 for this small test set)
    assert coverage >= 0.75


def test_crps_optimal_ensemble(synthetic_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Tests CRPS-optimal SLSQP blending across linear and tree models."""
    X, y = synthetic_data
    m1 = LinearBenchmark(quantiles=[0.1, 0.5, 0.9])
    m1.fit(X.iloc[:60], y.iloc[:60])

    m2 = LGBMQuantileForecaster(quantiles=[0.1, 0.5, 0.9], n_estimators=20)
    m2.fit(X.iloc[:60], y.iloc[:60])

    val_preds = [m1.predict_quantiles(X.iloc[60:90]), m2.predict_quantiles(X.iloc[60:90])]
    ensemble = CRPSOptimalEnsemble(models=[m1, m2], quantiles=[0.1, 0.5, 0.9])
    ensemble.fit_weights(val_preds, y.iloc[60:90].values)

    assert len(ensemble.weights) == 2
    assert pytest.approx(ensemble.weights.sum(), abs=1e-4) == 1.0
    assert (ensemble.weights >= 0.0).all()

    ens_preds = ensemble.predict_quantiles(X.iloc[90:])
    assert ens_preds.shape == (30, 3)
    assert (ens_preds[:, 0] <= ens_preds[:, 1]).all()
