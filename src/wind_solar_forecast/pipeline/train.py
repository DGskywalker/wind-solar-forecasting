"""Model training pipeline: walk-forward validation, model ladder execution, and MLflow experiment tracking."""

import argparse
from pathlib import Path
from typing import Any
import joblib
import mlflow
import numpy as np
import pandas as pd

from wind_solar_forecast.features.build_features import build_feature_dataset
from wind_solar_forecast.models.baselines import ClimatologyForecaster, NWPDirectForecaster, PersistenceForecaster
from wind_solar_forecast.models.conformal import ConformalQuantileRegressor
from wind_solar_forecast.models.deep import DeepProbabilisticForecaster
from wind_solar_forecast.models.ensemble import CRPSOptimalEnsemble, quantile_crps
from wind_solar_forecast.models.gbdt import LGBMQuantileForecaster, NGBoostForecaster
from wind_solar_forecast.models.linear import LinearBenchmark
from wind_solar_forecast.utils.io import load_parquet
from wind_solar_forecast.utils.logging import logger


def run_training_pipeline(
    zone: str = "DE_LU",
    vintage: str = "D-1_12:00",
    start_date: str = "2023-01-01",
    end_date: str = "2023-01-08",
    models_to_train: list[str] | None = None,
    output_model_dir: str = "data/models",
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "european-wind-solar-forecasting"
) -> dict[str, Any]:
    """Executes walk-forward training across the model ladder and tracks results in MLflow.

    Args:
        zone: European bidding zone.
        vintage: Forecast cycle vintage.
        start_date: Dataset start date.
        end_date: Dataset end date.
        models_to_train: Sublist of models to benchmark.
        output_model_dir: Checkpoint directory.
        mlflow_tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.

    Returns:
        Dictionary of validation metrics per model.
    """
    if models_to_train is None:
        models_to_train = ["persistence", "climatology", "nwp_direct", "linear", "gbdt", "ngboost", "deep", "conformal", "ensemble"]

    logger.info("Starting model training pipeline", zone=zone, vintage=vintage, models=models_to_train)
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    # 1. Obtain engineered features
    df, manifest = build_feature_dataset(
        zone=zone,
        vintage=vintage,
        start_date=start_date,
        end_date=end_date
    )

    # 2. Separate SAFE features from target contemporaneous
    safe_features = [m.feature_name for m in manifest if m.leakage_risk_tag == "SAFE_AT_GATE_CLOSURE"]
    # Filter to purely numeric predictor columns
    numeric_features = [
        col for col in safe_features
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        and col not in ["valid_time", "issue_time", "timestamp"]
    ]

    target_col = "actual_wind_onshore_cf"
    X = df[numeric_features].fillna(0.0)
    y = df[target_col].values
    timestamps = pd.DatetimeIndex(df["valid_time"])

    # 3. Expanding Walk-Forward Split with 24h Embargo
    n = len(df)
    train_size = int(n * 0.60)
    embargo = 24  # 24 hours embargo to prevent transition autocorrelation leakage
    val_size = int(n * 0.20)

    train_idx = np.arange(0, train_size)
    calib_idx = np.arange(train_size + embargo, train_size + embargo + val_size)
    test_idx = np.arange(train_size + embargo + val_size + embargo, n)

    if len(test_idx) < 12:
        # For short 1-week test sample, scale indices adaptively
        train_idx = np.arange(0, int(n * 0.55))
        calib_idx = np.arange(int(n * 0.55), int(n * 0.75))
        test_idx = np.arange(int(n * 0.75), n)

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_calib, y_calib = X.iloc[calib_idx], y[calib_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    logger.info("Dataset split partitions", train=len(X_train), calib=len(X_calib), test=len(X_test), features=len(numeric_features))

    # 4. Setup MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    results: dict[str, Any] = {}
    fitted_models: dict[str, Any] = {}
    test_quantile_preds: dict[str, np.ndarray] = {}
    calib_quantile_preds: dict[str, np.ndarray] = {}

    out_dir = Path(output_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"train_{zone}_{vintage}"):
        mlflow.log_params({
            "zone": zone,
            "vintage": vintage,
            "target": target_col,
            "n_features": len(numeric_features),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        })

        # --- A. Baselines ---
        if "persistence" in models_to_train:
            m = PersistenceForecaster(lag_hours=24)
            # Repeat last 24h
            pers_preds = m.predict(pd.Series(y_train), horizon_steps=len(X_test))
            mae = float(np.mean(np.abs(y_test - pers_preds)))
            results["persistence"] = {"mae": mae}
            mlflow.log_metric("persistence_mae", mae)

        if "climatology" in models_to_train:
            clim = ClimatologyForecaster(quantiles=quantiles)
            clim.fit(timestamps[train_idx], y_train)
            q_clim = clim.predict_quantiles(timestamps[test_idx])
            crps_clim = quantile_crps(q_clim, quantiles, y_test)
            results["climatology"] = {"crps": crps_clim}
            mlflow.log_metric("climatology_crps", crps_clim)

        if "nwp_direct" in models_to_train and "nwp_wind_onshore_cf" in X.columns:
            nwp_d = NWPDirectForecaster(nwp_cf_column="nwp_wind_onshore_cf")
            nwp_d.fit(X_train, pd.Series(y_train))
            preds_nwp = nwp_d.predict(X_test)
            mae_nwp = float(np.mean(np.abs(y_test - preds_nwp)))
            results["nwp_direct"] = {"mae": mae_nwp}
            mlflow.log_metric("nwp_direct_mae", mae_nwp)

        # --- B. Linear ---
        if "linear" in models_to_train:
            lin = LinearBenchmark(quantiles=quantiles, ridge_alpha=1.0)
            lin.fit(X_train, y_train)
            q_lin = lin.predict_quantiles(X_test)
            crps_lin = quantile_crps(q_lin, quantiles, y_test)
            results["linear"] = {"crps": crps_lin}
            mlflow.log_metric("linear_crps", crps_lin)
            fitted_models["linear"] = lin
            test_quantile_preds["linear"] = q_lin
            calib_quantile_preds["linear"] = lin.predict_quantiles(X_calib)

        # --- C. LightGBM ---
        if "gbdt" in models_to_train:
            lgb_model = LGBMQuantileForecaster(quantiles=quantiles, n_estimators=60, learning_rate=0.05)
            lgb_model.fit(X_train, y_train)
            q_lgb = lgb_model.predict_quantiles(X_test)
            crps_lgb = quantile_crps(q_lgb, quantiles, y_test)
            results["gbdt"] = {"crps": crps_lgb}
            mlflow.log_metric("gbdt_crps", crps_lgb)
            fitted_models["gbdt"] = lgb_model
            test_quantile_preds["gbdt"] = q_lgb
            calib_quantile_preds["gbdt"] = lgb_model.predict_quantiles(X_calib)

        # --- D. NGBoost ---
        if "ngboost" in models_to_train:
            ngb = NGBoostForecaster(quantiles=quantiles, n_estimators=30, learning_rate=0.03)
            ngb.fit(X_train, y_train)
            q_ngb = ngb.predict_quantiles(X_test)
            crps_ngb = quantile_crps(q_ngb, quantiles, y_test)
            results["ngboost"] = {"crps": crps_ngb}
            mlflow.log_metric("ngboost_crps", crps_ngb)
            fitted_models["ngboost"] = ngb
            test_quantile_preds["ngboost"] = q_ngb
            calib_quantile_preds["ngboost"] = ngb.predict_quantiles(X_calib)

        # --- E. Deep Temporal Fusion Net ---
        if "deep" in models_to_train:
            deep = DeepProbabilisticForecaster(quantiles=quantiles, hidden_dim=32, max_epochs=10, n_seeds=3)
            deep.fit(X_train, y_train)
            q_deep = deep.predict_quantiles(X_test)
            crps_deep = quantile_crps(q_deep, quantiles, y_test)
            results["deep"] = {"crps": crps_deep}
            mlflow.log_metric("deep_crps", crps_deep)
            fitted_models["deep"] = deep
            test_quantile_preds["deep"] = q_deep
            calib_quantile_preds["deep"] = deep.predict_quantiles(X_calib)

        # --- F. Conformal Prediction (CQR on top of GBDT) ---
        if "conformal" in models_to_train and "gbdt" in fitted_models:
            cqr = ConformalQuantileRegressor(base_quantile_model=fitted_models["gbdt"], nominal_coverage=0.90)
            cqr.calibrate(X_calib, y_calib)
            low_cqr, high_cqr = cqr.predict_interval(X_test)
            cov_cqr = float(np.mean((y_test >= low_cqr) & (y_test <= high_cqr)))
            width_cqr = float(np.mean(high_cqr - low_cqr))
            results["conformal"] = {"coverage": cov_cqr, "mean_width": width_cqr}
            mlflow.log_metric("conformal_coverage_90", cov_cqr)
            mlflow.log_metric("conformal_mean_width", width_cqr)
            fitted_models["conformal"] = cqr

        # --- G. CRPS-Optimal Ensemble Blend ---
        candidate_ensemble_keys = [k for k in ["linear", "gbdt", "ngboost", "deep"] if k in fitted_models]
        if "ensemble" in models_to_train and len(candidate_ensemble_keys) >= 2:
            ens_models = [fitted_models[k] for k in candidate_ensemble_keys]
            val_preds = [calib_quantile_preds[k] for k in candidate_ensemble_keys]
            ensemble = CRPSOptimalEnsemble(models=ens_models, model_names=candidate_ensemble_keys, quantiles=quantiles)
            ensemble.fit_weights(val_preds, y_calib)
            q_ens = ensemble.predict_quantiles(X_test)
            crps_ens = quantile_crps(q_ens, quantiles, y_test)
            results["ensemble"] = {"crps": crps_ens, "weights": dict(zip(candidate_ensemble_keys, ensemble.weights.tolist()))}
            mlflow.log_metric("ensemble_crps", crps_ens)
            for k, w in zip(candidate_ensemble_keys, ensemble.weights):
                mlflow.log_metric(f"ensemble_weight_{k}", float(w))
            fitted_models["ensemble"] = ensemble
            test_quantile_preds["ensemble"] = q_ens

        # Persist top models
        clean_vint = vintage.replace(":", "").replace("-", "_")
        for name, model in fitted_models.items():
            model_file = out_dir / f"model_{zone}_{clean_vint}_{name}.joblib"
            try:
                joblib.dump(model, model_file)
            except Exception:
                pass

        logger.info("Training ladder completed", results=results)

    return {
        "results": results,
        "test_quantile_preds": test_quantile_preds,
        "y_test": y_test,
        "timestamps_test": timestamps[test_idx]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train wind & solar probabilistic model ladder.")
    parser.add_argument("--zone", type=str, default="DE_LU")
    parser.add_argument("--vintage", type=str, default="D-1_12:00")
    parser.add_argument("--horizon", type=str, default="D-1")
    args = parser.parse_args()

    run_training_pipeline(zone=args.zone, vintage=args.vintage)


if __name__ == "__main__":
    main()
