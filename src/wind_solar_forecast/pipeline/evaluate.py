"""Evaluation pipeline: computes comprehensive metrics, reliability diagnostics, and Diebold-Mariano tests."""

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from wind_solar_forecast.evaluation.error_decomposition import decompose_forecast_error
from wind_solar_forecast.evaluation.metrics import (
    compute_calibration_ece,
    compute_crps,
    compute_mae,
    compute_pinball_loss,
    compute_rmse,
    compute_winkler_score,
)
from wind_solar_forecast.evaluation.regime_analysis import analyze_wind_ramps, detect_dunkelflaute_events
from wind_solar_forecast.evaluation.significance import diebold_mariano_test, stationary_block_bootstrap
from wind_solar_forecast.models.calibration import compute_pit_values, compute_reliability_table
from wind_solar_forecast.pipeline.train import run_training_pipeline
from wind_solar_forecast.utils.logging import logger


def run_evaluation_pipeline(
    zone: str = "DE_LU",
    vintage: str = "D-1_12:00",
    output_dir: str = "reports"
) -> dict[str, Any]:
    """Runs end-to-end evaluation suite on out-of-sample forecast predictions."""
    logger.info("Starting evaluation and significance testing pipeline", zone=zone, vintage=vintage)

    # Execute training or fetch cached predictions
    train_res = run_training_pipeline(zone=zone, vintage=vintage)
    y_test = train_res["y_test"]
    q_preds_dict = train_res["test_quantile_preds"]
    timestamps_test = train_res["timestamps_test"]
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    eval_summary: dict[str, Any] = {"zone": zone, "vintage": vintage, "models": {}}

    # Baseline comparison target (linear model or gbdt)
    ref_key = "linear" if "linear" in q_preds_dict else list(q_preds_dict.keys())[0]
    ref_loss = np.mean(np.abs(y_test - q_preds_dict[ref_key][:, len(quantiles) // 2]))
    ref_losses_series = np.abs(y_test - q_preds_dict[ref_key][:, len(quantiles) // 2])

    for model_name, q_preds in q_preds_dict.items():
        point_pred = q_preds[:, len(quantiles) // 2]
        mae = compute_mae(y_test, point_pred)
        rmse = compute_rmse(y_test, point_pred)
        crps = compute_crps(q_preds, quantiles, y_test)
        ece = compute_calibration_ece(q_preds, quantiles, y_test)
        winkler = compute_winkler_score(q_preds[:, 0], q_preds[:, -1], y_test, alpha=0.10)

        # Diebold-Mariano test vs reference
        model_losses = np.abs(y_test - point_pred)
        dm_res = diebold_mariano_test(model_losses, ref_losses_series, horizon_steps=24)

        # Stationary Block Bootstrap for 95% CI on CRPS
        _, _, (crps_ci_low, crps_ci_high) = stationary_block_bootstrap(
            y_test,
            lambda y_sample: compute_crps(q_preds[:len(y_sample)], quantiles, y_sample),
            n_bootstraps=300
        )

        rel_df = compute_reliability_table(q_preds, quantiles, y_test)

        eval_summary["models"][model_name] = {
            "mae": np.round(mae, 4),
            "rmse": np.round(rmse, 4),
            "crps": np.round(crps, 4),
            "crps_95_ci": [np.round(crps_ci_low, 4), np.round(crps_ci_high, 4)],
            "winkler_score_90": np.round(winkler, 4),
            "ece": np.round(ece, 4),
            "dm_stat_vs_ref": np.round(dm_res.dm_statistic, 3),
            "dm_p_val": np.round(dm_res.p_value, 4),
            "is_significantly_better": bool(dm_res.dm_statistic < 0 and dm_res.p_value < 0.05)
        }

    # Error decomposition on best ensemble or gbdt
    best_key = "ensemble" if "ensemble" in q_preds_dict else list(q_preds_dict.keys())[0]
    best_point = q_preds_dict[best_key][:, len(quantiles) // 2]
    decomp = decompose_forecast_error(y_test, best_point)
    eval_summary["error_decomposition"] = {
        "model": best_key,
        "total_mse": np.round(decomp.total_mse, 6),
        "unconditional_bias": np.round(decomp.unconditional_bias, 6),
        "fraction_bias": np.round(decomp.fraction_bias, 4),
        "fraction_variance": np.round(decomp.fraction_variance, 4),
        "portfolio_smoothing_ratio": np.round(decomp.portfolio_smoothing_ratio, 4)
    }

    # Ramp analysis
    ramps = analyze_wind_ramps(y_test, best_point)
    eval_summary["ramp_analysis"] = ramps

    out_path = Path(output_dir) / f"evaluation_summary_{zone}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(eval_summary, f, indent=2)

    logger.info("Evaluation pipeline completed", summary_path=str(out_path), models=list(eval_summary["models"].keys()))
    return eval_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate probabilistic forecast models.")
    parser.add_argument("--zone", type=str, default="DE_LU")
    parser.add_argument("--vintage", type=str, default="D-1_12:00")
    args = parser.parse_args()

    run_evaluation_pipeline(zone=args.zone, vintage=args.vintage)


if __name__ == "__main__":
    main()
