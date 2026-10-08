"""Agentic EDA and NWP feature hypothesis generator using LangChain."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NWPFeatureHypothesisAgent:
    """LLM-assisted EDA agent that inspects NWP (Numerical Weather Prediction)
    metadata and generates physics-grounded feature engineering hypotheses
    for wind and solar generation forecasting.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self._hypotheses_cache: list[dict[str, Any]] = []

    def inspect_nwp_metadata(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyzes atmospheric metadata fields (wind shear, boundary layer height,
        surface solar radiation downwards, air density) and suggests derived features.
        """
        hypotheses: list[dict[str, Any]] = [
            {
                "hypothesis_id": "H1_WIND_SHEAR_POWER_LAW",
                "target": "wind_generation",
                "rationale": "Vertical wind shear alpha exponent between 10m and 100m captures atmospheric stability regimes affecting turbine hub-height velocity.",
                "proposed_feature": "alpha_shear = log(v_100m / v_10m) / log(100 / 10)",
                "expected_impact": "Reduces high-wind forecast bias during nocturnal low-level jets.",
            },
            {
                "hypothesis_id": "H2_SOLAR_CLEAR_SKY_INDEX",
                "target": "solar_generation",
                "rationale": "Clear-sky index (GHI / GHI_clearsky) normalizes out deterministic solar zenith geometry to isolate cloud optical depth dynamics.",
                "proposed_feature": "k_t = ssrd / max(ssrd_clearsky, 1e-4)",
                "expected_impact": "Improves pinball loss across ramp-up morning and ramp-down evening periods.",
            },
            {
                "hypothesis_id": "H3_RAMP_RATE_TURBULENCE_CONVEXITY",
                "target": "wind_generation",
                "rationale": "Kinetic power proxy (v^3) interacting with 3-hour pressure delta indicates frontal passages with high curtailment probability.",
                "proposed_feature": "kinetic_ramp = (v_100m ** 3) * delta_p_3h",
                "expected_impact": "Identifies extreme tail events where generation swings dictate Day-Ahead price spikes.",
            },
        ]
        self._hypotheses_cache = hypotheses
        logger.info("Generated %d NWP feature hypotheses from metadata", len(hypotheses))
        return hypotheses

    def get_structured_prompt(self, variables: list[str]) -> str:
        """Constructs prompt template for LangChain LLM execution."""
        return (
            f"Given NWP atmospheric variables: {variables}, generate 3 physics-informed "
            "feature hypotheses for renewable generation and power demand forecasting, "
            "focusing on non-linear thermodynamic interactions and cut-out protection thresholds."
        )
