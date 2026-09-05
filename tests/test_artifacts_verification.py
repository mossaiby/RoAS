"""Tests for repository artifact integrity, JSON schemas, and verification tooling.

Covers:
- Execution and validation of all checks in scripts/verify_artifacts.py
- Failure mode testing (detection of corrupted bounds, bad probabilities, missing keys)
- Full inventory check across all primary committed benchmark artifacts in results/
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from verify_artifacts import (
    load_json,
    probability,
    require,
    validate_calibration,
    validate_cancellation,
    validate_manuscript_language,
    validate_rank_one_baseline,
    validate_routing,
    validate_steering,
)

ROOT = Path(__file__).resolve().parents[1]


class TestArtifactVerificationScript:
    """Verifies that all committed artifacts conform to the structural invariants."""

    def test_all_verification_functions_pass(self):
        validate_calibration()
        validate_steering()
        validate_rank_one_baseline()
        validate_routing()
        validate_cancellation()
        validate_manuscript_language()

    def test_probability_validator_rejects_out_of_bounds(self):
        probability(0.0, "zero")
        probability(1.0, "one")
        probability(0.5, "half")

        with pytest.raises(AssertionError):
            probability(-0.01, "negative")

        with pytest.raises(AssertionError):
            probability(1.01, "excessive")

    def test_require_raises_on_false(self):
        require(True, "Should not raise")
        with pytest.raises(AssertionError, match="Condition failed"):
            require(False, "Condition failed")


class TestResultsDirectoryIntegrity:
    """Checks that every expected top-level benchmark JSON artifact exists and parses."""

    EXPECTED_ARTIFACTS = [
        "results/adaptive_retrieval_results.json",
        "results/counterfact_calibration_results.json",
        "results/counterfact_grace_results.json",
        "results/counterfact_retrieval_results.json",
        "results/counterfact_rome_results.json",
        "results/decoupled_steering_results.json",
        "results/finetune_baseline_results.json",
        "results/generative_results.json",
        "results/lifelong_results.json",
        "results/lipschitz_margin_results.json",
        "results/multi_seed_results.json",
        "results/rome_baseline_results.json",
        "results/signed_cancellation_stress_results.json",
        "results/summary_v3.json",
        "results/tri_space_seed_sweep.json",
        "results/tri_space_summary.json",
        "results/unlearning_results.json",
    ]

    @pytest.mark.parametrize("rel_path", EXPECTED_ARTIFACTS)
    def test_artifact_exists_and_is_valid_json(self, rel_path: str):
        full_path = ROOT / rel_path
        assert full_path.is_file(), f"Missing required result artifact: {rel_path}"

        with full_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        assert data is not None, f"Artifact {rel_path} parsed to None"
        assert isinstance(data, (dict, list)), f"Artifact {rel_path} root must be dict or list"
