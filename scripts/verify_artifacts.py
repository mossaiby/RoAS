"""Validate the scope and schema of the committed RoAS result artifacts.

This is deliberately lightweight: it performs no model downloads and does not
claim to reproduce floating-point values. It protects against accidentally
publishing partial benchmark outputs after an exploratory run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.is_file():
        raise AssertionError(f"Missing required artifact: {relative_path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def probability(value: float, name: str) -> None:
    require(0.0 <= value <= 1.0, f"{name} must be in [0, 1], found {value}")


def validate_calibration() -> None:
    artifact = load_json("results/counterfact_calibration_results.json")
    require(artifact["calibration_set_size"] == 500, "Calibration split must contain 500 records")
    require(artifact["test_set_size"] == 500, "Test split must contain 500 records")
    probability(artifact["calibration_auc"], "Calibration AUC")
    probability(artifact["test_auc"], "Test AUC")
    require(0.0 < artifact["optimal_threshold"] < 1.0, "Calibration threshold must be in (0, 1)")
    for name in ("fixed_0_60_results", "calibrated_results"):
        require(set(artifact[name]) == {"rewrite", "paraphrase", "neighborhood"}, f"{name} query groups are incomplete")


def validate_steering() -> None:
    artifact = load_json("results/decoupled_steering_results.json")
    require(artifact["num_records"] == 50, "Steering study must contain 50 records")
    require(artifact["calibrated_threshold"] == 0.55, "Steering threshold must match the reported calibration")
    for metric in ("rewrite_efficacy_score", "rewrite_argmax_accuracy", "paraphrase_score", "paraphrase_argmax_accuracy"):
        for system in ("base_gpt2", "steered_gpt2"):
            values = artifact[metric][system]
            probability(values["mean"], f"{metric}.{system}.mean")
            require(len(values["ci_95"]) == 2, f"{metric}.{system} must include a two-sided CI")


def validate_rank_one_baseline() -> None:
    artifact = load_json("results/counterfact_rome_results.json")
    require(artifact["num_records"] == 500, "Rank-one baseline must contain 500 records")
    require(artifact["covariance_corpus_texts"] == 1500, "Rank-one covariance corpus must contain 1,500 texts")
    for metric in ("efficacy_score", "canonical_argmax_accuracy", "paraphrase_score", "neighborhood_score"):
        probability(artifact[metric], metric)


def validate_routing() -> None:
    artifact = load_json("results/tri_space_seed_sweep.json")
    require(artifact["metadata"]["seeds"] == [40, 41, 42, 43, 44], "Routing seeds must match the reported five-seed sweep")
    aggregate = artifact["surface_matched_aggregate"]
    require(set(aggregate) == {"context", "corpus", "parameter"}, "Routing task families are incomplete")
    for family, metrics in aggregate.items():
        for metric in ("task_accuracy", "route_accuracy", "dedicated_space_ablated_accuracy"):
            probability(metrics[metric]["mean"], f"{family}.{metric}")


def validate_cancellation() -> None:
    artifact = load_json("results/signed_cancellation_stress_results.json")
    config = artifact["config"]
    require(config["trials"] == 50, "Signed-cancellation run must have 50 trials per condition")
    require(config["background_sizes"] == [10, 100, 1000], "Signed-cancellation background sizes do not match the manuscript")
    require(len(artifact["results"]) == 60, "Signed-cancellation artifact must contain 60 conditions")
    require(all(row["bound_coverage"] == 1.0 for row in artifact["results"]), "Signed-cancellation bound coverage failed")


def validate_manuscript_language() -> None:
    manuscript = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    require("ROME-inspired" in manuscript, "Manuscript must label the reduced baseline as ROME-inspired")
    require("not an end-to-end RAG or model-editing comparison" in manuscript, "Manuscript must state the GPT-2 study scope")
    require("not a probability measure" in manuscript, "Manuscript must distinguish signed from probability measures")


def main() -> None:
    validators = (
        validate_calibration,
        validate_steering,
        validate_rank_one_baseline,
        validate_routing,
        validate_cancellation,
        validate_manuscript_language,
    )
    for validator in validators:
        validator()
    print(f"Validated {len(validators)} manuscript and artifact invariants.")


if __name__ == "__main__":
    main()