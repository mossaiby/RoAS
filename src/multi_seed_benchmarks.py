"""Run insertion, lifelong, and suppression engines across multiple seeds."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [40, 41, 42, 43, 44]
CANONICAL_SEED = 42
ENGINES = {
    "insertion": (ROOT / "src" / "atom_space_engine.py", ROOT / "results" / "summary_v3.json"),
    "lifelong": (ROOT / "src" / "atom_space_lifelong_engine.py", ROOT / "results" / "lifelong_results.json"),
    "unlearning": (ROOT / "src" / "atom_space_unlearning_engine.py", ROOT / "results" / "unlearning_results.json"),
}
OUT_DIR = ROOT / "results" / "details" / "multi_seed_outputs"


def describe(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def run_engine(name: str, seed: int) -> Dict:
    script, result_path = ENGINES[name]
    environment = os.environ.copy()
    environment["ROAS_SEED"] = str(seed)
    environment["PYTHONHASHSEED"] = str(seed)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        print(completed.stderr.strip())
    print(f"{name}: seed={seed} complete")
    with open(result_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_insertion(runs: Dict[str, Dict]) -> Dict:
    densities = runs[next(iter(runs))].keys()
    scalar_metrics = [
        "efficacy",
        "generality",
        "specificity",
        "base_training_accuracy",
        "specificity_mean_kl",
        "specificity_max_probability_shift",
        "novel_atom_attention_mass",
        "adapter_spectral_product",
    ]
    return {
        density: {
            metric: describe([runs[str(seed)][density][metric] for seed in map(int, runs)])
            for metric in scalar_metrics
        }
        for density in densities
    }


def aggregate_lifelong(runs: Dict[str, List[Dict]]) -> List[Dict]:
    first_run = runs[next(iter(runs))]
    metrics = [
        "cumulative_efficacy",
        "cumulative_generality",
        "background_specificity",
        "mean_attention_mass",
    ]
    return [
        {
            "num_edits_inserted": checkpoint["num_edits_inserted"],
            **{
                metric: describe([
                    runs[str(seed)][index][metric] for seed in map(int, runs)
                ])
                for metric in metrics
            },
        }
        for index, checkpoint in enumerate(first_run)
    ]


def aggregate_unlearning(runs: Dict[str, Dict]) -> Dict:
    def seed_means(section: str, metric: str) -> List[float]:
        return [
            float(np.mean([float(row[metric]) for row in run[section]]))
            for run in runs.values()
        ]

    return {
        "counterfactual_updates": {
            "success_rate": describe(seed_means("counterfactual_updates", "override_success")),
            "new_target_probability": describe(seed_means("counterfactual_updates", "prob_new")),
            "old_target_probability": describe(seed_means("counterfactual_updates", "prob_old")),
            "background_specificity": describe(
                seed_means("counterfactual_updates", "background_specificity")
            ),
        },
        "matched_anti_atom_suppression": {
            "success_rate": describe(
                seed_means("anti_atom_suppression", "successfully_suppressed")
            ),
            "pre_suppression_probability": describe(
                seed_means("anti_atom_suppression", "pre_suppression_probability")
            ),
            "residual_probability": describe(
                seed_means("anti_atom_suppression", "residual_probability")
            ),
            "background_specificity": describe(
                seed_means("anti_atom_suppression", "background_specificity")
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name in args.engines:
        runs = {str(seed): copy.deepcopy(run_engine(name, seed)) for seed in args.seeds}
        if name == "insertion":
            aggregate = aggregate_insertion(runs)
        elif name == "lifelong":
            aggregate = aggregate_lifelong(runs)
        else:
            aggregate = aggregate_unlearning(runs)
        all_results[name] = {"seeds": args.seeds, "aggregate": aggregate, "runs": runs}
        if CANONICAL_SEED in args.seeds:
            run_engine(name, CANONICAL_SEED)

    payload = {"benchmark": "multi_seed_controlled_studies", "results": all_results}
    output = json.dumps(payload, indent=2)
    for path in [ROOT / "results" / "multi_seed_results.json", OUT_DIR / "multi_seed_results.json"]:
        path.write_text(output, encoding="utf-8")
    print("Multi-seed results written; canonical seed 42 artifacts restored where applicable.")


if __name__ == "__main__":
    main()