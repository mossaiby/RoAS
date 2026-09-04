"""Stress-test signed anti-atom cancellation away from exact matching."""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


OUT_DIR = os.path.join("results", "details", "signed_cancellation_stress_outputs")
SUMMARY_PATH = os.path.join("results", "signed_cancellation_stress_results.json")


@dataclass
class StressConfig:
    seed: int = 42
    dimension: int = 128
    temperature: float = 0.07
    trials: int = 50
    background_sizes: List[int] = field(default_factory=lambda: [10, 100, 1000])
    perturbations: List[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.05, 0.1, 0.2]
    )


def unit_noise(reference: torch.Tensor, scale: float) -> torch.Tensor:
    return F.normalize(reference + scale * torch.randn_like(reference), dim=-1)


def measure_trial(
    config: StressConfig,
    background_size: int,
    key_noise: float,
    value_noise: float,
    mass_error: float,
    query_noise: float,
) -> Dict[str, float]:
    positive_key = F.normalize(torch.randn(config.dimension, dtype=torch.float64), dim=-1)
    positive_value = F.normalize(torch.randn(config.dimension, dtype=torch.float64), dim=-1)
    anti_key = unit_noise(positive_key, key_noise)
    anti_value = unit_noise(positive_value, value_noise)
    query = unit_noise(positive_key, query_noise)
    background_keys = F.normalize(
        torch.randn(background_size, config.dimension, dtype=torch.float64), dim=-1
    )

    all_keys = torch.cat([background_keys, positive_key[None], anti_key[None]], dim=0)
    scores = (query @ all_keys.T) / config.temperature
    kernels = torch.exp(scores - scores.max())
    positive_kernel = kernels[-2]
    anti_kernel = kernels[-1]
    anti_mass = 1.0 + mass_error
    denominator = kernels[:-2].sum() + positive_kernel + anti_mass * anti_kernel

    pair_numerator = positive_kernel * positive_value - anti_mass * anti_kernel * anti_value
    pair_residual = pair_numerator.norm() / denominator
    numerator_bound = (
        torch.abs(positive_kernel - anti_mass * anti_kernel) * positive_value.norm()
        + anti_mass * anti_kernel * (positive_value - anti_value).norm()
    )
    pair_bound = numerator_bound / denominator
    pre_pair_contribution = positive_kernel * positive_value.norm() / (
        kernels[:-2].sum() + positive_kernel
    )
    relative_residual = pair_residual / pre_pair_contribution.clamp_min(1e-12)
    return {
        "pair_residual": pair_residual.item(),
        "pair_bound": pair_bound.item(),
        "relative_residual": relative_residual.item(),
    }


def run_axis(config: StressConfig, axis: str, background_size: int) -> List[Dict]:
    rows = []
    for perturbation in config.perturbations:
        measurements = []
        for _ in range(config.trials):
            arguments = {
                "key_noise": 0.0,
                "value_noise": 0.0,
                "mass_error": 0.0,
                "query_noise": 0.0,
            }
            arguments[axis] = perturbation
            measurements.append(
                measure_trial(config, background_size=background_size, **arguments)
            )
        residuals = np.array([item["pair_residual"] for item in measurements])
        bounds = np.array([item["pair_bound"] for item in measurements])
        relative = np.array([item["relative_residual"] for item in measurements])
        rows.append(
            {
                "axis": axis,
                "background_size": background_size,
                "perturbation": perturbation,
                "mean_pair_residual": float(residuals.mean()),
                "max_pair_residual": float(residuals.max()),
                "mean_pair_bound": float(bounds.mean()),
                "mean_relative_residual": float(relative.mean()),
                "bound_coverage": float(np.mean(residuals <= bounds + 1e-7)),
            }
        )
    return rows


def plot_results(rows: List[Dict]) -> None:
    axes_names = ["key_noise", "value_noise", "mass_error", "query_noise"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for axis_plot, axis_name in zip(axes.flat, axes_names):
        for background_size in sorted({row["background_size"] for row in rows}):
            selected = [
                row
                for row in rows
                if row["axis"] == axis_name and row["background_size"] == background_size
            ]
            axis_plot.plot(
                [row["perturbation"] for row in selected],
                [row["mean_relative_residual"] for row in selected],
                marker="o",
                label=f"N={background_size}",
            )
        axis_plot.set(title=axis_name.replace("_", " "), ylabel="Relative pair residual")
        axis_plot.set_yscale("symlog", linthresh=1e-7)
    axes[-1, 0].set_xlabel("Perturbation magnitude")
    axes[-1, 1].set_xlabel("Perturbation magnitude")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(os.path.join(OUT_DIR, "signed_cancellation_stress.png"), dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, help="Override trials per condition")
    parser.add_argument("--background-sizes", type=int, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StressConfig()
    if args.trials:
        config.trials = args.trials
    if args.background_sizes:
        config.background_sizes = args.background_sizes
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    rows = []
    for axis in ["key_noise", "value_noise", "mass_error", "query_noise"]:
        for background_size in config.background_sizes:
            rows.extend(run_axis(config, axis, background_size))
    if any(row["bound_coverage"] < 1.0 for row in rows):
        raise AssertionError("Approximate cancellation bound failed")

    payload = {
        "benchmark": "approximate_signed_pair_cancellation",
        "bound": "||a v - b v_tilde|| / Z <= (|a-b| ||v|| + b ||v-v_tilde||) / Z",
        "config": asdict(config),
        "results": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_results(rows)
    for path in [SUMMARY_PATH, os.path.join(OUT_DIR, "signed_cancellation_stress_results.json")]:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()