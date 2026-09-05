"""Benchmark error-budgeted exact top-k reads across corpus sizes."""

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


OUT_DIR = os.path.join("results", "details", "adaptive_retrieval_outputs")
SUMMARY_PATH = os.path.join("results", "adaptive_retrieval_results.json")


@dataclass
class BenchmarkConfig:
    seed: int = 42
    dimension: int = 128
    num_queries: int = 64
    corpus_sizes: List[int] = field(default_factory=lambda: [1000, 5000, 10000, 25000])
    temperatures: List[float] = field(default_factory=lambda: [0.05, 0.1])
    error_budgets: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    fixed_k_values: List[int] = field(default_factory=lambda: [1, 10, 100])
    query_noise: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def adaptive_topk_read(
    sorted_scores: torch.Tensor,
    sorted_values: torch.Tensor,
    full_read: torch.Tensor,
    error_budget: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return minimum certified k, certificate, actual error, and tail mass."""
    probabilities = F.softmax(sorted_scores, dim=-1)
    cumulative_mass = probabilities.cumsum(dim=-1)
    value_norm_bound = sorted_values.norm(dim=-1).amax(dim=-1)
    allowed_tail = error_budget / (2.0 * value_norm_bound.clamp_min(1e-12))
    meets_budget = cumulative_mass >= (1.0 - allowed_tail).unsqueeze(-1)
    
    # Safe index resolution: fallback to corpus size if no prefix meets budget
    has_match = meets_budget.any(dim=-1)
    first_match = meets_budget.to(torch.int64).argmax(dim=-1) + 1
    selected_k = torch.where(has_match, first_match, sorted_scores.size(-1))

    positions = torch.arange(sorted_scores.size(1), device=sorted_scores.device)
    mask = positions.unsqueeze(0) < selected_k.unsqueeze(1)
    retained_probabilities = probabilities * mask
    retained_mass = retained_probabilities.sum(dim=-1)
    adaptive_weights = retained_probabilities / retained_mass.unsqueeze(-1)
    adaptive_read = torch.bmm(
        adaptive_weights.unsqueeze(1), sorted_values
    ).squeeze(1)

    tail_mass = 1.0 - retained_mass
    certificate = 2.0 * value_norm_bound * tail_mass
    actual_error = (full_read - adaptive_read).norm(dim=-1)
    return selected_k, certificate, actual_error, tail_mass


def fixed_topk_error(
    sorted_scores: torch.Tensor,
    sorted_values: torch.Tensor,
    full_read: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    selected_scores = sorted_scores[:, :top_k]
    selected_values = sorted_values[:, :top_k]
    weights = F.softmax(selected_scores, dim=-1)
    read = torch.bmm(weights.unsqueeze(1), selected_values).squeeze(1)
    return (full_read - read).norm(dim=-1)


def make_problem(config: BenchmarkConfig, corpus_size: int) -> Tuple[torch.Tensor, ...]:
    keys = F.normalize(
        torch.randn(corpus_size, config.dimension, device=config.device), dim=-1
    )
    values = F.normalize(
        torch.randn(corpus_size, config.dimension, device=config.device), dim=-1
    )
    anchors = torch.randint(corpus_size, (config.num_queries,), device=config.device)
    queries = F.normalize(
        keys[anchors] + config.query_noise * torch.randn_like(keys[anchors]), dim=-1
    )
    return keys, values, queries


def benchmark_configuration(
    config: BenchmarkConfig, corpus_size: int, temperature: float
) -> Dict:
    keys, values, queries = make_problem(config, corpus_size)
    synchronize = torch.cuda.synchronize if config.device.startswith("cuda") else lambda: None

    synchronize()
    start = time.perf_counter()
    scores = (queries @ keys.T) / temperature
    sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
    sorted_values = values[sorted_indices]
    probabilities = F.softmax(sorted_scores, dim=-1)
    full_read = torch.bmm(probabilities.unsqueeze(1), sorted_values).squeeze(1)
    synchronize()
    exact_elapsed_ms = (time.perf_counter() - start) * 1000.0

    fixed_results = {}
    for top_k in config.fixed_k_values:
        effective_k = min(top_k, corpus_size)
        errors = fixed_topk_error(sorted_scores, sorted_values, full_read, effective_k)
        fixed_results[str(top_k)] = {
            "effective_k": effective_k,
            "mean_error": errors.mean().item(),
            "max_error": errors.max().item(),
        }

    adaptive_results = {}
    for error_budget in config.error_budgets:
        synchronize()
        start = time.perf_counter()
        selected_k, certificate, errors, tail_mass = adaptive_topk_read(
            sorted_scores, sorted_values, full_read, error_budget
        )
        synchronize()
        adaptive_elapsed_ms = (time.perf_counter() - start) * 1000.0
        if torch.any(errors > certificate + 1e-5):
            raise AssertionError("Adaptive read violated its tail-mass certificate")
        if torch.any(certificate > error_budget + 1e-5):
            raise AssertionError("Selected k did not satisfy the requested error budget")

        adaptive_results[str(error_budget)] = {
            "mean_k": selected_k.float().mean().item(),
            "median_k": selected_k.float().median().item(),
            "max_k": selected_k.max().item(),
            "mean_k_fraction": (selected_k.float() / corpus_size).mean().item(),
            "mean_tail_mass": tail_mass.mean().item(),
            "mean_certificate": certificate.mean().item(),
            "max_certificate": certificate.max().item(),
            "mean_error": errors.mean().item(),
            "max_error": errors.max().item(),
            "empirical_coverage": (errors <= error_budget + 1e-5).float().mean().item(),
            "selection_and_read_ms": adaptive_elapsed_ms,
        }

    return {
        "corpus_size": corpus_size,
        "temperature": temperature,
        "exact_score_sort_and_read_ms": exact_elapsed_ms,
        "fixed_k": fixed_results,
        "adaptive": adaptive_results,
    }


def plot_results(results: List[Dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    temperatures = sorted({row["temperature"] for row in results})
    budgets = sorted(float(key) for key in results[0]["adaptive"])

    for temperature in temperatures:
        rows = [row for row in results if row["temperature"] == temperature]
        rows.sort(key=lambda row: row["corpus_size"])
        axes[0].plot(
            [row["corpus_size"] for row in rows],
            [row["exact_score_sort_and_read_ms"] for row in rows],
            marker="o",
            label=fr"$\tau={temperature}$",
        )
    axes[0].set(xlabel="Corpus size", ylabel="Exact scoring + sort + read (ms)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend()

    temperature = min(temperatures)
    rows = sorted(
        [row for row in results if row["temperature"] == temperature],
        key=lambda row: row["corpus_size"],
    )
    for budget in budgets:
        axes[1].plot(
            [row["corpus_size"] for row in rows],
            [row["adaptive"][str(budget)]["mean_k"] for row in rows],
            marker="o",
            label=fr"$\varepsilon={budget}$",
        )
    axes[1].set(xlabel="Corpus size", ylabel="Mean certified k")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(os.path.join(OUT_DIR, "adaptive_retrieval_scaling.png"), dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", help="Override corpus sizes")
    parser.add_argument("--queries", type=int, help="Override query count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig()
    if args.sizes:
        config.corpus_sizes = args.sizes
    if args.queries:
        config.num_queries = args.queries

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    results = [
        benchmark_configuration(config, corpus_size, temperature)
        for corpus_size in config.corpus_sizes
        for temperature in config.temperatures
    ]
    payload = {
        "benchmark": "adaptive_error_budgeted_exact_topk",
        "certificate": "||R-R_k||_2 <= 2 * V_max * rho_tail <= epsilon",
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": config.device,
        },
        "results": results,
    }
    plot_results(results)
    for path in [SUMMARY_PATH, os.path.join(OUT_DIR, "adaptive_retrieval_results.json")]:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()