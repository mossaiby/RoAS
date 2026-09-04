"""Run the tri-space surface-matched evaluation across multiple seeds."""

import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


SEEDS = [40, 41, 42, 43, 44]
CANONICAL_SEED = 42
ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "atom_space_tri_gating_engine.py"
RESULT = ROOT / "results" / "tri_space_summary.json"
DETAIL_DIR = ROOT / "results" / "details" / "tri_gating_probe_outputs"


def run_seed(seed: int) -> dict:
    env = os.environ.copy()
    env["ROAS_SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)
    completed = subprocess.run(
        [sys.executable, str(ENGINE)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"seed={seed} complete")
    if completed.stderr:
        print(completed.stderr.strip())
    return json.loads(RESULT.read_text(encoding="utf-8"))


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def plot_aggregate(summary: dict) -> None:
    aggregate = summary["surface_matched_aggregate"]
    families = ("context", "corpus", "parameter")
    labels = ["Context Tasks", "Corpus Tasks", "Parameter Tasks"]
    gate_matrix = np.array([
        [aggregate[family]["mean_gate"][gate]["mean"] for gate in families]
        for family in families
    ]) * 100
    task_means = np.array([aggregate[family]["task_accuracy"]["mean"] for family in families]) * 100
    task_stds = np.array([aggregate[family]["task_accuracy"]["sample_std"] for family in families]) * 100
    ablated_means = np.array([
        aggregate[family]["dedicated_space_ablated_accuracy"]["mean"]
        for family in families
    ]) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.imshow(gate_matrix, cmap="Blues", vmin=0, vmax=100)
    ax1.set_xticks(range(3), [r"$\Omega_{\mathrm{ctx}}$", r"$\Omega_{\mathrm{corpus}}$", r"$\Omega_{\mathrm{params}}$"])
    ax1.set_yticks(range(3), labels)
    ax1.set_title(r"Five-Seed Mean Router Allocation $\lambda(x)$")
    for row in range(3):
        for column in range(3):
            ax1.text(
                column,
                row,
                f"{gate_matrix[row, column]:.1f}%",
                ha="center",
                va="center",
                color="white" if gate_matrix[row, column] > 50 else "black",
                fontweight="bold",
            )

    positions = np.arange(3)
    width = 0.35
    ax2.bar(
        positions - width / 2,
        task_means,
        width,
        yerr=task_stds,
        capsize=4,
        label="Full Model (Mean +/- SD)",
        color="#1f77b4",
    )
    ax2.bar(
        positions + width / 2,
        ablated_means,
        width,
        label="Dedicated Space Ablated",
        color="#d62728",
    )
    ax2.set_xticks(positions, labels)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 115)
    ax2.set_title("Five-Seed Surface-Matched Ablation")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend()
    plt.tight_layout()
    for output in (
        DETAIL_DIR / "tri_space_gating_ablation.png",
        ROOT / "paper" / "figures" / "tri_space_gating_ablation.png",
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    runs = {str(seed): copy.deepcopy(run_seed(seed)) for seed in SEEDS}
    families = ("context", "corpus", "parameter")
    metrics = ("task_accuracy", "route_accuracy", "dedicated_space_ablated_accuracy")
    aggregate = {
        family: {
            **{
                metric: describe([
                    runs[str(seed)]["surface_matched_formulations"][family][metric]
                    for seed in SEEDS
                ])
                for metric in metrics
            },
            "mean_gate": {
                gate_name: describe([
                    runs[str(seed)]["surface_matched_formulations"][family]["mean_gate"][gate_index]
                    for seed in SEEDS
                ])
                for gate_index, gate_name in enumerate(("context", "corpus", "parameter"))
            },
        }
        for family in families
    }
    summary = {
        "metadata": {
            "seeds": SEEDS,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": importlib.metadata.version("torch"),
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "transformers": importlib.metadata.version("transformers"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "surface_matched_aggregate": aggregate,
        "runs": runs,
    }
    output = json.dumps(summary, indent=2)
    (ROOT / "results" / "tri_space_seed_sweep.json").write_text(output, encoding="utf-8")
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    (DETAIL_DIR / "tri_space_seed_sweep.json").write_text(output, encoding="utf-8")

    run_seed(CANONICAL_SEED)
    plot_aggregate(summary)
    print("Five-seed summary written; canonical seed 42 artifacts restored.")


if __name__ == "__main__":
    main()