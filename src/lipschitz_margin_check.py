"""
src/lipschitz_margin_check.py

Empirical verification of Proposition 1 and Equation (1):
Validates the Lipschitz-margin sufficient condition for top-k inclusion:
  <phi_q(x*), phi_k(omega_0)> - ||phi_q(x*)||_2 * L * d_Omega(omega*, omega_0) > tau_k(x*)
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from atom_space_engine import EngineConfig, RAW_DOMAINS, UnifiedResidualTiedLM

OUT_DIR = os.path.join("results", "details", "lipschitz_probe_outputs")
SUMMARY_PATH = os.path.join("results", "lipschitz_margin_results.json")


def compute_lipschitz_bounds(model: UnifiedResidualTiedLM, raw_features: torch.Tensor) -> Tuple[float, float]:
    """
    Computes both:
    1. Analytical upper bound on L for phi(x) = Normalize(x + 0.1 * MLP(x))
    2. Empirical maximum directional derivative (finite-difference estimate)
    """
    # 1. Analytical bound:
    # Let g(x) = x + 0.1 * MLP(x). L_g <= 1 + 0.1 * ||W_2||_2 * ||W_1||_2 (assuming ReLU has L=1)
    w1 = model.adapter[0].weight
    w2 = model.adapter[3].weight
    norm_w1 = torch.linalg.matrix_norm(w1, ord=2).item()
    norm_w2 = torch.linalg.matrix_norm(w2, ord=2).item()
    l_g = 1.0 + 0.1 * norm_w1 * norm_w2

    # Normalization map h(u) = u / ||u|| has Lipschitz constant <= 2 / min(||u||)
    with torch.no_grad():
        adapted_raw = raw_features + 0.1 * model.adapter(raw_features)
        min_norm = torch.linalg.vector_norm(adapted_raw, dim=-1).min().item()
    l_norm = 2.0 / max(min_norm, 1e-6)
    l_analytical = l_g * l_norm

    # 2. Empirical maximum ratio over all pairwise combinations in the benchmark:
    with torch.no_grad():
        phi_feats = model.phi(raw_features)
        dist_in = torch.cdist(raw_features, raw_features, p=2)
        dist_out = torch.cdist(phi_feats, phi_feats, p=2)
        mask = dist_in > 1e-5
        l_empirical = (dist_out[mask] / dist_in[mask]).max().item()

    return l_analytical, l_empirical


def evaluate_margin_condition(
    model: UnifiedResidualTiedLM,
    train_k_raw: torch.Tensor,
    hold_k_raw: torch.Tensor,
    hold_q_raw: torch.Tensor,
    top_k_list: List[int] = [1, 3, 5, 10]
) -> Dict:
    """
    Evaluates Eq. (1) for each holdout fact omega* against its nearest anchor omega_0 in Omega_train.
    """
    model.eval()
    with torch.no_grad():
        phi_q = model.phi(hold_q_raw)
        phi_train_k = model.phi(train_k_raw)
        phi_hold_k = model.phi(hold_k_raw)

        # Baseline scores for all training atoms: [num_holdout, num_train]
        train_scores = phi_q @ phi_train_k.T
        sorted_train_scores, _ = torch.sort(train_scores, dim=-1, descending=True)

        all_raw = torch.cat([train_k_raw, hold_k_raw], dim=0)
        l_analytical, l_empirical = compute_lipschitz_bounds(model, all_raw)

        # d_Omega is the L2 distance in the base encoder space
        pairwise_d_omega = torch.cdist(hold_k_raw, train_k_raw, p=2)  # [num_holdout, num_train]
        nearest_anchor_dist, nearest_anchor_idx = torch.min(pairwise_d_omega, dim=-1)

        # Vectorized anchor score computation: <phi_q(x*), phi_k(omega_0)>
        anchor_scores = torch.sum(phi_q * phi_train_k[nearest_anchor_idx], dim=-1)
        q_norm = torch.linalg.vector_norm(phi_q, dim=-1)

        results_by_k = {}
        for k in top_k_list:
            tau_k = sorted_train_scores[:, k - 1]  # k-th highest training score

            # Evaluate Equation (1) with empirical Lipschitz constant
            lhs_empirical = anchor_scores - q_norm * l_empirical * nearest_anchor_dist
            empirical_certified = (lhs_empirical > tau_k).float().mean().item()

            # Actual observed top-k entry: <phi_q(x*), phi_k(omega*)> > tau_k
            actual_scores = torch.sum(phi_q * phi_hold_k, dim=-1)
            actual_entered = (actual_scores > tau_k).float().mean().item()

            results_by_k[f"top_{k}"] = {
                "empirical_certified_ratio": empirical_certified,
                "actual_entry_ratio": actual_entered,
                "mean_lhs": lhs_empirical.mean().item(),
                "mean_tau_k": tau_k.mean().item(),
                "mean_anchor_score": anchor_scores.mean().item(),
                "mean_nearest_distance": nearest_anchor_dist.mean().item()
            }

    return {
        "l_analytical_bound": l_analytical,
        "l_empirical": l_empirical,
        "results_by_k": results_by_k
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    cfg = EngineConfig()
    device = cfg.device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"=== Evaluating Lipschitz Margin Condition (Eq. 1) on {device} ===")
    backbone = SentenceTransformer(cfg.encoder_name, device=device)

    # .clone() strips the InferenceTensor tag so autograd works during model training
    def encode_clean(texts: List[str]) -> torch.Tensor:
        return backbone.encode(texts, convert_to_tensor=True, device=device).clone()

    all_targets = set()
    for cat in RAW_DOMAINS:
        for s, r, o in RAW_DOMAINS[cat]:
            all_targets.add(o)
    target2id = {tgt: i for i, tgt in enumerate(sorted(all_targets))}

    train_triplets, holdout_triplets = [], []
    for cat in RAW_DOMAINS:
        train_triplets.extend(RAW_DOMAINS[cat][:40])
        holdout_triplets.extend(RAW_DOMAINS[cat][40:])

    train_sents = [f"{s} {r} {o}." for s, r, o in train_triplets]
    train_queries = [f"Complete the statement: {s} {r} ___" for s, r, o in train_triplets]
    train_tgts = torch.tensor([target2id[o] for _, _, o in train_triplets], device=device)

    train_k_raw = encode_clean(train_sents)
    train_q_raw = encode_clean(train_queries)

    hold_sents = [f"{s} {r} {o}." for s, r, o in holdout_triplets]
    hold_queries = [f"Complete the statement: {s} {r} ___" for s, r, o in holdout_triplets]
    hold_k_raw = encode_clean(hold_sents)
    hold_q_raw = encode_clean(hold_queries)

    model = UnifiedResidualTiedLM(cfg, train_k_raw.shape[-1], len(target2id)).to(device)
    with torch.no_grad():
        norm_E = model.get_normalized_embeddings()
        model.omega_corpus.set_atoms(train_k_raw, norm_E[train_tgts])

    # Base model training
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    for ep in range(400):
        model.train()
        norm_E = model.get_normalized_embeddings()
        model.omega_corpus.values = norm_E[train_tgts]
        logits, _ = model(train_q_raw)
        loss = F.cross_entropy(logits, train_tgts)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < 0.01:
            break

    print(f"Base model converged at epoch {ep} (loss: {loss.item():.4f})")
    eval_results = evaluate_margin_condition(model, train_k_raw, hold_k_raw, hold_q_raw)

    print("\n--- Lipschitz Evaluation Results ---")
    print(f"Analytical Lipschitz Bound: {eval_results['l_analytical_bound']:.3f}")
    print(f"Empirical Maximum Lipschitz Ratio: {eval_results['l_empirical']:.3f}")
    for k_key, metrics in eval_results["results_by_k"].items():
        print(f"  {k_key}: Actual Entry: {metrics['actual_entry_ratio']*100:.1f}% | "
              f"Certifiable Entry by Eq.(1): {metrics['empirical_certified_ratio']*100:.1f}%")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    with open(os.path.join(OUT_DIR, "lipschitz_margin_results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    main()