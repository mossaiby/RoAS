"""
src/counterfact_threshold_calibration.py

CounterFact-500 ROC Analysis and Held-Out Threshold Calibration.
Resolves Reviewer M6 & Question 4:
  - Replaces the fixed 0.60 threshold with an empirical ROC/AUC curve.
  - Calibrates optimal threshold tau* on held-out CounterFact records (500-999).
  - Evaluates test performance on records 0-499.
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from counterfact_retrieval_benchmark import load_counterfact, prepare_records, flatten_queries

OUT_DIR = os.path.join("results", "details", "counterfact_calibration_outputs")
SUMMARY_PATH = os.path.join("results", "counterfact_calibration_results.json")
TOTAL_RECORDS_NEEDED = 1000
TEST_SPLIT_SIZE = 500


def extract_split_features(
    records: List[Dict], encoder: SentenceTransformer
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    memory_texts, targets, evaluations = prepare_records(records)
    query_texts, metadata = flatten_queries(evaluations, max_per_type=2)

    mem_emb = encoder.encode(memory_texts, convert_to_tensor=True, normalize_embeddings=True).cpu()
    query_emb = encoder.encode(query_texts, convert_to_tensor=True, normalize_embeddings=True).cpu()
    return mem_emb, query_emb, metadata


def compute_roc_metrics(
    similarities: np.ndarray, is_positive: np.ndarray, thresholds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Computes TPR, FPR, AUC, and optimal threshold via Youden's J."""
    tpr_list, fpr_list = [], []
    num_pos = np.sum(is_positive)
    num_neg = len(is_positive) - num_pos

    for tau in thresholds:
        pred_pos = similarities >= tau
        tp = np.sum(pred_pos & is_positive)
        fp = np.sum(pred_pos & (~is_positive))
        tpr_list.append(tp / num_pos if num_pos > 0 else 0.0)
        fpr_list.append(fp / num_neg if num_neg > 0 else 0.0)

    tpr = np.array(tpr_list)
    fpr = np.array(fpr_list)

    # Sort descending by FPR to integrate AUC via trapezoidal rule
    sort_idx = np.argsort(fpr)
    auc = float(np.trapezoid(tpr[sort_idx], fpr[sort_idx]))

    # Youden's J statistic = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_tau = float(thresholds[best_idx])

    return tpr, fpr, auc, optimal_tau


def evaluate_at_threshold(
    sims: np.ndarray,
    top_indices: np.ndarray,
    metadata: List[Dict],
    threshold: float
) -> Dict:
    results = {}
    for qtype in ["rewrite", "paraphrase", "neighborhood"]:
        pos = [i for i, item in enumerate(metadata) if item["query_type"] == qtype]
        pred = top_indices[pos]
        expected = np.array([metadata[i]["memory_index"] for i in pos])
        activated = sims[pos] >= threshold

        if qtype == "neighborhood":
            results[qtype] = {
                "count": len(pos),
                "false_activation_rate": float(activated.mean()),
                "wrong_edit_retrieval_rate": float(((pred == expected) & activated).mean()),
                "mean_similarity": float(sims[pos].mean()),
            }
        else:
            results[qtype] = {
                "count": len(pos),
                "top1_recall": float((pred == expected).mean()),
                "activated_correct_recall": float(((pred == expected) & activated).mean()),
                "mean_similarity": float(sims[pos].mean()),
            }
    return results


def run_calibration_study():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print(f"=== Running CounterFact ROC Analysis & Calibration on 1000 Records ===")

    all_records = load_counterfact(TOTAL_RECORDS_NEEDED)
    test_records = all_records[:TEST_SPLIT_SIZE]
    calib_records = all_records[TEST_SPLIT_SIZE:TOTAL_RECORDS_NEEDED]

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 1. Calibration Split (records 500-999)
    print("Encoding calibration split (records 500-999)...")
    c_mem, c_query, c_meta = extract_split_features(calib_records, encoder)
    c_sims_matrix = (c_query @ c_mem.T).numpy()
    c_top_sims = np.max(c_sims_matrix, axis=1)

    c_is_pos = np.array([item["query_type"] in ["rewrite", "paraphrase"] for item in c_meta])
    thresholds = np.linspace(0.1, 0.9, 161)

    c_tpr, c_fpr, calib_auc, optimal_tau = compute_roc_metrics(c_top_sims, c_is_pos, thresholds)
    print(f"Calibration AUC: {calib_auc:.4f} | Optimal Tau (Youden's J): {optimal_tau:.3f}")

    # 2. Test Split (records 0-499)
    print("Encoding test split (records 0-499)...")
    t_mem, t_query, t_meta = extract_split_features(test_records, encoder)
    t_sims_matrix = (t_query @ t_mem.T).numpy()
    t_top_indices = np.argmax(t_sims_matrix, axis=1)
    t_top_sims = np.max(t_sims_matrix, axis=1)

    t_is_pos = np.array([item["query_type"] in ["rewrite", "paraphrase"] for item in t_meta])
    t_tpr, t_fpr, test_auc, _ = compute_roc_metrics(t_top_sims, t_is_pos, thresholds)

    # Compare fixed 0.60 vs Calibrated optimal_tau
    strawman_results = evaluate_at_threshold(t_top_sims, t_top_indices, t_meta, threshold=0.60)
    calibrated_results = evaluate_at_threshold(t_top_sims, t_top_indices, t_meta, threshold=optimal_tau)

    print("\n--- Test Set Comparison (500 Records) ---")
    print(f"Fixed 0.60: Paraphrase Activated Recall = {strawman_results['paraphrase']['activated_correct_recall']*100:.1f}% | "
          f"Neighborhood False Activation = {strawman_results['neighborhood']['false_activation_rate']*100:.1f}%")
    print(f"Calibrated ({optimal_tau:.2f}): Paraphrase Activated Recall = {calibrated_results['paraphrase']['activated_correct_recall']*100:.1f}% | "
          f"Neighborhood False Activation = {calibrated_results['neighborhood']['false_activation_rate']*100:.1f}%")

    # 3. Plot ROC & Calibration curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ROC Curve
    ax1.plot(t_fpr, t_tpr, color="#1f77b4", lw=2, label=f"ROC curve (AUC = {test_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--")
    # Mark the two operating points
    idx_straw = np.argmin(np.abs(thresholds - 0.60))
    idx_opt = np.argmin(np.abs(thresholds - optimal_tau))
    ax1.scatter([t_fpr[idx_straw]], [t_tpr[idx_straw]], color="red", zorder=5, label=r"Fixed $\tau=0.60$")
    ax1.scatter([t_fpr[idx_opt]], [t_tpr[idx_opt]], color="green", zorder=5, label=rf"Calibrated $\tau={optimal_tau:.2f}$")
    ax1.set_xlabel("False Positive Rate (Neighborhoods)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Rewrites + Paraphrases)", fontsize=11)
    ax1.set_title("CounterFact Retrieval ROC Curve", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right")

    # Trade-off vs Threshold
    para_recalls = []
    neigh_false = []
    for tau in thresholds:
        res = evaluate_at_threshold(t_top_sims, t_top_indices, t_meta, threshold=tau)
        para_recalls.append(res["paraphrase"]["activated_correct_recall"] * 100)
        neigh_false.append(res["neighborhood"]["false_activation_rate"] * 100)

    ax2.plot(thresholds, para_recalls, color="#2ca02c", lw=2, label="Paraphrase Activated Recall (%)")
    ax2.plot(thresholds, neigh_false, color="#d62728", lw=2, label="Neighborhood False Activation (%)")
    ax2.axvline(x=0.60, color="red", linestyle=":", label=r"Strawman $\tau=0.60$")
    ax2.axvline(x=optimal_tau, color="green", linestyle="--", label=rf"Calibrated $\tau={optimal_tau:.2f}$")
    ax2.set_xlabel("Cosine Activation Threshold", fontsize=11)
    ax2.set_ylabel("Rate (%)", fontsize=11)
    ax2.set_title("Paraphrase Recall vs. False Activation Trade-off", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "roc_calibration_curve.png"), dpi=200)
    if os.path.exists("paper/figures"):
        fig.savefig("paper/figures/roc_calibration_curve.png", dpi=200)
    plt.close()

    summary = {
        "calibration_set_size": len(calib_records),
        "test_set_size": len(test_records),
        "calibration_auc": calib_auc,
        "test_auc": test_auc,
        "optimal_threshold": optimal_tau,
        "fixed_0_60_results": strawman_results,
        "calibrated_results": calibrated_results,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "counterfact_calibration_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCalibration study complete. Outputs saved to '{OUT_DIR}/' and '{SUMMARY_PATH}'.")


if __name__ == "__main__":
    run_calibration_study()