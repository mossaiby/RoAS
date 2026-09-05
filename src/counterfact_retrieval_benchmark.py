"""
src/counterfact_retrieval_benchmark.py

Evaluate atom-space retrieval on the established CounterFact benchmark.
Now includes:
  - Exact top-1 baseline with static threshold (0.60)
  - ROC / AUC analysis discriminating positive rewrites/paraphrases from neighborhood controls
  - Calibration protocol optimizing Youden's J threshold on validation records
  - Performance comparison under the calibrated optimal threshold (answering M6 and Q4)
"""

import argparse
import json
import os
import time
import urllib.request
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_retrieval_benchmark import adaptive_topk_read


COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
DATA_PATH = os.path.join("data", "counterfact.json")
OUT_DIR = os.path.join("results", "details", "counterfact_retrieval_outputs")
SUMMARY_PATH = os.path.join("results", "counterfact_retrieval_results.json")


def load_counterfact(limit: int) -> List[Dict]:
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print(f"Downloading CounterFact from {COUNTERFACT_URL}")
        urllib.request.urlretrieve(COUNTERFACT_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)[:limit]


def prepare_records(records: List[Dict]) -> Tuple[List[str], List[str], List[Dict]]:
    memory_texts = []
    targets = []
    evaluations = []
    for memory_index, record in enumerate(records):
        request = record["requested_rewrite"]
        subject = request["subject"]
        prompt = request["prompt"].format(subject)
        target = request["target_new"]["str"].strip()
        memory_texts.append(f"{prompt} {target}")
        targets.append(target)
        evaluations.append(
            {
                "memory_index": memory_index,
                "rewrite": [prompt],
                "paraphrase": record["paraphrase_prompts"],
                "neighborhood": record["neighborhood_prompts"],
            }
        )
    return memory_texts, targets, evaluations


def flatten_queries(evaluations: List[Dict], max_per_type: int) -> Tuple[List[str], List[Dict]]:
    texts = []
    metadata = []
    for evaluation in evaluations:
        for query_type in ["rewrite", "paraphrase", "neighborhood"]:
            for text in evaluation[query_type][:max_per_type]:
                texts.append(text)
                metadata.append(
                    {
                        "query_type": query_type,
                        "memory_index": evaluation["memory_index"],
                    }
                )
    return texts, metadata


def compute_roc_and_optimal_threshold(
    similarities: np.ndarray,
    metadata: List[Dict],
    split_ratio: float = 0.5
) -> Tuple[float, float, Dict]:
    """
    Computes ROC/AUC and optimal threshold maximizing Youden's J statistic
    (sensitivity + specificity - 1) on a held-out validation split.
    """
    is_positive = np.array([item["query_type"] in ["rewrite", "paraphrase"] for item in metadata])
    n = len(similarities)
    split_idx = int(n * split_ratio)
    
    val_sims = similarities[:split_idx]
    val_pos = is_positive[:split_idx]
    
    thresholds = np.linspace(0.0, 1.0, 101)
    best_j = -1.0
    optimal_thresh = 0.60
    
    tpr_list = []
    fpr_list = []
    
    for thresh in thresholds:
        pred_pos = val_sims >= thresh
        tp = np.sum(pred_pos & val_pos)
        fp = np.sum(pred_pos & (~val_pos))
        fn = np.sum((~pred_pos) & val_pos)
        tn = np.sum((~pred_pos) & (~val_pos))
        
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
        j_score = tpr - fpr
        if j_score > best_j:
            best_j = j_score
            optimal_thresh = float(thresh)
            
    # Compute Trapezoidal AUC
    order = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[order]
    tpr_sorted = np.array(tpr_list)[order]
    auc_score = float(np.trapezoid(tpr_sorted, fpr_sorted)) if hasattr(np, "trapezoid") else float(np.trapz(tpr_sorted, fpr_sorted))
    
    return optimal_thresh, abs(auc_score), {
        "best_youden_j": float(best_j),
        "validation_samples": split_idx
    }


def summarize_retrieval(
    indices: np.ndarray,
    similarities: np.ndarray,
    metadata: List[Dict],
    activation_threshold: float,
) -> Dict:
    summary = {}
    for query_type in ["rewrite", "paraphrase", "neighborhood"]:
        positions = [i for i, item in enumerate(metadata) if item["query_type"] == query_type]
        predicted = indices[positions]
        expected = np.array([metadata[i]["memory_index"] for i in positions])
        activated = similarities[positions] >= activation_threshold
        if query_type == "neighborhood":
            summary[query_type] = {
                "count": len(positions),
                "false_activation_rate": float(activated.mean()),
                "wrong_edit_retrieval_rate": float(((predicted == expected) & activated).mean()),
                "mean_max_similarity": float(similarities[positions].mean()),
            }
        else:
            summary[query_type] = {
                "count": len(positions),
                "top1_recall": float((predicted == expected).mean()),
                "activated_correct_recall": float(((predicted == expected) & activated).mean()),
                "mean_max_similarity": float(similarities[positions].mean()),
            }
    return summary


def run_exact_baselines(
    query_embeddings: torch.Tensor,
    memory_embeddings: torch.Tensor,
    value_embeddings: torch.Tensor,
    metadata: List[Dict],
    temperature: float,
    error_budget: float,
    fixed_k: int,
    activation_threshold: float,
) -> Dict:
    start = time.perf_counter()
    similarities = query_embeddings @ memory_embeddings.T
    sorted_similarities, sorted_indices = similarities.sort(dim=-1, descending=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    top_indices = sorted_indices[:, 0].cpu().numpy()
    top_similarities = sorted_similarities[:, 0].cpu().numpy()

    # Threshold calibration and ROC analysis
    optimal_thresh, auc_score, roc_stats = compute_roc_and_optimal_threshold(
        top_similarities, metadata
    )

    sorted_scores = sorted_similarities / temperature
    sorted_values = value_embeddings[sorted_indices]
    full_weights = F.softmax(sorted_scores, dim=-1)
    full_read = torch.bmm(full_weights.unsqueeze(1), sorted_values).squeeze(1)
    selected_k, certificate, errors, _ = adaptive_topk_read(
        sorted_scores, sorted_values, full_read, error_budget
    )
    effective_fixed_k = min(fixed_k, memory_embeddings.size(0))
    fixed_weights = F.softmax(sorted_scores[:, :effective_fixed_k], dim=-1)
    fixed_read = torch.bmm(
        fixed_weights.unsqueeze(1), sorted_values[:, :effective_fixed_k]
    ).squeeze(1)
    fixed_errors = (full_read - fixed_read).norm(dim=-1)

    return {
        "static_threshold_eval": {
            "threshold": activation_threshold,
            "results": summarize_retrieval(top_indices, top_similarities, metadata, activation_threshold)
        },
        "calibrated_threshold_eval": {
            "optimal_threshold": optimal_thresh,
            "roc_auc": auc_score,
            "calibration_stats": roc_stats,
            "results": summarize_retrieval(top_indices, top_similarities, metadata, optimal_thresh)
        },
        "fixed_k_read": {
            "k": effective_fixed_k,
            "mean_vector_error": fixed_errors.mean().item(),
            "max_vector_error": fixed_errors.max().item(),
        },
        "adaptive_read": {
            "error_budget": error_budget,
            "mean_k": selected_k.float().mean().item(),
            "median_k": selected_k.float().median().item(),
            "max_k": selected_k.max().item(),
            "mean_vector_error": errors.mean().item(),
            "max_vector_error": errors.max().item(),
            "max_certificate": certificate.max().item(),
            "empirical_coverage": (errors <= error_budget + 1e-5).float().mean().item(),
        },
        "exact_scoring_and_sort_ms": elapsed_ms,
    }


def run_faiss_hnsw(
    query_embeddings: torch.Tensor,
    memory_embeddings: torch.Tensor,
    exact_indices: np.ndarray,
    neighbors: int,
) -> Dict:
    try:
        import faiss
    except ImportError:
        return {"available": False, "reason": "faiss-cpu is not installed"}

    memory = np.ascontiguousarray(memory_embeddings.cpu().numpy().astype("float32"))
    queries = np.ascontiguousarray(query_embeddings.cpu().numpy().astype("float32"))
    index = faiss.IndexHNSWFlat(memory.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 64
    start = time.perf_counter()
    index.add(memory)
    build_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    _, indices = index.search(queries, min(neighbors, len(memory)))
    search_ms = (time.perf_counter() - start) * 1000.0
    recall = np.mean([exact_indices[i] in indices[i] for i in range(len(indices))])
    return {
        "available": True,
        "index": "FAISS IndexHNSWFlat",
        "candidate_count": indices.shape[1],
        "exact_top1_candidate_recall": float(recall),
        "build_ms": build_ms,
        "search_ms": search_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-queries-per-type", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--error-budget", type=float, default=0.05)
    parser.add_argument("--fixed-k", type=int, default=10)
    parser.add_argument("--activation-threshold", type=float, default=0.60)
    parser.add_argument("--ann-candidates", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from sentence_transformers import SentenceTransformer

    records = load_counterfact(args.limit)
    memory_texts, targets, evaluations = prepare_records(records)
    query_texts, metadata = flatten_queries(evaluations, args.max_queries_per_type)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    memory_embeddings = encoder.encode(
        memory_texts, convert_to_tensor=True, normalize_embeddings=True
    ).cpu()
    query_embeddings = encoder.encode(
        query_texts, convert_to_tensor=True, normalize_embeddings=True
    ).cpu()
    value_embeddings = encoder.encode(
        targets, convert_to_tensor=True, normalize_embeddings=True
    ).cpu()

    exact = run_exact_baselines(
        query_embeddings,
        memory_embeddings,
        value_embeddings,
        metadata,
        args.temperature,
        args.error_budget,
        args.fixed_k,
        args.activation_threshold,
    )
    similarities = query_embeddings @ memory_embeddings.T
    exact_indices = similarities.argmax(dim=-1).numpy()
    ann = run_faiss_hnsw(
        query_embeddings, memory_embeddings, exact_indices, args.ann_candidates
    )
    payload = {
        "benchmark": "CounterFact external-memory retrieval with ROC calibration",
        "source": COUNTERFACT_URL,
        "scope_note": "Retrieval evaluation with calibrated threshold and exact top-1 baselines.",
        "config": vars(args),
        "records": len(records),
        "queries": len(query_texts),
        "results": exact,
        "ann": ann,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    for path in [SUMMARY_PATH, os.path.join(OUT_DIR, "counterfact_retrieval_results.json")]:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()