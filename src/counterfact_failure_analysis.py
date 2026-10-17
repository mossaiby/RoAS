"""
src/counterfact_failure_analysis.py

Revision analysis of the CounterFact-500 retrieval gate (Section 6.4, Table 4).

Answers, in order:
  R1.6  Explicit counting rules. Every query is assigned to exactly one outcome bucket
        (roas_analysis.POSITIVE_BUCKETS / NEGATIVE_BUCKETS); Table 4's "activated recall"
        is the activated_correct bucket.
  R1.2  Which paraphrases are retrieved correctly but rejected by the threshold, and why
        (prefix distractors, lexical overlap, similarity/margin distributions, examples).
  R1.5  Which neighborhood prompts account for the extra false activations when theta moves
        from 0.60 to theta* (same relation as the retrieved record? own record?).
  R1.7  Effect of the stored-key construction: prompt+target (submitted) versus prompt-only
        keys, each with its own held-out calibration.
  R3.2  Coverage of the query-side certified radius (Proposition in Section 4), which
        replaces the anchor-based Lipschitz condition.
  Also: a rank-margin gate (s_(1) - s_(2) >= gamma), calibrated the same way, as a
        diagnostic of whether rank information recovers the rejected-but-correct queries.

Protocol is identical to src/counterfact_threshold_calibration.py: records 500-999 calibrate,
records 0-499 evaluate, Youden's J on a 161-point grid over [0.1, 0.9], frozen
all-MiniLM-L6-v2 with unit-normalized embeddings, two paraphrases and two neighborhood
prompts per record.

Writes results/counterfact_failure_analysis.json and
results/details/counterfact_failure_outputs/{counterfact_failure_analysis.json, examples.json}.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import numpy as np

from counterfact_retrieval_benchmark import load_counterfact
from roas_analysis import (
    bucket_queries,
    bucket_rates,
    certified_radii,
    contains_ci,
    jaccard,
    query_displacements,
    roc_auc,
    split_prefix,
    youden_threshold,
)

OUT_DIR = os.path.join("results", "details", "counterfact_failure_outputs")
SUMMARY_PATH = os.path.join("results", "counterfact_failure_analysis.json")
GRID = np.linspace(0.1, 0.9, 161)
MARGIN_GRID = np.linspace(0.0, 0.5, 201)
STATIC_THETA = 0.60
EXAMPLES_PER_BUCKET = 8
SEED = 42


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_split(records: List[Dict], key_mode: str, per_type: int = 2) -> Dict:
    """Keys, canonical queries and flattened evaluation queries for one split."""
    keys, canon, rel, subj, tgt = [], [], [], [], []
    q_text, q_type, q_rec = [], [], []
    for i, r in enumerate(records):
        req = r["requested_rewrite"]
        prompt = req["prompt"].format(req["subject"])
        target = req["target_new"]["str"].strip()
        if key_mode == "prompt_target":
            keys.append(f"{prompt} {target}")
        elif key_mode == "prompt_only":
            keys.append(prompt)
        else:
            raise ValueError(key_mode)
        canon.append(prompt)
        rel.append(req.get("relation_id", ""))
        subj.append(req["subject"])
        tgt.append(target)
        for qt, texts in (("rewrite", [prompt]),
                          ("paraphrase", r["paraphrase_prompts"][:per_type]),
                          ("neighborhood", r["neighborhood_prompts"][:per_type])):
            for t in texts:
                q_text.append(t)
                q_type.append(qt)
                q_rec.append(i)
    return {
        "key_text": keys, "canonical_text": canon, "relation": rel, "subject": subj, "target": tgt,
        "query_text": q_text, "query_type": np.array(q_type), "query_record": np.array(q_rec),
    }


def encode(encoder, texts: Sequence[str]) -> np.ndarray:
    emb = encoder.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True,
                         batch_size=64, show_progress_bar=False)
    return emb.astype(np.float64)


def top2(sim: np.ndarray):
    order = np.argsort(-sim, axis=1)[:, :2]
    rows = np.arange(sim.shape[0])
    return order[:, 0], sim[rows, order[:, 0]], sim[rows, order[:, 1]]


def score_split(encoder, split: Dict, query_override: Dict[int, str] | None = None) -> Dict:
    texts = list(split["query_text"])
    if query_override:
        for idx, t in query_override.items():
            texts[idx] = t
    k = encode(encoder, split["key_text"])
    q = encode(encoder, texts)
    sim = q @ k.T
    top1, s1, s2 = top2(sim)
    rows = np.arange(len(texts))
    own = sim[rows, split["query_record"]]
    return {"keys": k, "queries": q, "top1": top1, "s1": s1, "s2": s2, "own": own}


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def calibrate(scores: np.ndarray, qtypes: np.ndarray, grid: np.ndarray, positives=("rewrite", "paraphrase")):
    labels = np.isin(qtypes, positives)
    theta, j = youden_threshold(scores, labels, grid)
    return theta, j, roc_auc(scores, labels)


def operating_point(split: Dict, sc: Dict, theta: float, gate: str = "absolute") -> Dict:
    stat = sc["s1"] if gate == "absolute" else sc["s1"] - sc["s2"]
    buckets = bucket_queries(split["query_type"], split["query_record"], sc["top1"], stat, theta)
    rates = bucket_rates(buckets, split["query_type"])
    qt = split["query_type"]
    rates["exact_top1_recall"] = {
        t: float(np.mean(sc["top1"][qt == t] == split["query_record"][qt == t])) for t in ("rewrite", "paraphrase")
    }
    rates["threshold"] = float(theta)
    return {"rates": rates, "buckets": buckets}


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def describe(values: np.ndarray) -> Dict[str, float]:
    if len(values) == 0:
        return {"n": 0}
    return {"n": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))}


def analyse(encoder, calib_records, test_records) -> Dict:
    rng = np.random.default_rng(SEED)
    out: Dict = {}

    # ---------------- Submitted key construction (prompt + target) ----------------
    cal = build_split(calib_records, "prompt_target")
    tst = build_split(test_records, "prompt_target")
    cal_sc = score_split(encoder, cal)
    tst_sc = score_split(encoder, tst)

    theta_star, j_star, auc_cal = calibrate(cal_sc["s1"], cal["query_type"], GRID)
    _, _, auc_test = calibrate(tst_sc["s1"], tst["query_type"], GRID)
    # Paraphrase-only ROC (rewrites trivially high; excluding them is the harder discrimination)
    para_mask = tst["query_type"] != "rewrite"
    auc_para_only = roc_auc(tst_sc["s1"][para_mask], tst["query_type"][para_mask] == "paraphrase")

    static = operating_point(tst, tst_sc, STATIC_THETA)
    calib = operating_point(tst, tst_sc, theta_star)
    out["submitted_keys"] = {
        "key_construction": "rewrite prompt + ' ' + target_new",
        "theta_star": theta_star, "youden_j_calibration": j_star,
        "auc_calibration_split": auc_cal, "auc_evaluation_split": auc_test,
        "auc_evaluation_paraphrase_vs_neighborhood_only": auc_para_only,
        "at_0.60": static["rates"], "at_theta_star": calib["rates"],
    }

    # ---------------- R1.2: retrieved but rejected paraphrases ----------------
    qt, rec = tst["query_type"], tst["query_record"]
    b_star = np.array(calib["buckets"])
    para = np.where(qt == "paraphrase")[0]
    prefix_len, core_text, overlap, subj_in = {}, {}, {}, {}
    for i in para:
        pre, core = split_prefix(tst["query_text"][i])
        prefix_len[i] = len(pre.split())
        core_text[i] = core
        overlap[i] = jaccard(tst["query_text"][i], tst["key_text"][rec[i]])
        subj_in[i] = contains_ci(tst["query_text"][i], tst["subject"][rec[i]])

    out["paraphrase_prefix_rate_all"] = float(np.mean([prefix_len[i] > 0 for i in para]))
    groups = {name: [i for i in para if b_star[i] == name]
              for name in ("activated_correct", "retrieved_rejected", "wrong_record_activated", "wrong_record_rejected")}
    out["paraphrase_failure_profile"] = {
        name: {
            "count": len(idx),
            "has_prefix_rate": float(np.mean([prefix_len[i] > 0 for i in idx])) if idx else None,
            "prefix_words": describe(np.array([prefix_len[i] for i in idx])),
            "token_jaccard_with_key": describe(np.array([overlap[i] for i in idx])),
            "subject_mentioned_rate": float(np.mean([subj_in[i] for i in idx])) if idx else None,
            "top1_similarity": describe(tst_sc["s1"][idx]),
            "rank_margin_s1_minus_s2": describe((tst_sc["s1"] - tst_sc["s2"])[idx]),
        }
        for name, idx in groups.items()
    }

    # Counterfactual: strip the distractor prefix from paraphrases and re-score (same keys, same theta*)
    override = {i: core_text[i] for i in para}
    tst_core = score_split(encoder, tst, query_override=override)
    core_point = operating_point(tst, tst_core, theta_star)
    rr = groups["retrieved_rejected"]
    out["prefix_stripped_paraphrases"] = {
        "note": "Paraphrase prompts reduced to their final sentence; keys and theta* unchanged.",
        "at_theta_star": core_point["rates"]["paraphrase"],
        "exact_top1_recall": core_point["rates"]["exact_top1_recall"]["paraphrase"],
        "recovered_fraction_of_retrieved_rejected": float(np.mean(tst_core["s1"][rr] >= theta_star)) if rr else None,
    }

    # ---------------- R1.5: extra false activations between 0.60 and theta* ----------------
    neigh = np.where(qt == "neighborhood")[0]
    fires_static = tst_sc["s1"] >= STATIC_THETA
    fires_star = tst_sc["s1"] >= theta_star
    extra = [i for i in neigh if fires_star[i] and not fires_static[i]]
    all_fa = [i for i in neigh if fires_star[i]]

    def fa_profile(idx):
        if not idx:
            return {"count": 0}
        retrieved = tst_sc["top1"][idx]
        own = rec[idx]
        same_rel = [tst["relation"][r] == tst["relation"][o] for r, o in zip(retrieved, own)]
        tgt_true = [test_records[o]["requested_rewrite"]["target_true"]["str"] for o in own]
        return {
            "count": len(idx),
            "rate_of_all_neighborhood": len(idx) / len(neigh),
            "retrieved_own_record_rate": float(np.mean(retrieved == own)),
            "retrieved_same_relation_rate": float(np.mean(same_rel)),
            "retrieved_other_relation_rate": float(1 - np.mean(same_rel)),
            "query_mentions_true_object_rate": float(np.mean([contains_ci(tst["query_text"][i], t) for i, t in zip(idx, tgt_true)])),
            "top1_similarity": describe(tst_sc["s1"][idx]),
        }

    rel_counts: Dict[str, int] = {}
    for i in extra:
        rel_counts[tst["relation"][rec[i]]] = rel_counts.get(tst["relation"][rec[i]], 0) + 1
    out["neighborhood_false_activations"] = {
        "all_at_theta_star": fa_profile(all_fa),
        "additional_between_0.60_and_theta_star": fa_profile(extra),
        "additional_by_relation_id": dict(sorted(rel_counts.items(), key=lambda kv: -kv[1])),
        "baseline_same_relation_rate_random_pair": float(np.mean(
            [tst["relation"][a] == tst["relation"][b] for a, b in rng.integers(0, len(test_records), size=(5000, 2))])),
    }

    # ---------------- R1.7: key construction ablation ----------------
    cal_p = build_split(calib_records, "prompt_only")
    tst_p = build_split(test_records, "prompt_only")
    cal_p_sc = score_split(encoder, cal_p)
    tst_p_sc = score_split(encoder, tst_p)
    theta_p, _, auc_p_cal = calibrate(cal_p_sc["s1"], cal_p["query_type"], GRID)
    _, _, auc_p_test = calibrate(tst_p_sc["s1"], tst_p["query_type"], GRID)
    out["key_ablation_prompt_only"] = {
        "theta_star": theta_p, "auc_calibration_split": auc_p_cal, "auc_evaluation_split": auc_p_test,
        "at_theta_star": operating_point(tst_p, tst_p_sc, theta_p)["rates"],
        "mean_own_similarity_paraphrase": {
            "prompt_target": float(np.mean(tst_sc["own"][qt == "paraphrase"])),
            "prompt_only": float(np.mean(tst_p_sc["own"][qt == "paraphrase"])),
        },
    }

    # ---------------- Rank-margin gate (diagnostic) ----------------
    cal_margin = cal_sc["s1"] - cal_sc["s2"]
    gamma, _, auc_margin_cal = calibrate(cal_margin, cal["query_type"], MARGIN_GRID)
    _, _, auc_margin_test = calibrate(tst_sc["s1"] - tst_sc["s2"], tst["query_type"], MARGIN_GRID)
    out["rank_margin_gate"] = {
        "gamma_star": gamma, "auc_calibration_split": auc_margin_cal, "auc_evaluation_split": auc_margin_test,
        "at_gamma_star": operating_point(tst, tst_sc, gamma, gate="margin")["rates"],
    }

    # ---------------- R3.2: query-side certified radius ----------------
    canon = encode(encoder, tst["canonical_text"])
    radii = certified_radii(canon, tst_sc["keys"], theta=theta_star)
    disp = query_displacements(tst_sc["queries"], canon, rec)
    disp_core = query_displacements(tst_core["queries"], canon, rec)
    cert = {}
    for name, d, sc in (("as_given", disp, tst_sc), ("prefix_stripped", disp_core, tst_core)):
        for t in ("rewrite", "paraphrase"):
            m = qt == t
            r_top, r_act = radii["top1"][rec[m]], radii["activation"][rec[m]]
            cert[f"{name}/{t}"] = {
                "certified_top1_rate": float(np.mean(d[m] < r_top)),
                "actual_top1_rate": float(np.mean(sc["top1"][m] == rec[m])),
                # Activation is certified only inside the top-1 ball as well (Proposition 2):
                # records that are not top-1 at their canonical query have radius 0 and certify nothing.
                "certified_activation_rate": float(np.mean((d[m] < r_top) & (d[m] <= r_act))),
                "actual_activation_rate": float(np.mean((sc["top1"][m] == rec[m]) & (sc["s1"][m] >= theta_star))),
                "certificate_violations": int(np.sum((d[m] < r_top) & (sc["top1"][m] != rec[m]))
                                              + np.sum((d[m] < r_top) & (d[m] <= r_act) & (sc["s1"][m] < theta_star))),
                "displacement": describe(d[m]),
            }
    out["query_side_certificate"] = {
        "records_top1_at_canonical_query": float(np.mean(radii["is_top1_at_canonical"])),
        "top1_radius": describe(radii["top1"]),
        "activation_radius": describe(radii["activation"]),
        "coverage": cert,
    }

    # ---------------- Examples ----------------
    examples = {}
    def pick(idx, n):
        idx = list(idx)
        return [idx[i] for i in rng.choice(len(idx), size=min(n, len(idx)), replace=False)] if idx else []
    for name, idx in groups.items():
        examples[f"paraphrase/{name}"] = [{
            "query": tst["query_text"][i], "core": core_text[i], "stored_key": tst["key_text"][rec[i]],
            "retrieved_key": tst["key_text"][tst_sc["top1"][i]],
            "top1_similarity": round(float(tst_sc["s1"][i]), 4), "own_similarity": round(float(tst_sc["own"][i]), 4),
            "core_top1_similarity": round(float(tst_core["s1"][i]), 4),
        } for i in pick(idx, EXAMPLES_PER_BUCKET)]
    examples["neighborhood/additional_false_activation"] = [{
        "query": tst["query_text"][i], "own_record_key": tst["key_text"][rec[i]],
        "retrieved_key": tst["key_text"][tst_sc["top1"][i]],
        "own_relation": tst["relation"][rec[i]], "retrieved_relation": tst["relation"][tst_sc["top1"][i]],
        "top1_similarity": round(float(tst_sc["s1"][i]), 4),
    } for i in pick(extra, 2 * EXAMPLES_PER_BUCKET)]
    return out, examples


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    return p.parse_args()


def main():
    args = parse_args()
    from sentence_transformers import SentenceTransformer

    records = load_counterfact(1000)
    test_records, calib_records = records[:500], records[500:1000]
    encoder = SentenceTransformer(args.encoder)
    summary, examples = analyse(encoder, calib_records, test_records)
    summary["config"] = {"encoder": args.encoder, "calibration_records": "500-999", "evaluation_records": "0-499",
                         "grid": [0.1, 0.9, 161], "static_threshold": STATIC_THETA, "seed": SEED}

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    for path in (SUMMARY_PATH, os.path.join(OUT_DIR, "counterfact_failure_analysis.json")):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "examples.json"), "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
