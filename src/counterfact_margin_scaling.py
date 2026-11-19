"""
src/counterfact_margin_scaling.py

Store-size sweep for the rank-margin gate (follow-up to Section 6.4).

At N = 500 the rank-margin gate (fire iff s_(1) - s_(2) >= gamma) beats the calibrated absolute
gate (fire iff s_(1) >= theta) on both axes: 70.8% paraphrase activation at 7.0% neighborhood
false activation, against 53.2% at 16.1%. The margin depends on the rest of the store, so the
open question is whether this survives a denser store. This script answers it on CounterFact
with the protocol of src/counterfact_failure_analysis.py, varying only the store size N.

For every N and every record ordering:
  * evaluation store = the first N records of the ordering, calibration store = the next N
    (disjoint), so N = 500 under the file order is exactly the setting of the paper;
  * keys = MiniLM("rewrite prompt + target_new"), unit-normalized; queries = the rewrite prompt,
    two paraphrases and two neighborhood prompts per record, scored against all N keys;
  * theta*(N) and gamma*(N) by Youden's J on the calibration store (same grids as the paper);
  * reported on the evaluation store: outcome buckets, exact top-1 recall, held-out AUC.
Also reported:
  * transfer: theta* and gamma* calibrated at N = 500 applied unchanged at every N
    (does the threshold have to be refit as the store grows?);
  * matched false activation: for each gate, the threshold whose calibration-store false
    activation is at most 5% / 10%, and the activation it gives on the evaluation store
    (compares the gates at equal cost instead of at their Youden points);
  * near-duplicate records: share of evaluation records whose (subject, relation) occurs in
    another record of the same store, which shrinks margins for a reason other than density.

Embeddings of all records are computed once and cached in data/ (git-ignored).

Usage (from the repository root):
    python src/counterfact_margin_scaling.py                      # file order only
    python src/counterfact_margin_scaling.py --orderings 3        # + 3 random orderings

Writes results/counterfact_margin_scaling.json and
results/details/counterfact_margin_scaling_outputs/{counterfact_margin_scaling.json, margin_scaling.png}.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from typing import Dict, List

import numpy as np
import torch

from counterfact_failure_analysis import GRID, MARGIN_GRID
from counterfact_retrieval_benchmark import load_counterfact
from roas_analysis import bucket_queries, bucket_rates, roc_auc, youden_threshold

OUT_DIR = os.path.join("results", "details", "counterfact_margin_scaling_outputs")
SUMMARY_PATH = os.path.join("results", "counterfact_margin_scaling.json")
CACHE_PATH = os.path.join("data", "counterfact_minilm_embeddings.npz")
SIZES = (500, 1000, 2000, 5000, 10000)
FA_TARGETS = (0.05, 0.10)
QTYPES = ("rewrite", "paraphrase", "paraphrase", "neighborhood", "neighborhood")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Embeddings for every record (computed once)
# ---------------------------------------------------------------------------

def record_texts(r: Dict):
    req = r["requested_rewrite"]
    prompt = req["prompt"].format(req["subject"])
    key = f"{prompt} {req['target_new']['str'].strip()}"
    para = (r["paraphrase_prompts"] + ["", ""])[:2]
    neigh = (r["neighborhood_prompts"] + ["", ""])[:2]
    return key, [prompt, *para, *neigh]


def embeddings(records: List[Dict], encoder_name: str):
    if os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH)
        if z["keys"].shape[0] == len(records) and str(z["encoder"]) == encoder_name:
            return z["keys"], z["queries"]
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(encoder_name, device=DEVICE)
    keys, queries = [], []
    for r in records:
        k, q = record_texts(r)
        keys.append(k)
        queries.extend(q)
    t0 = time.perf_counter()
    kv = enc.encode(keys, convert_to_numpy=True, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    qv = enc.encode(queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    qv = qv.reshape(len(records), 5, -1)
    print(f"Encoded {len(keys)} keys and {len(queries)} queries in {time.perf_counter() - t0:.0f}s")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez(CACHE_PATH, keys=kv.astype(np.float32), queries=qv.astype(np.float32), encoder=encoder_name)
    return kv.astype(np.float32), qv.astype(np.float32)


# ---------------------------------------------------------------------------
# Scoring one store
# ---------------------------------------------------------------------------

def score_store(keys_all, queries_all, idx: np.ndarray, records: List[Dict]) -> Dict:
    """Top-2 scores of every query of the store against the store's N keys."""
    k = torch.from_numpy(keys_all[idx]).to(DEVICE)
    q = torch.from_numpy(queries_all[idx].reshape(-1, keys_all.shape[1])).to(DEVICE)
    top1, s1, s2 = [], [], []
    for start in range(0, q.shape[0], 8192):
        sim = q[start:start + 8192] @ k.T
        vals, ind = torch.topk(sim, 2, dim=1)
        top1.append(ind[:, 0].cpu()); s1.append(vals[:, 0].cpu()); s2.append(vals[:, 1].cpu())
    n = len(idx)
    qtype = np.array(QTYPES * n)
    qrec = np.repeat(np.arange(n), 5)
    # Records with fewer than two paraphrase / neighborhood prompts leave empty slots: drop them.
    present = np.array([bool(t) for i in idx for t in record_texts(records[i])[1]])
    return {
        "top1": torch.cat(top1).numpy()[present], "s1": torch.cat(s1).double().numpy()[present],
        "s2": torch.cat(s2).double().numpy()[present], "qtype": qtype[present], "qrec": qrec[present],
    }


def stat_of(sc: Dict, gate: str) -> np.ndarray:
    return sc["s1"] if gate == "absolute" else sc["s1"] - sc["s2"]


def calibrate(sc: Dict, gate: str):
    grid = GRID if gate == "absolute" else MARGIN_GRID
    labels = np.isin(sc["qtype"], ("rewrite", "paraphrase"))
    t, j = youden_threshold(stat_of(sc, gate), labels, grid)
    return t, j


def operating_point(sc: Dict, gate: str, t: float) -> Dict:
    b = bucket_queries(sc["qtype"], sc["qrec"], sc["top1"], stat_of(sc, gate), t)
    r = bucket_rates(b, sc["qtype"])
    neigh = r["neighborhood"]
    return {
        "threshold": float(t),
        "paraphrase_activation": r["paraphrase"]["activated_correct"],
        "paraphrase_retrieved_rejected": r["paraphrase"]["retrieved_rejected"],
        "rewrite_activation": r["rewrite"]["activated_correct"],
        "neighborhood_false_activation": neigh["false_activation_own_record"] + neigh["false_activation_other_record"],
        "neighborhood_wrong_edit": neigh["false_activation_own_record"],
    }


def threshold_at_fa(sc: Dict, gate: str, target: float) -> float:
    """Smallest grid threshold whose neighborhood false activation is at most `target`."""
    grid = GRID if gate == "absolute" else MARGIN_GRID
    s = stat_of(sc, gate)[sc["qtype"] == "neighborhood"]
    for t in grid:
        if np.mean(s >= t) <= target:
            return float(t)
    return float(grid[-1])


def duplicate_rate(records: List[Dict], idx: np.ndarray) -> float:
    pairs = [(records[i]["requested_rewrite"]["subject"].strip().lower(),
              records[i]["requested_rewrite"].get("relation_id", "")) for i in idx]
    c = Counter(pairs)
    return float(np.mean([c[p] > 1 for p in pairs]))


def evaluate(keys_all, queries_all, records, order: np.ndarray, sizes) -> List[Dict]:
    rows, ref = [], {}
    for n in sizes:
        ev, ca = order[:n], order[n:2 * n]
        sc_ev, sc_ca = score_store(keys_all, queries_all, ev, records), score_store(keys_all, queries_all, ca, records)
        para = sc_ev["qtype"] == "paraphrase"
        pos = np.isin(sc_ev["qtype"], ("rewrite", "paraphrase"))
        correct = sc_ev["top1"] == sc_ev["qrec"]
        row = {"N": n, "duplicate_subject_relation_rate": duplicate_rate(records, ev),
               "paraphrase_exact_top1_recall": float(np.mean(correct[para])),
               "median_margin": {
                   "paraphrase_correct": float(np.median((sc_ev["s1"] - sc_ev["s2"])[para & correct])),
                   "neighborhood": float(np.median((sc_ev["s1"] - sc_ev["s2"])[sc_ev["qtype"] == "neighborhood"]))},
               "gates": {}}
        for gate in ("absolute", "margin"):
            t, j = calibrate(sc_ca, gate)
            if n == sizes[0]:
                ref[gate] = t
            g = {"youden": operating_point(sc_ev, gate, t), "youden_j_calibration": j,
                 "auc_evaluation": roc_auc(stat_of(sc_ev, gate), pos),
                 "transfer_from_N500": operating_point(sc_ev, gate, ref[gate])}
            for fa in FA_TARGETS:
                g[f"matched_fa_{fa:.2f}"] = operating_point(sc_ev, gate, threshold_at_fa(sc_ca, gate, fa))
            row["gates"][gate] = g
        rows.append(row)
        a, m = row["gates"]["absolute"]["youden"], row["gates"]["margin"]["youden"]
        print(f"  N={n:5d}  top1 {row['paraphrase_exact_top1_recall']:.3f} | abs th={a['threshold']:.3f} "
              f"act {a['paraphrase_activation']:.3f} fa {a['neighborhood_false_activation']:.3f} | "
              f"margin g={m['threshold']:.4f} act {m['paraphrase_activation']:.3f} fa {m['neighborhood_false_activation']:.3f}"
              f" | dup {row['duplicate_subject_relation_rate']:.3f}")
    return rows


def plot(result: Dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, runs in result["orderings"].items():
        ns = [r["N"] for r in runs]
        style = "-" if name == "file_order" else ":"
        for gate, color in (("absolute", "#1f77b4"), ("margin", "#d62728")):
            lab = gate if name == "file_order" else None
            axes[0].plot(ns, [100 * r["gates"][gate]["youden"]["paraphrase_activation"] for r in runs], style, color=color, marker="o", ms=3, label=lab)
            axes[1].plot(ns, [100 * r["gates"][gate]["youden"]["neighborhood_false_activation"] for r in runs], style, color=color, marker="o", ms=3, label=lab)
            axes[2].plot(ns, [100 * r["gates"][gate]["matched_fa_0.10"]["paraphrase_activation"] for r in runs], style, color=color, marker="o", ms=3, label=lab)
        axes[0].plot(ns, [100 * r["paraphrase_exact_top1_recall"] for r in runs], style, color="gray", lw=1, label="exact top-1" if name == "file_order" else None)
    titles = ("Paraphrase activation at Youden point (%)", "Neighborhood false activation (%)", "Activation at calibrated FA <= 10% (%)")
    for ax, t in zip(axes, titles):
        ax.set_xscale("log"); ax.set_xlabel("store size N"); ax.set_title(t, fontsize=10); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--orderings", type=int, default=0, help="number of random record orderings in addition to file order")
    args = p.parse_args()

    records = load_counterfact(10 ** 9)
    if 2 * max(args.sizes) > len(records):
        raise SystemExit(f"need {2 * max(args.sizes)} records, have {len(records)}")
    keys_all, queries_all = embeddings(records, args.encoder)

    result = {"config": {"encoder": args.encoder, "records_available": len(records), "sizes": args.sizes,
                         "evaluation_store": "first N records of the ordering", "calibration_store": "next N records",
                         "grid_absolute": [0.1, 0.9, 161], "grid_margin": [0.0, 0.5, 201],
                         "matched_fa_targets": list(FA_TARGETS)},
              "orderings": {}}
    orders = {"file_order": np.arange(len(records))}
    for s in range(args.orderings):
        orders[f"random_seed{s}"] = np.random.default_rng(s).permutation(len(records))
    for name, order in orders.items():
        print(f"=== {name} ===")
        result["orderings"][name] = evaluate(keys_all, queries_all, records, order, args.sizes)

    os.makedirs(OUT_DIR, exist_ok=True)
    for path in (SUMMARY_PATH, os.path.join(OUT_DIR, os.path.basename(SUMMARY_PATH))):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    plot(result, os.path.join(OUT_DIR, "margin_scaling.png"))
    print(f"Wrote {SUMMARY_PATH} and {OUT_DIR}/")


if __name__ == "__main__":
    main()
