"""
src/margin_gate_scaling.py

Store-size sweep for the rank-margin gate (follow-up to Section 6.4), on CounterFact or zsRE.

At N = 500 on CounterFact the rank-margin gate (fire iff s_(1) - s_(2) >= gamma) beats the
calibrated absolute gate (fire iff s_(1) >= theta) on both axes: 70.8% paraphrase activation at
7.0% neighborhood false activation, against 53.2% at 16.1%. The margin depends on the rest of the
store, so the open question is whether this survives a denser store, and whether it is specific
to CounterFact. This script answers both with the protocol of
src/counterfact_failure_analysis.py, varying only the store size N and the dataset.

For every N and every record ordering:
  * evaluation store = the first N records of the ordering, calibration store = the next N
    (disjoint), so N = 500 on CounterFact in file order is exactly the setting of the paper;
  * keys = MiniLM("rewrite prompt + new target"), unit-normalized; every query of the store is
    scored against all N keys;
  * theta*(N) and gamma*(N) by Youden's J on the calibration store (same grids as the paper);
  * reported on the evaluation store: outcome buckets, exact top-1 recall, held-out AUC.
Also reported:
  * transfer: theta* and gamma* calibrated at the smallest N applied unchanged at every N;
  * matched false activation: for each gate, the threshold whose calibration-store false
    activation is at most 5% / 10%, and the activation it gives on the evaluation store;
  * near-duplicate records: share of evaluation records that share a subject (and, for
    CounterFact, a relation) with another record of the same store.

Datasets (queries per record):
  counterfact  rewrite prompt, 2 paraphrases, 2 neighborhood prompts (same relation template,
               other subject). Duplicates: same (subject, relation_id).
  zsre         the MEMIT zsRE evaluation set: question `src`, 1 paraphrase `rephrase`,
               1 neighborhood = the unrelated Natural Questions query `loc` (its "nq question: "
               prefix removed). zsRE has no relation id, so duplicates are same subject.
               zsRE neighborhoods are far easier negatives than CounterFact's.

Embeddings are computed once per dataset and cached in data/ (git-ignored).

Usage (from the repository root):
    python src/margin_gate_scaling.py --dataset counterfact --orderings 3
    python src/margin_gate_scaling.py --dataset zsre --orderings 3

Writes results/<dataset>_margin_scaling.json and
results/details/<dataset>_margin_scaling_outputs/{<dataset>_margin_scaling.json, margin_scaling.png}.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch

from counterfact_failure_analysis import GRID, MARGIN_GRID
from counterfact_retrieval_benchmark import load_counterfact
from roas_analysis import bucket_queries, bucket_rates, roc_auc, youden_threshold

DEFAULT_SIZES = {"counterfact": (500, 1000, 2000, 5000, 10000), "zsre": (500, 1000, 2000, 5000, 9000)}
ZSRE_PATH = os.path.join("data", "zsre_mem_eval.jsonl")
# Mirror of the MEMIT zsRE evaluation set (zsre_mem_eval.json), which is no longer served at
# rome.baulab.info. Same fields: subject, src, pred, rephrase, alt, answers, loc, loc_ans, cond.
ZSRE_URL = "https://huggingface.co/datasets/CrowdMind/ZsRE/resolve/main/test.jsonl"
FA_TARGETS = (0.05, 0.10)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def load_zsre() -> List[Dict]:
    if not os.path.exists(ZSRE_PATH):
        os.makedirs(os.path.dirname(ZSRE_PATH), exist_ok=True)
        print(f"Downloading zsRE from {ZSRE_URL}")
        urllib.request.urlretrieve(ZSRE_URL, ZSRE_PATH)
    with open(ZSRE_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record_texts(dataset: str, r: Dict) -> Tuple[str, List[Tuple[str, str]]]:
    """(key text, [(query type, query text), ...]) for one record."""
    if dataset == "counterfact":
        req = r["requested_rewrite"]
        prompt = req["prompt"].format(req["subject"])
        key = f"{prompt} {req['target_new']['str'].strip()}"
        qs = [("rewrite", prompt)]
        qs += [("paraphrase", t) for t in r["paraphrase_prompts"][:2]]
        qs += [("neighborhood", t) for t in r["neighborhood_prompts"][:2]]
        return key, qs
    if dataset == "zsre":
        loc = r["loc"]
        if loc.startswith("nq question: "):
            loc = loc[len("nq question: "):]
        return f"{r['src']} {r['alt'].strip()}", [("rewrite", r["src"]), ("paraphrase", r["rephrase"]),
                                                  ("neighborhood", loc)]
    raise ValueError(dataset)


def duplicate_key(dataset: str, r: Dict):
    if dataset == "counterfact":
        req = r["requested_rewrite"]
        return req["subject"].strip().lower(), req.get("relation_id", "")
    return r["subject"].strip().lower()


def embeddings(dataset: str, records: List[Dict], encoder_name: str):
    """Unit-normalized key embeddings [R, m] and query embeddings with per-query type/record."""
    cache = os.path.join("data", f"{dataset}_minilm_embeddings.npz")
    keys, q_text, q_type, q_rec = [], [], [], []
    for i, r in enumerate(records):
        k, qs = record_texts(dataset, r)
        keys.append(k)
        for t, text in qs:
            q_type.append(t); q_text.append(text); q_rec.append(i)
    q_type, q_rec = np.array(q_type), np.array(q_rec)
    if os.path.exists(cache):
        z = np.load(cache)
        if z["keys"].shape[0] == len(records) and z["queries"].shape[0] == len(q_text) and str(z["encoder"]) == encoder_name:
            return z["keys"], z["queries"], q_type, q_rec
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(encoder_name, device=DEVICE)
    t0 = time.perf_counter()
    kv = enc.encode(keys, convert_to_numpy=True, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    qv = enc.encode(q_text, convert_to_numpy=True, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    print(f"Encoded {len(keys)} keys and {len(q_text)} queries in {time.perf_counter() - t0:.0f}s")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    kv, qv = kv.astype(np.float32), qv.astype(np.float32)
    np.savez(cache, keys=kv, queries=qv, encoder=encoder_name)
    return kv, qv, q_type, q_rec


# ---------------------------------------------------------------------------
# Scoring one store
# ---------------------------------------------------------------------------

def score_store(emb, idx: np.ndarray) -> Dict:
    """Top-2 scores of every query of the store's records against the store's N keys."""
    keys_all, queries_all, q_type, q_rec = emb
    local = -np.ones(keys_all.shape[0], dtype=np.int64)
    local[idx] = np.arange(len(idx))
    sel = np.where(local[q_rec] >= 0)[0]
    k = torch.from_numpy(keys_all[idx]).to(DEVICE)
    q = torch.from_numpy(queries_all[sel]).to(DEVICE)
    top1, s1, s2 = [], [], []
    for start in range(0, q.shape[0], 8192):
        vals, ind = torch.topk(q[start:start + 8192] @ k.T, 2, dim=1)
        top1.append(ind[:, 0].cpu()); s1.append(vals[:, 0].cpu()); s2.append(vals[:, 1].cpu())
    return {"top1": torch.cat(top1).numpy(), "s1": torch.cat(s1).double().numpy(),
            "s2": torch.cat(s2).double().numpy(), "qtype": q_type[sel], "qrec": local[q_rec[sel]]}


def stat_of(sc: Dict, gate: str) -> np.ndarray:
    return sc["s1"] if gate == "absolute" else sc["s1"] - sc["s2"]


def calibrate(sc: Dict, gate: str):
    grid = GRID if gate == "absolute" else MARGIN_GRID
    labels = np.isin(sc["qtype"], ("rewrite", "paraphrase"))
    return youden_threshold(stat_of(sc, gate), labels, grid)


def operating_point(sc: Dict, gate: str, t: float) -> Dict:
    r = bucket_rates(bucket_queries(sc["qtype"], sc["qrec"], sc["top1"], stat_of(sc, gate), t), sc["qtype"])
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


def duplicate_rate(dataset: str, records: List[Dict], idx: np.ndarray) -> float:
    keys = [duplicate_key(dataset, records[i]) for i in idx]
    c = Counter(keys)
    return float(np.mean([c[k] > 1 for k in keys]))


def evaluate(dataset, emb, records, order: np.ndarray, sizes) -> List[Dict]:
    rows, ref = [], {}
    for n in sizes:
        sc_ev, sc_ca = score_store(emb, order[:n]), score_store(emb, order[n:2 * n])
        para = sc_ev["qtype"] == "paraphrase"
        pos = np.isin(sc_ev["qtype"], ("rewrite", "paraphrase"))
        correct = sc_ev["top1"] == sc_ev["qrec"]
        margin = sc_ev["s1"] - sc_ev["s2"]
        row = {"N": n, "duplicate_rate": duplicate_rate(dataset, records, order[:n]),
               "paraphrase_exact_top1_recall": float(np.mean(correct[para])),
               "median_margin": {"paraphrase_correct": float(np.median(margin[para & correct])),
                                 "neighborhood": float(np.median(margin[sc_ev["qtype"] == "neighborhood"]))},
               "gates": {}}
        for gate in ("absolute", "margin"):
            t, j = calibrate(sc_ca, gate)
            if n == sizes[0]:
                ref[gate] = t
            g = {"youden": operating_point(sc_ev, gate, t), "youden_j_calibration": j,
                 "auc_evaluation": roc_auc(stat_of(sc_ev, gate), pos),
                 "transfer_from_smallest_N": operating_point(sc_ev, gate, ref[gate])}
            for fa in FA_TARGETS:
                g[f"matched_fa_{fa:.2f}"] = operating_point(sc_ev, gate, threshold_at_fa(sc_ca, gate, fa))
            row["gates"][gate] = g
        rows.append(row)
        a, m = row["gates"]["absolute"]["youden"], row["gates"]["margin"]["youden"]
        print(f"  N={n:5d}  top1 {row['paraphrase_exact_top1_recall']:.3f} | abs th={a['threshold']:.3f} "
              f"act {a['paraphrase_activation']:.3f} fa {a['neighborhood_false_activation']:.3f} | "
              f"margin g={m['threshold']:.4f} act {m['paraphrase_activation']:.3f} fa {m['neighborhood_false_activation']:.3f}"
              f" | dup {row['duplicate_rate']:.3f}")
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
        axes[0].plot(ns, [100 * r["paraphrase_exact_top1_recall"] for r in runs], style, color="gray", lw=1,
                     label="exact top-1" if name == "file_order" else None)
    titles = ("Paraphrase activation at Youden point (%)", "Neighborhood false activation (%)",
              "Activation at calibrated FA <= 10% (%)")
    for ax, t in zip(axes, titles):
        ax.set_xscale("log"); ax.set_xlabel("store size N"); ax.set_title(t, fontsize=10); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(result["config"]["dataset"], fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("counterfact", "zsre"), default="counterfact")
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--sizes", type=int, nargs="+", default=None)
    p.add_argument("--orderings", type=int, default=0, help="number of random record orderings in addition to file order")
    args = p.parse_args()
    sizes = args.sizes or list(DEFAULT_SIZES[args.dataset])

    records = load_counterfact(10 ** 9) if args.dataset == "counterfact" else load_zsre()
    if 2 * max(sizes) > len(records):
        raise SystemExit(f"need {2 * max(sizes)} records, have {len(records)}")
    emb = embeddings(args.dataset, records, args.encoder)

    result = {"config": {"dataset": args.dataset, "encoder": args.encoder, "records_available": len(records),
                         "sizes": sizes, "evaluation_store": "first N records of the ordering",
                         "calibration_store": "next N records", "grid_absolute": [0.1, 0.9, 161],
                         "grid_margin": [0.0, 0.5, 201], "matched_fa_targets": list(FA_TARGETS)},
              "orderings": {}}
    orders = {"file_order": np.arange(len(records))}
    for s in range(args.orderings):
        orders[f"random_seed{s}"] = np.random.default_rng(s).permutation(len(records))
    for name, order in orders.items():
        print(f"=== {args.dataset} / {name} ===")
        result["orderings"][name] = evaluate(args.dataset, emb, records, order, sizes)

    out_dir = os.path.join("results", "details", f"{args.dataset}_margin_scaling_outputs")
    summary = os.path.join("results", f"{args.dataset}_margin_scaling.json")
    os.makedirs(out_dir, exist_ok=True)
    for path in (summary, os.path.join(out_dir, os.path.basename(summary))):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    plot(result, os.path.join(out_dir, "margin_scaling.png"))
    print(f"Wrote {summary} and {out_dir}/")


if __name__ == "__main__":
    main()
