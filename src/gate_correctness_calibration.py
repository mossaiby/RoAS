"""
src/gate_correctness_calibration.py

Correctness-aware calibration of the read gates (follow-up to src/margin_gate_scaling.py).

The store-size sweep calibrates each gate by Youden's J with "rewrite or paraphrase" as the
positive label. That objective rewards a positive query for firing even when its top-1 key is
another record's, so it counts injecting the wrong edit as a success. On zsRE, where 47% of a
9,000-record store shares a subject with another record, this is what makes the absolute gate
look best: it fires on shared-subject paraphrases and injects the wrong record's edit on 46-63%
of them.

Here a firing is scored by its outcome, not by the query type:
    +1  positive query (rewrite / paraphrase) fires and its top-1 is its own record
    -1  positive query fires on another record (wrong edit injected)
    -1  neighborhood query fires (any record)
     0  no firing (the base model answers)
The calibration objective is the per-type-normalized utility
    U = P(fire, correct | pos) - P(fire, wrong | pos) - P(fire | neighborhood),
which reduces to Youden's J when top-1 retrieval is always correct. Rewrite queries nearly
always retrieve their own key (the key is the rewrite prompt plus the target), so U is also
computed with paraphrases as the only positives (utility_para), the case the gate exists for.

Three gates, all on unit-normalized MiniLM scores s_(1) >= s_(2) of the store:
    absolute   s_(1) >= theta
    margin     s_(1) - s_(2) >= gamma
    combined   s_(1) >= theta  AND  s_(1) - s_(2) >= gamma   (2-D search)
Each gate is calibrated twice on the calibration store (next N records, disjoint from the
evaluation store; same orderings and grids as margin_gate_scaling.py):
    youden        the original objective (for the combined gate, J over the 2-D grid)
    utility       U above
    utility_para  U with paraphrases as the only positives
and additionally at a matched error budget: the largest calibration-store correct activation
subject to neighborhood false activation <= 5% AND wrong-record activation <= 5%; when no
threshold meets the budget the gate never fires (reported as infeasible).

Reported on the evaluation store for every operating point: correct activation, wrong-record
activation and neighborhood false activation for paraphrases and rewrites, the utility U, and
the same split over unique-subject vs shared-subject records.

Usage (from the repository root; embeddings are reused from data/):
    python src/gate_correctness_calibration.py --dataset counterfact --orderings 3
    python src/gate_correctness_calibration.py --dataset zsre --orderings 3

Writes results/<dataset>_gate_correctness.json and
results/details/<dataset>_gate_correctness_outputs/{<dataset>_gate_correctness.json, gate_correctness.png}.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, Tuple

import numpy as np

import margin_gate_scaling as M
from counterfact_failure_analysis import GRID, MARGIN_GRID
from counterfact_retrieval_benchmark import load_counterfact

GATES = ("absolute", "margin", "combined")
BUDGET = 0.05
OBJECTIVES = ("youden", "utility", "utility_para", "budget")


POSITIVE_TYPES = {"youden": ("rewrite", "paraphrase"), "utility": ("rewrite", "paraphrase"),
                  "utility_para": ("paraphrase",), "budget": ("rewrite", "paraphrase")}


def query_sets(sc: Dict, pos_types=("rewrite", "paraphrase")) -> Dict[str, np.ndarray]:
    pos = np.isin(sc["qtype"], pos_types)
    correct = sc["top1"] == sc["qrec"]
    return {"pos_correct": pos & correct, "pos_wrong": pos & ~correct,
            "neigh": sc["qtype"] == "neighborhood", "n_pos": max(int(pos.sum()), 1),
            "n_neigh": max(int((sc["qtype"] == "neighborhood").sum()), 1)}


def _count_ge(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Number of entries of `values` that are >= each grid point."""
    v = np.sort(values)
    return len(v) - np.searchsorted(v, grid, side="left")


def rate_table(sc: Dict, gate: str, pos_types=("rewrite", "paraphrase")) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Rates of each outcome for every threshold of the gate's grid (1-D, or 2-D for combined).

    Returns (thresholds, rates) where thresholds[i] is a (theta, gamma) pair and rates has keys
    act (correct activation over positives), wrong (wrong-record activation over positives),
    pos_fire (any firing over positives), fa (false activation over neighborhoods).
    """
    q = query_sets(sc, pos_types)
    s1, m = sc["s1"], sc["s1"] - sc["s2"]
    if gate == "absolute":
        th = [(float(t), 0.0) for t in GRID]
        c = {k: _count_ge(s1[q[k]], GRID) for k in ("pos_correct", "pos_wrong", "neigh")}
    elif gate == "margin":
        th = [(-1.0, float(g)) for g in MARGIN_GRID]
        c = {k: _count_ge(m[q[k]], MARGIN_GRID) for k in ("pos_correct", "pos_wrong", "neigh")}
    else:
        th, parts = [], {k: [] for k in ("pos_correct", "pos_wrong", "neigh")}
        for t in GRID:
            th += [(float(t), float(g)) for g in MARGIN_GRID]
            for k in parts:
                sel = q[k] & (s1 >= t)
                parts[k].append(_count_ge(m[sel], MARGIN_GRID))
        c = {k: np.concatenate(v) for k, v in parts.items()}
    rates = {"act": c["pos_correct"] / q["n_pos"], "wrong": c["pos_wrong"] / q["n_pos"],
             "fa": c["neigh"] / q["n_neigh"]}
    rates["pos_fire"] = rates["act"] + rates["wrong"]
    return np.array(th), rates


def pick(sc: Dict, gate: str, objective: str) -> Tuple[Tuple[float, float], float]:
    th, r = rate_table(sc, gate, POSITIVE_TYPES[objective])
    if objective == "youden":
        score = r["pos_fire"] - r["fa"]
    elif objective in ("utility", "utility_para"):
        score = r["act"] - r["wrong"] - r["fa"]
    elif objective == "budget":
        score = np.where((r["fa"] <= BUDGET) & (r["wrong"] <= BUDGET), r["act"], -np.inf)
    else:
        raise ValueError(objective)
    i = int(np.argmax(score))        # first maximizer, as in roas_analysis.youden_threshold
    if not np.isfinite(score[i]):    # budget infeasible on the calibration store: never fire
        return (np.inf, np.inf), float("nan")
    return (float(th[i][0]), float(th[i][1])), float(score[i])


def fires(sc: Dict, t: Tuple[float, float]) -> np.ndarray:
    return (sc["s1"] >= t[0]) & ((sc["s1"] - sc["s2"]) >= t[1])


def outcome(sc: Dict, f: np.ndarray, mask: np.ndarray = None) -> Dict:
    mask = np.ones(len(f), dtype=bool) if mask is None else mask
    correct = sc["top1"] == sc["qrec"]
    out = {}
    for qt in ("rewrite", "paraphrase"):
        sel = mask & (sc["qtype"] == qt)
        if sel.any():
            out[qt] = {"correct_activation": float(np.mean((f & correct)[sel])),
                       "wrong_record_activation": float(np.mean((f & ~correct)[sel])),
                       "top1_correct": float(np.mean(correct[sel])), "count": int(sel.sum())}
    sel = mask & (sc["qtype"] == "neighborhood")
    if sel.any():
        out["neighborhood"] = {"false_activation": float(np.mean(f[sel])), "count": int(sel.sum())}
    pos = mask & np.isin(sc["qtype"], ("rewrite", "paraphrase"))
    para = mask & (sc["qtype"] == "paraphrase")
    if pos.any() and sel.any():
        out["utility"] = float(np.mean((f & correct)[pos]) - np.mean((f & ~correct)[pos]) - np.mean(f[sel]))
        out["utility_para"] = float(np.mean((f & correct)[para]) - np.mean((f & ~correct)[para]) - np.mean(f[sel]))
    return out


def evaluate(dataset, emb, records, order, sizes):
    rows = []
    for n in sizes:
        sc_ev, sc_ca = M.score_store(emb, order[:n]), M.score_store(emb, order[n:2 * n])
        dk = [M.duplicate_key(dataset, records[i]) for i in order[:n]]
        cnt = Counter(dk)
        shared_rec = np.array([cnt[k] > 1 for k in dk])
        shared_q = shared_rec[sc_ev["qrec"]]
        row = {"N": n, "shared_subject_rate": float(shared_rec.mean()), "gates": {}}
        for gate in GATES:
            row["gates"][gate] = {}
            for obj in OBJECTIVES:
                t, val = pick(sc_ca, gate, obj)
                f = fires(sc_ev, t)
                row["gates"][gate][obj] = {
                    "feasible": bool(np.isfinite(t[0])),
                    "theta": t[0] if gate != "margin" and np.isfinite(t[0]) else None,
                    "gamma": t[1] if gate != "absolute" and np.isfinite(t[1]) else None,
                    "calibration_objective": val,
                    "all": outcome(sc_ev, f), "unique_subject": outcome(sc_ev, f, ~shared_q),
                    "shared_subject": outcome(sc_ev, f, shared_q)}
        rows.append(row)
        line = f"  N={n:5d} shared {row['shared_subject_rate']:.2f} |"
        for obj in OBJECTIVES:
            line += f" {obj}:"
            for gate in GATES:
                a = row["gates"][gate][obj]["all"]
                line += (f" {gate[:3]} {100 * a['paraphrase']['correct_activation']:4.1f}"
                         f"/{100 * a['paraphrase']['wrong_record_activation']:4.1f}"
                         f"/{100 * a['neighborhood']['false_activation']:4.1f}")
            line += " |"
        print(line)
    return rows


def plot(result: Dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"absolute": "#1f77b4", "margin": "#d62728", "combined": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    runs = list(result["orderings"].values())
    ns = [r["N"] for r in runs[0]]
    panels = (("utility_para", "correct_activation", "Paraphrase correct activation (%), utility"),
              ("utility_para", "wrong_record_activation", "Paraphrase wrong-record activation (%), utility"),
              ("budget", "correct_activation", "Correct activation (%), FA and wrong <= 5%"))
    for ax, (obj, key, title) in zip(axes, panels):
        for gate in GATES:
            vals = np.array([[100 * r["gates"][gate][obj]["all"]["paraphrase"][key] for r in run] for run in runs])
            ax.errorbar(ns, vals.mean(0), yerr=vals.std(0), color=colors[gate], marker="o", ms=3, capsize=2, label=gate)
        ax.set_xscale("log"); ax.set_xlabel("store size N"); ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
    axes[0].plot(ns, [100 * np.mean([run[i]["gates"]["absolute"]["utility"]["all"]["paraphrase"]["top1_correct"]
                                     for run in runs]) for i in range(len(ns))], color="gray", lw=1, label="exact top-1")
    axes[0].legend(fontsize=8)
    fig.suptitle(result["config"]["dataset"], fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("counterfact", "zsre"), default="counterfact")
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--sizes", type=int, nargs="+", default=None)
    p.add_argument("--orderings", type=int, default=3, help="number of random record orderings")
    p.add_argument("--file-order", action="store_true", help="also run the file order")
    args = p.parse_args()
    sizes = args.sizes or list(M.DEFAULT_SIZES[args.dataset])

    records = load_counterfact(10 ** 9) if args.dataset == "counterfact" else M.load_zsre()
    if 2 * max(sizes) > len(records):
        raise SystemExit(f"need {2 * max(sizes)} records, have {len(records)}")
    emb = M.embeddings(args.dataset, records, args.encoder)

    orders = {"file_order": np.arange(len(records))} if args.file_order else {}
    for s in range(args.orderings):
        orders[f"random_seed{s}"] = np.random.default_rng(s).permutation(len(records))
    result = {"config": {"dataset": args.dataset, "encoder": args.encoder, "sizes": sizes,
                         "objectives": {"youden": "P(fire|pos) - P(fire|neigh)",
                                        "utility": "P(fire,correct|pos) - P(fire,wrong|pos) - P(fire|neigh)",
                                        "utility_para": "as utility, positives = paraphrases only",
                                        "budget": f"max P(fire,correct|pos) s.t. FA <= {BUDGET} and wrong <= {BUDGET}"},
                         "grid_absolute": [0.1, 0.9, 161], "grid_margin": [0.0, 0.5, 201]},
              "orderings": {}}
    print("  columns per gate: paraphrase correct / wrong-record / neighborhood false activation (%)")
    for name, order in orders.items():
        print(f"=== {args.dataset} / {name} ===")
        result["orderings"][name] = evaluate(args.dataset, emb, records, order, sizes)

    out_dir = os.path.join("results", "details", f"{args.dataset}_gate_correctness_outputs")
    summary = os.path.join("results", f"{args.dataset}_gate_correctness.json")
    os.makedirs(out_dir, exist_ok=True)
    for path in (summary, os.path.join(out_dir, os.path.basename(summary))):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    plot(result, os.path.join(out_dir, "gate_correctness.png"))
    print(f"Wrote {summary} and {out_dir}/")


if __name__ == "__main__":
    main()
