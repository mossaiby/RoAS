"""
src/counterfact_score_decomposition.py

Per-query decomposition of the read's CounterFact scores by gate outcome (second-round
Reviewer 1, comment 2).

Table 6 reports a paraphrase score of 62.6% for the read against a paraphrase firing rate of
53.8%, so the score cannot be "capped by activation". The read leaves the forward pass
untouched when its gate does not fire, so on those queries it scores exactly what the unedited
model scores, and the unedited model already prefers the new target on 28.1% of paraphrases.
This script measures the decomposition per query instead of per record:

    score = P(fired) * score | fired  +  P(not fired) * score | not fired,

checks that score | not fired is identical query-by-query to the base model, and gives the
bound that does hold:

    score <= P(fired) + P(not fired) * base score | not fired.

The same decomposition is reported for argmax and for neighborhood prompts. Scoring uses
counterfact_grace_benchmark.score_prompt unchanged, the store, keys, values, theta* and
temperature of src/counterfact_unified_editors.py, and the same 500 records.

Usage (from the repository root):
    python src/counterfact_score_decomposition.py --model gpt2
    python src/counterfact_score_decomposition.py --model gpt2-xl --lambda-read 0.85   # CPU is fine

Writes results/counterfact_score_decomposition_<model>.json and a per-query file under
results/details/counterfact_unified_outputs/.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import counterfact_grace_benchmark as grace
from counterfact_retrieval_benchmark import load_counterfact
from counterfact_rome_benchmark import first_subtoken_id
from counterfact_unified_editors import CALIBRATION_PATH, DEVICE, ReadAdaptor, SEED

OUT_DIR = os.path.join("results", "details", "counterfact_unified_outputs")


def score_queries(model, tokenizer, records, base, read):
    rows = []
    t0 = time.perf_counter()
    for idx, record in enumerate(records):
        req = record["requested_rewrite"]
        new_id = first_subtoken_id(tokenizer, req["target_new"]["str"])
        true_id = first_subtoken_id(tokenizer, req["target_true"]["str"])
        prompts = ([("rewrite", req["prompt"].format(req["subject"]))]
                   + [("paraphrase", p) for p in record["paraphrase_prompts"][:2]]
                   + [("neighborhood", p) for p in record["neighborhood_prompts"][:2]])
        for kind, prompt in prompts:
            b = grace.score_prompt(model, base, tokenizer, prompt, new_id, true_id)
            r = grace.score_prompt(model, read, tokenizer, prompt, new_id, true_id)
            # Neighborhood success means the edit did NOT leak: P(true) > P(new).
            succ = (lambda s: s["p_true"] > s["p_new"]) if kind == "neighborhood" else (lambda s: s["p_new"] > s["p_true"])
            rows.append({
                "case_id": record.get("case_id"), "kind": kind, "fired": r["fired"],
                "top1_cosine": r["distance"],
                "base_success": bool(succ(b)), "read_success": bool(succ(r)),
                "base_argmax_new": b["argmax_new"], "read_argmax_new": r["argmax_new"],
                "base_p_new": b["p_new"], "read_p_new": r["p_new"],
            })
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(records)} records ({time.perf_counter() - t0:.0f}s)")
    return rows


def decompose(rows, kind):
    q = [r for r in rows if r["kind"] == kind]
    fired = np.array([r["fired"] for r in q])
    out = {"queries": len(q), "fired_rate": float(fired.mean())}
    for metric, read_key, base_key in (("score", "read_success", "base_success"),
                                       ("argmax", "read_argmax_new", "base_argmax_new")):
        rd = np.array([r[read_key] for r in q], dtype=float)
        bs = np.array([r[base_key] for r in q], dtype=float)
        inactive = ~fired
        out[metric] = {
            "read": float(rd.mean()),
            "base": float(bs.mean()),
            "read_given_fired": float(rd[fired].mean()) if fired.any() else None,
            "read_given_not_fired": float(rd[inactive].mean()) if inactive.any() else None,
            "base_given_fired": float(bs[fired].mean()) if fired.any() else None,
            "base_given_not_fired": float(bs[inactive].mean()) if inactive.any() else None,
            # contributions to the overall rate (they sum to "read")
            "contribution_fired": float(rd[fired].sum() / len(q)),
            "contribution_not_fired": float(rd[inactive].sum() / len(q)),
            # bound that does hold: fired queries can contribute at most P(fired)
            "upper_bound": float(fired.mean() + rd[inactive].sum() / len(q)),
            # sanity: on unfired queries the read must equal the base model exactly
            "not_fired_mismatches_vs_base": int(np.sum(rd[inactive] != bs[inactive])),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--lambda-read", type=float, default=0.60)
    p.add_argument("--temperature", type=float, default=0.07)
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with open(CALIBRATION_PATH, encoding="utf-8") as f:
        theta = float(json.load(f)["optimal_threshold"])

    from sentence_transformers import SentenceTransformer
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    records = load_counterfact(args.limit)

    key_texts, value_ids = [], []
    for r in records:
        req = r["requested_rewrite"]
        key_texts.append(f"{req['prompt'].format(req['subject'])} {req['target_new']['str'].strip()}")
        value_ids.append(first_subtoken_id(tokenizer, req["target_new"]["str"]))
    read = ReadAdaptor(model, encoder, key_texts, value_ids, theta, args.lambda_read, args.temperature)

    # The read's forward hook is registered on ln_f; NullAdaptor scoring must bypass it.
    class _Base:
        last_fired, last_distance = False, None

        def prepare(self, prompt):
            read.clear()

    rows = score_queries(model, tokenizer, records, _Base(), read)
    read.remove()

    results = {"model": args.model, "records": len(records), "theta": theta, "lambda_read": args.lambda_read,
               "temperature": args.temperature, "scorer": "counterfact_grace_benchmark.score_prompt",
               "decomposition": {k: decompose(rows, k) for k in ("rewrite", "paraphrase", "neighborhood")}}
    suffix = "" if abs(args.lambda_read - 0.60) < 1e-9 else f"_lambda{args.lambda_read:g}"
    os.makedirs(OUT_DIR, exist_ok=True)
    name = f"counterfact_score_decomposition_{args.model}{suffix}.json"
    for out in (os.path.join("results", name), os.path.join(OUT_DIR, name)):
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    with open(os.path.join(OUT_DIR, f"score_decomposition_per_query_{args.model}{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print(json.dumps(results["decomposition"], indent=2))


if __name__ == "__main__":
    main()
