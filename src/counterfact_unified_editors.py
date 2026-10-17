"""
src/counterfact_unified_editors.py

One scorer for every system on CounterFact (Reviewer 3, comments 4 and 5).

In the submitted Table 5 the atom-space read was reported only through activation rates,
while R1C and GRACE were reported through efficacy / paraphrase / neighborhood scores, so the
rows were not comparable. This script scores the read with *exactly* the function used for
GRACE (counterfact_grace_benchmark.evaluate_record / aggregate): same records, same two
paraphrases and two neighborhood prompts per record, same first-subtoken P(new) vs P(true)
comparison, same argmax and activation bookkeeping. The unedited model is scored the same way.

The read:
  * store = all N records at once (the analogue of GRACE's sequential protocol);
  * key = frozen MiniLM embedding of "rewrite prompt + target_new"; value = the LM's input
    embedding row of target_new's first subtoken;
  * gate = top-1 cosine >= theta* (read from results/counterfact_calibration_results.json);
  * when the gate fires, the final-layer hidden state at the last prompt position is replaced
    by (1 - lambda) h + lambda * sum_i mu_i v_i (softmax at temperature 0.07), as in
    src/generative_steering_decoupled.py. Nothing else in the network changes.

`--model` selects any GPT-2 checkpoint (gpt2, gpt2-medium, gpt2-large, gpt2-xl). The read and
the base model need only forward passes, so gpt2-xl is feasible on CPU (a few thousand
forward passes). GRACE is optional (--with-grace) because its per-edit value optimization is
the expensive part; on gpt2-xl it uses the published GRACE layer (35).

Writes results/counterfact_unified_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import counterfact_grace_benchmark as grace
from atom_space_generative_engine import CONTROL_TEXTS
from counterfact_retrieval_benchmark import load_counterfact
from counterfact_rome_benchmark import first_subtoken_id

DEVICE = grace.DEVICE
OUT_DIR = os.path.join("results", "details", "counterfact_unified_outputs")
CALIBRATION_PATH = os.path.join("results", "counterfact_calibration_results.json")
SEED = 42
METRICS = ("efficacy_success", "canonical_argmax_correct", "rewrite_fired", "paraphrase_success_rate",
           "paraphrase_argmax_rate", "paraphrase_fired_rate", "neighborhood_success_rate", "neighborhood_fired_rate")


class NullAdaptor:
    """The unedited model: never fires."""
    last_fired = False
    last_distance = None

    def prepare(self, prompt: str):
        self.last_fired = False


class ReadAdaptor:
    """Gated atom-space read injected at the final hidden state (post ln_f), last position only."""

    def __init__(self, model, encoder, key_texts, value_ids, theta, lam, temperature):
        self.encoder = encoder
        self.keys = encoder.encode(key_texts, convert_to_tensor=True, normalize_embeddings=True).to(DEVICE)
        self.values = model.get_input_embeddings().weight[torch.tensor(value_ids, device=DEVICE)].detach()
        self.theta, self.lam, self.temperature = theta, lam, temperature
        self.last_fired, self.last_distance, self._z = False, None, None
        self.handle = model.transformer.ln_f.register_forward_hook(self._hook)

    @torch.no_grad()
    def prepare(self, prompt: str):
        q = self.encoder.encode([prompt], convert_to_tensor=True, normalize_embeddings=True).to(DEVICE)
        sims = (q @ self.keys.T)[0]
        top = sims.max().item()
        self.last_distance = top
        self.last_fired = top >= self.theta
        self._z = (F.softmax(sims / self.temperature, dim=-1) @ self.values) if self.last_fired else None

    def _hook(self, module, inputs, output):
        if self._z is None:
            return output
        out = output.clone()
        out[:, -1, :] = (1.0 - self.lam) * out[:, -1, :] + self.lam * self._z
        return out

    def clear(self):
        self.last_fired, self._z = False, None

    def remove(self):
        self.handle.remove()


def bootstrap(per_record, field, n=2000, seed=SEED):
    vals = np.array([float(r[field]) for r in per_record])
    rng = np.random.default_rng(seed)
    boots = vals[rng.integers(0, len(vals), size=(n, len(vals)))].mean(axis=1)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def run_system(name, model, tokenizer, records, adaptor):
    per_record = []
    t0 = time.perf_counter()
    for idx, record in enumerate(records):
        per_record.append(grace.evaluate_record(model, adaptor, tokenizer, record))
        if (idx + 1) % 100 == 0:
            print(f"  [{name}] {idx + 1}/{len(records)} records ({time.perf_counter() - t0:.0f}s)")
    summary = grace.aggregate(per_record)
    summary["ci_95"] = {f: bootstrap(per_record, f) for f in METRICS}
    return summary, per_record


def control_activation(adaptor):
    fired = []
    for text in CONTROL_TEXTS:
        prompt = text.rsplit(" ", 1)[0]
        adaptor.prepare(prompt)
        fired.append(bool(adaptor.last_fired))
    if hasattr(adaptor, "clear"):
        adaptor.clear()
    return float(np.mean(fired))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--theta", type=float, default=None, help="default: optimal_threshold from the calibration artifact")
    p.add_argument("--lambda-read", type=float, default=0.60)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--with-grace", action="store_true")
    p.add_argument("--grace-eps", type=float, default=1.0)
    p.add_argument("--calibrate-lambda", action="store_true",
                   help="choose lambda_read on held-out records 500-999 (smallest grid value whose held-out "
                        "efficacy score is >= --target-efficacy), then evaluate on records 0-499")
    p.add_argument("--lambda-grid", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    p.add_argument("--target-efficacy", type=float, default=0.95)
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    theta = args.theta
    if theta is None:
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

    calibration = None
    if args.calibrate_lambda:
        # Selection rule fixed in advance; the evaluation records are never used to choose lambda.
        held_out = load_counterfact(1000)[500:1000]
        calibration = {"rule": f"smallest lambda with held-out efficacy >= {args.target_efficacy}", "grid": {}}
        chosen = None
        for lam in sorted(args.lambda_grid):
            keys_h = [f"{r['requested_rewrite']['prompt'].format(r['requested_rewrite']['subject'])} "
                      f"{r['requested_rewrite']['target_new']['str'].strip()}" for r in held_out]
            vals_h = [first_subtoken_id(tokenizer, r["requested_rewrite"]["target_new"]["str"]) for r in held_out]
            probe = ReadAdaptor(model, encoder, keys_h, vals_h, theta, lam, args.temperature)
            summ, _ = run_system(f"calib lambda={lam}", model, tokenizer, held_out, probe)
            probe.remove()
            calibration["grid"][str(lam)] = {k: summ[k] for k in ("efficacy_score", "paraphrase_score", "neighborhood_score")}
            if chosen is None and summ["efficacy_score"] >= args.target_efficacy:
                chosen = lam
        calibration["chosen_lambda"] = chosen if chosen is not None else max(args.lambda_grid)
        args.lambda_read = calibration["chosen_lambda"]
        print(f"Calibrated lambda_read = {args.lambda_read}")

    results = {"model": args.model, "records": len(records), "theta": theta, "lambda_read": args.lambda_read,
               "temperature": args.temperature, "scorer": "counterfact_grace_benchmark.evaluate_record",
               "lambda_calibration": calibration, "systems": {}}

    print(f"=== Base {args.model} ===")
    base, _ = run_system("base", model, tokenizer, records, NullAdaptor())
    results["systems"]["base"] = base

    print(f"=== Atom-space read on {args.model} (theta={theta}) ===")
    key_texts, value_ids = [], []
    for r in records:
        req = r["requested_rewrite"]
        key_texts.append(f"{req['prompt'].format(req['subject'])} {req['target_new']['str'].strip()}")
        value_ids.append(first_subtoken_id(tokenizer, req["target_new"]["str"]))
    read = ReadAdaptor(model, encoder, key_texts, value_ids, theta, args.lambda_read, args.temperature)
    read_summary, read_records = run_system("read", model, tokenizer, records, read)
    read_summary["control_text_activation_rate"] = control_activation(read)
    read.remove()
    results["systems"]["read"] = read_summary

    if args.with_grace:
        n_layer = model.config.n_layer
        grace.LM_NAME = args.model
        grace.EDIT_LAYER = 35 if args.model == "gpt2-xl" else round(0.75 * n_layer)
        print(f"=== GRACE sequential on {args.model}, layer {grace.EDIT_LAYER}, eps_init={args.grace_eps} ===")
        pre_ppl = [grace.text_perplexity(model, tokenizer, t) for t in CONTROL_TEXTS]
        torch.manual_seed(SEED)
        results["systems"]["grace_sequential"] = grace.run_sequential(model, tokenizer, records, args.grace_eps, pre_ppl)
        results["grace_layer"] = grace.EDIT_LAYER

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_lambda_calibrated" if args.calibrate_lambda else ""
    path = os.path.join("results", f"counterfact_unified_{args.model}{suffix}.json")
    for out in (path, os.path.join(OUT_DIR, os.path.basename(path))):
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    with open(os.path.join(OUT_DIR, f"read_per_record_{args.model}{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(read_records, f, indent=1)
    print(json.dumps({k: {m: v for m, v in s.items() if m != "ci_95"} for k, s in results["systems"].items()}, indent=2))


if __name__ == "__main__":
    main()
