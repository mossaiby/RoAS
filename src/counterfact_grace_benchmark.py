"""
src/counterfact_grace_benchmark.py

GRACE (Hartvigsen et al., 2023, "Aging with GRACE: Lifelong Model Editing with Discrete
Key-Value Adaptors") on GPT-2 (124M), evaluated on the same first 500 CounterFact records
and with the same first-subtoken metrics as src/counterfact_rome_benchmark.py, so that the
retrieval read (Section 6.3 of the paper), the rank-one parametric edit, and a memory-based
editor share one benchmark scale and one scorer.

Mechanism, following the reference implementation (github.com/Thartvigsen/GRACE):
  * One MLP up-projection layer (transformer.h.L.mlp.c_fc) is wrapped by an adaptor holding a
    codebook of (key, value, epsilon) triples. Keys are the layer's *input* activation at the
    last prompt token; values are learned replacements for the layer's *output* at that token.
  * Inference: if the last-token input activation lies within epsilon (Euclidean) of its
    nearest key, the layer output at that token is replaced by the stored value; otherwise the
    layer is untouched. Nothing else in the network changes.
  * Editing: if no key is within its epsilon of the new activation, add a new key with
    epsilon = eps_init and a cold (random) value; if the nearest key is within reach and has
    the same label, expand its epsilon to the observed distance and refine its value; if it has
    a different label, shrink both epsilons to half the distance so the balls do not overlap
    ("split"). The value is optimized by SGD (lr 1.0, 100 iterations) on the cross-entropy of
    the target's first subtoken.
  * Edits are applied sequentially with no reset (GRACE's lifelong protocol); we also report a
    per-record reset variant for comparison with the rank-one baseline.

Deviations from the published setup, stated for the record: GRACE's GPT-2 experiments used
GPT-2 XL (layer 35 of 48) on a hallucination task; here we use GPT-2 small and choose the layer
at the same relative depth (layer 9 of 12), and we train the value on the first subtoken of the
target rather than on a full continuation, to match the CounterFact scorer used throughout.

Writes results/counterfact_grace_results.json and results/details/counterfact_grace_outputs/.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from atom_space_generative_engine import CONTROL_TEXTS
from counterfact_retrieval_benchmark import load_counterfact
from counterfact_rome_benchmark import first_subtoken_id, text_perplexity

OUT_DIR = os.path.join("results", "details", "counterfact_grace_outputs")
SUMMARY_PATH = os.path.join("results", "counterfact_grace_results.json")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LM_NAME = "gpt2"
EVAL_LIMIT = 500
EDIT_LAYER = 9
EDIT_LR = 1.0
N_ITER = 100
EPS_INIT_DEFAULT = 1.0
EPS_SWEEP = [1.0, 3.0, 10.0]


class GRACEAdaptor(nn.Module):
    """Wraps an nn.Module layer with a discrete key-value codebook acting on the last token."""

    def __init__(self, layer: nn.Module, eps_init: float):
        super().__init__()
        self.layer = layer
        self.eps_init = eps_init
        self.keys = None      # [n, d_in]
        self.values = None    # [n, d_out]
        self.epsilons = None  # [n]
        self.labels = []      # python list of hashable labels
        self.edit_mode = False
        self.active_index = None
        self.last_fired = None
        self.last_distance = None

    def reset(self):
        self.keys = self.values = self.epsilons = None
        self.labels = []

    def forward(self, x):
        out = self.layer(x)
        if self.keys is None and not self.edit_mode:
            self.last_fired = False
            return out
        query = x[:, -1, :]
        if self.edit_mode:
            out = out.clone()
            out[:, -1, :] = self.values[self.active_index].unsqueeze(0)
            return out
        dists = torch.cdist(query, self.keys)  # [B, n]
        smallest, nearest = dists.min(dim=1)
        self.last_distance = smallest[0].item()
        fired = smallest[0] <= self.epsilons[nearest[0]]
        self.last_fired = bool(fired.item())
        if fired:
            out = out.clone()
            out[:, -1, :] = self.values[nearest[0]].detach().unsqueeze(0)
        return out

    @torch.no_grad()
    def _last_token_input(self, x):
        return x[:, -1, :].detach()

    def register_edit(self, key: torch.Tensor, label) -> str:
        """Insert or adjust codebook entries for a new key; returns the action taken."""
        key = key.detach()
        d_out = self.layer.nf if hasattr(self.layer, "nf") else self.layer.out_features
        if self.keys is None:
            self.keys = key.clone()
            self.values = nn.Parameter(torch.randn(1, d_out, device=key.device) * 0.1)
            self.epsilons = torch.full((1,), self.eps_init, device=key.device)
            self.labels = [label]
            self.active_index = 0
            return "new"
        dists = torch.cdist(key, self.keys)[0]
        smallest, nearest = dists.min(dim=0)
        nearest = nearest.item()
        smallest = smallest.item()
        if smallest > self.epsilons[nearest].item() + self.eps_init:
            action = "new"
        elif self.labels[nearest] == label:
            self.epsilons[nearest] = max(self.epsilons[nearest].item(), smallest)
            self.active_index = nearest
            return "expand"
        else:
            self.epsilons[nearest] = smallest / 2.0 - 1e-3
            action = "split"
        new_eps = self.eps_init if action == "new" else smallest / 2.0 - 1e-3
        self.keys = torch.cat([self.keys, key.clone()], dim=0)
        new_value = torch.randn(1, d_out, device=key.device) * 0.1
        self.values = nn.Parameter(torch.cat([self.values.data, new_value], dim=0))
        self.epsilons = torch.cat([self.epsilons, torch.tensor([new_eps], device=key.device)])
        self.labels.append(label)
        self.active_index = self.keys.shape[0] - 1
        return action


def wrap_layer(model, layer_idx: int, eps_init: float) -> GRACEAdaptor:
    block = model.transformer.h[layer_idx].mlp
    adaptor = GRACEAdaptor(block.c_fc, eps_init).to(DEVICE)
    block.c_fc = adaptor
    return adaptor


def unwrap_layer(model, layer_idx: int, adaptor: GRACEAdaptor):
    model.transformer.h[layer_idx].mlp.c_fc = adaptor.layer


def capture_key(model, adaptor: GRACEAdaptor, prompt_ids: torch.Tensor) -> torch.Tensor:
    captured = {}

    def hook(module, inp, out):
        captured["key"] = inp[0][:, -1, :].detach()

    handle = adaptor.register_forward_hook(hook)
    with torch.no_grad():
        model(prompt_ids)
    handle.remove()
    return captured["key"]


def grace_edit(model, adaptor: GRACEAdaptor, prompt_ids: torch.Tensor, target_id: int, label) -> dict:
    key = capture_key(model, adaptor, prompt_ids)
    action = adaptor.register_edit(key, label)
    adaptor.edit_mode = True
    optimizer = torch.optim.SGD([adaptor.values], lr=EDIT_LR)
    target = torch.tensor([target_id], device=DEVICE)
    mask = torch.zeros_like(adaptor.values)
    mask[adaptor.active_index] = 1.0
    final_loss = None
    for _ in range(N_ITER):
        optimizer.zero_grad()
        logits = model(prompt_ids).logits[:, -1, :]
        loss = F.cross_entropy(logits, target)
        loss.backward()
        adaptor.values.grad.mul_(mask)  # only the active codebook value is trained
        optimizer.step()
        final_loss = loss.item()
        if final_loss < 1e-2:
            break
    adaptor.edit_mode = False
    return {"action": action, "final_loss": final_loss}


@torch.no_grad()
def score_prompt(model, adaptor, tokenizer, prompt: str, new_id: int, true_id: int):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    probs = F.softmax(model(ids).logits[:, -1, :], dim=-1)[0]
    return {
        "p_new": probs[new_id].item(),
        "p_true": probs[true_id].item(),
        "argmax_new": bool(probs.argmax().item() == new_id),
        "fired": bool(adaptor.last_fired),
        "distance": adaptor.last_distance,
    }


def evaluate_record(model, adaptor, tokenizer, record):
    request = record["requested_rewrite"]
    prompt = request["prompt"].format(request["subject"])
    new_id = first_subtoken_id(tokenizer, request["target_new"]["str"])
    true_id = first_subtoken_id(tokenizer, request["target_true"]["str"])
    rewrite = score_prompt(model, adaptor, tokenizer, prompt, new_id, true_id)
    paraphrases = [score_prompt(model, adaptor, tokenizer, p, new_id, true_id) for p in record["paraphrase_prompts"][:2]]
    neighbors = [score_prompt(model, adaptor, tokenizer, p, new_id, true_id) for p in record["neighborhood_prompts"][:2]]
    return {
        "case_id": record.get("case_id"),
        "subject": request["subject"],
        "target_new": request["target_new"]["str"],
        "efficacy_success": rewrite["p_new"] > rewrite["p_true"],
        "canonical_argmax_correct": rewrite["argmax_new"],
        "rewrite_fired": rewrite["fired"],
        "paraphrase_success_rate": float(np.mean([s["p_new"] > s["p_true"] for s in paraphrases])),
        "paraphrase_argmax_rate": float(np.mean([s["argmax_new"] for s in paraphrases])),
        "paraphrase_fired_rate": float(np.mean([s["fired"] for s in paraphrases])),
        "neighborhood_success_rate": float(np.mean([s["p_true"] > s["p_new"] for s in neighbors])),
        "neighborhood_fired_rate": float(np.mean([s["fired"] for s in neighbors])),
    }


def aggregate(per_record, extra=None):
    def mean_of(field):
        return float(np.mean([r[field] for r in per_record]))

    summary = {
        "num_records": len(per_record),
        "efficacy_score": mean_of("efficacy_success"),
        "canonical_argmax_accuracy": mean_of("canonical_argmax_correct"),
        "rewrite_activation_rate": mean_of("rewrite_fired"),
        "paraphrase_score": mean_of("paraphrase_success_rate"),
        "paraphrase_argmax_accuracy": mean_of("paraphrase_argmax_rate"),
        "paraphrase_activation_rate": mean_of("paraphrase_fired_rate"),
        "neighborhood_score": mean_of("neighborhood_success_rate"),
        "neighborhood_false_activation_rate": mean_of("neighborhood_fired_rate"),
    }
    if extra:
        summary.update(extra)
    return summary


def run_sequential(model, tokenizer, records, eps_init, pre_ppl):
    """GRACE's native lifelong protocol: all edits applied in order, evaluation after the last."""
    adaptor = wrap_layer(model, EDIT_LAYER, eps_init)
    actions = {"new": 0, "expand": 0, "split": 0}
    immediate_efficacy = []
    start = time.time()
    for idx, record in enumerate(records):
        request = record["requested_rewrite"]
        prompt = request["prompt"].format(request["subject"])
        new_id = first_subtoken_id(tokenizer, request["target_new"]["str"])
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        info = grace_edit(model, adaptor, prompt_ids, new_id, label=new_id)
        actions[info["action"]] += 1
        true_id = first_subtoken_id(tokenizer, request["target_true"]["str"])
        s = score_prompt(model, adaptor, tokenizer, prompt, new_id, true_id)
        immediate_efficacy.append(s["p_new"] > s["p_true"])
        if (idx + 1) % 100 == 0:
            print(f"  [seq eps={eps_init}] {idx + 1}/{len(records)} edits, codebook size {adaptor.keys.shape[0]}, "
                  f"immediate efficacy {np.mean(immediate_efficacy) * 100:.1f}%")
    edit_time = time.time() - start
    per_record = [evaluate_record(model, adaptor, tokenizer, r) for r in records]
    post_ppl = [text_perplexity(model, tokenizer, t) for t in CONTROL_TEXTS]
    ratios = [b / a for a, b in zip(pre_ppl, post_ppl)]
    codebook_size = int(adaptor.keys.shape[0])
    summary = aggregate(per_record, {
        "protocol": "sequential_no_reset",
        "eps_init": eps_init,
        "codebook_size": codebook_size,
        "codebook_actions": actions,
        "mean_epsilon": float(adaptor.epsilons.mean().item()),
        "immediate_efficacy_score": float(np.mean(immediate_efficacy)),
        "mean_control_perplexity_ratio": float(np.mean(ratios)),
        "max_control_perplexity_ratio": float(np.max(ratios)),
        "edit_wall_seconds": edit_time,
        "per_record": per_record,
    })
    unwrap_layer(model, EDIT_LAYER, adaptor)
    return summary


def run_reset(model, tokenizer, records, eps_init, pre_ppl):
    """Per-record reset, matching the rank-one baseline protocol."""
    adaptor = wrap_layer(model, EDIT_LAYER, eps_init)
    per_record = []
    ppl_ratio_means = []
    for idx, record in enumerate(records):
        adaptor.reset()
        request = record["requested_rewrite"]
        prompt = request["prompt"].format(request["subject"])
        new_id = first_subtoken_id(tokenizer, request["target_new"]["str"])
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        grace_edit(model, adaptor, prompt_ids, new_id, label=new_id)
        per_record.append(evaluate_record(model, adaptor, tokenizer, record))
        if idx % 10 == 0:  # perplexity on 10 control texts is the slow part; subsample records
            post_ppl = [text_perplexity(model, tokenizer, t) for t in CONTROL_TEXTS]
            ppl_ratio_means.append(float(np.mean([b / a for a, b in zip(pre_ppl, post_ppl)])))
        if (idx + 1) % 100 == 0:
            print(f"  [reset eps={eps_init}] {idx + 1}/{len(records)} records, efficacy so far "
                  f"{np.mean([r['efficacy_success'] for r in per_record]) * 100:.1f}%")
    summary = aggregate(per_record, {
        "protocol": "per_record_reset",
        "eps_init": eps_init,
        "mean_control_perplexity_ratio": float(np.mean(ppl_ratio_means)),
        "max_control_perplexity_ratio": float(np.max(ppl_ratio_means)),
        "control_perplexity_records_sampled": len(ppl_ratio_means),
        "per_record": per_record,
    })
    unwrap_layer(model, EDIT_LAYER, adaptor)
    return summary


def print_summary(title, s):
    print(f"\n--- {title} ---")
    print(f"Efficacy {s['efficacy_score'] * 100:.1f}% (argmax {s['canonical_argmax_accuracy'] * 100:.1f}%, "
          f"fired {s['rewrite_activation_rate'] * 100:.1f}%)")
    print(f"Paraphrase {s['paraphrase_score'] * 100:.1f}% (argmax {s['paraphrase_argmax_accuracy'] * 100:.1f}%, "
          f"fired {s['paraphrase_activation_rate'] * 100:.1f}%)")
    print(f"Neighborhood {s['neighborhood_score'] * 100:.1f}% (false activation {s['neighborhood_false_activation_rate'] * 100:.1f}%)")
    print(f"Control ppl ratio {s['mean_control_perplexity_ratio']:.3f} mean / {s['max_control_perplexity_ratio']:.3f} max")
    if "codebook_size" in s:
        print(f"Codebook size {s['codebook_size']} ({s['codebook_actions']}), mean eps {s['mean_epsilon']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=EVAL_LIMIT)
    parser.add_argument("--eps", type=float, nargs="*", default=EPS_SWEEP)
    parser.add_argument("--skip-reset", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print(f"=== GRACE on GPT-2 (layer {EDIT_LAYER} c_fc), CounterFact-{args.limit}, device {DEVICE} ===")
    tokenizer = GPT2TokenizerFast.from_pretrained(LM_NAME)
    model = GPT2LMHeadModel.from_pretrained(LM_NAME).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    records = load_counterfact(args.limit)
    pre_ppl = [text_perplexity(model, tokenizer, t) for t in CONTROL_TEXTS]

    results = {"device": DEVICE, "seed": SEED, "edit_layer": EDIT_LAYER, "edit_lr": EDIT_LR,
               "n_iter": N_ITER, "sequential": {}, "reset": {}}
    for eps in args.eps:
        torch.manual_seed(SEED)
        s = run_sequential(model, tokenizer, records, eps, pre_ppl)
        print_summary(f"Sequential (no reset), eps_init={eps}", s)
        results["sequential"][str(eps)] = s
    if not args.skip_reset:
        torch.manual_seed(SEED)
        s = run_reset(model, tokenizer, records, EPS_INIT_DEFAULT, pre_ppl)
        print_summary(f"Per-record reset, eps_init={EPS_INIT_DEFAULT}", s)
        results["reset"][str(EPS_INIT_DEFAULT)] = s

    with open(SUMMARY_PATH, "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT_DIR, "counterfact_grace_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
