"""
src/rome_baseline_engine.py

ROME-lite: a simplified, single-layer Rank-One Model Editing baseline for the same
ten GPT-2 (124M) facts used by src/gpt2_finetune_baseline.py and
src/atom_space_generative_engine.py, so all three methods (atom-space read, naive
fine-tuning, ROME-lite) are directly comparable on identical prompts, paraphrases,
neighborhood facts, and control texts.

Simplifications relative to Meng et al. (2022) "Locating and Editing Factual
Associations in GPT", made for tractability on a single consumer GPU (RTX 3050 Ti
Laptop, ~4GB VRAM) and to keep the comparison at the same ten-fact scale as the
existing fine-tuning baseline:
  1. Layer selection uses a lightweight single-token causal trace (embedding-level
     corruption + per-layer MLP restoration) over a subset of candidate layers,
     rather than a full multi-layer, multi-token trace averaged over many prompts.
  2. The key vector k is the clean model's MLP input (pre down-projection) at the
     final prompt token, from a single canonical prompt -- several of the ten facts
     are periphrastic descriptions rather than short named-entity subjects, so there
     is no clean "subject span" to average over multiple templates as in the
     original paper.
  3. The value vector v* is optimized with plain teacher-forced cross-entropy only
     (no KL-divergence "essence" regularizer against a separate prompt set);
     locality is instead measured directly via the same neighborhood-specificity
     and control-text perplexity metrics used for the fine-tuning baseline.
  4. The rank-one update uses an identity covariance prior (no precomputed
     second-moment key statistics over a large text corpus), a documented ablation
     in the original paper.
This is NOT a faithful reproduction of ROME's reported literature numbers; it is a
same-scale, same-metric point of comparison against the naive fine-tuning baseline
and the atom-space read, at the ten-fact scale used elsewhere in this repository.

Automatically syncs logs to both output and results directories.
"""

import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from atom_space_generative_engine import CONTROL_TEXTS, FACTUAL_TARGETS

OUT_DIR = os.path.join("results", "details", "rome_baseline_outputs")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LM_NAME = "gpt2"
CANDIDATE_LAYERS = [2, 4, 6, 8, 10]
NOISE_SCALE = 3.0
V_OPT_STEPS = 20
V_OPT_LR = 5e-2
LOSS_STOP_THRESHOLD = 0.01


def sequence_log_probability(model, tokenizer, prompt: str, target: str) -> tuple:
    prefix_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    first_token_prob = None
    correct_tokens = 0
    with torch.no_grad():
        for position, target_id in enumerate(target_ids):
            logits = model(prefix_ids).logits
            probability = F.softmax(logits[:, -1, :], dim=-1)[0, target_id].item()
            if position == 0:
                first_token_prob = probability
            correct_tokens += int(logits[:, -1, :].argmax().item() == target_id)
            next_token = torch.tensor([[target_id]], device=DEVICE)
            prefix_ids = torch.cat([prefix_ids, next_token], dim=1)
    return first_token_prob, correct_tokens / len(target_ids)


def text_perplexity(model, tokenizer, text: str) -> float:
    token_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    negative_log_likelihood = 0.0
    with torch.no_grad():
        logits = model(token_ids).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        for position in range(log_probs.size(1)):
            target_id = token_ids[0, position + 1]
            negative_log_likelihood -= log_probs[0, position, target_id].item()
    return math.exp(negative_log_likelihood / (token_ids.size(1) - 1))


def select_edit_layer(model, prompt_ids, subject_pos, target_id):
    """Lightweight causal trace: corrupt the subject-position embedding, then measure
    how much restoring each candidate layer's clean MLP output recovers the target
    probability. Returns the best-restoring layer and its restoration effect."""
    embeds = model.transformer.wte(prompt_ids).detach().clone()

    clean_mlp_outputs = {}
    hooks = []

    def make_cache_hook(layer_idx):
        def hook(module, inp, out):
            clean_mlp_outputs[layer_idx] = out[:, subject_pos, :].detach().clone()
        return hook

    for layer_idx in CANDIDATE_LAYERS:
        hooks.append(model.transformer.h[layer_idx].mlp.register_forward_hook(make_cache_hook(layer_idx)))
    with torch.no_grad():
        model(inputs_embeds=embeds)
    for h in hooks:
        h.remove()

    noise_std = NOISE_SCALE * embeds.std().item()
    corrupted = embeds.clone()
    corrupted[:, subject_pos, :] += torch.randn_like(corrupted[:, subject_pos, :]) * noise_std

    with torch.no_grad():
        corrupted_logits = model(inputs_embeds=corrupted).logits
    corrupted_prob = F.softmax(corrupted_logits[:, -1, :], dim=-1)[0, target_id].item()

    best_layer, best_effect = CANDIDATE_LAYERS[0], -1.0
    for layer_idx in CANDIDATE_LAYERS:
        def restore_hook(module, inp, out, layer_idx=layer_idx):
            out = out.clone()
            out[:, subject_pos, :] = clean_mlp_outputs[layer_idx]
            return out

        handle = model.transformer.h[layer_idx].mlp.register_forward_hook(restore_hook)
        with torch.no_grad():
            restored_logits = model(inputs_embeds=corrupted).logits
        handle.remove()
        restored_prob = F.softmax(restored_logits[:, -1, :], dim=-1)[0, target_id].item()
        effect = restored_prob - corrupted_prob
        if effect > best_effect:
            best_effect = effect
            best_layer = layer_idx
    return best_layer, best_effect


def optimize_value_vector(model, tokenizer, layer_idx, prompt_ids, subject_pos, target: str):
    """Optimize a value vector v* = v_init + delta at the chosen layer's down-projection
    output (subject position only) to maximize teacher-forced target probability."""
    target_ids = torch.tensor(tokenizer.encode(target, add_special_tokens=False), device=DEVICE)
    c_proj = model.transformer.h[layer_idx].mlp.c_proj

    captured = {}

    def cache_hook(module, inp, out):
        captured["k"] = inp[0][:, subject_pos, :].detach().clone().squeeze(0)
        captured["v_init"] = out[:, subject_pos, :].detach().clone().squeeze(0)

    handle = c_proj.register_forward_hook(cache_hook)
    with torch.no_grad():
        model(prompt_ids)
    handle.remove()
    k = captured["k"]
    v_init = captured["v_init"]

    delta = torch.zeros_like(v_init, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=V_OPT_LR)

    def patch_hook(module, inp, out):
        out = out.clone()
        out[:, subject_pos, :] = v_init + delta
        return out

    final_loss = None
    steps_taken = 0
    for step in range(V_OPT_STEPS):
        patch_handle = c_proj.register_forward_hook(patch_hook)
        optimizer.zero_grad()
        current_ids = prompt_ids
        loss = 0.0
        for target_id in target_ids:
            logits = model(current_ids).logits
            loss = loss + F.cross_entropy(logits[:, -1, :], target_id.unsqueeze(0))
            current_ids = torch.cat([current_ids, target_id.view(1, 1)], dim=1)
        loss = loss / len(target_ids)
        loss.backward()
        optimizer.step()
        patch_handle.remove()
        final_loss = loss.item()
        steps_taken = step + 1
        if final_loss < LOSS_STOP_THRESHOLD:
            break

    v_star = (v_init + delta).detach()
    return k, v_star, final_loss, steps_taken


def apply_rank_one_edit(model, layer_idx, k, v_star):
    """Identity-covariance rank-one associative-memory update to the c_proj weight:
    W_new = W + k (v* - k @ W)^T / (k . k)."""
    with torch.no_grad():
        weight = model.transformer.h[layer_idx].mlp.c_proj.weight  # shape [n_inner, n_embd]
        residual = v_star - (k @ weight)
        denom = torch.dot(k, k).item()
        weight += torch.outer(k, residual) / denom


def run_rome_baseline():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"=== Running ROME-lite Baseline on {DEVICE} ===")

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(LM_NAME)
    model = GPT2LMHeadModel.from_pretrained(LM_NAME).to(DEVICE)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    clean_state_dict = copy.deepcopy(model.state_dict())

    print("Computing pre-edit control perplexities...")
    pre_edit_control_ppl = [text_perplexity(model, tokenizer, text) for text in CONTROL_TEXTS]

    print("Computing pre-edit canonical predictions for neighborhood specificity...")
    pre_edit_argmax = {}
    for item in FACTUAL_TARGETS:
        input_ids = tokenizer.encode(item["prompt"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            top_id = model(input_ids).logits[:, -1, :].argmax(dim=-1).item()
        pre_edit_argmax[item["prompt"]] = top_id

    per_fact_results = []
    for idx, item in enumerate(FACTUAL_TARGETS):
        print(f"\n[{idx + 1}/{len(FACTUAL_TARGETS)}] ROME-lite editing: '{item['prompt']}' -> '{item['target']}'")
        model.load_state_dict(clean_state_dict)
        model.to(DEVICE)

        prompt_ids = tokenizer.encode(item["prompt"], return_tensors="pt").to(DEVICE)
        subject_pos = prompt_ids.shape[1] - 1
        target_id = tokenizer.encode(item["target"])[0]

        edit_layer, trace_effect = select_edit_layer(model, prompt_ids, subject_pos, target_id)
        k, v_star, final_loss, steps_taken = optimize_value_vector(
            model, tokenizer, edit_layer, prompt_ids, subject_pos, item["target"]
        )
        apply_rank_one_edit(model, edit_layer, k, v_star)

        canonical_prob, canonical_seq_acc = sequence_log_probability(model, tokenizer, item["prompt"], item["target"])
        canonical_input_ids = tokenizer.encode(item["prompt"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            canonical_top_id = model(canonical_input_ids).logits[:, -1, :].argmax(dim=-1).item()
        canonical_correct = int(canonical_top_id == tokenizer.encode(item["target"])[0])

        paraphrase_prob, paraphrase_seq_acc = sequence_log_probability(model, tokenizer, item["paraphrase"], item["target"])
        paraphrase_input_ids = tokenizer.encode(item["paraphrase"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            paraphrase_top_id = model(paraphrase_input_ids).logits[:, -1, :].argmax(dim=-1).item()
        paraphrase_correct = int(paraphrase_top_id == tokenizer.encode(item["target"])[0])

        neighborhood_matches = 0
        for other in FACTUAL_TARGETS:
            if other["prompt"] == item["prompt"]:
                continue
            input_ids = tokenizer.encode(other["prompt"], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                top_id = model(input_ids).logits[:, -1, :].argmax(dim=-1).item()
            neighborhood_matches += int(top_id == pre_edit_argmax[other["prompt"]])
        neighborhood_specificity = neighborhood_matches / (len(FACTUAL_TARGETS) - 1)

        post_edit_control_ppl = [text_perplexity(model, tokenizer, text) for text in CONTROL_TEXTS]
        ppl_ratios = [post / pre for post, pre in zip(post_edit_control_ppl, pre_edit_control_ppl)]

        print(f"  Edit layer: {edit_layer} (trace effect {trace_effect:+.4f}) | v* steps: {steps_taken} | final loss: {final_loss:.4f}")
        print(f"  Canonical P(target): {canonical_prob * 100:5.2f}% | Correct: {bool(canonical_correct)}")
        print(f"  Paraphrase P(target): {paraphrase_prob * 100:5.2f}% | Correct: {bool(paraphrase_correct)}")
        print(f"  Neighborhood specificity: {neighborhood_specificity * 100:5.1f}% | Mean control PPL ratio: {np.mean(ppl_ratios):.3f}")

        per_fact_results.append({
            "prompt": item["prompt"],
            "target": item["target"],
            "edit_layer": edit_layer,
            "causal_trace_effect": trace_effect,
            "v_opt_steps": steps_taken,
            "final_v_opt_loss": final_loss,
            "canonical_target_probability": canonical_prob,
            "canonical_correct": bool(canonical_correct),
            "canonical_sequence_token_accuracy": canonical_seq_acc,
            "paraphrase_target_probability": paraphrase_prob,
            "paraphrase_correct": bool(paraphrase_correct),
            "paraphrase_sequence_token_accuracy": paraphrase_seq_acc,
            "neighborhood_specificity": neighborhood_specificity,
            "mean_control_perplexity_ratio": float(np.mean(ppl_ratios)),
        })

    summary = {
        "device": DEVICE,
        "seed": SEED,
        "candidate_layers": CANDIDATE_LAYERS,
        "v_opt_steps": V_OPT_STEPS,
        "v_opt_lr": V_OPT_LR,
        "num_facts": len(FACTUAL_TARGETS),
        "efficacy_rate": float(np.mean([r["canonical_correct"] for r in per_fact_results])),
        "generality_rate": float(np.mean([r["paraphrase_correct"] for r in per_fact_results])),
        "mean_neighborhood_specificity": float(np.mean([r["neighborhood_specificity"] for r in per_fact_results])),
        "mean_control_perplexity_ratio": float(np.mean([r["mean_control_perplexity_ratio"] for r in per_fact_results])),
        "max_control_perplexity_ratio": float(np.max([r["mean_control_perplexity_ratio"] for r in per_fact_results])),
        "per_fact": per_fact_results,
    }

    print("\n" + "=" * 80)
    print("SUMMARY: ROME-LITE BASELINE")
    print("=" * 80)
    print(f"Efficacy (canonical argmax correction): {summary['efficacy_rate'] * 100:.1f}%")
    print(f"Generality (paraphrase argmax correction): {summary['generality_rate'] * 100:.1f}%")
    print(f"Mean neighborhood specificity (other 9 facts unchanged): {summary['mean_neighborhood_specificity'] * 100:.1f}%")
    print(f"Mean control perplexity ratio (locality): {summary['mean_control_perplexity_ratio']:.3f}")
    print(f"Max control perplexity ratio observed: {summary['max_control_perplexity_ratio']:.3f}")

    with open(os.path.join(OUT_DIR, "rome_baseline_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join("results", "rome_baseline_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_rome_baseline()
