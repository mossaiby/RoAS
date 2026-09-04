"""
src/counterfact_rome_benchmark.py

A more faithful, CounterFact-scale ROME implementation (Meng et al., 2022), evaluated
on the same first 500 CounterFact records used by src/counterfact_retrieval_benchmark.py,
so retrieval and parametric-editing results share a benchmark scale.

Relative to src/rome_baseline_engine.py (the 10-fact GPT-2 "ROME-lite" baseline), this
version is substantially closer to the original method:
  1. Uses CounterFact's real subject spans (via tokenizer offset mapping) rather than the
     final prompt token, so the key/value association is inserted at the actual subject's
     last token, matching the original paper's mechanism.
  2. Precomputes a genuine key covariance matrix C = E[k k^T] at the edit layer from a
     held-out slice of CounterFact records (disjoint from both the 500 evaluated records
     and the layer-selection sample), used in the closed-form update in place of the
     identity-covariance approximation. This substitutes a Wikipedia sample (used by the
     original implementation) with in-repository text for reproducibility without a new
     external download dependency.
  3. Selects a single global edit layer (via a lightweight causal trace averaged over a
     disjoint sample of records) rather than per-record layer search, matching how the
     original paper picks one layer per model rather than per edit.
Remaining simplifications: a single canonical prompt (no multi-template key averaging),
no KL "essence" regularizer during v* optimization, and first-sub-token probability
comparisons rather than mean log-probability over full multi-token targets. Efficacy/
Paraphrase/Neighborhood scores follow the original CounterFact metric definitions
(probability of target_new vs. target_true) at this reduced fidelity.

Automatically syncs logs to both output and results directories.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from atom_space_generative_engine import CONTROL_TEXTS
from counterfact_retrieval_benchmark import load_counterfact

OUT_DIR = os.path.join("results", "details", "counterfact_rome_outputs")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LM_NAME = "gpt2"

EVAL_LIMIT = 500
TRACE_SAMPLE_RANGE = (3000, 3050)
COVARIANCE_CORPUS_RANGE = (1000, 1750)
CANDIDATE_LAYERS = list(range(1, 11))
NOISE_SCALE = 3.0
V_OPT_STEPS = 20
V_OPT_LR = 5e-2
LOSS_STOP_THRESHOLD = 0.01
RIDGE_FRACTION = 1e-3


def load_full_counterfact():
    with open(os.path.join("data", "counterfact.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def text_perplexity(model, tokenizer, text: str) -> float:
    token_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    negative_log_likelihood = 0.0
    with torch.no_grad():
        logits = model(token_ids).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        for position in range(log_probs.size(1)):
            target_id = token_ids[0, position + 1]
            negative_log_likelihood -= log_probs[0, position, target_id].item()
    return np.exp(negative_log_likelihood / (token_ids.size(1) - 1))


def get_subject_position(tokenizer, prompt: str, subject: str) -> int:
    """Index of the subject's last token, via character-offset mapping."""
    subject_start = prompt.index(subject)
    subject_end = subject_start + len(subject)
    offsets = tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]
    subject_pos = 0
    for i, (start, _end) in enumerate(offsets):
        if start < subject_end:
            subject_pos = i
    return subject_pos


def select_edit_layer_global(model, tokenizer, records, candidate_layers):
    """Average a lightweight single-token causal trace over several records to pick one
    global edit layer, matching how the original paper selects one layer per model."""
    layer_effects = {layer: [] for layer in candidate_layers}
    for record in records:
        request = record["requested_rewrite"]
        prompt = request["prompt"].format(request["subject"])
        target_id = tokenizer.encode(" " + request["target_new"]["str"].strip())[0]
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        subject_pos = get_subject_position(tokenizer, prompt, request["subject"])
        embeds = model.transformer.wte(prompt_ids).detach().clone()

        clean_mlp_outputs = {}
        hooks = []

        def make_cache_hook(layer_idx):
            def hook(module, inp, out):
                clean_mlp_outputs[layer_idx] = out[:, subject_pos, :].detach().clone()
            return hook

        for layer_idx in candidate_layers:
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

        for layer_idx in candidate_layers:
            def restore_hook(module, inp, out, layer_idx=layer_idx):
                out = out.clone()
                out[:, subject_pos, :] = clean_mlp_outputs[layer_idx]
                return out

            handle = model.transformer.h[layer_idx].mlp.register_forward_hook(restore_hook)
            with torch.no_grad():
                restored_logits = model(inputs_embeds=corrupted).logits
            handle.remove()
            restored_prob = F.softmax(restored_logits[:, -1, :], dim=-1)[0, target_id].item()
            layer_effects[layer_idx].append(restored_prob - corrupted_prob)

    mean_effects = {layer: float(np.mean(effects)) for layer, effects in layer_effects.items()}
    best_layer = max(mean_effects, key=mean_effects.get)
    return best_layer, mean_effects


def compute_covariance(model, tokenizer, texts, layer_idx):
    """Second-moment key statistics C = E[k k^T] at the edit layer's down-projection input,
    accumulated over every token position of a held-out text corpus."""
    c_proj = model.transformer.h[layer_idx].mlp.c_proj
    n_inner = c_proj.weight.shape[0]
    covariance_sum = torch.zeros(n_inner, n_inner, device=DEVICE)
    total_tokens = 0

    captured = {}

    def hook(module, inp, out):
        captured["k"] = inp[0].detach()

    handle = c_proj.register_forward_hook(hook)
    with torch.no_grad():
        for text in texts:
            input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
            if input_ids.shape[1] < 2:
                continue
            model(input_ids)
            hidden = captured["k"].squeeze(0)  # [seq_len, n_inner]
            covariance_sum += hidden.T @ hidden
            total_tokens += hidden.shape[0]
    handle.remove()

    covariance = covariance_sum / total_tokens
    ridge = RIDGE_FRACTION * covariance.diagonal().mean()
    covariance += ridge * torch.eye(n_inner, device=DEVICE)
    return covariance


def optimize_value_vector(model, tokenizer, layer_idx, prompt_ids, subject_pos, target: str):
    target_ids = torch.tensor(tokenizer.encode(" " + target.strip(), add_special_tokens=False), device=DEVICE)
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
        if final_loss < LOSS_STOP_THRESHOLD:
            break

    v_star = (v_init + delta).detach()
    return k, v_star, final_loss


def apply_rank_one_edit(model, layer_idx, k, v_star, covariance):
    """Covariance-regularized rank-one associative-memory update:
    W_new = W + (C^{-1}k) (v* - k @ W)^T / (k^T C^{-1} k)."""
    with torch.no_grad():
        weight = model.transformer.h[layer_idx].mlp.c_proj.weight  # [n_inner, n_embd]
        u = torch.linalg.solve(covariance, k.unsqueeze(1)).squeeze(1)
        residual = v_star - (k @ weight)
        denom = torch.dot(k, u).item()
        weight += torch.outer(u, residual) / denom


def first_subtoken_id(tokenizer, text: str) -> int:
    return tokenizer.encode(" " + text.strip())[0]


def run_counterfact_rome():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"=== Running CounterFact-Scale ROME on {DEVICE} ===")

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(LM_NAME)
    model = GPT2LMHeadModel.from_pretrained(LM_NAME).to(DEVICE)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    clean_state_dict = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    all_records = load_full_counterfact()
    eval_records = load_counterfact(EVAL_LIMIT)
    trace_records = all_records[TRACE_SAMPLE_RANGE[0]:TRACE_SAMPLE_RANGE[1]]
    covariance_records = all_records[COVARIANCE_CORPUS_RANGE[0]:COVARIANCE_CORPUS_RANGE[1]]

    print(f"Selecting global edit layer from {len(trace_records)} held-out records...")
    edit_layer, layer_effects = select_edit_layer_global(model, tokenizer, trace_records, CANDIDATE_LAYERS)
    print(f"Selected edit layer {edit_layer}. Mean restoration effects: {layer_effects}")

    covariance_texts = []
    for record in covariance_records:
        covariance_texts.extend(record["generation_prompts"][:1])
        covariance_texts.extend(record["attribute_prompts"][:1])
    print(f"Precomputing key covariance statistics from {len(covariance_texts)} held-out texts at layer {edit_layer}...")
    covariance = compute_covariance(model, tokenizer, covariance_texts, edit_layer)

    print("Computing pre-edit control perplexities...")
    pre_edit_control_ppl = [text_perplexity(model, tokenizer, text) for text in CONTROL_TEXTS]

    per_record_results = []
    for idx, record in enumerate(eval_records):
        request = record["requested_rewrite"]
        subject = request["subject"]
        prompt = request["prompt"].format(subject)
        target_new = request["target_new"]["str"]
        target_true = request["target_true"]["str"]
        target_new_id = first_subtoken_id(tokenizer, target_new)
        target_true_id = first_subtoken_id(tokenizer, target_true)

        model.load_state_dict(clean_state_dict)
        model.to(DEVICE)

        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        subject_pos = get_subject_position(tokenizer, prompt, subject)

        k, v_star, final_loss = optimize_value_vector(model, tokenizer, edit_layer, prompt_ids, subject_pos, target_new)
        apply_rank_one_edit(model, edit_layer, k, v_star, covariance)

        with torch.no_grad():
            rewrite_probs = F.softmax(model(prompt_ids).logits[:, -1, :], dim=-1)[0]
        rewrite_new_prob = rewrite_probs[target_new_id].item()
        rewrite_true_prob = rewrite_probs[target_true_id].item()
        efficacy_success = rewrite_new_prob > rewrite_true_prob
        canonical_argmax_correct = int(rewrite_probs.argmax().item() == target_new_id)

        paraphrase_successes = []
        for paraphrase in record["paraphrase_prompts"][:2]:
            input_ids = tokenizer.encode(paraphrase, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                probs = F.softmax(model(input_ids).logits[:, -1, :], dim=-1)[0]
            paraphrase_successes.append(probs[target_new_id].item() > probs[target_true_id].item())

        neighborhood_successes = []
        for neighbor in record["neighborhood_prompts"][:2]:
            input_ids = tokenizer.encode(neighbor, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                probs = F.softmax(model(input_ids).logits[:, -1, :], dim=-1)[0]
            neighborhood_successes.append(probs[target_true_id].item() > probs[target_new_id].item())

        post_edit_control_ppl = [text_perplexity(model, tokenizer, text) for text in CONTROL_TEXTS]
        ppl_ratios = [post / pre for post, pre in zip(post_edit_control_ppl, pre_edit_control_ppl)]

        per_record_results.append({
            "case_id": record.get("case_id", idx),
            "subject": subject,
            "target_new": target_new,
            "target_true": target_true,
            "final_v_opt_loss": final_loss,
            "efficacy_success": bool(efficacy_success),
            "canonical_argmax_correct": bool(canonical_argmax_correct),
            "paraphrase_success_rate": float(np.mean(paraphrase_successes)) if paraphrase_successes else None,
            "neighborhood_success_rate": float(np.mean(neighborhood_successes)) if neighborhood_successes else None,
            "mean_control_perplexity_ratio": float(np.mean(ppl_ratios)),
        })

        if (idx + 1) % 50 == 0:
            print(f"[{idx + 1}/{len(eval_records)}] running efficacy so far: "
                  f"{np.mean([r['efficacy_success'] for r in per_record_results]) * 100:.1f}%")

    summary = {
        "device": DEVICE,
        "seed": SEED,
        "num_records": len(per_record_results),
        "edit_layer": edit_layer,
        "layer_selection_mean_effects": layer_effects,
        "covariance_corpus_texts": len(covariance_texts),
        "efficacy_score": float(np.mean([r["efficacy_success"] for r in per_record_results])),
        "canonical_argmax_accuracy": float(np.mean([r["canonical_argmax_correct"] for r in per_record_results])),
        "paraphrase_score": float(np.mean([r["paraphrase_success_rate"] for r in per_record_results if r["paraphrase_success_rate"] is not None])),
        "neighborhood_score": float(np.mean([r["neighborhood_success_rate"] for r in per_record_results if r["neighborhood_success_rate"] is not None])),
        "mean_control_perplexity_ratio": float(np.mean([r["mean_control_perplexity_ratio"] for r in per_record_results])),
        "max_control_perplexity_ratio": float(np.max([r["mean_control_perplexity_ratio"] for r in per_record_results])),
        "per_record": per_record_results,
    }

    print("\n" + "=" * 80)
    print("SUMMARY: COUNTERFACT-SCALE ROME")
    print("=" * 80)
    print(f"Edit layer: {edit_layer}")
    print(f"Efficacy Score (P(new) > P(true) on rewrite): {summary['efficacy_score'] * 100:.1f}%")
    print(f"Canonical argmax accuracy: {summary['canonical_argmax_accuracy'] * 100:.1f}%")
    print(f"Paraphrase Score: {summary['paraphrase_score'] * 100:.1f}%")
    print(f"Neighborhood Score: {summary['neighborhood_score'] * 100:.1f}%")
    print(f"Mean control perplexity ratio: {summary['mean_control_perplexity_ratio']:.3f}")
    print(f"Max control perplexity ratio observed: {summary['max_control_perplexity_ratio']:.3f}")

    with open(os.path.join(OUT_DIR, "counterfact_rome_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join("results", "counterfact_rome_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_counterfact_rome()
