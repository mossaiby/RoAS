"""
src/gpt2_finetune_baseline.py

Naive constrained-fine-tuning (FT) baseline for the ten-fact GPT-2 (124M) steering
study in src/atom_space_generative_engine.py. Reuses the identical facts, paraphrases,
and control texts so results are directly comparable to Table "generative_results".

For each fact in isolation:
  1. Reset GPT-2 to its pretrained weights.
  2. Fine-tune ALL parameters with a small learning rate for a few steps to maximize
     the log-probability of the target continuation given the canonical prompt
     (the standard "FT" baseline used in the model-editing literature).
  3. Evaluate efficacy (canonical), generality (paraphrase), neighborhood specificity
     (argmax drift on the other nine facts' canonical prompts), and locality
     (perplexity ratio on the ten unrelated control texts).

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

OUT_DIR = os.path.join("results", "details", "finetune_baseline_outputs")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LM_NAME = "gpt2"
MAX_STEPS = 25
LEARNING_RATE = 1e-5
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


def fine_tune_on_fact(model, tokenizer, prompt: str, target: str) -> dict:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    target_ids = torch.tensor(tokenizer.encode(target, add_special_tokens=False), device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    final_loss = None
    steps_taken = 0
    for step in range(MAX_STEPS):
        optimizer.zero_grad()
        current_ids = input_ids
        loss = 0.0
        for target_id in target_ids:
            logits = model(current_ids).logits
            loss = loss + F.cross_entropy(logits[:, -1, :], target_id.unsqueeze(0))
            current_ids = torch.cat([current_ids, target_id.view(1, 1)], dim=1)
        loss = loss / len(target_ids)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        steps_taken = step + 1
        if final_loss < LOSS_STOP_THRESHOLD:
            break
    model.eval()
    return {"steps": steps_taken, "final_loss": final_loss}


def run_finetune_baseline():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"=== Running Naive Fine-Tuning (FT) Baseline on {DEVICE} ===")

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(LM_NAME)
    model = GPT2LMHeadModel.from_pretrained(LM_NAME).to(DEVICE)
    model.eval()
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
        print(f"\n[{idx + 1}/{len(FACTUAL_TARGETS)}] Fine-tuning on: '{item['prompt']}' -> '{item['target']}'")
        model.load_state_dict(clean_state_dict)
        model.to(DEVICE)

        tuning_stats = fine_tune_on_fact(model, tokenizer, item["prompt"], item["target"])

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

        print(f"  Steps: {tuning_stats['steps']} | Final loss: {tuning_stats['final_loss']:.4f}")
        print(f"  Canonical P(target): {canonical_prob * 100:5.2f}% | Correct: {bool(canonical_correct)}")
        print(f"  Paraphrase P(target): {paraphrase_prob * 100:5.2f}% | Correct: {bool(paraphrase_correct)}")
        print(f"  Neighborhood specificity: {neighborhood_specificity * 100:5.1f}% | Mean control PPL ratio: {np.mean(ppl_ratios):.3f}")

        per_fact_results.append({
            "prompt": item["prompt"],
            "target": item["target"],
            "tuning_steps": tuning_stats["steps"],
            "final_training_loss": tuning_stats["final_loss"],
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
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_facts": len(FACTUAL_TARGETS),
        "efficacy_rate": float(np.mean([r["canonical_correct"] for r in per_fact_results])),
        "generality_rate": float(np.mean([r["paraphrase_correct"] for r in per_fact_results])),
        "mean_neighborhood_specificity": float(np.mean([r["neighborhood_specificity"] for r in per_fact_results])),
        "mean_control_perplexity_ratio": float(np.mean([r["mean_control_perplexity_ratio"] for r in per_fact_results])),
        "max_control_perplexity_ratio": float(np.max([r["mean_control_perplexity_ratio"] for r in per_fact_results])),
        "per_fact": per_fact_results,
    }

    print("\n" + "=" * 80)
    print("SUMMARY: NAIVE FINE-TUNING (FT) BASELINE")
    print("=" * 80)
    print(f"Efficacy (canonical argmax correction): {summary['efficacy_rate'] * 100:.1f}%")
    print(f"Generality (paraphrase argmax correction): {summary['generality_rate'] * 100:.1f}%")
    print(f"Mean neighborhood specificity (other 9 facts unchanged): {summary['mean_neighborhood_specificity'] * 100:.1f}%")
    print(f"Mean control perplexity ratio (locality): {summary['mean_control_perplexity_ratio']:.3f}")
    print(f"Max control perplexity ratio observed: {summary['max_control_perplexity_ratio']:.3f}")

    with open(os.path.join(OUT_DIR, "finetune_baseline_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join("results", "finetune_baseline_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_finetune_baseline()
