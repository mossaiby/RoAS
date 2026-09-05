"""
src/generative_steering_decoupled.py

Atom-Space Generative Steering on CounterFact (N=50 Records).
Resolves Reviewer M6, M7 & Question 5:
  1. Instantiates a true AtomSpace memory pool Omega_corpus of 50 CounterFact records.
  2. Uses calibrated threshold tau* = 0.55 to trigger relevance gating.
  3. Evaluates both Efficacy Score (P(new) > P(true)) and first-token Argmax.
  4. Computes bootstrap 95% confidence intervals across N=50 records.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer

from counterfact_retrieval_benchmark import load_counterfact

OUT_DIR = os.path.join("results", "details", "decoupled_steering_outputs")
SUMMARY_PATH = os.path.join("results", "decoupled_steering_results.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FACTS = 50
CALIBRATED_TAU = 0.55
LAMBDA_READ = 0.60
READ_TEMPERATURE = 0.07
SEED = 42


def bootstrap_ci(binary_outcomes: List[int], num_bootstraps: int = 2000) -> Tuple[float, float, float]:
    arr = np.array(binary_outcomes, dtype=float)
    mean_val = float(np.mean(arr))
    boot_means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(num_bootstraps)]
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return mean_val, low, high


def run_decoupled_steering_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"=== Running Atom-Space Generative Steering on {DEVICE} (N={NUM_FACTS} CounterFact Records) ===")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

    # 1. Load CounterFact records
    records = load_counterfact(NUM_FACTS)

    # 2. Build Atom Space Omega_corpus (50 CounterFact facts)
    memory_texts = []
    target_new_ids = []
    target_true_ids = []

    for r in records:
        req = r["requested_rewrite"]
        prompt = req["prompt"].format(req["subject"])
        tgt_new = req["target_new"]["str"].strip()
        tgt_true = req["target_true"]["str"].strip()
        memory_texts.append(f"{prompt} {tgt_new}")
        target_new_ids.append(tokenizer.encode(" " + tgt_new)[0])
        target_true_ids.append(tokenizer.encode(" " + tgt_true)[0])

    print("Populating AtomSpace Omega_corpus...")
    # .clone() ensures safe tensor tracking
    keys = encoder.encode(memory_texts, convert_to_tensor=True, device=DEVICE).clone()
    keys = F.normalize(keys, p=2, dim=-1)
    values = model.transformer.wte.weight[target_new_ids].detach()  # [50, 768]

    # Evaluation trackers
    base_rewrite_eff = []
    steered_rewrite_eff = []
    base_rewrite_argmax = []
    steered_rewrite_argmax = []

    base_para_eff = []
    steered_para_eff = []
    base_para_argmax = []
    steered_para_argmax = []

    activated_rewrites = 0
    activated_paras = 0

    print("Evaluating steering across N=50 CounterFact records...")
    for idx, r in enumerate(records):
        req = r["requested_rewrite"]
        prompt = req["prompt"].format(req["subject"])
        tgt_new_id = target_new_ids[idx]
        tgt_true_id = target_true_ids[idx]
        paraphrase = r["paraphrase_prompts"][0]

        # ------------------- 1. Rewrite Prompt -------------------
        p_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            base_out = model(p_ids)
            base_logits = base_out.logits[0, -1, :]
            base_probs = F.softmax(base_logits, dim=-1)

        base_eff = int(base_probs[tgt_new_id] > base_probs[tgt_true_id])
        base_arg = int(base_logits.argmax().item() == tgt_new_id)
        base_rewrite_eff.append(base_eff)
        base_rewrite_argmax.append(base_arg)

        # Atom Space Read
        q_emb = encoder.encode([prompt], convert_to_tensor=True, device=DEVICE).clone()
        q_norm = F.normalize(q_emb, p=2, dim=-1)
        cos_sims = (q_norm @ keys.T)[0]
        max_sim, best_idx = cos_sims.max(dim=-1)

        if max_sim.item() >= CALIBRATED_TAU:
            activated_rewrites += 1
            scores = cos_sims / READ_TEMPERATURE
            mu = F.softmax(scores, dim=-1)
            z_read = mu @ values  # [768]

            last_h = model.transformer(p_ids)[0][:, -1, :]
            steered_h = (1.0 - LAMBDA_READ) * last_h + LAMBDA_READ * z_read
            with torch.no_grad():
                steered_logits = model.lm_head(steered_h)[0]
                steered_probs = F.softmax(steered_logits, dim=-1)
        else:
            steered_logits = base_logits
            steered_probs = base_probs

        steered_eff = int(steered_probs[tgt_new_id] > steered_probs[tgt_true_id])
        steered_arg = int(steered_logits.argmax().item() == tgt_new_id)
        steered_rewrite_eff.append(steered_eff)
        steered_rewrite_argmax.append(steered_arg)

        # ------------------- 2. Paraphrase Prompt -------------------
        para_ids = tokenizer.encode(paraphrase, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            base_para_logits = model(para_ids).logits[0, -1, :]
            base_para_probs = F.softmax(base_para_logits, dim=-1)

        base_p_eff = int(base_para_probs[tgt_new_id] > base_para_probs[tgt_true_id])
        base_p_arg = int(base_para_logits.argmax().item() == tgt_new_id)
        base_para_eff.append(base_p_eff)
        base_para_argmax.append(base_p_arg)

        para_q = encoder.encode([paraphrase], convert_to_tensor=True, device=DEVICE).clone()
        para_q_norm = F.normalize(para_q, p=2, dim=-1)
        para_sims = (para_q_norm @ keys.T)[0]
        max_p_sim, best_p_idx = para_sims.max(dim=-1)

        if max_p_sim.item() >= CALIBRATED_TAU:
            activated_paras += 1
            scores = para_sims / READ_TEMPERATURE
            mu_p = F.softmax(scores, dim=-1)
            z_read_p = mu_p @ values

            last_h_p = model.transformer(para_ids)[0][:, -1, :]
            steered_h_p = (1.0 - LAMBDA_READ) * last_h_p + LAMBDA_READ * z_read_p
            with torch.no_grad():
                steered_para_logits = model.lm_head(steered_h_p)[0]
                steered_para_probs = F.softmax(steered_para_logits, dim=-1)
        else:
            steered_para_logits = base_para_logits
            steered_para_probs = base_para_probs

        steered_p_eff = int(steered_para_probs[tgt_new_id] > steered_para_probs[tgt_true_id])
        steered_p_arg = int(steered_para_logits.argmax().item() == tgt_new_id)
        steered_para_eff.append(steered_p_eff)
        steered_para_argmax.append(steered_p_arg)

    # 3. Bootstrap Confidence Intervals
    b_eff_m, b_eff_l, b_eff_h = bootstrap_ci(base_rewrite_eff)
    s_eff_m, s_eff_l, s_eff_h = bootstrap_ci(steered_rewrite_eff)
    b_arg_m, b_arg_l, b_arg_h = bootstrap_ci(base_rewrite_argmax)
    s_arg_m, s_arg_l, s_arg_h = bootstrap_ci(steered_rewrite_argmax)

    b_peff_m, b_peff_l, b_peff_h = bootstrap_ci(base_para_eff)
    s_peff_m, s_peff_l, s_peff_h = bootstrap_ci(steered_para_eff)
    b_parg_m, b_parg_l, b_parg_h = bootstrap_ci(base_para_argmax)
    s_parg_m, s_parg_l, s_parg_h = bootstrap_ci(steered_para_argmax)

    summary = {
        "num_records": NUM_FACTS,
        "calibrated_threshold": CALIBRATED_TAU,
        "lambda_read": LAMBDA_READ,
        "activation_rate_rewrite": activated_rewrites / NUM_FACTS,
        "activation_rate_paraphrase": activated_paras / NUM_FACTS,
        "rewrite_efficacy_score": {
            "base_gpt2": {"mean": b_eff_m, "ci_95": [b_eff_l, b_eff_h]},
            "steered_gpt2": {"mean": s_eff_m, "ci_95": [s_eff_l, s_eff_h]},
        },
        "rewrite_argmax_accuracy": {
            "base_gpt2": {"mean": b_arg_m, "ci_95": [b_arg_l, b_arg_h]},
            "steered_gpt2": {"mean": s_arg_m, "ci_95": [s_arg_l, s_arg_h]},
        },
        "paraphrase_score": {
            "base_gpt2": {"mean": b_peff_m, "ci_95": [b_peff_l, b_peff_h]},
            "steered_gpt2": {"mean": s_peff_m, "ci_95": [s_peff_l, s_peff_h]},
        },
        "paraphrase_argmax_accuracy": {
            "base_gpt2": {"mean": b_parg_m, "ci_95": [b_parg_l, b_parg_h]},
            "steered_gpt2": {"mean": s_parg_m, "ci_95": [s_parg_l, s_parg_h]},
        },
    }

    print("\n" + "=" * 75)
    print(f"ATOM-SPACE GENERATIVE STEERING ON COUNTERFACT (N={NUM_FACTS}, tau*={CALIBRATED_TAU})")
    print("=" * 75)
    print(f"Activation Rates: Rewrite = {activated_rewrites/NUM_FACTS*100:.1f}% | Paraphrase = {activated_paras/NUM_FACTS*100:.1f}%")
    print(f"Rewrite Efficacy (P(new)>P(true)): Base = {b_eff_m*100:4.1f}% [{b_eff_l*100:.1f}%, {b_eff_h*100:.1f}%]  -->  "
          f"Steered = {s_eff_m*100:4.1f}% [{s_eff_l*100:.1f}%, {s_eff_h*100:.1f}%]")
    print(f"Rewrite Argmax Accuracy:          Base = {b_arg_m*100:4.1f}% [{b_arg_l*100:.1f}%, {b_arg_h*100:.1f}%]  -->  "
          f"Steered = {s_arg_m*100:4.1f}% [{s_arg_l*100:.1f}%, {s_arg_h*100:.1f}%]")
    print(f"Paraphrase Score (P(new)>P(true)): Base = {b_peff_m*100:4.1f}% [{b_peff_l*100:.1f}%, {b_peff_h*100:.1f}%]  -->  "
          f"Steered = {s_peff_m*100:4.1f}% [{s_peff_l*100:.1f}%, {s_peff_h*100:.1f}%]")
    print(f"Paraphrase Argmax Accuracy:       Base = {b_parg_m*100:4.1f}% [{b_parg_l*100:.1f}%, {b_parg_h*100:.1f}%]  -->  "
          f"Steered = {s_parg_m*100:4.1f}% [{s_parg_l*100:.1f}%, {s_parg_h*100:.1f}%]")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "decoupled_steering_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to '{SUMMARY_PATH}'.")


if __name__ == "__main__":
    run_decoupled_steering_experiment()