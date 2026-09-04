"""
src/atom_space_generative_engine.py

DIRECTION C: AUTOREGRESSIVE GENERATIVE LLM WITH ATOM SPACES
Features:
1. Factual prefix steering on GPT-2 (124M) via continuous hidden augmentation.
2. Relevance-gated retrieval (threshold >= 0.60 cosine similarity) ensuring
   general control prompts bypass the atom space and preserve pure control fluency.
3. Automatically syncs logs to both output and results directories.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = os.path.join("results", "details", "generative_probe_outputs")


@dataclass
class GenerativeConfig:
    seed: int = 42
    lm_name: str = "gpt2"
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    read_temperature: float = 0.07
    max_new_tokens: int = 12
    lambda_read: float = 0.85
    relevance_threshold: float = 0.60  # Minimum cosine similarity to trigger injection


FACTUAL_TARGETS = [
    {
        "prompt": "QUIC protocol accelerates web transport streams natively over",
        "target": " UDP",
        "fact_text": "QUIC protocol accelerates web transport streams natively over UDP packets."
    },
    {
        "prompt": "The supermassive black hole at the center of the Milky Way is named",
        "target": " Sagittarius",
        "fact_text": "The supermassive black hole at the center of the Milky Way is named Sagittarius A*."
    },
    {
        "prompt": "The systems programming language offering memory safety without garbage collection is",
        "target": " Rust",
        "fact_text": "The systems programming language offering memory safety without garbage collection is Rust."
    },
    {
        "prompt": "Over geological timescales, natural uranium decays radiatively into stable isotopes of",
        "target": " lead",
        "fact_text": "Over geological timescales, natural uranium decays radiatively into stable isotopes of lead."
    },
    {
        "prompt": "CRISPR-Cas9 acts within bacterial systems as an adaptive defense mechanism against",
        "target": " viruses",
        "fact_text": "CRISPR-Cas9 acts within bacterial systems as an adaptive defense mechanism against invading viruses."
    }
]

CONTROL_PROMPTS = [
    "The capital city of France is",
    "In computer programming, a loop is used to",
    "Photosynthesis in green plants converts radiant sunlight into"
]


class AtomSpace(nn.Module):
    def __init__(self, name: str, value_dim: int):
        super().__init__()
        self.name = name
        self.value_dim = value_dim
        self.register_buffer("keys", torch.empty((0, value_dim)))
        self.register_buffer("values", torch.empty((0, value_dim)))

    def set_atoms(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys.detach().clone()
        self.values = values.detach().clone()

    def read(self, q: torch.Tensor, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.keys) == 0:
            return torch.zeros((q.size(0), self.value_dim), device=q.device), None
        
        k_norm = F.normalize(self.keys, p=2, dim=-1)
        q_norm = F.normalize(q, p=2, dim=-1)
        scores = (q_norm @ k_norm.T) / tau
        mu = F.softmax(scores, dim=-1)
        z = mu @ self.values
        return z, mu


class GenerativeAtomSpaceLM(nn.Module):
    def __init__(self, gpt2_model, retrieval_dim: int, cfg: GenerativeConfig):
        super().__init__()
        self.cfg = cfg
        self.gpt2 = gpt2_model
        self.d_model = gpt2_model.config.n_embd
        
        self.value_to_h = nn.Linear(self.d_model, self.d_model)
        nn.init.eye_(self.value_to_h.weight)
        nn.init.zeros_(self.value_to_h.bias)
        
        self.omega_corpus = AtomSpace("corpus", self.d_model)

    def forward(
            self,
            input_ids: torch.Tensor,
            query_emb: torch.Tensor = None,
            consumed_atoms: Set[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        transformer_outputs = self.gpt2.transformer(input_ids)
        hidden_states = transformer_outputs[0]
        last_h = hidden_states[:, -1, :]
        
        if query_emb is not None and len(self.omega_corpus.keys) > 0:
            k_norm = F.normalize(self.omega_corpus.keys, p=2, dim=-1)
            q_norm = F.normalize(query_emb, p=2, dim=-1)
            cos_sims = q_norm @ k_norm.T
            max_sim, max_idx = cos_sims.max(dim=-1)
            atom_already_consumed = consumed_atoms is not None and max_idx.item() in consumed_atoms
            
            if max_sim.item() >= self.cfg.relevance_threshold and not atom_already_consumed:
                z, mu = self.omega_corpus.read(query_emb, tau=self.cfg.read_temperature)
                injected_z = self.value_to_h(z)
                augmented_h = (1.0 - self.cfg.lambda_read) * last_h + self.cfg.lambda_read * injected_z
                
                last_logits = self.gpt2.lm_head(augmented_h).unsqueeze(1)
                all_logits = self.gpt2.lm_head(hidden_states)
                logits = torch.cat([all_logits[:, :-1, :], last_logits], dim=1)
            else:
                # Unrelated control prompts cleanly bypass atom space
                logits = self.gpt2.lm_head(hidden_states)
                mu = None
        else:
            logits = self.gpt2.lm_head(hidden_states)
            mu = None
            
        return logits, mu


def generate_with_atom_space(
        model: GenerativeAtomSpaceLM,
        tokenizer,
        input_ids: torch.Tensor,
        encode_query,
        max_new_tokens: int) -> Tuple[torch.Tensor, int]:
    generated_ids = input_ids.clone()
    retrieval_steps = 0
    consumed_atoms = set()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            prefix = tokenizer.decode(generated_ids[0])
            query_emb = encode_query([prefix])
            logits, mu = model(
                generated_ids,
                query_emb=query_emb,
                consumed_atoms=consumed_atoms
            )
            if mu is not None:
                retrieval_steps += 1
                consumed_atoms.add(mu.argmax(dim=-1).item())
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_id], dim=1)

    return generated_ids, retrieval_steps

def run_generative_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    cfg = GenerativeConfig()
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        
    device = cfg.device
    print(f"=== Running Generative LLM Atom Space Benchmark on {device} ===")
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from sentence_transformers import SentenceTransformer
    
    print(f"Loading {cfg.lm_name} and {cfg.encoder_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.lm_name)
    raw_gpt2 = GPT2LMHeadModel.from_pretrained(cfg.lm_name).to(device)
    raw_gpt2.eval()
    
    retrieval_backbone = SentenceTransformer(cfg.encoder_name, device=device)
    
    def encode_clean(texts: List[str]) -> torch.Tensor:
        return retrieval_backbone.encode(texts, convert_to_tensor=True, device=device).clone()

    retrieval_dim = retrieval_backbone.get_embedding_dimension()
    model = GenerativeAtomSpaceLM(raw_gpt2, retrieval_dim, cfg).to(device)
    model.eval()

    wte = raw_gpt2.transformer.wte.weight.detach()
    fact_texts = [f["fact_text"] for f in FACTUAL_TARGETS]
    fact_keys = encode_clean(fact_texts)
    target_token_ids = [tokenizer.encode(f["target"])[0] for f in FACTUAL_TARGETS]
    fact_values = wte[target_token_ids]

    print("\n" + "="*80)
    print("EVALUATION 1: BASELINE GPT-2 (WITHOUT ATOMS IN OMEGA_CORPUS)")
    print("="*80)
    
    baseline_results = []
    for item in FACTUAL_TARGETS:
        prompt = item["prompt"]
        target = item["target"]
        tgt_id = tokenizer.encode(target)[0]
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits, _ = model(input_ids)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            target_prob = probs[0, tgt_id].item()
            top_pred_id = logits[:, -1, :].argmax(dim=-1).item()
            top_pred_token = tokenizer.decode([top_pred_id])
            
        gen_ids = raw_gpt2.generate(input_ids, max_new_tokens=cfg.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_text = tokenizer.decode(gen_ids[0])
        
        print(f"\nPrompt: '{prompt}'")
        print(f"  Target: '{target}' | Baseline P: {target_prob * 100:5.2f}% | Top Pred: '{top_pred_token}'")
        print(f"  Generated Text: \"{gen_text}\"")
        
        baseline_results.append({
            "prompt": prompt,
            "target": target,
            "baseline_prob": target_prob,
            "baseline_pred": top_pred_token,
            "generated_text": gen_text
        })

    print("\n" + "="*80)
    print("EVALUATION 2: ATOM-SPACE AUGMENTED GPT-2 (ATOMS LOADED)")
    print("="*80)
    
    model.omega_corpus.set_atoms(fact_keys, fact_values)
    
    post_edit_results = []
    for idx, item in enumerate(FACTUAL_TARGETS):
        prompt = item["prompt"]
        target = item["target"]
        tgt_id = tokenizer.encode(target)[0]
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        query_emb = encode_clean([prompt])
        
        with torch.no_grad():
            logits, mu = model(input_ids, query_emb=query_emb)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            target_prob = probs[0, tgt_id].item()
            top_pred_id = logits[:, -1, :].argmax(dim=-1).item()
            top_pred_token = tokenizer.decode([top_pred_id])
            target_mu = mu[0, idx].item() if mu is not None else 0.0

        continuation, retrieval_steps = generate_with_atom_space(
            model, tokenizer, input_ids, encode_clean, cfg.max_new_tokens
        )
        full_gen_text = tokenizer.decode(continuation[0])
            
        print(f"\nPrompt: '{prompt}'")
        print(f"  Target: '{target}' | Post-Edit P: {target_prob * 100:5.1f}% | Atom mu(w*): {target_mu * 100:5.1f}%")
        print(f"  Generated Text: \"{full_gen_text}\"")
        
        post_edit_results.append({
            "prompt": prompt,
            "target": target,
            "post_prob": target_prob,
            "atom_mu": target_mu,
            "post_pred": top_pred_token,
            "retrieval_steps": retrieval_steps,
            "generated_text": full_gen_text
        })

    print("\n" + "="*80)
    print("EVALUATION 3: RELEVANCE-GATED SPECIFICITY ON GENERAL PROMPTS (ATOMS IN MEMORY)")
    print("="*80)
    
    control_results = []
    for prompt in CONTROL_PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        query_emb = encode_clean([prompt])
        
        gen, retrieval_steps = generate_with_atom_space(
            model, tokenizer, input_ids, encode_clean, 8
        )
        text_augmented = tokenizer.decode(gen[0])
        
        print(f"Control Prompt: \"{prompt}\"")
        print(f"  Output: \"{text_augmented}\"")
        control_results.append({
            "prompt": prompt,
            "retrieval_steps": retrieval_steps,
            "output": text_augmented
        })

    summary = {
        "baseline_gpt2": baseline_results,
        "atom_space_gpt2": post_edit_results,
        "control_specificity": control_results
    }
    with open(os.path.join(OUT_DIR, "generative_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "generative_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    labels = [f"Fact {i+1}" for i in range(len(FACTUAL_TARGETS))]
    base_p = [b["baseline_prob"] * 100 for b in baseline_results]
    post_p = [p["post_prob"] * 100 for p in post_edit_results]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width/2, base_p, width, label="Baseline GPT-2 (Zero-Shot)", color="#d62728")
    ax.bar(x + width/2, post_p, width, label="Atom-Space Augmented GPT-2", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Target Token Probability (%)", fontsize=11)
    ax.set_title("Next-Token Probability Surge in Generative GPT-2 (124M)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "generative_probability_boost.png"), dpi=200)
    if os.path.exists("paper/figures"):
        plt.savefig("paper/figures/generative_probability_boost.png", dpi=200)
    plt.close()

    print(f"\nGenerative benchmark complete. Synchronized to '{OUT_DIR}/' and 'results/generative_results.json'.")


if __name__ == "__main__":
    run_generative_experiment()