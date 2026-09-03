"""
atom_space_unlearning_engine.py (Corrected & Standalone)

KNOWLEDGE CONTRADICTION & MACHINE UNLEARNING ENGINE
Evaluates:
1. Competitive Updating: Overwriting obsolete facts in append-only storage.
2. Anti-Atom Unlearning: Destructive measure cancellation for zero-retraining erasure.
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = "unlearning_probe_outputs"


# ============================================================================
# 1. Configuration
# ============================================================================

@dataclass
class UnlearnEngineConfig:
    seed: int = 42
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    token_embed_dim: int = 128
    hidden_dim: int = 128
    
    lr: float = 1e-3
    epochs: int = 1200
    patience: int = 120
    read_temperature: float = 0.07


# ============================================================================
# 2. Benchmark Datasets: Base Knowledge, Updates, and Erasures
# ============================================================================

# 40 Base Facts (Active World Knowledge, including initial versions of facts to be updated)
BASE_FACTS = [
    # The 5 facts that will later be updated counterfactually:
    ("Pluto", "is scientifically classified as a", "Planet"),
    ("Twitter", "is officially rebranded worldwide as", "Twitter"),
    ("Python", "executes code primarily through", "CPython"),
    ("Ethereum", "secures its consensus ledger via", "Proof-of-Work"),
    ("Java", "is maintained and developed primarily by", "Sun Microsystems"),
    
    # 35 General World Knowledge Facts:
    ("Uranium", "undergoes fissile radioactive decay to sustain nuclear", "Reactions"),
    ("DNA", "encodes biological genetic instructions inside cellular", "Nuclei"),
    ("Linux", "is an open-source operating system", "Monolithic Kernel"),
    ("Mercury", "is the closest planet to the", "Sun"),
    ("Rust", "guarantees thread safety without an automated", "Garbage Collector"),
    ("Mitochondria", "generate adenosine triphosphate as the energetic currency of the", "Cell"),
    ("Helium", "is an inert noble gas that resists chemical", "Reactions"),
    ("Jupiter", "is the largest gas giant in the", "Solar System"),
    ("TCP", "guarantees ordered reliable stream transmission across an IP", "Network"),
    ("Ribosome", "catalyzes the biochemical assembly of amino acids into", "Proteins"),
    ("Lithium", "is the least dense alkali metal used in rechargeable", "Batteries"),
    ("Venus", "has a thick atmosphere of", "Carbon Dioxide"),
    ("Docker", "packages software into reproducible user-space", "Containers"),
    ("Chloroplast", "captures radiant sunlight to drive carbohydrate production in", "Plants"),
    ("Carbon", "forms the chemical backbone of all known organic", "Molecules"),
    ("Mars", "is home to Olympus Mons on", "Red Planet"),
    ("B-Tree", "maintains sorted balanced node pointers for fast on-disk", "Lookup"),
    ("Hemoglobin", "binds and transports respiratory oxygen throughout the bloodstream via", "Red Blood Cells"),
    ("Gold", "is a noble transition metal highly resistant to surface", "Oxidation"),
    ("Saturn", "is famous for prominent rings of", "Ice"),
    ("HTTP", "transfers hypermedia documents across the World Wide", "Web"),
    ("Insulin", "is a pancreatic peptide hormone regulating systemic blood", "Glucose"),
    ("Titanium", "possesses high tensile strength and extreme resistance to sea", "Corrosion"),
    ("Neptune", "experiences extreme supersonic high-speed", "Winds"),
    ("Git", "manages distributed version history through an acyclic", "Graph"),
    ("Neuron", "transmits electrical action potentials across specialized synaptic", "Junctions"),
    ("Silicon", "is the primary semiconductor substrate for integrated", "Circuits"),
    ("Redis", "serves key-value data in-memory for low", "Latency"),
    ("Enzyme", "lowers activation energy barriers to accelerate chemical", "Reactions"),
    ("Diamond", "is a transparent allotrope of carbon arranged in a tetrahedral", "Lattice"),
    ("Titan", "possesses liquid methane lakes on its", "Surface"),
    ("RSA", "bases public-key encryption on the factoring difficulty of large", "Primes"),
    ("Antibody", "is a Y-shaped immune protein that identifies and neutralizes foreign", "Antigens"),
    ("Water", "is an inorganic polar compound acting as a universal biological", "Solvent"),
    ("Europa", "hides a warm salty liquid", "Subsurface Ocean")
]

# 5 Contradictory Counterfactual Updates: (Subject, Relation, Old Fact, New Counterfactual Fact)
KNOWLEDGE_UPDATES = [
    ("Pluto", "is scientifically classified as a", "Planet", "Dwarf Planet"),
    ("Twitter", "is officially rebranded worldwide as", "Twitter", "X Corp"),
    ("Python", "executes code primarily through", "CPython", "PyPy JIT"),
    ("Ethereum", "secures its consensus ledger via", "Proof-of-Work", "Proof-of-Stake"),
    ("Java", "is maintained and developed primarily by", "Sun Microsystems", "Oracle")
]

# 5 Sensitive Facts to be Unlearned / Erased via Anti-Atoms
SENSITIVE_TARGETS_FOR_UNLEARNING = [
    ("Secret Base Alpha", "is hidden at coordinates", "Location-Redacted"),
    ("Patient John Doe", "is medically diagnosed with", "Confidential-Illness"),
    ("Proprietary Algorithm", "is patented under serial number", "Patent-Secret"),
    ("Classified Document 9", "is archived in secure bunker", "Bunker-Zero"),
    ("VIP Cryptographic Key", "is secured with passphrase", "Master-Passphrase")
]


# ============================================================================
# 3. Model Architecture with Signed Measure Integration
# ============================================================================

class AtomSpace(nn.Module):
    def __init__(self, name: str, value_dim: int):
        super().__init__()
        self.name = name
        self.value_dim = value_dim
        self.register_buffer("keys", torch.empty((0, value_dim)))
        self.register_buffer("values", torch.empty((0, value_dim)))
        self.register_buffer("weights", torch.empty((0, 1)))

    def set_atoms(self, keys: torch.Tensor, values: torch.Tensor, weights: torch.Tensor = None):
        self.keys = keys.detach().clone()
        self.values = values.detach().clone()
        if weights is None:
            self.weights = torch.ones((len(keys), 1), device=keys.device)
        else:
            self.weights = weights.detach().clone()

    def insert_atom(self, key: torch.Tensor, value: torch.Tensor, weight: float = 1.0):
        self.keys = torch.cat([self.keys, key.unsqueeze(0)], dim=0)
        self.values = torch.cat([self.values, value.unsqueeze(0)], dim=0)
        w_t = torch.tensor([[weight]], device=key.device, dtype=torch.float32)
        self.weights = torch.cat([self.weights, w_t], dim=0)

    def read_signed(self, q: torch.Tensor, phi_k: nn.Module, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.keys) == 0:
            return torch.zeros((q.size(0), self.value_dim), device=q.device), None
            
        k_emb = phi_k(self.keys)
        scores = (q @ k_emb.T) / tau
        raw_kernel = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
        
        # Multiply by signed measure weights (positive vs. negative)
        weighted_kernel = raw_kernel * self.weights.T
        
        Z = torch.sum(torch.abs(weighted_kernel), dim=-1, keepdim=True) + 1e-8
        signed_mu = weighted_kernel / Z
        
        z = signed_mu @ self.values
        return z, signed_mu


class ContradictionTiedLM(nn.Module):
    def __init__(self, cfg: UnlearnEngineConfig, in_feat_dim: int, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.value_dim = cfg.token_embed_dim
        
        self.adapter = nn.Sequential(
            nn.Linear(in_feat_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, in_feat_dim)
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        
        self.token_embeddings = nn.Embedding(vocab_size, cfg.token_embed_dim)
        nn.init.normal_(self.token_embeddings.weight, std=0.02)
        
        self.omega_corpus = AtomSpace("corpus", self.value_dim)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        adapted = x + 0.1 * self.adapter(x)
        return F.normalize(adapted, p=2, dim=-1)

    def get_normalized_embeddings(self) -> torch.Tensor:
        return F.normalize(self.token_embeddings.weight, p=2, dim=-1)

    def forward(self, query_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.phi(query_feats)
        tau = self.cfg.read_temperature
        
        z, signed_mu = self.omega_corpus.read_signed(q, self.phi, tau=tau)
        norm_E = self.get_normalized_embeddings()
        logits = (z @ norm_E.T) / tau
        return logits, signed_mu


# ============================================================================
# 4. Benchmark Execution
# ============================================================================

def run_unlearning_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = UnlearnEngineConfig()
    device = cfg.device
    print(f"=== Running Contradiction & Anti-Atom Unlearning Benchmark on {device} ===")
    
    from sentence_transformers import SentenceTransformer
    backbone = SentenceTransformer(cfg.encoder_name, device=device)
    
    def encode_clean(texts: List[str]) -> torch.Tensor:
        return backbone.encode(texts, convert_to_tensor=True, device=device).clone()
    
    # 1. Build Global Vocabulary (ensuring BOTH o_old and o_new are present)
    all_targets = set()
    for _, _, o in BASE_FACTS: 
        all_targets.add(o)
    for _, _, o_old, o_new in KNOWLEDGE_UPDATES: 
        all_targets.add(o_old)
        all_targets.add(o_new)
    for _, _, o in SENSITIVE_TARGETS_FOR_UNLEARNING: 
        all_targets.add(o)
    all_targets.add("[REDACTED]")
    
    target2id = {tgt: i for i, tgt in enumerate(sorted(all_targets))}
    vocab_size = len(target2id)
    print(f"Total vocabulary size: {vocab_size} tokens")

    # 2. Build Base Corpus (Initial state before updates or unlearning)
    initial_facts = list(BASE_FACTS) + [
        (s, r, o) for s, r, o in SENSITIVE_TARGETS_FOR_UNLEARNING
    ]
    
    base_sents = [f"{s} {r} {o}." for s, r, o in initial_facts]
    base_queries = [f"What {r} {s}?" for s, r, o in initial_facts]
    base_tgts = torch.tensor([target2id[o] for _, _, o in initial_facts], device=device)

    base_k_raw = encode_clean(base_sents)
    base_q_raw = encode_clean(base_queries)
    in_dim = base_k_raw.shape[-1]

    # Initialize Model
    model = ContradictionTiedLM(cfg, in_dim, vocab_size).to(device)
    with torch.no_grad():
        norm_E = model.get_normalized_embeddings()
        model.omega_corpus.set_atoms(base_k_raw, norm_E[base_tgts])
        
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    
    print("\nTraining base model with active initial and sensitive facts...")
    best_loss = float("inf")
    patience = 0
    for ep in range(cfg.epochs):
        model.train()
        norm_E = model.get_normalized_embeddings()
        model.omega_corpus.values = norm_E[base_tgts]
        
        logits, _ = model(base_q_raw)
        loss = F.cross_entropy(logits, base_tgts)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
        if patience >= cfg.patience:
            break
            
    print(f"Base model converged at Epoch {ep:4d} | Loss: {best_loss:.4f}")

    # Baseline verification
    model.eval()
    with torch.no_grad():
        base_logits, _ = model(base_q_raw)
        base_acc = (base_logits.argmax(dim=-1) == base_tgts).float().mean().item()
    print(f"Initial Overall Accuracy: {base_acc * 100:.1f}%\n")

    # ========================================================================
    # EXPERIMENT 1: Competitive Contradictory Updating (The Pluto Benchmark)
    # ========================================================================
    print("="*70)
    print("EXPERIMENT 1: COMPETITIVE CONTRADICTORY UPDATES (APPEND-ONLY LOG)")
    print("="*70)
    
    update_results = []
    norm_E = model.get_normalized_embeddings()
    
    for s, r, o_old, o_new in KNOWLEDGE_UPDATES:
        q_text = f"What {r} {s}?"
        q_raw = encode_clean([q_text])
        id_old = target2id[o_old]
        id_new = target2id[o_new]
        
        # APPEND-ONLY INSERTION: Insert new fact with recency/priority weight w=2.5
        new_sent = f"{s} {r} {o_new}."
        k_new_raw = encode_clean([new_sent])[0]
        v_new = norm_E[id_new]
        
        # Insert without deleting old fact:
        model.omega_corpus.insert_atom(k_new_raw, v_new, weight=2.5)
        
        with torch.no_grad():
            post_logits, _ = model(q_raw)
            post_pred = post_logits.argmax(dim=-1).item()
            prob_new = F.softmax(post_logits, dim=-1)[0, id_new].item()
            prob_old = F.softmax(post_logits, dim=-1)[0, id_old].item()
            
        success = (post_pred == id_new)
        print(f"Update: '{s}' | Target: '{o_new}' | Correctly Overrode: {success} | "
              f"P(New Target): {prob_new*100:5.1f}% | P(Obsolete Old): {prob_old*100:4.1f}%")
        
        update_results.append({
            "subject": s,
            "old_target": o_old,
            "new_target": o_new,
            "override_success": bool(success),
            "prob_new": prob_new,
            "prob_old": prob_old
        })

    # ========================================================================
    # EXPERIMENT 2: Machine Unlearning via Anti-Atoms (Destructive Interference)
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: ZERO-RETRAINING UNLEARNING VIA ANTI-ATOMS (SIGNED MEASURES)")
    print("="*70)
    
    unlearn_results = []
    clean_base_q = base_q_raw[:35]
    clean_base_tgts = base_tgts[:35]
    
    for s, r, o_sensitive in SENSITIVE_TARGETS_FOR_UNLEARNING:
        q_text = f"What {r} {s}?"
        q_raw = encode_clean([q_text])
        id_sensitive = target2id[o_sensitive]
        
        # INSERT ANTI-ATOM: Key matches sensitive fact, weight is NEGATIVE (-1.0)
        sensitive_sent = f"{s} {r} {o_sensitive}."
        k_sensitive_raw = encode_clean([sensitive_sent])[0]
        v_sensitive = norm_E[id_sensitive]
        
        # Destructive anti-atom insertion:
        model.omega_corpus.insert_atom(k_sensitive_raw, v_sensitive, weight=-1.0)
        
        # Test post-unlearning recall & suppression
        with torch.no_grad():
            post_l, _ = model(q_raw)
            post_pred = post_l.argmax(dim=-1).item()
            post_prob = F.softmax(post_l, dim=-1)[0, id_sensitive].item()
            
            clean_l, _ = model(clean_base_q)
            clean_acc = (clean_l.argmax(dim=-1) == clean_base_tgts).float().mean().item()
            
        erased = (post_pred != id_sensitive)
        print(f"Unlearning: '{s}' | Sensitive Target: '{o_sensitive}' | "
              f"Erased: {erased} | P(Sensitive): {post_prob*100:5.2f}% | "
              f"Background Specificity: {clean_acc*100:5.1f}%")
        
        unlearn_results.append({
            "target": o_sensitive,
            "successfully_erased": bool(erased),
            "residual_probability": post_prob,
            "background_specificity": clean_acc
        })

    # Summary Export
    summary = {
        "counterfactual_updates": update_results,
        "anti_atom_unlearning": unlearn_results
    }
    with open(os.path.join(OUT_DIR, "unlearning_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    up_names = [u["subject"] for u in update_results]
    p_news = [u["prob_new"] * 100 for u in update_results]
    p_olds = [u["prob_old"] * 100 for u in update_results]
    x_idx = np.arange(len(up_names))
    width = 0.35
    
    ax1.bar(x_idx - width/2, p_news, width, label="P(New Target)", color="#2ca02c")
    ax1.bar(x_idx + width/2, p_olds, width, label="P(Old Obsolete)", color="#d62728")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(up_names, rotation=20, ha="right")
    ax1.set_ylabel("Probability Allocated (%)")
    ax1.set_title("Competitive Fact Updating (Append-Only Log)")
    ax1.set_ylim(0, 110)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend()

    un_names = [f"Item {i+1}" for i in range(len(unlearn_results))]
    res_probs = [u["residual_probability"] * 100 for u in unlearn_results]
    
    ax2.bar(un_names, res_probs, color="#9467bd", width=0.4)
    ax2.set_ylabel("Residual Sensitive Probability (%)")
    ax2.set_title("Anti-Atom Machine Unlearning (Residual < 1%)")
    ax2.set_ylim(0, 10)
    ax2.axhline(y=1.0, color="red", linestyle="--", label="1% Safety Threshold")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "unlearning_and_updates.png"), dpi=200)
    plt.close()

    print(f"\nBenchmark complete. Summary and figures saved to '{OUT_DIR}/'.")


if __name__ == "__main__":
    run_unlearning_experiment()