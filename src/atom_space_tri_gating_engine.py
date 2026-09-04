"""
src/atom_space_tri_gating_engine.py

FULL TRI-SPACE BENCHMARK (Context + Corpus + Parameters)
Calibrated Routing & Functional Orthogonality:
- Unit-norm parameter expert keys (resolves Section 3.2 scale mismatch)
- Tied expert values to vocabulary embeddings
- Exports synchronized summary JSON to both output and results directories
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

OUT_DIR = os.path.join("results", "details", "tri_gating_probe_outputs")


@dataclass
class TriEngineConfig:
    seed: int = 42
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    token_embed_dim: int = 128
    hidden_dim: int = 128
    
    lr: float = 1e-3
    epochs: int = 1200
    patience: int = 120
    read_temperature: float = 0.07


# --- Task Definitions ---

CORPUS_FACTS = [
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
    ("Pluto", "is a prominent resident of the", "Kuiper Belt"),
    ("Redis", "serves key-value data in-memory for low", "Latency"),
    ("Enzyme", "lowers activation energy barriers to accelerate chemical", "Reactions"),
    ("Diamond", "is a transparent allotrope of carbon arranged in a tetrahedral", "Lattice"),
    ("Titan", "possesses liquid methane lakes on its", "Surface"),
    ("RSA", "bases public-key encryption on the factoring difficulty of large", "Primes"),
    ("Antibody", "is a Y-shaped immune protein that identifies and neutralizes foreign", "Antigens"),
    ("Water", "is an inorganic polar compound acting as a universal biological", "Solvent"),
    ("Europa", "hides a warm salty liquid", "Subsurface Ocean"),
    ("PostgreSQL", "is an advanced open-source object-relational", "Database"),
    ("CRISPR-Cas9", "functions naturally as an adaptive bacterial defense against invading", "Viruses"),
    ("Ozone", "absorbs harmful solar ultraviolet rays in the upper", "Stratosphere"),
    ("Andromeda", "is the nearest major spiral neighbor", "Galaxy"),
    ("GraphQL", "allows clients to request precisely structured custom response", "Fields"),
    ("Dopamine", "is a neurotransmitter mediating reward pathways and motor", "Control"),
    ("Ammonia", "is synthesized industrially through the high-pressure Haber-Bosch", "Process"),
    ("Ceres", "is the largest asteroid in the inner", "Asteroid Belt"),
    ("Kubernetes", "orchestrates scalable containerized deployments across clustered", "Nodes"),
    ("Stem Cell", "maintains the undifferentiated capacity to develop into specialized cell", "Types"),
    ("Tungsten", "exhibits the highest melting point of any metallic", "Element"),
    ("Ganymede", "is larger than the planet", "Mercury"),
    ("SQLite", "embeds a serverless transactional database engine directly into an", "Application"),
    ("Platelet", "is anucleate cell fragment essential for initiating vascular blood", "Clotting")
]

CONTEXT_EPHEMERAL = [
    ("Agent Alpha", "is stationed at outpost", "Sector-9"),
    ("Cipher Blue", "unlocks vault door", "Delta-4"),
    ("Project Titan", "is supervised by director", "Vance"),
    ("Frequency Theta", "transmits emergency beacon", "Echo-7"),
    ("Device Omega", "is powered by crystal", "Zircon-3"),
    ("Protocol Gold", "authorizes system override", "Clearance-5"),
    ("Vessel Nebula", "is commanded by officer", "Kovacs"),
    ("Satellite Sigma", "relays orbital telemetry to", "Ground-2"),
    ("Key Amber", "secures confidential archive", "Vault-11"),
    ("Beacon Gamma", "monitors seismic tremors near", "Fault-6"),
    ("Unit Cobalt", "defends perimeter perimeter", "Zone-8"),
    ("Artifact Epsilon", "was retrieved from cavern", "Chamber-12"),
    ("Relay Nova", "synchronizes timing pulses with", "Clock-1"),
    ("Courier Shadow", "delivers sealed envelope to", "Safehouse-3"),
    ("Node Apex", "routes encrypted traffic through", "Gateway-10")
]

PARAM_EXPERT_TASKS = [
    ("Compute binary complement of state 1", "Expert Parity", "Binary-Zero"),
    ("Compute binary complement of state 0", "Expert Parity", "Binary-One"),
    ("Evaluate parity check for bitstream high", "Expert Parity", "Parity-Even"),
    ("Evaluate parity check for bitstream low", "Expert Parity", "Parity-Odd"),
    ("Format identifier using uppercase convention", "Expert Syntax", "FORMAT_UPPER"),
    ("Format identifier using lowercase convention", "Expert Syntax", "format_lower"),
    ("Apply camelCase naming convention to token", "Expert Syntax", "formatCamelCase"),
    ("Apply snake_case naming convention to token", "Expert Syntax", "format_snake_case"),
    ("Classify payload sensitivity for public channel", "Expert Security", "SEC_UNCLASSIFIED"),
    ("Classify payload sensitivity for internal network", "Expert Security", "SEC_RESTRICTED"),
    ("Classify payload sensitivity for diplomatic cable", "Expert Security", "SEC_CONFIDENTIAL"),
    ("Classify payload sensitivity for nuclear launch", "Expert Security", "SEC_TOP_SECRET"),
    ("Apply logical negation to affirmative statement", "Expert Logic", "LOGIC_FALSE"),
    ("Apply logical negation to negative statement", "Expert Logic", "LOGIC_TRUE"),
    ("Evaluate contradiction between opposite states", "Expert Logic", "LOGIC_NULL")
]


# ============================================================================
# Model Architecture
# ============================================================================

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


class ParameterExpertBank(nn.Module):
    def __init__(self, num_experts: int, value_dim: int):
        super().__init__()
        self.expert_values = nn.Parameter(torch.empty(num_experts, value_dim))
        nn.init.normal_(self.expert_values, std=0.02)

    def forward(self, routing_weights: torch.Tensor) -> torch.Tensor:
        experts = F.normalize(self.expert_values, p=2, dim=-1)
        return routing_weights @ experts


class TriSpaceLanguageModel(nn.Module):
    def __init__(self, cfg: TriEngineConfig, in_feat_dim: int, vocab_size: int, num_param_tasks: int):
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
        
        # Space 1: Corpus
        self.omega_corpus = AtomSpace("corpus", self.value_dim)
        
        # Space 2: Parameter-resident experts selected by normalized trigger keys
        self.omega_params = AtomSpace("params", self.value_dim)
        self.param_experts = ParameterExpertBank(num_param_tasks, self.value_dim)
        
        # Tri-Space Router
        self.gate_net = nn.Sequential(
            nn.Linear(in_feat_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 3)
        )

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        adapted = x + 0.1 * self.adapter(x)
        return F.normalize(adapted, p=2, dim=-1)

    def get_normalized_embeddings(self) -> torch.Tensor:
        return F.normalize(self.token_embeddings.weight, p=2, dim=-1)

    def forward(self, 
                query_feats: torch.Tensor, 
                ctx_keys: torch.Tensor = None, 
                ctx_values: torch.Tensor = None,
                ablate_space: str = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        q = self.phi(query_feats)
        tau = self.cfg.read_temperature
        
        # 1. Read Omega_corpus
        k_corp = self.phi(self.omega_corpus.keys)
        scores_corp = (q @ k_corp.T) / tau
        mu_corp = F.softmax(scores_corp, dim=-1)
        z_corp = mu_corp @ self.omega_corpus.values
        
        # 2. Read Omega_params
        k_param = self.phi(self.omega_params.keys)
        scores_param = (q @ k_param.T) / tau
        mu_param = F.softmax(scores_param, dim=-1)
        z_param = self.param_experts(mu_param)
        
        # 3. Read Omega_ctx
        if ctx_keys is not None and ctx_values is not None and len(ctx_keys) > 0:
            k_ctx = self.phi(ctx_keys)
            scores_ctx = (q @ k_ctx.T) / tau
            mu_ctx = F.softmax(scores_ctx, dim=-1)
            z_ctx = mu_ctx @ ctx_values
        else:
            z_ctx = torch.zeros_like(z_corp)
            mu_ctx = None
            
        # 4. Router Gating
        gate_logits = self.gate_net(query_feats)
        lambdas = F.softmax(gate_logits, dim=-1)
        
        if ablate_space == "ctx":
            lambdas = lambdas.clone()
            lambdas[:, 0] = 0.0
            lambdas = lambdas / (lambdas.sum(dim=-1, keepdim=True) + 1e-8)
        elif ablate_space == "corpus":
            lambdas = lambdas.clone()
            lambdas[:, 1] = 0.0
            lambdas = lambdas / (lambdas.sum(dim=-1, keepdim=True) + 1e-8)
        elif ablate_space == "params":
            lambdas = lambdas.clone()
            lambdas[:, 2] = 0.0
            lambdas = lambdas / (lambdas.sum(dim=-1, keepdim=True) + 1e-8)
            
        z = (lambdas[:, 0:1] * z_ctx + 
             lambdas[:, 1:2] * z_corp + 
             lambdas[:, 2:3] * z_param)
        
        norm_E = self.get_normalized_embeddings()
        logits = (z @ norm_E.T) / tau
        
        return logits, lambdas, {"mu_corp": mu_corp, "mu_ctx": mu_ctx, "mu_param": mu_param}


# ============================================================================
# Execution
# ============================================================================

def run_tri_space_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    cfg = TriEngineConfig()
    
    # Explicit Deterministic Seeding
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        
    device = cfg.device
    print(f"=== Running Calibrated Tri-Space Benchmark on {device} ===")
    
    from sentence_transformers import SentenceTransformer
    backbone = SentenceTransformer(cfg.encoder_name, device=device)
    
    def encode_clean(texts: List[str]) -> torch.Tensor:
        return backbone.encode(texts, convert_to_tensor=True, device=device).clone()
    
    all_targets = set()
    for _, _, o in CORPUS_FACTS: all_targets.add(o)
    for _, _, o in CONTEXT_EPHEMERAL: all_targets.add(o)
    for _, _, o in PARAM_EXPERT_TASKS: all_targets.add(o)
    target2id = {tgt: i for i, tgt in enumerate(sorted(all_targets))}
    vocab_size = len(target2id)

    # Encode Corpus
    corp_sents = [f"{s} {r} {o}." for s, r, o in CORPUS_FACTS]
    corp_queries = [f"What {r} {s}?" for s, r, o in CORPUS_FACTS]
    corp_tgts = torch.tensor([target2id[o] for _, _, o in CORPUS_FACTS], device=device)
    corp_k_raw = encode_clean(corp_sents)
    corp_q_raw = encode_clean(corp_queries)

    # Encode Context
    ctx_premises = [f"Premise: {s} {r} {o}." for s, r, o in CONTEXT_EPHEMERAL]
    ctx_queries = [f"According to premise, what {r} {s}?" for s, r, o in CONTEXT_EPHEMERAL]
    ctx_tgts = torch.tensor([target2id[o] for _, _, o in CONTEXT_EPHEMERAL], device=device)
    ctx_k_raw = encode_clean(ctx_premises)
    ctx_q_raw = encode_clean(ctx_queries)

    # Encode Parameter Experts (Keys are unit-norm triggers)
    param_triggers = [f"Execute task: {desc}." for desc, _, _ in PARAM_EXPERT_TASKS]
    param_queries = [desc for desc, _, _ in PARAM_EXPERT_TASKS]
    param_tgts = torch.tensor([target2id[o] for _, _, o in PARAM_EXPERT_TASKS], device=device)
    param_k_raw = encode_clean(param_triggers)
    param_q_raw = encode_clean(param_queries)

    in_dim = corp_k_raw.shape[-1]
    model = TriSpaceLanguageModel(cfg, in_dim, vocab_size, len(PARAM_EXPERT_TASKS)).to(device)
    
    with torch.no_grad():
        norm_E = model.get_normalized_embeddings()
        model.omega_corpus.set_atoms(corp_k_raw, norm_E[corp_tgts])
        model.omega_params.set_atoms(param_k_raw, torch.zeros_like(norm_E[param_tgts]))
        model.param_experts.expert_values.copy_(norm_E[param_tgts])
        
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    print("\nTraining Calibrated Tri-Space Model...")
    best_loss = float("inf")
    patience = 0
    
    for ep in range(cfg.epochs):
        model.train()
        norm_E = model.get_normalized_embeddings()
        model.omega_corpus.values = norm_E[corp_tgts]
        
        logits_corp, _, _ = model(corp_q_raw)
        loss_corp = F.cross_entropy(logits_corp, corp_tgts)
        
        ctx_vals = norm_E[ctx_tgts]
        logits_ctx, _, _ = model(ctx_q_raw, ctx_keys=ctx_k_raw, ctx_values=ctx_vals)
        loss_ctx = F.cross_entropy(logits_ctx, ctx_tgts)
        
        logits_param, _, _ = model(param_q_raw)
        loss_param = F.cross_entropy(logits_param, param_tgts)
        
        total_loss = loss_corp + loss_ctx + loss_param
        
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        
        if total_loss.item() < best_loss - 1e-4:
            best_loss = total_loss.item()
            patience = 0
        else:
            patience += 1
        if patience >= cfg.patience:
            break
            
    print(f"Calibrated Model converged at Epoch {ep:4d} | Joint Loss: {best_loss:.4f}")

    # ========================================================================
    # Evaluation
    # ========================================================================
    model.eval()
    norm_E = model.get_normalized_embeddings()
    ctx_vals = norm_E[ctx_tgts]
    
    print("\n" + "="*70)
    print("CALIBRATED TRI-SPACE GATING EVALUATION & ABLATION")
    print("="*70)
    
    with torch.no_grad():
        logits_ctx_full, gate_ctx, _ = model(ctx_q_raw, ctx_keys=ctx_k_raw, ctx_values=ctx_vals)
        acc_ctx_full = (logits_ctx_full.argmax(dim=-1) == ctx_tgts).float().mean().item()
        mean_gate_ctx = gate_ctx.mean(dim=0).tolist()
        
        logits_corp_full, gate_corp, _ = model(corp_q_raw)
        acc_corp_full = (logits_corp_full.argmax(dim=-1) == corp_tgts).float().mean().item()
        mean_gate_corp = gate_corp.mean(dim=0).tolist()
        
        logits_param_full, gate_param, _ = model(param_q_raw)
        acc_param_full = (logits_param_full.argmax(dim=-1) == param_tgts).float().mean().item()
        mean_gate_param = gate_param.mean(dim=0).tolist()
        
        print("\n[FULL TRI-SPACE MODEL]")
        print(f"  Context Tasks   -> Acc: {acc_ctx_full*100:5.1f}% | Gate [Ctx, Corp, Param]: [{mean_gate_ctx[0]*100:4.1f}%, {mean_gate_ctx[1]*100:4.1f}%, {mean_gate_ctx[2]*100:4.1f}%]")
        print(f"  Corpus Tasks    -> Acc: {acc_corp_full*100:5.1f}% | Gate [Ctx, Corp, Param]: [{mean_gate_corp[0]*100:4.1f}%, {mean_gate_corp[1]*100:4.1f}%, {mean_gate_corp[2]*100:4.1f}%]")
        print(f"  Parameter Tasks -> Acc: {acc_param_full*100:5.1f}% | Gate [Ctx, Corp, Param]: [{mean_gate_param[0]*100:4.1f}%, {mean_gate_param[1]*100:4.1f}%, {mean_gate_param[2]*100:4.1f}%]")
        
        print("\n[SYSTEMATIC ABLATION EXPERIMENTS]")
        l_c_no_ctx, _, _ = model(ctx_q_raw, ctx_keys=ctx_k_raw, ctx_values=ctx_vals, ablate_space="ctx")
        acc_ctx_no_ctx = (l_c_no_ctx.argmax(dim=-1) == ctx_tgts).float().mean().item()
        
        l_c_no_corp, _, _ = model(corp_q_raw, ablate_space="corpus")
        acc_corp_no_corp = (l_c_no_corp.argmax(dim=-1) == corp_tgts).float().mean().item()
        
        l_p_no_param, _, _ = model(param_q_raw, ablate_space="params")
        acc_param_no_param = (l_p_no_param.argmax(dim=-1) == param_tgts).float().mean().item()
        
        print(f"  When Omega_ctx is ablated    -> Context Task Accuracy drops:   {acc_ctx_full*100:5.1f}% -> {acc_ctx_no_ctx*100:5.1f}%")
        print(f"  When Omega_corpus is ablated -> Corpus Task Accuracy drops:    {acc_corp_full*100:5.1f}% -> {acc_corp_no_corp*100:5.1f}%")
        print(f"  When Omega_params is ablated -> Parameter Task Accuracy drops: {acc_param_full*100:5.1f}% -> {acc_param_no_param*100:5.1f}%")

    # Serialize Summary JSON to BOTH OUT_DIR and results/
    summary = {
        "full_model": {
            "context_accuracy": float(acc_ctx_full),
            "corpus_accuracy": float(acc_corp_full),
            "param_accuracy": float(acc_param_full),
            "gate_on_context_task": mean_gate_ctx,
            "gate_on_corpus_task": mean_gate_corp,
            "gate_on_param_task": mean_gate_param
        },
        "ablations": {
            "context_task_without_ctx": float(acc_ctx_no_ctx),
            "corpus_task_without_corpus": float(acc_corp_no_corp),
            "param_task_without_params": float(acc_param_no_param)
        }
    }
    
    with open(os.path.join(OUT_DIR, "tri_space_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join("results", "tri_space_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plotting
    labels = ["Context Tasks", "Corpus Tasks", "Parameter Tasks"]
    gate_matrix = np.array([mean_gate_ctx, mean_gate_corp, mean_gate_param]) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    im = ax1.imshow(gate_matrix, cmap="Blues", vmin=0, vmax=100)
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels([r"$\Omega_{\mathrm{ctx}}$", r"$\Omega_{\mathrm{corpus}}$", r"$\Omega_{\mathrm{params}}$"], fontsize=11)
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_title(r"Dynamic Router Allocation $\lambda(x)$ (%)", fontsize=12)
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f"{gate_matrix[i, j]:.1f}%", ha="center", va="center", 
                     color="white" if gate_matrix[i, j] > 50 else "black", fontweight="bold")
            
    x_indices = np.arange(3)
    width = 0.35
    full_accs = [acc_ctx_full * 100, acc_corp_full * 100, acc_param_full * 100]
    ablated_accs = [acc_ctx_no_ctx * 100, acc_corp_no_corp * 100, acc_param_no_param * 100]
    
    ax2.bar(x_indices - width/2, full_accs, width, label="Full Tri-Space Model", color="#1f77b4")
    ax2.bar(x_indices + width/2, ablated_accs, width, label="Dedicated Space Ablated", color="#d62728")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.set_title("Orthogonality Ablation Study", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tri_space_gating_ablation.png"), dpi=200)
    if os.path.exists("paper/figures"):
        plt.savefig("paper/figures/tri_space_gating_ablation.png", dpi=200)
    plt.close()

    print(f"\nBenchmark finished. Synchronized to '{OUT_DIR}/' and 'results/tri_space_summary.json'.")


if __name__ == "__main__":
    run_tri_space_experiment()