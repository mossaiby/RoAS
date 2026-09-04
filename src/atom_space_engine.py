"""
src/atom_space_engine.py

Complete, self-contained experimental engine for:
"Language Models as Reads over Atom Spaces: A Unified Measure-Theoretic Formulation"

Key Architecture:
  - Metric-Preserving Residual Read Kernel: phi(x) = L2_Norm(x + 0.1 * MLP(x))
  - Tied Key-Value Alignment: v_i = Normalize(E[y_i])
  - Full Post-Hoc Model Editing Triad: Efficacy, Generality, and Specificity
  - Quadrature Truncation Bound: True Euclidean norm ||z_exact - z_k||_2
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

OUT_DIR = os.path.join("results", "details", "advanced_probe_outputs")


# ============================================================================
# 1. Configuration
# ============================================================================

@dataclass
class EngineConfig:
    seed: int = int(os.environ.get("ROAS_SEED", "42"))
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model dimensions
    token_embed_dim: int = 128
    hidden_dim: int = 128
    
    # Training
    lr: float = 1e-3
    epochs: int = 1500
    patience: int = 150
    densities: List[int] = field(default_factory=lambda: [1, 3, 10, 40])
    
    # Softmax read temperature
    read_temperature: float = 0.07
    
    # Truncation & Quadrature parameters
    top_k_list: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    temperatures: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.5, 1.0])


# ============================================================================
# 2. Structured Real-World Dataset (4 Domains x 50 Items = 200 Facts)
# ============================================================================

RAW_DOMAINS = {
    "space": [
        ("Mercury", "is the closest planet to the", "Sun"),
        ("Venus", "has a thick atmosphere of", "Carbon Dioxide"),
        ("Earth", "supports abundant life and liquid", "Water"),
        ("Mars", "is home to Olympus Mons on", "Red Planet"),
        ("Jupiter", "is the largest gas giant in the", "Solar System"),
        ("Saturn", "is famous for prominent rings of", "Ice"),
        ("Uranus", "rotates on a tilted sideways", "Axis"),
        ("Neptune", "experiences extreme supersonic high-speed", "Winds"),
        ("Pluto", "is a prominent resident of the", "Kuiper Belt"),
        ("Titan", "possesses liquid methane lakes on its", "Surface"),
        ("Europa", "hides a warm salty liquid", "Subsurface Ocean"),
        ("Io", "is the most volcanically active body in the", "Solar System"),
        ("Ganymede", "is larger than the planet", "Mercury"),
        ("Callisto", "has an ancient heavily cratered icy", "Crust"),
        ("Ceres", "is the largest asteroid in the inner", "Asteroid Belt"),
        ("Vesta", "is a bright basaltic protoplanet in the", "Asteroid Belt"),
        ("Enceladus", "ejects geothermal water vapor cryovolcanic", "Geysers"),
        ("Triton", "orbits Neptune in a retrograde backward", "Direction"),
        ("Phobos", "orbits extraordinarily close to the planet", "Mars"),
        ("Deimos", "is the smaller outer moon of", "Mars"),
        ("Proxima Centauri", "is the nearest known star to the", "Sun"),
        ("Sirius", "is the brightest optical star in the night", "Sky"),
        ("Betelgeuse", "is a massive red supergiant star in", "Orion"),
        ("Rigel", "is a prominent blue supergiant star in", "Orion"),
        ("Polaris", "marks the approximate northern celestial", "Pole"),
        ("Vega", "was the first star photographed other than the", "Sun"),
        ("Andromeda", "is the nearest major spiral neighbor", "Galaxy"),
        ("Triangulum", "is the third largest member of our Local", "Group"),
        ("Sagittarius A*", "is the supermassive black hole at the center of the", "Milky Way"),
        ("Cygnus X-1", "was the first widely accepted stellar-mass", "Black Hole"),
        ("Crab Nebula", "is the remnant of an ancient recorded", "Supernova"),
        ("Halley's Comet", "returns to the inner solar system every 76", "Years"),
        ("Oumuamua", "was the first discovered macroscopic", "Interstellar Object"),
        ("Borisov", "was the first clearly recognized interstellar", "Comet"),
        ("Voyager 1", "has crossed beyond the solar", "Heliopause"),
        ("Hubble", "operates as an optical space telescope in low Earth", "Orbit"),
        ("James Webb", "observes primarily deep cosmic", "Infrared"),
        ("Kepler", "discovered thousands of transit", "Exoplanets"),
        ("Chandra", "observes high-energy emissions in", "X-Rays"),
        ("Spitzer", "studied cool astronomical targets using", "Infrared"),
        # Holdouts (10 novel items)
        ("Eris", "is a massive trans-Neptunian dwarf planet in the", "Scattered Disc"),
        ("Haumea", "is a rapidly spinning elongated dwarf", "Planet"),
        ("Makemake", "is a distant frozen dwarf planet in the", "Kuiper Belt"),
        ("Sedna", "has an extraordinarily distant eccentric", "Orbit"),
        ("Charon", "is the largest gravitationally locked moon of", "Pluto"),
        ("Aldebaran", "is an orange giant star in the constellation", "Taurus"),
        ("Antares", "is a red supergiant in the heart of", "Scorpius"),
        ("Sombrero", "is an unbarred spiral galaxy featuring a prominent dust", "Lane"),
        ("Centaurus A", "is a prominent nearby radio-loud active", "Galaxy"),
        ("Canopus", "is the second brightest star visible in the night", "Sky")
    ],
    "computing": [
        ("Python", "emphasizes human readability and dynamic", "Typing"),
        ("Rust", "guarantees thread safety without an automated", "Garbage Collector"),
        ("C", "provides low-level access to system", "Memory"),
        ("C++", "introduces object-oriented features with zero-cost", "Abstractions"),
        ("Java", "compiles source code into portable virtual machine", "Bytecode"),
        ("JavaScript", "executes natively inside modern internet web", "Browsers"),
        ("TypeScript", "adds compile-time static type checking to", "JavaScript"),
        ("Go", "simplifies concurrent programming via lightweight", "Goroutines"),
        ("Swift", "serves as the primary language for Apple", "Platforms"),
        ("Kotlin", "runs interoperably on the Java Virtual", "Machine"),
        ("Haskell", "enforces purely functional programming with lazy", "Evaluation"),
        ("Lisp", "represents code and data uniformly using symbolic", "S-Expressions"),
        ("SQL", "queries and manipulates structured relational database", "Tables"),
        ("Linux", "is an open-source operating system", "Monolithic Kernel"),
        ("Git", "manages distributed version history through an acyclic", "Graph"),
        ("Redis", "serves key-value data in-memory for low", "Latency"),
        ("PostgreSQL", "is an advanced open-source object-relational", "Database"),
        ("SQLite", "embeds a serverless transactional database engine directly into an", "Application"),
        ("MongoDB", "stores unstructured documents using binary", "JSON"),
        ("Docker", "packages software into reproducible user-space", "Containers"),
        ("Kubernetes", "orchestrates scalable containerized deployments across clustered", "Nodes"),
        ("HTTP", "transfers hypermedia documents across the World Wide", "Web"),
        ("TCP", "guarantees ordered reliable stream transmission across an IP", "Network"),
        ("UDP", "delivers lightweight datagrams without establishing a persistent", "Connection"),
        ("DNS", "translates human-readable domain names into numerical IP", "Addresses"),
        ("TLS", "encrypts communication channels over transport", "Protocols"),
        ("SSH", "secures remote command-line login sessions across a", "Network"),
        ("B-Tree", "maintains sorted balanced node pointers for fast on-disk", "Lookup"),
        ("Hash Table", "maps unique keys to associative values in amortized constant", "Time"),
        ("Quicksort", "sorts elements efficiently by recursively choosing a", "Pivot"),
        ("Dijkstra", "finds the shortest path between weighted graph", "Nodes"),
        ("PageRank", "ranks hyperlinked web pages by counting directional incoming", "Links"),
        ("RSA", "bases public-key encryption on the factoring difficulty of large", "Primes"),
        ("AES", "implements symmetric block cipher encryption across fixed", "Blocks"),
        ("SHA-256", "computes a deterministic one-way cryptographic cryptographic", "Digest"),
        ("LLVM", "provides modular compiler target intermediate", "Representations"),
        ("Nginx", "acts as an asynchronous high-performance reverse", "Proxy"),
        ("Kafka", "distributes streaming partitioned message logs across fault-tolerant", "Clusters"),
        ("GraphQL", "allows clients to request precisely structured custom response", "Fields"),
        ("WebAssembly", "executes near-native binary instructions inside modern web", "Browsers"),
        # Holdouts
        ("QUIC", "accelerates web transport streams natively over", "UDP"),
        ("Raft", "achieves distributed state machine agreement through leader", "Election"),
        ("Paxos", "is a foundational distributed network consensus", "Protocol"),
        ("Erlang", "implements resilient fault-tolerant messaging using the", "Actor Model"),
        ("Vulkan", "provides explicit low-overhead graphics control over the", "GPU"),
        ("CUDA", "enables massively parallel computing directly on Nvidia", "GPUs"),
        ("Zstandard", "provides real-time high-ratio lossless data", "Compression"),
        ("Protobuf", "serializes structured typed data into compact binary", "Payloads"),
        ("gRPC", "executes cross-language remote procedure calls over", "HTTP/2"),
        ("ClickHouse", "processes analytical queries rapidly using a column-oriented", "Storage Engine")
    ],
    "chemistry": [
        ("Hydrogen", "is the lightest chemical element containing a single", "Proton"),
        ("Helium", "is an inert noble gas that resists chemical", "Reactions"),
        ("Lithium", "is the least dense alkali metal used in rechargeable", "Batteries"),
        ("Carbon", "forms the chemical backbone of all known organic", "Molecules"),
        ("Nitrogen", "constitutes seventy-eight percent of Earth's", "Atmosphere"),
        ("Oxygen", "drives aerobic cellular respiration and chemical", "Combustion"),
        ("Fluorine", "is the most electronegative reactive halogen", "Gas"),
        ("Sodium", "is a soft alkali metal that reacts violently with liquid", "Water"),
        ("Chlorine", "combines with sodium to form common table", "Salt"),
        ("Iron", "is the primary transition metal forming the structural alloy", "Steel"),
        ("Copper", "conducts electrical currents with high thermal", "Conductivity"),
        ("Gold", "is a noble transition metal highly resistant to surface", "Oxidation"),
        ("Silver", "exhibits the highest electrical conductivity of any pure", "Metal"),
        ("Mercury", "remains in a liquid state at standard room", "Temperature"),
        ("Lead", "is a dense toxic heavy metal used for radiation", "Shielding"),
        ("Uranium", "undergoes fissile radioactive decay to sustain nuclear", "Reactions"),
        ("Plutonium", "is an actinide synthesized inside nuclear", "Reactors"),
        ("Silicon", "is the primary semiconductor substrate for integrated", "Circuits"),
        ("Phosphorus", "forms the structural nucleotide backbone of genetic", "DNA"),
        ("Sulfur", "is a yellow nonmetal historically known as", "Brimstone"),
        ("Water", "is an inorganic polar compound acting as a universal biological", "Solvent"),
        ("Ammonia", "is synthesized industrially through the high-pressure Haber-Bosch", "Process"),
        ("Methane", "is the primary combustible hydrocarbon component of natural", "Gas"),
        ("Carbon Dioxide", "is produced by respiration and consumed during plant", "Photosynthesis"),
        ("Ozone", "absorbs harmful solar ultraviolet rays in the upper", "Stratosphere"),
        ("Sulfuric Acid", "is an intensely corrosive industrial mineral", "Acid"),
        ("Hydrochloric Acid", "serves as the primary digestive fluid inside human", "Stomachs"),
        ("Benzene", "is an aromatic hydrocarbon ring stabilized by conjugated pi", "Bonds"),
        ("Ethanol", "is the volatile alcohol produced during biochemical", "Fermentation"),
        ("Glucose", "is a simple six-carbon monosaccharide circulating in animal", "Blood"),
        ("Diamond", "is a transparent allotrope of carbon arranged in a tetrahedral", "Lattice"),
        ("Graphite", "is an opaque carbon allotrope arranged in layered hexagonal", "Sheets"),
        ("Graphene", "is an atom-thick two-dimensional hexagonal lattice of pure", "Carbon"),
        ("Rust", "forms on iron surfaces through electrochemical hydrated", "Oxidation"),
        ("Titanium", "possesses high tensile strength and extreme resistance to sea", "Corrosion"),
        ("Platinum", "catalyzes toxic exhaust conversion inside vehicle catalytic", "Converters"),
        ("Tungsten", "exhibits the highest melting point of any metallic", "Element"),
        ("Bromine", "is a fuming reddish-brown halogen that exists as a liquid at", "Room Temperature"),
        ("Iodine", "is a purple halogen essential for regulating the human thyroid", "Gland"),
        ("Xenon", "is a heavy noble gas used in specialized ion thruster", "Engines"),
        # Holdouts
        ("Bismuth", "exhibits extremely weak radioactivity with half-life exceeding the universe", "Age"),
        ("Gallium", "is a post-transition metal that melts inside the warmth of a human", "Hand"),
        ("Cesium", "is an alkali metal used to define the precise scientific second in atomic", "Clocks"),
        ("Argon", "is the third most abundant gas in Earth's", "Atmosphere"),
        ("Neodymium", "is a rare-earth metal used to manufacture extraordinarily powerful permanent", "Magnets"),
        ("Tritium", "is a radioactive isotope of hydrogen featuring two added", "Neutrons"),
        ("Heavy Water", "incorporates deuterium atoms to moderate neutron speeds in nuclear", "Reactors"),
        ("Fullerene", "is a spherical cage allotrope composed of sixty bonded", "Carbons"),
        ("Acetone", "is a simple organic ketone solvent capable of dissolving", "Plastics"),
        ("Potassium", "is an essential dietary electrolyte critical for nerve impulse", "Transmission")
    ],
    "biology": [
        ("DNA", "encodes biological genetic instructions inside cellular", "Nuclei"),
        ("RNA", "transcribes genetic codes to direct protein", "Synthesis"),
        ("Ribosome", "catalyzes the biochemical assembly of amino acids into", "Proteins"),
        ("Mitochondria", "generate adenosine triphosphate as the energetic currency of the", "Cell"),
        ("Chloroplast", "captures radiant sunlight to drive carbohydrate production in", "Plants"),
        ("Endoplasmic Reticulum", "folds and transports newly synthesized biochemical", "Proteins"),
        ("Golgi Apparatus", "packages and sorts cellular macromolecules into secretory", "Vesicles"),
        ("Lysosome", "contains hydrolytic enzymes designed to break down cellular", "Waste"),
        ("Vacuole", "maintains hydrostatic turgor pressure inside plant", "Cells"),
        ("Enzyme", "lowers activation energy barriers to accelerate chemical", "Reactions"),
        ("Hemoglobin", "binds and transports respiratory oxygen throughout the bloodstream via", "Red Blood Cells"),
        ("Insulin", "is a pancreatic peptide hormone regulating systemic blood", "Glucose"),
        ("Glucagon", "signals liver cells to convert stored glycogen back into", "Glucose"),
        ("Adrenaline", "triggers acute sympathetic fight-or-flight physiological", "Responses"),
        ("Melatonin", "is a pineal hormone regulating circadian sleep-wake", "Cycles"),
        ("Dopamine", "is a neurotransmitter mediating reward pathways and motor", "Control"),
        ("Serotonin", "modulates systemic mood, digestion, and neurological", "Sleep"),
        ("Neuron", "transmits electrical action potentials across specialized synaptic", "Junctions"),
        ("Myelin", "insulates neuronal axons to accelerate nerve impulse", "Conduction"),
        ("Antibody", "is a Y-shaped immune protein that identifies and neutralizes foreign", "Antigens"),
        ("Macrophage", "is a specialized white blood cell that engulfs pathogenic", "Debris"),
        ("T-Cell", "orchestrates targeted cell-mediated adaptive immune", "Responses"),
        ("B-Cell", "matures within bone marrow to secrete protective circulatory", "Antibodies"),
        ("CRISPR-Cas9", "functions naturally as an adaptive bacterial defense against invading", "Viruses"),
        ("Virus", "is an obligate intracellular parasite lacking autonomous cellular", "Metabolism"),
        ("Bacterium", "is a single-celled prokaryote that reproduces primarily through binary", "Fission"),
        ("Fungus", "absorbs organic nutrients through external hyphal digestion and cell walls of", "Chitin"),
        ("Mitosis", "divides a somatic nucleus into two genetically identical daughter", "Nuclei"),
        ("Meiosis", "reduces somatic chromosomal numbers by half to yield reproductive", "Gametes"),
        ("Stem Cell", "maintains the undifferentiated capacity to develop into specialized cell", "Types"),
        ("Collagen", "is the most abundant fibrous structural protein found in animal connective", "Tissue"),
        ("Keratin", "forms the protective outer structural protein in animal hair and", "Nails"),
        ("Actin", "forms microfilaments that interact with myosin to power muscular", "Contraction"),
        ("Myosin", "is a motor protein that hydrolyzes ATP to generate mechanical muscle", "Movement"),
        ("Platelet", "is anucleate cell fragment essential for initiating vascular blood", "Clotting"),
        ("Alveoli", "are microscopic pulmonary air sacs where respiratory blood gas", "Exchange occurs"),
        ("Nephron", "is the microscopic structural filtering unit of the mammalian", "Kidney"),
        ("Liver", "synthesizes bile salts and detoxifies metabolic compounds in the", "Body"),
        ("Pancreas", "secretes both digestive enzymes and vital endocrine metabolic", "Hormones"),
        ("Thyroid", "synthesizes iodinated hormones to govern baseline metabolic", "Rates"),
        # Holdouts
        ("Telomere", "caps chromosome ends to protect genomic integrity during repetitive", "Replication"),
        ("Prion", "is a misfolded infectious protein that induces neurological degenerative", "Diseases"),
        ("Epigenetics", "modulates gene expression without altering underlying nucleotide DNA", "Sequences"),
        ("Histone", "acts as a protein spool around which genomic DNA strands", "Wind"),
        ("Centrosome", "organizes microtubule spindle apparatuses during mitotic cell", "Division"),
        ("Peroxisome", "neutralizes toxic metabolic peroxides through catalase enzyme", "Activity"),
        ("Synapse", "is the microscopic gap across which neurotransmitters relay chemical", "Signals"),
        ("Glial Cell", "provides structural metabolic support and insulation for central nervous", "Neurons"),
        ("Aquaporin", "is a specialized membrane channel facilitating rapid passive movement of", "Water"),
        ("Cytokine", "acts as a small signaling peptide coordinating systemic immune and inflammatory", "Responses")
    ]
}


# ============================================================================
# 3. Model Architecture
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
        
    def insert_atom(self, key: torch.Tensor, value: torch.Tensor):
        self.keys = torch.cat([self.keys, key.unsqueeze(0)], dim=0)
        self.values = torch.cat([self.values, value.unsqueeze(0)], dim=0)

    def delete_last_atom(self):
        self.keys = self.keys[:-1]
        self.values = self.values[:-1]


class UnifiedResidualTiedLM(nn.Module):
    """
    Unified LM with Metric-Preserving Residual Projection:
      phi(x) = Normalize(x + 0.1 * MLP(x))
        The zero-initialized residual preserves the backbone geometry at initialization.
    """
    def __init__(self, cfg: EngineConfig, in_feat_dim: int, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.value_dim = cfg.token_embed_dim
        
        # Zero-initialized residual adapter
        self.adapter = nn.Sequential(
            nn.Linear(in_feat_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, in_feat_dim)
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        
        # Tied Token Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, cfg.token_embed_dim)
        nn.init.normal_(self.token_embeddings.weight, std=0.02)
        
        # Corpus Atom Space
        self.omega_corpus = AtomSpace("corpus", self.value_dim)
        
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        adapted = x + 0.1 * self.adapter(x)
        return F.normalize(adapted, p=2, dim=-1)

    def get_normalized_embeddings(self) -> torch.Tensor:
        return F.normalize(self.token_embeddings.weight, p=2, dim=-1)

    def read_vector(self, query_feats: torch.Tensor, tau: float = None, top_k: int = None) -> torch.Tensor:
        """
        Directly evaluates the continuous or top-k truncated context vector:
          z(x) = sum mu_i v_i
        Used to measure true quadrature integration error ||z_exact - z_k||_2.
        """
        if tau is None:
            tau = self.cfg.read_temperature
            
        q = self.phi(query_feats)
        k = self.phi(self.omega_corpus.keys)
        scores = (q @ k.T) / tau
        
        if top_k is not None and top_k < scores.size(-1):
            top_scores, top_idx = torch.topk(scores, top_k, dim=-1)
            mu_k = F.softmax(top_scores, dim=-1)
            v_selected = self.omega_corpus.values[top_idx]
            z = torch.bmm(mu_k.unsqueeze(1), v_selected).squeeze(1)
        else:
            mu = F.softmax(scores, dim=-1)
            z = mu @ self.omega_corpus.values
            
        return z

    def forward(self, query_feats: torch.Tensor, tau: float = None, top_k: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if tau is None:
            tau = self.cfg.read_temperature
            
        q = self.phi(query_feats)
        k = self.phi(self.omega_corpus.keys)
        scores = (q @ k.T) / tau
        
        if top_k is not None and top_k < scores.size(-1):
            top_scores, top_idx = torch.topk(scores, top_k, dim=-1)
            mu_k = F.softmax(top_scores, dim=-1)
            v_selected = self.omega_corpus.values[top_idx]
            z = torch.bmm(mu_k.unsqueeze(1), v_selected).squeeze(1)
            mu = torch.zeros_like(scores).scatter_(-1, top_idx, mu_k)
        else:
            mu = F.softmax(scores, dim=-1)
            z = mu @ self.omega_corpus.values
            
        norm_E = self.get_normalized_embeddings()
        logits = (z @ norm_E.T) / tau
        return logits, mu


def compute_adapter_spectral_product(model: nn.Module) -> float:
    """
    Computes the product of linear layer spectral norms within the residual adapter MLP.
    Tracks the empirical inflation of layerwise operator norms under gradient updates.
    """
    L = 1.0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            L *= torch.linalg.matrix_norm(m.weight, ord=2).item()
    return L


# ============================================================================
# 4. Core Execution Script
# ============================================================================

def run_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    cfg = EngineConfig()
    
    # Explicit Deterministic Seeding
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        
    device = cfg.device
    print(f"=== Running Residual Tied Atom Space Engine on {device} ===")
    
    from sentence_transformers import SentenceTransformer
    backbone = SentenceTransformer(cfg.encoder_name, device=device)
    
    def encode_clean(texts: List[str]) -> torch.Tensor:
        return backbone.encode(texts, convert_to_tensor=True, device=device).clone()
    
    # Build Global Vocabulary
    all_targets = set()
    for cat in RAW_DOMAINS:
        for s, r, o in RAW_DOMAINS[cat]:
            all_targets.add(o)
    target2id = {tgt: i for i, tgt in enumerate(sorted(all_targets))}
    vocab_size = len(target2id)
    print(f"Corpus size: 200 facts | Target vocabulary: {vocab_size} tokens")

    # Split into 160 Training and 40 Holdout facts
    train_triplets, holdout_triplets = [], []
    for cat in RAW_DOMAINS:
        train_triplets.extend(RAW_DOMAINS[cat][:40])
        holdout_triplets.extend(RAW_DOMAINS[cat][40:])

    # Pre-embed Training Data
    train_sents = [f"{s} {r} {o}." for s, r, o in train_triplets]
    train_queries = [f"What {r} {s}?" for s, r, o in train_triplets]
    train_paras = [f"Regarding {s}, what {r} it?" for s, r, o in train_triplets]
    train_tgts = torch.tensor([target2id[o] for _, _, o in train_triplets], device=device)

    train_k_raw = encode_clean(train_sents)
    train_q_raw = encode_clean(train_queries)
    train_p_raw = encode_clean(train_paras)
    in_dim = train_k_raw.shape[-1]
    
    # Pre-embed Holdout Data
    hold_sents = [f"{s} {r} {o}." for s, r, o in holdout_triplets]
    hold_queries = [f"What {r} {s}?" for s, r, o in holdout_triplets]
    hold_paras = [f"Regarding {s}, what {r} it?" for s, r, o in holdout_triplets]
    hold_tgts = [target2id[o] for _, _, o in holdout_triplets]

    hold_k_raw = encode_clean(hold_sents)
    hold_q_raw = encode_clean(hold_queries)
    hold_p_raw = encode_clean(hold_paras)

    results_across_densities = {}

    for density in cfg.densities:
        print(f"\n{'='*20} TESTING DENSITY: {density}/topic ({density*4} total atoms) {'='*20}")
        
        # Subsample precisely 'density' items from each category
        sub_indices = []
        for c_idx in range(4):
            offset = c_idx * 40
            sub_indices.extend(range(offset, offset + density))
            
        cur_k = train_k_raw[sub_indices]
        cur_q = train_q_raw[sub_indices]
        cur_tgts = train_tgts[sub_indices]
        
        model = UnifiedResidualTiedLM(cfg, in_dim, vocab_size).to(device)
        
        with torch.no_grad():
            norm_E = model.get_normalized_embeddings()
            cur_values = norm_E[cur_tgts]
            model.omega_corpus.set_atoms(cur_k, cur_values)
            
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        
        # Supervised Training Loop
        best_loss = float("inf")
        patience = 0
        for ep in range(cfg.epochs):
            model.train()
            norm_E = model.get_normalized_embeddings()
            model.omega_corpus.values = norm_E[cur_tgts]
            
            logits, _ = model(cur_q)
            loss = F.cross_entropy(logits, cur_tgts)
            
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
                
        L_bound = compute_adapter_spectral_product(model.adapter)
        print(f"  Converged at Epoch {ep:4d} | Loss: {best_loss:.4f} | Adapter L: {L_bound:.2f}")

        # Post-Hoc Model Editing Benchmark
        model.eval()
        efficacy_list, generality_list, specificity_list, mu_mass_list = [], [], [], []
        specificity_kl_list, specificity_max_prob_shift_list = [], []
        baseline_efficacy_list, baseline_generality_list, baseline_specificity_list = [], [], []
        
        # Pre-edit baseline on existing atoms
        with torch.no_grad():
            base_logits, _ = model(cur_q)
            base_preds = base_logits.argmax(dim=-1)
            base_probs = F.softmax(base_logits, dim=-1)
            base_accuracy = (base_preds == cur_tgts).float().mean().item()
            normalized_cur_keys = F.normalize(cur_k, p=2, dim=-1)
            normalized_cur_queries = F.normalize(cur_q, p=2, dim=-1)
            baseline_base_indices = (normalized_cur_queries @ normalized_cur_keys.T).argmax(dim=-1)
            baseline_base_preds = cur_tgts[baseline_base_indices]

        norm_E = model.get_normalized_embeddings()
        for i in range(len(holdout_triplets)):
            k_star = hold_k_raw[i]
            q_star = hold_q_raw[i:i+1]
            p_star = hold_p_raw[i:i+1]
            tgt_star = hold_tgts[i]
            
            with torch.no_grad():
                v_star = norm_E[tgt_star]
                model.omega_corpus.insert_atom(k_star, v_star)
                
                # 1. Efficacy
                logits_q, mu_q = model(q_star)
                pred_q = logits_q.argmax(dim=-1).item()
                efficacy_list.append(int(pred_q == tgt_star))
                mu_mass_list.append(mu_q[0, -1].item())
                
                # 2. Generality
                logits_p, _ = model(p_star)
                pred_p = logits_p.argmax(dim=-1).item()
                generality_list.append(int(pred_p == tgt_star))
                
                # 3. Specificity
                post_logits, _ = model(cur_q)
                post_preds = post_logits.argmax(dim=-1)
                post_probs = F.softmax(post_logits, dim=-1)
                spec = (base_preds == post_preds).float().mean().item()
                specificity_list.append(spec)
                specificity_kl_list.append(
                    F.kl_div(post_probs.log(), base_probs, reduction="batchmean").item()
                )
                specificity_max_prob_shift_list.append(
                    torch.max(torch.abs(post_probs - base_probs)).item()
                )

                # Frozen MiniLM exact 1-nearest-neighbor baseline.
                baseline_keys = torch.cat([cur_k, k_star.unsqueeze(0)], dim=0)
                baseline_targets = torch.cat([
                    cur_tgts,
                    torch.tensor([tgt_star], device=device)
                ])
                normalized_baseline_keys = F.normalize(baseline_keys, p=2, dim=-1)
                baseline_q_index = (
                    F.normalize(q_star, p=2, dim=-1) @ normalized_baseline_keys.T
                ).argmax(dim=-1)
                baseline_p_index = (
                    F.normalize(p_star, p=2, dim=-1) @ normalized_baseline_keys.T
                ).argmax(dim=-1)
                baseline_post_indices = (
                    normalized_cur_queries @ normalized_baseline_keys.T
                ).argmax(dim=-1)
                baseline_efficacy_list.append(
                    int(baseline_targets[baseline_q_index].item() == tgt_star)
                )
                baseline_generality_list.append(
                    int(baseline_targets[baseline_p_index].item() == tgt_star)
                )
                baseline_specificity_list.append(
                    (baseline_targets[baseline_post_indices] == baseline_base_preds).float().mean().item()
                )
                
                model.omega_corpus.delete_last_atom()

        # TRUE QUADRATURE TRUNCATION ERROR: ||z_exact - z_k||_2 on context vectors
        quad_errors = {tau: {} for tau in cfg.temperatures}
        quad_diagnostics = {tau: {} for tau in cfg.temperatures}
        with torch.no_grad():
            normalized_values = F.normalize(model.omega_corpus.values, p=2, dim=-1)
            value_max = torch.linalg.vector_norm(normalized_values, dim=-1).max().item()
            query_vectors = model.phi(cur_q)
            key_vectors = model.phi(model.omega_corpus.keys)
            raw_scores = query_vectors @ key_vectors.T
            sorted_scores, _ = torch.sort(raw_scores, dim=-1, descending=True)
            for tau in cfg.temperatures:
                z_exact = model.read_vector(cur_q, tau=tau, top_k=None)
                for k in cfg.top_k_list:
                    if k <= len(cur_k):
                        z_k = model.read_vector(cur_q, tau=tau, top_k=k)
                        per_query_error = torch.norm(z_exact - z_k, p=2, dim=-1)
                        quad_errors[tau][k] = per_query_error.mean().item()
                        if k < len(cur_k):
                            first_gap = sorted_scores[:, 0] - sorted_scores[:, k]
                            boundary_gap = sorted_scores[:, k - 1] - sorted_scores[:, k]
                            bound_1 = 2 * value_max * (len(cur_k) - k) * torch.exp(-first_gap / tau)
                            bound_2 = 2 * value_max * ((len(cur_k) - k) / k) * torch.exp(-boundary_gap / tau)
                            best_bound = torch.minimum(bound_1, bound_2)
                            quad_diagnostics[tau][k] = {
                                "mean_error": per_query_error.mean().item(),
                                "max_error": per_query_error.max().item(),
                                "mean_bound": best_bound.mean().item(),
                                "min_bound_slack": (best_bound - per_query_error).min().item(),
                                "empirical_coverage": (per_query_error <= best_bound + 1e-6).float().mean().item(),
                                "mean_first_gap": first_gap.mean().item(),
                                "mean_boundary_gap": boundary_gap.mean().item()
                            }

        results_across_densities[density] = {
            "efficacy": float(np.mean(efficacy_list)),
            "generality": float(np.mean(generality_list)),
            "specificity": float(np.mean(specificity_list)),
            "base_training_accuracy": base_accuracy,
            "specificity_mean_kl": float(np.mean(specificity_kl_list)),
            "specificity_max_probability_shift": float(np.max(specificity_max_prob_shift_list)),
            "frozen_exact_1nn_baseline": {
                "efficacy": float(np.mean(baseline_efficacy_list)),
                "generality": float(np.mean(baseline_generality_list)),
                "specificity": float(np.mean(baseline_specificity_list))
            },
            "novel_atom_attention_mass": float(np.mean(mu_mass_list)),
            "adapter_spectral_product": L_bound,
            "truncation_error": quad_errors,
            "truncation_bound_diagnostics": quad_diagnostics
        }
        
        print(
            f"  [Results] Efficacy: {np.mean(efficacy_list)*100:5.1f}% | "
            f"Generality: {np.mean(generality_list)*100:5.1f}% | "
            f"Specificity: {np.mean(specificity_list)*100:5.1f}% | "
            f"Novel Attention Mass: {np.mean(mu_mass_list)*100:5.1f}%"
        )
        print(
            f"  [Frozen Exact 1-NN] Efficacy: {np.mean(baseline_efficacy_list)*100:5.1f}% | "
            f"Generality: {np.mean(baseline_generality_list)*100:5.1f}% | "
            f"Specificity: {np.mean(baseline_specificity_list)*100:5.1f}%"
        )

    # Save to both OUT_DIR and results/
    with open(os.path.join(OUT_DIR, "summary_v3.json"), "w") as f:
        json.dump(results_across_densities, f, indent=2)
    with open(os.path.join("results", "summary_v3.json"), "w") as f:
        json.dump(results_across_densities, f, indent=2)

    # Plotting
    densities = list(results_across_densities.keys())
    
    plt.figure(figsize=(7, 4.5))
    eff = [results_across_densities[d]["efficacy"] * 100 for d in densities]
    gen = [results_across_densities[d]["generality"] * 100 for d in densities]
    spec = [results_across_densities[d]["specificity"] * 100 for d in densities]
    
    plt.plot(densities, eff, marker="o", label="Efficacy (Recall on w*)", color="#1f77b4", linewidth=2)
    plt.plot(densities, gen, marker="s", label="Generality (Paraphrase)", color="#2ca02c", linewidth=2)
    plt.plot(densities, spec, marker="^", label="Specificity (Locality Retention)", color="#d62728", linewidth=2)
    plt.xlabel("Training Atoms per Domain", fontsize=11)
    plt.ylabel("Accuracy / Retention (%)", fontsize=11)
    plt.title("Post-Hoc Knowledge Editing Triad vs. Density", fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "editing_triad_vs_density.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    high_d = densities[-1]
    for tau in cfg.temperatures:
        k_keys = sorted(results_across_densities[high_d]["truncation_error"][tau].keys())
        errs = [results_across_densities[high_d]["truncation_error"][tau][k] for k in k_keys]
        plt.plot(k_keys, errs, marker="x", label=rf"Temperature $\tau={tau}$")
    plt.xlabel("Top-$k$ Cutoff", fontsize=11)
    plt.ylabel(r"Log Truncation Error $\|R_{\mathrm{exact}} - \hat{R}_k\|_2$", fontsize=11)
    plt.yscale("log")
    plt.title("Approximation Error of Exact Top-$k$ Truncation", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "truncation_error.png"), dpi=200)
    plt.close()

    print(f"\nAll benchmark runs complete. Outputs synchronized to '{OUT_DIR}/' and 'results/'.")


if __name__ == "__main__":
    run_experiment()