# RoAS: Reads over Atom Spaces

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

This repository provides the official implementation and experimental benchmarks for the paper:  
**"Language Models as Reads over Atom Spaces: A Unified Measure-Theoretic Formulation of Context, Retrieval, and Parameters"**  
by **Farshid Mossaiby** (*University of Isfahan*).

---

## 📌 Overview

Modern language models bake factual knowledge directly into frozen, dense parameter matrices. This causes catastrophic forgetting, high training costs, and severe difficulty in editing or erasing obsolete and sensitive facts.

**RoAS** reformulates language model attention as an integral of a query-conditioned probability measure against a value map over arbitrary **atom spaces** $(\Omega, \mathcal{F}, v)$:
$$R(x; \Omega) = \int_\Omega v(\omega)\, d\mu_x(\omega)$$

This unified framework proves that:
1. **In-Context Attention ($\Omega_{\text{ctx}}$)**, **Dense Retrieval ($\Omega_{\text{corpus}}$)**, and **Parameter-Expert Routing ($\Omega_{\text{params}}$)** are instances of the *same continuous read operator*.
2. **Top-$k$ Truncation Has an Exponential Error Bound:** The **Quadrature Truncation Bound Theorem** bounds exact ranked top-$k$ approximation under calibrated temperatures ($\tau = 0.05$, measured error $\approx 10^{-6}$). ANN indexes can supply candidates in deployment, but ANN recall and runtime are not benchmarked here.
3. **Lifelong Factual Scaling Without Damage:** Inserting 40 facts sequentially into an active pool of 200 atoms achieves **100% recall across all past edits and 100% background specificity**—completely eliminating the catastrophic interference of parametric editing methods (ROME/MEMIT).
4. **External-Memory Suppression via "Anti-Atoms":** Negative atoms cancel matched records in the signed external read, collapsing target probabilities to **uniform maximum entropy ($1/|\mathcal{V}| = 2.04\%$)** without retraining. This does not remove information from pretrained weights.
5. **Generative LLM Steering:** A relevance-gated read is applied at every greedy GPT-2 decoding step. On five factual prompts, it boosts first-token target probabilities up to **$1100\times$** and flips all first-token predictions.

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/mossaiby/RoAS.git
cd RoAS
pip install -r requirements.txt
```

### 2. Reproducing the Benchmarks

All scripts are standalone and run on both CPU and CUDA-enabled GPUs:

```bash
# Benchmark 1: Density Sweeps, Model-Editing Triad & Quadrature Truncation Error
python src/atom_space_engine.py

# Benchmark 2: Lifelong Sequential Scaling (40 edits without teardown to 200 atoms)
python src/atom_space_lifelong_engine.py

# Benchmark 3: Dynamic Tri-Space Routing (Context + Corpus + Parameters) & Ablation
python src/atom_space_tri_gating_engine.py

# Benchmark 4: Signed-Measure External-Memory Suppression & Counterfactual Updates
python src/atom_space_unlearning_engine.py

# Benchmark 5: Autoregressive Generative Steering with GPT-2 (124M)
python src/atom_space_generative_engine.py
```

---

## 📊 Summary of Experimental Results

| Benchmark | Key Metric | Result | Theoretical Validation |
| :--- | :--- | :--- | :--- |
| **Model-Editing Triad** | Efficacy / Generality / Specificity | **100% / 100% / 100%** | Locality & Margin Condition |
| **Quadrature Bounds** | Truncation Error ($\tau = 0.05, k=1$) | **$3.75 \times 10^{-6}$** | Theorem 1 (Exponential decay) |
| **Lifelong Editing** | 40 Sequential Edits (to 200 atoms) | **100% Retention / 100% Specificity** | Non-parametric measure scaling |
| **Tri-Space Gating** | Dynamic Routing Purity ($\Delta^2$) | **$>91\%$ per domain** | Functional ablation |
| **External-Memory Suppression** | Anti-Atom Cancellation | **$P = 2.04\% = 1/\|\mathcal{V}\|$** | Signed-read cancellation |
| **Causal GPT-2** | Target Prediction Flip & Boost | **100% Flipped / Up to $1100\times$** | Continuous hidden augmentation |

---

## 📖 Citation

If you find this work or repository useful, please cite our paper:

```bibtex
@article{mossaiby2026roas,
  title     = {Language Models as Reads over Atom Spaces: A Unified Measure-Theoretic Formulation of Context, Retrieval, and Parameters},
  author    = {Mossaiby, Farshid},
  journal   = {Neurocomputing},
  year      = {2026},
  note      = {Repository: \url{https://github.com/mossaiby/RoAS}}
}
```