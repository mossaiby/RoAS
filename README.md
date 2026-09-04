# RoAS: Reads over Atom Spaces

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

This repository provides the official implementation and experimental benchmarks for the paper:  
**"Language Models as Reads over Atom Spaces: A Unified Measure-Theoretic Formulation of Context, Retrieval, and Parameters"**  
by **Farshid Mossaiby** (*University of Isfahan*).

---

## 📌 Overview

Language models can encode factual knowledge in dense parameters, while retrieval systems and expert routers provide alternative mechanisms for accessing mutable or specialized information.

**RoAS** reformulates language model attention as an integral of a query-conditioned probability measure against a value map over arbitrary **atom spaces** $(\Omega, \mathcal{F}, v)$:
$$R(x; \Omega) = \int_\Omega v(\omega)\, d\mu_x(\omega)$$

The repository studies these aspects of the framework:
1. **Common Output-Space Reads:** finite attention, dense retrieval, and expert-output routing share a normalized weighted-read structure while retaining different internal computations. Frozen exact 1-NN matches the trained read on the templated insertion task.
2. **Error-Budgeted Exact Reads:** exact softmax tail mass selects the minimum certified $k$ for a requested vector-error budget. Scaling runs show that the certificate can require a large corpus fraction.
3. **Established Retrieval Evaluation:** CounterFact-500 measures rewrite, paraphrase, and neighborhood behavior, fixed/adaptive reads, and FAISS HNSW candidate recall.
4. **Cumulative External-Memory Insertion:** 40 sequential records retain target and training-control argmax predictions across five seeds in a templated 200-record benchmark.
5. **Signed-Read Suppression:** negative atoms exactly cancel matched contributions; 3,000 perturbation trials quantify sensitivity to key, value, and mass mismatch. This is not physical deletion or certified unlearning.
6. **GPT-2 Steering:** ten facts, ten paraphrases, ten controls, multi-token scoring, perplexity checks, and threshold/strength sensitivity extend the direct target-embedding mechanism study.

---

## 🚀 Quickstart

### 1. Installation
The committed results were generated with Python 3.11.9 and the exact package versions in `requirements.txt` (GPU-accelerated via CUDA 12.4; CPU-only execution also works).

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

# Benchmark 3b: Five-seed surface-matched routing statistics
python src/tri_space_seed_sweep.py

# Benchmark 4: Signed-Measure External-Memory Suppression & Counterfactual Updates
python src/atom_space_unlearning_engine.py

# Benchmark 5: Autoregressive Generative Steering with GPT-2 (124M)
python src/atom_space_generative_engine.py

# Benchmark 6: Adaptive exact-read error budgets and scaling
python src/adaptive_retrieval_benchmark.py

# Benchmark 7: CounterFact-500 retrieval and FAISS HNSW candidates
python src/counterfact_retrieval_benchmark.py

# Benchmark 8: Approximate signed-cancellation stress test
python src/signed_cancellation_stress.py

# Five-seed insertion, lifelong, and update/suppression aggregation
python src/multi_seed_benchmarks.py

# Benchmark 9: Naive full-parameter fine-tuning baseline for the ten GPT-2 facts
python src/gpt2_finetune_baseline.py

# Benchmark 10: Simplified single-layer rank-one edit (ROME-lite) baseline for the ten GPT-2 facts
python src/rome_baseline_engine.py
```

The CounterFact script downloads the official dataset to `data/counterfact.json` on first use. Use `--limit`, `--max-queries-per-type`, and the other command-line flags for smaller diagnostic runs.

---

## 📊 Summary of Experimental Results

| Benchmark | Key Metric | Reported Result | Scope |
| :--- | :--- | :--- | :--- |
| **Inserted-Record Retrieval** | Trained read and frozen exact 1-NN | **Both 100% / 100% / 100%** | Generated templates, five seeds; no adapter advantage |
| **Exact Truncation** | Mean error ($\tau = 0.05, k=1$) | **$3.82 \times 10^{-6}$** | Exact scoring over 160 records |
| **Adaptive Scaling** | $N=25{,}000$, $\varepsilon=0.05$, $\tau=0.05$ | **100% coverage; mean $k=13{,}026$** | Exact scoring; certificate is conservative |
| **CounterFact Retrieval** | Rewrite / paraphrase top-1 | **99.2% / 91.0%** | 500 records; activated paraphrase recall 38.2% |
| **HNSW Candidates** | Exact top-1 in 10 candidates | **99.96% recall** | CounterFact-500; not large-scale ANN evidence |
| **Cumulative Insertion** | 40 additions to 200 records | **100.0±0.0% target / control argmax** | Five seeds; training controls |
| **Tri-Space Gating** | Five-seed surface-matched task / route accuracy | **100/96.8/100% task; 100/82.4/100% route** | Mean over five seeds; known tasks and labels |
| **Matched Suppression** | Residual target probability | **$2.0407\pm0.00002\%$** | Five seeds; exact paired-key construction |
| **Cancellation Stress** | Analytical residual-bound coverage | **100% over 3,000 trials** | Key mismatch degrades rapidly |
| **GPT-2 Steering** | Canonical / paraphrase first-token correction | **10/10 / 9/10** | Ten controls bypassed intervention |
| **Naive Fine-Tuning Baseline** | Efficacy / generality / neighborhood specificity | **100% / 70% / 58.9%** | Full-parameter FT on the same ten facts; higher drift than the atom-space read |
| **ROME-Lite Baseline** | Efficacy / generality / neighborhood specificity | **100% / 40% / 87.8%** | Simplified single-layer rank-one edit on the same ten facts; better locality than FT, weaker paraphrase generality |

---

## 📖 Citation

If you find this work or repository useful, please cite our paper:

```bibtex
@article{mossaiby2026roas,
  title     = {Language Models as Reads over Atom Spaces: A Unified Measure-Theoretic Formulation of Context, Retrieval, and Parameters},
  author    = {Mossaiby, Farshid},
  note      = {Manuscript and code repository},
  year      = {2026},
  url       = {https://github.com/mossaiby/RoAS}
}
```