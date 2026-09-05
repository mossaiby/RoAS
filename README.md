# RoAS: Reads over Atom Spaces

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

This repository provides the official implementation and experimental benchmarks for the paper:  
**"Reads over Atom Spaces: A Mathematical View of Context, Retrieval, and Parameter Routing in Language Models"**  
by **Farshid Mossaiby** (*University of Isfahan*).

---

## 📌 Overview

Language models can encode factual knowledge in dense parameters, while retrieval systems and expert routers provide alternative mechanisms for accessing mutable or specialized information.

**RoAS** formulates language model attention as an integral of a query-conditioned probability measure against a value map over arbitrary **atom spaces** $(\Omega, \mathcal{F}, v)$:
$$R(x; \Omega) = \int_\Omega v(\omega)\, d\mu_x(\omega)$$

The repository studies these aspects of the framework:
1. **Common Output-Space Reads:** finite attention, dense retrieval, and expert-output routing share a normalized weighted-read structure while retaining different internal computations. Frozen exact 1-NN matches the trained read on the templated insertion task.
2. **Error-Budgeted Exact Reads:** exact softmax tail mass selects the minimum certified $k$ for a requested vector-error budget. Scaling runs show that the certificate can require a large corpus fraction.
3. **Established Retrieval Evaluation & Data-Driven Calibration:** CounterFact-500 measures rewrite, paraphrase, and neighborhood behavior. Replacing a static 0.60 threshold with held-out ROC calibration ($\tau^* = 0.55$, AUC = 0.817) increases activated paraphrase recall from 38.2% to 53.2%.
4. **Lipschitz Margin Condition Diagnostics:** Evaluates the checkable sufficient condition of Proposition 1; demonstrates that while actual top-1 entry is 100%, worst-case global Cauchy–Schwarz Lipschitz bounds certify 0.0% in sparse semantic spaces.
5. **Cumulative External-Memory Insertion:** 40 sequential records retain target and training-control argmax predictions across five seeds in a templated 200-record benchmark.
6. **Signed-Read Suppression:** negative atoms exactly cancel matched contributions; 3,000 perturbation trials quantify sensitivity to key, value, and mass mismatch. This is external-read cancellation, not physical deletion or certified parameter unlearning.
7. **GPT-2 Steering & Scale:** Ten facts, ten paraphrases, ten controls, multi-token scoring, perplexity checks, and threshold/strength sensitivity extend the direct target-embedding mechanism study, complemented by an expanded $N=50$ CounterFact generative study under calibrated thresholding.

---

## 🚀 Quickstart

### 1. Installation
The committed results were generated with Python 3.11.9 and the exact package versions in `requirements.txt` (GPU-accelerated via CUDA 12.4; CPU-only execution also works).

```bash
git clone https://github.com/mossaiby/RoAS.git
cd RoAS
pip install -r requirements.txt

# Validate the committed artifact manifest before a full re-run.
python scripts/verify_artifacts.py
```

The scripts write canonical outputs into `results/` and `results/details/`. Run them from the repository root. A full run can replace committed artifacts; preserve a clean checkout or copy artifacts before exploratory changes. CPU and GPU runs may differ at floating-point precision, while the validator checks structural invariants and reported experimental scope rather than exact floating-point equality.

### 2. Reproducing the Benchmarks

All scripts are standalone and run on both CPU and CUDA-enabled GPUs:

```bash
# Benchmark 1: Density Sweeps, Model-Editing Triad & Softmax Truncation Error
python src/atom_space_engine.py

# Benchmark 1b: Empirical verification of Proposition 1 / Eq. (1) Lipschitz bounds
python src/lipschitz_margin_check.py

# Benchmark 2: Lifelong Sequential Scaling (40 edits without teardown to 200 atoms)
python src/atom_space_lifelong_engine.py

# Benchmark 3: Dynamic Tri-Space Routing (Context + Corpus + Parameters) & Ablation
python src/atom_space_tri_gating_engine.py

# Benchmark 3b: Five-seed surface-matched routing statistics
python src/tri_space_seed_sweep.py

# Benchmark 4: Signed-Measure External-Memory Suppression & Counterfactual Updates
python src/atom_space_unlearning_engine.py

# Benchmark 5: Autoregressive Generative Steering with GPT-2 (124M) - 10-Fact Study
python src/atom_space_generative_engine.py

# Benchmark 6: Adaptive exact-read error budgets and scaling
python src/adaptive_retrieval_benchmark.py

# Benchmark 7: CounterFact-500 retrieval and FAISS HNSW candidates
python src/counterfact_retrieval_benchmark.py

# Benchmark 7b: Held-out CounterFact ROC analysis and optimal threshold calibration
python src/counterfact_threshold_calibration.py

# Benchmark 8: Approximate signed-cancellation stress test (3,000 trials)
python src/signed_cancellation_stress.py

# Five-seed insertion, lifelong, and update/suppression aggregation
python src/multi_seed_benchmarks.py

# Benchmark 9: Naive full-parameter fine-tuning baseline for the ten GPT-2 facts
python src/gpt2_finetune_baseline.py

# Benchmark 10: Simplified single-layer rank-one edit (ROME-lite) baseline for the ten GPT-2 facts
python src/rome_baseline_engine.py

# Benchmark 11: Covariance-regularized ROME reproduction on the full 500-record CounterFact set
python src/counterfact_rome_benchmark.py

# Benchmark 12: Expanded CounterFact generative steering (N=50) with bootstrap confidence intervals
python src/generative_steering_decoupled.py
```

The CounterFact scripts download the official dataset to `data/counterfact.json` on first use.

### 3. Artifact Validation

After reproducing a benchmark, verify that the committed-result schema and the study sizes described in the manuscript remain intact:

```bash
python scripts/verify_artifacts.py
```

This lightweight check has no model-download or GPU requirement. It validates the calibration split, the $N=50$ steering scope, the $N=500$ rank-one baseline scope, the five-seed routing aggregate, and the 3,000-trial signed-cancellation design.

### 4. Scope of the Evidence

The synthetic insertion, routing, and signed-cancellation experiments are controlled mechanism studies. They are not intended as a general performance comparison with production attention, RAG, MoE, or certified-unlearning systems. The GPT-2 experiment uses retrieved target-token embeddings mixed into the final hidden state and measures first-token control conditional on retrieval. The CounterFact rank-one baseline is ROME-inspired, not a reproduction of the full ROME protocol. See the manuscript limitations for the corresponding evaluation gaps.

---

## 📊 Summary of Experimental Results

| Benchmark | Key Metric | Reported Result | Scope |
| :--- | :--- | :--- | :--- |
| **Inserted-Record Retrieval** | Trained read and frozen exact 1-NN | **Both 100% / 100% / 100%** | Generated templates, five seeds; no adapter advantage |
| **Exact Truncation** | Mean error ($\tau = 0.05, k=1$) | **$3.82 \times 10^{-6}$** | Exact scoring over 160 records |
| **Adaptive Scaling** | $N=25{,}000$, $\varepsilon=0.05$, $\tau=0.05$ | **100% coverage; mean $k=13{,}026$** | Exact scoring; certificate is conservative |
| **Lipschitz Margin Condition** | Analytical bound / Eq. (1) certification | **$L \le 2.105$ / 0.0% certified vs 100% actual** | Proves global Cauchy–Schwarz certificate is vacuous in sparse spaces |
| **CounterFact Retrieval** | Rewrite / paraphrase top-1 | **99.2% / 91.0%** | 500 records; activated paraphrase recall 38.2% at $\tau = 0.60$ |
| **CounterFact ROC / Calibration** | AUC / Calibrated Paraphrase Recall | **AUC = 0.817 / 53.2% (at $\tau^* = 0.55$)** | Held-out calibration replaces 0.60 strawman (was 38.2%) |
| **CounterFact Generative Steering** | Rewrite / Paraphrase Score ($N=50$) | **100.0% / 66.0% [52%, 78%]** | Base: 16% / 22%; non-overlapping 95% bootstrap CIs |
| **HNSW Candidates** | Exact top-1 in 10 candidates | **99.96% recall** | CounterFact-500; not large-scale ANN evidence |
| **Cumulative Insertion** | 40 additions to 200 records | **100.0±0.0% target / control argmax** | Five seeds; training controls |
| **Tri-Space Gating** | Five-seed surface-matched task / route accuracy | **100/96.8/100% task; 100/82.4/100% route** | Mean over five seeds; known tasks and labels |
| **Matched Suppression** | Residual target probability | **$2.0407\pm0.00003\%$** | Five seeds; exact paired-key construction |
| **Cancellation Stress** | Analytical residual-bound coverage | **100% over 3,000 trials** | Key mismatch degrades rapidly |
| **GPT-2 Steering (Exploratory)** | Canonical / paraphrase first-token correction | **10/10 / 9/10** | Ten controls bypassed intervention |
| **Naive Fine-Tuning Baseline** | Efficacy / generality / neighborhood specificity | **100% / 70% / 58.9%** | Full-parameter FT on 10 facts; higher drift than atom-space read |
| **ROME-Lite Baseline** | Efficacy / generality / neighborhood specificity | **100% / 40% / 87.8%** | Simplified single-layer rank-one edit on 10 facts |
| **CounterFact-Scale ROME** | Efficacy / Paraphrase / Neighborhood Score | **85.2% / 59.5% / 70.1%** | Covariance-regularized rank-one edit, 500 CounterFact records |

---

## 📖 Citation

If you find this work or repository useful, please cite our paper:

```bibtex
@article{mossaiby2026roas,
  title     = {Reads over Atom Spaces: A Mathematical View of Context, Retrieval, and Parameter Routing in Language Models},
  author    = {Mossaiby, Farshid},
  note      = {Manuscript and code repository},
  year      = {2026},
  url       = {https://github.com/mossaiby/RoAS}
}
```