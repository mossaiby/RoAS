"""Tests for retrieval evaluation, threshold calibration, and statistical estimators.

Covers:
- CounterFact record formatting and query flattening
- Retrieval summary metrics (top-1 recall, activation, false activation)
- ROC curve generation, trapezoidal AUC, and Youden's J threshold calibration
- Bootstrap confidence interval estimation
"""

from __future__ import annotations

import numpy as np
import pytest

from counterfact_retrieval_benchmark import flatten_queries, prepare_records, summarize_retrieval
from counterfact_threshold_calibration import compute_roc_metrics
from generative_steering_decoupled import bootstrap_ci


class TestCounterFactDataPreparation:
    """Tests for record preparation and query flattening."""

    def test_prepare_records_and_flattening(self, sample_counterfact_record):
        records = [sample_counterfact_record]
        memory_texts, targets, evaluations = prepare_records(records)

        assert len(memory_texts) == 1
        assert "Danielle Darrieux speaks the language English" in memory_texts[0]
        assert targets[0] == "English"
        assert len(evaluations) == 1

        query_texts, metadata = flatten_queries(evaluations, max_per_type=2)
        # 1 rewrite + 2 paraphrases + 2 neighborhoods = 5 queries
        assert len(query_texts) == 5
        assert len(metadata) == 5

        qtypes = [m["query_type"] for m in metadata]
        assert qtypes.count("rewrite") == 1
        assert qtypes.count("paraphrase") == 2
        assert qtypes.count("neighborhood") == 2


class TestRetrievalSummaries:
    """Tests for retrieval summary metrics."""

    def test_summarize_retrieval_metrics(self):
        # 2 rewrites, 2 paraphrases, 2 neighborhoods
        metadata = [
            {"query_type": "rewrite", "memory_index": 0},
            {"query_type": "rewrite", "memory_index": 1},
            {"query_type": "paraphrase", "memory_index": 0},
            {"query_type": "paraphrase", "memory_index": 1},
            {"query_type": "neighborhood", "memory_index": 0},
            {"query_type": "neighborhood", "memory_index": 1},
        ]
        # Predict index 0 for all, with similarities
        indices = np.array([0, 1, 0, 0, 0, 1])
        similarities = np.array([0.9, 0.85, 0.7, 0.4, 0.65, 0.3])
        threshold = 0.60

        summary = summarize_retrieval(indices, similarities, metadata, activation_threshold=threshold)

        # Rewrite: both predicted correctly and both >= 0.60
        assert summary["rewrite"]["top1_recall"] == 1.0
        assert summary["rewrite"]["activated_correct_recall"] == 1.0

        # Paraphrase: item 0 correct (0.7 >= 0.60), item 1 wrong index (predicted 0 != 1)
        assert summary["paraphrase"]["top1_recall"] == 0.5
        assert summary["paraphrase"]["activated_correct_recall"] == 0.5

        # Neighborhood: item 0 has similarity 0.65 >= 0.60 (false activation), item 1 has 0.30
        assert summary["neighborhood"]["false_activation_rate"] == 0.5
        # Item 0 predicted expected index 0 and activated -> wrong edit retrieval
        assert summary["neighborhood"]["wrong_edit_retrieval_rate"] == 0.5


class TestROCCalibration:
    """Tests for ROC metrics, AUC calculation, and Youden's J calibration."""

    def test_perfect_separation_auc(self):
        # 10 positive queries with high similarities, 10 negative with low
        pos_sims = np.linspace(0.7, 0.95, 10)
        neg_sims = np.linspace(0.1, 0.4, 10)
        similarities = np.concatenate([pos_sims, neg_sims])
        is_positive = np.array([True] * 10 + [False] * 10)

        thresholds = np.linspace(0.0, 1.0, 101)
        tpr, fpr, auc, optimal_tau = compute_roc_metrics(similarities, is_positive, thresholds)

        # Perfect separation must yield AUC = 1.0
        assert pytest.approx(auc, abs=1e-3) == 1.0
        # Optimal tau should cleanly separate pos from neg (between 0.4 and 0.7)
        assert 0.4 <= optimal_tau <= 0.71

    def test_random_separation_auc(self):
        np.random.seed(999)
        n = 200
        similarities = np.random.uniform(0.0, 1.0, n)
        is_positive = np.random.choice([True, False], size=n)

        thresholds = np.linspace(0.0, 1.0, 51)
        _, _, auc, _ = compute_roc_metrics(similarities, is_positive, thresholds)

        # Random predictions should yield AUC approximately 0.5
        assert 0.40 <= auc <= 0.60


class TestBootstrapEstimator:
    """Tests for percentile bootstrap confidence interval calculations."""

    def test_deterministic_extremes(self):
        all_ones = [1] * 50
        mean, low, high = bootstrap_ci(all_ones, num_bootstraps=200)
        assert mean == 1.0
        assert low == 1.0
        assert high == 1.0

        all_zeros = [0] * 50
        mean, low, high = bootstrap_ci(all_zeros, num_bootstraps=200)
        assert mean == 0.0
        assert low == 0.0
        assert high == 0.0

    def test_interval_ordering(self):
        np.random.seed(42)
        binary_data = list(np.random.binomial(1, 0.6, size=100))
        mean, low, high = bootstrap_ci(binary_data, num_bootstraps=500)

        assert 0.0 <= low <= mean <= high <= 1.0
        # 95% CI around p=0.6 with N=100 should contain 0.6
        assert low <= 0.6 <= high
