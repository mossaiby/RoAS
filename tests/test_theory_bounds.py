"""Tests for theoretical theorems and bounds derived in the manuscript.

Covers:
- Theorem 1: Softmax tail truncation error bounds (Bound 1 and Bound 2)
- Corollary 1: Error-budgeted adaptive top-k stopping rule
- Proposition 1: Score stability on unseen atoms (Lipschitz margin)
- Proposition 3: Exact matched-pair anti-atom cancellation
- Proposition 4: Approximate signed-pair cancellation perturbation bound
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from adaptive_retrieval_benchmark import adaptive_topk_read, fixed_topk_error
from signed_cancellation_stress import measure_trial, StressConfig


class TestSoftmaxTailTruncationBounds:
    """Verifies Theorem 1: Tail truncation bounds for exact ranked top-k reads."""

    @pytest.mark.parametrize("tau", [0.05, 0.1, 0.5, 1.0])
    @pytest.mark.parametrize("k", [1, 3, 5, 10, 20])
    def test_theorem_1_bounds_hold(self, synthetic_read_problem, tau: float, k: int):
        queries, keys, values = synthetic_read_problem
        n_items = keys.shape[0]
        v_max = values.norm(dim=-1).max().item()

        scores = (queries @ keys.T) / tau  # [B, N]
        sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
        sorted_values = values[sorted_indices]  # [B, N, D]

        # Full dense read
        weights_full = F.softmax(sorted_scores, dim=-1)
        full_read = torch.bmm(weights_full.unsqueeze(1), sorted_values).squeeze(1)

        # Truncated read at k
        weights_k = F.softmax(sorted_scores[:, :k], dim=-1)
        k_read = torch.bmm(weights_k.unsqueeze(1), sorted_values[:, :k]).squeeze(1)

        actual_error = (full_read - k_read).norm(dim=-1)  # [B]

        # Bound 1: 2 * V_max * (N - k) * exp(-(s_(1) - s_(k+1)) / tau)
        # Note: (s_1 - s_(k+1)) is the gap between top-1 and the first omitted score
        delta_k = (sorted_scores[:, 0] - sorted_scores[:, k])  # gap: s_(1) - s_(k+1) in 0-indexed terms
        bound_1 = 2.0 * v_max * (n_items - k) * torch.exp(-delta_k)

        # Bound 2: 2 * V_max * ((N - k) / k) * exp(-(s_(k) - s_(k+1)) / tau)
        boundary_gap = (sorted_scores[:, k - 1] - sorted_scores[:, k])  # s_(k) - s_(k+1)
        bound_2 = 2.0 * v_max * ((n_items - k) / k) * torch.exp(-boundary_gap)

        # Both bounds must cover actual error up to numerical precision
        slack_1 = bound_1 - actual_error
        slack_2 = bound_2 - actual_error

        assert torch.all(slack_1 >= -1e-10), f"Bound 1 violated: min slack = {slack_1.min().item()}"
        assert torch.all(slack_2 >= -1e-10), f"Bound 2 violated: min slack = {slack_2.min().item()}"

    def test_decay_with_increasing_k(self, synthetic_read_problem):
        """Verifies that truncation error monotonically decreases as k grows."""
        queries, keys, values = synthetic_read_problem
        tau = 0.05
        scores = (queries @ keys.T) / tau
        sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
        sorted_values = values[sorted_indices]

        weights_full = F.softmax(sorted_scores, dim=-1)
        full_read = torch.bmm(weights_full.unsqueeze(1), sorted_values).squeeze(1)

        errors = []
        for k in [1, 2, 5, 10, 20]:
            err = fixed_topk_error(sorted_scores, sorted_values, full_read, k)
            errors.append(err.mean().item())

        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-12, "Truncation error did not decay monotonically"


class TestAdaptiveErrorBudget:
    """Verifies Corollary 1: Adaptive truncation with guaranteed error budget."""

    @pytest.mark.parametrize("error_budget", [0.01, 0.05, 0.1, 0.25])
    def test_corollary_1_coverage(self, synthetic_read_problem, error_budget: float):
        queries, keys, values = synthetic_read_problem
        tau = 0.07

        scores = (queries @ keys.T) / tau
        sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
        sorted_values = values[sorted_indices]

        weights_full = F.softmax(sorted_scores, dim=-1)
        full_read = torch.bmm(weights_full.unsqueeze(1), sorted_values).squeeze(1)

        selected_k, certificate, actual_error, tail_mass = adaptive_topk_read(
            sorted_scores, sorted_values, full_read, error_budget
        )

        # Certificate must be within requested error budget
        assert torch.all(certificate <= error_budget + 1e-8), "Certificate exceeded requested error budget"
        # Actual error must be bounded by the certificate
        assert torch.all(actual_error <= certificate + 1e-8), "Actual error exceeded the tail-mass certificate"
        # Selected k must be within valid index range
        assert torch.all(selected_k >= 1) and torch.all(selected_k <= keys.shape[0])


class TestLipschitzScoreControl:
    """Verifies Proposition 1: Score stability under Lipschitz encoder mapping."""

    def test_score_stability_bound(self):
        dim = 64
        q = F.normalize(torch.randn(dim, dtype=torch.float64), dim=-1)

        # Linear mapping W: R^d -> R^d as key encoder phi_k
        torch.manual_seed(123)
        w = torch.randn(dim, dim, dtype=torch.float64)
        # Operator norm of W is its largest singular value
        l_constant = torch.linalg.matrix_norm(w, ord=2).item()

        omega_0 = torch.randn(dim, dtype=torch.float64)
        omega_star = omega_0 + 0.2 * torch.randn(dim, dtype=torch.float64)

        phi_k_0 = w @ omega_0
        phi_k_star = w @ omega_star

        score_0 = torch.dot(q, phi_k_0).item()
        score_star = torch.dot(q, phi_k_star).item()
        score_diff = abs(score_star - score_0)

        d_omega = torch.norm(omega_star - omega_0, p=2).item()
        q_norm = torch.norm(q, p=2).item()
        rhs_bound = q_norm * l_constant * d_omega

        assert score_diff <= rhs_bound + 1e-10, (
            f"Proposition 1 bound violated: |<q, phi(w*)> - <q, phi(w0)>| = {score_diff} > {rhs_bound}"
        )


class TestSignedSpaceCancellation:
    """Verifies Propositions 3 and 4: Signed external-memory cancellation and perturbation bounds."""

    def test_proposition_3_exact_pair_cancellation(self):
        """When keys and values match with equal opposite mass, pair read is exactly 0."""
        dim = 128
        tau = 0.07

        q = F.normalize(torch.randn(1, dim, dtype=torch.float64), dim=-1)
        key_target = F.normalize(torch.randn(dim, dtype=torch.float64), dim=-1)
        val_target = F.normalize(torch.randn(dim, dtype=torch.float64), dim=-1)

        # Matched pair: positive atom (+1) and anti-atom (-1)
        pair_keys = torch.stack([key_target, key_target], dim=0)    # [2, D]
        pair_values = torch.stack([val_target, val_target], dim=0)  # [2, D]
        weights = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)  # [+1, -1]

        scores = (q @ pair_keys.T) / tau  # [1, 2]
        raw_kernel = torch.exp(scores - scores.max())
        weighted_kernel = raw_kernel * weights.T  # [+k, -k]

        denominator = torch.sum(torch.abs(weighted_kernel), dim=-1, keepdim=True)
        signed_mu = weighted_kernel / denominator  # [+0.5, -0.5]
        z_pair = signed_mu @ pair_values

        assert torch.norm(z_pair).item() < 1e-15, "Exact matched-pair cancellation did not evaluate to 0"

    @pytest.mark.parametrize("key_noise", [0.0, 0.01, 0.05, 0.1])
    @pytest.mark.parametrize("value_noise", [0.0, 0.01, 0.05, 0.1])
    @pytest.mark.parametrize("mass_error", [0.0, 0.02, 0.05])
    def test_proposition_4_approximate_bound(self, key_noise, value_noise, mass_error):
        """Verifies ||a v - b v_tilde|| / Z <= (|a - b| ||v|| + b ||v - v_tilde||) / Z."""
        config = StressConfig(seed=42, dimension=64, temperature=0.07)
        result = measure_trial(
            config=config,
            background_size=20,
            key_noise=key_noise,
            value_noise=value_noise,
            mass_error=mass_error,
            query_noise=0.0,
        )

        residual = result["pair_residual"]
        bound = result["pair_bound"]
        assert residual <= bound + 1e-9, f"Proposition 4 bound violated: residual {residual} > bound {bound}"
