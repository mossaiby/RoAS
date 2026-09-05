"""Unit and architectural tests for AtomSpace and Tri-Space components.

Covers:
- AtomSpace buffer management (dense and signed)
- Metric-preserving residual adapter phi(x)
- Dynamic atom insertion and buffer concatenation
- ParameterExpertBank output routing
- Tri-space routing distribution lambda(x) in simplex Delta^2
- Targeted space ablation (context, corpus, parameters)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from atom_space_engine import AtomSpace as DenseAtomSpace, EngineConfig, UnifiedResidualTiedLM
from atom_space_unlearning_engine import AtomSpace as SignedAtomSpace, UnlearnEngineConfig
from atom_space_tri_gating_engine import (
    AtomSpace as TriAtomSpace,
    ParameterExpertBank,
    TriEngineConfig,
    TriSpaceLanguageModel,
)


class TestDenseAtomSpace:
    """Tests for the standard dense AtomSpace container."""

    def test_buffer_initialization_and_set(self):
        space = DenseAtomSpace("corpus_test", value_dim=64)
        assert space.name == "corpus_test"
        assert space.keys.shape == (0, 64)
        assert space.values.shape == (0, 64)

        dummy_keys = torch.randn(10, 64)
        dummy_values = torch.randn(10, 64)
        space.set_atoms(dummy_keys, dummy_values)

        assert space.keys.shape == (10, 64)
        assert space.values.shape == (10, 64)

    def test_read_vector_shape_and_convex_sum(self):
        cfg = EngineConfig()
        dim = 32
        vocab_size = 10
        model = UnifiedResidualTiedLM(cfg, in_feat_dim=dim, vocab_size=vocab_size)

        n_atoms = 15
        batch_size = 4

        keys = F.normalize(torch.randn(n_atoms, dim), dim=-1)
        values = F.normalize(torch.randn(n_atoms, cfg.token_embed_dim), dim=-1)
        model.omega_corpus.set_atoms(keys, values)

        queries = F.normalize(torch.randn(batch_size, dim), dim=-1)
        z_read = model.read_vector(queries, tau=0.1)

        assert z_read.shape == (batch_size, cfg.token_embed_dim)

        # Forward output verification
        logits, mu = model(queries, tau=0.1)
        assert logits.shape == (batch_size, vocab_size)
        assert mu.shape == (batch_size, n_atoms)
        torch.testing.assert_close(mu.sum(dim=-1), torch.ones(batch_size))


class TestSignedAtomSpace:
    """Tests for the signed-measure AtomSpace supporting anti-atoms."""

    def test_dynamic_atom_insertion(self):
        space = SignedAtomSpace("unlearn_test", value_dim=64)
        assert len(space.keys) == 0

        # Insert positive atom
        key_pos = torch.randn(64)
        val_pos = torch.randn(64)
        space.insert_atom(key_pos, val_pos, weight=1.0)
        assert len(space.keys) == 1
        assert space.weights[0].item() == 1.0

        # Insert negative anti-atom
        key_neg = key_pos.clone()
        val_neg = val_pos.clone()
        space.insert_atom(key_neg, val_neg, weight=-1.0)
        assert len(space.keys) == 2
        assert space.weights[1].item() == -1.0

    def test_empty_signed_space_returns_zeros(self):
        space = SignedAtomSpace("empty_test", value_dim=48)
        q = torch.randn(2, 48)
        z, mu = space.read_signed(q, nn.Identity(), tau=0.1)
        assert z.shape == (2, 48)
        assert torch.all(z == 0.0)
        assert mu is None


class TestParameterExpertBank:
    """Tests for the modular parameter expert bank."""

    def test_expert_output_dimensions_and_normalization(self):
        num_experts = 6
        value_dim = 32
        bank = ParameterExpertBank(num_experts=num_experts, value_dim=value_dim)

        batch_size = 3
        # Routing weights over the 6 experts
        routing_weights = F.softmax(torch.randn(batch_size, num_experts), dim=-1)

        output = bank(routing_weights)
        assert output.shape == (batch_size, value_dim)

        # Internal expert values must be normalized to unit L2 norm during forward
        norm_experts = F.normalize(bank.expert_values, p=2, dim=-1)
        expected = routing_weights @ norm_experts
        torch.testing.assert_close(output, expected)


class TestTriSpaceArchitecture:
    """Tests for the integrated TriSpaceLanguageModel and routing gating."""

    @pytest.fixture
    def tri_model(self):
        cfg = TriEngineConfig(token_embed_dim=64, hidden_dim=64, read_temperature=0.07)
        in_dim = 64
        vocab_size = 20
        num_param_tasks = 4
        model = TriSpaceLanguageModel(
            cfg=cfg, in_feat_dim=in_dim, vocab_size=vocab_size, num_param_tasks=num_param_tasks
        )

        # Set mock corpus atoms
        corp_k = F.normalize(torch.randn(8, in_dim), dim=-1)
        corp_v = F.normalize(torch.randn(8, cfg.token_embed_dim), dim=-1)
        model.omega_corpus.set_atoms(corp_k, corp_v)

        # Set mock parameter trigger keys
        param_k = F.normalize(torch.randn(num_param_tasks, in_dim), dim=-1)
        model.omega_params.keys = param_k
        return model

    def test_phi_adapter_preserves_unit_norm(self, tri_model):
        raw = torch.randn(5, 64)
        adapted = tri_model.phi(raw)
        norms = torch.linalg.vector_norm(adapted, dim=-1)
        torch.testing.assert_close(norms, torch.ones(5), atol=1e-6, rtol=1e-6)

    def test_tri_space_router_simplex(self, tri_model):
        query = torch.randn(4, 64)
        logits, gating, reads = tri_model(query)

        assert logits.shape == (4, tri_model.vocab_size)
        assert gating.shape == (4, 3)

        # Router distribution must be in standard 2-simplex Delta^2: sum = 1, each >= 0
        gate_sums = gating.sum(dim=-1)
        torch.testing.assert_close(gate_sums, torch.ones(4))
        assert torch.all(gating >= 0.0)

    @pytest.mark.parametrize("ablate_target,gate_idx", [("ctx", 0), ("corpus", 1), ("params", 2)])
    def test_space_ablation_zeroes_gate(self, tri_model, ablate_target, gate_idx):
        query = torch.randn(2, 64)
        _, lambdas, _ = tri_model(query, ablate_space=ablate_target)
        # The ablated space's gate weight must be zeroed out
        assert torch.all(lambdas[:, gate_idx] == 0.0)
        # The remaining gate weights must re-normalize to 1.0
        torch.testing.assert_close(lambdas.sum(dim=-1), torch.ones(2))
