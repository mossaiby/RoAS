"""Pytest configuration and shared fixtures for the RoAS test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Ensure src and scripts are on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Ensure determinism across tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def synthetic_read_problem() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates normalized queries, keys, and values in R^128."""
    dim = 128
    num_keys = 50
    num_queries = 8

    keys = F.normalize(torch.randn(num_keys, dim, dtype=torch.float64), dim=-1)
    values = F.normalize(torch.randn(num_keys, dim, dtype=torch.float64), dim=-1)
    queries = F.normalize(torch.randn(num_queries, dim, dtype=torch.float64), dim=-1)
    return queries, keys, values


@pytest.fixture
def sample_counterfact_record() -> Dict:
    """Provides a synthetic CounterFact-structured record."""
    return {
        "case_id": 0,
        "paraphrase_prompts": [
            "The mother tongue of Danielle Darrieux is",
            "Danielle Darrieux spoke the language"
        ],
        "neighborhood_prompts": [
            "The official language of France is",
            "In Paris, people primarily converse in"
        ],
        "requested_rewrite": {
            "prompt": "{} speaks the language",
            "subject": "Danielle Darrieux",
            "target_new": {"str": "English", "id": 3594},
            "target_true": {"str": "French", "id": 4141}
        }
    }
