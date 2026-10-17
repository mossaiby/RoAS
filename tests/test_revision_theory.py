"""Tests for the propositions added in the first revision (src/roas_analysis.py).

These tests use NumPy only. They can be run without torch via
    pytest --noconftest tests/test_revision_theory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roas_analysis import (  # noqa: E402
    anti_atom_vs_deletion,
    bucket_queries,
    bucket_rates,
    certified_radii,
    null_atom_mass_bounds,
    null_atom_real_mass,
    query_displacements,
    roc_auc,
    split_prefix,
    youden_threshold,
)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def store():
    rng = np.random.default_rng(0)
    n, m = 60, 32
    keys = unit(rng.normal(size=(n, m)))
    # canonical queries close to their own key
    q0 = unit(keys + 0.3 * unit(rng.normal(size=(n, m))))
    return keys, q0, rng


class TestQuerySideCertificate:
    def test_certificate_is_sound_for_top1(self, store):
        keys, q0, rng = store
        radii = certified_radii(q0, keys)["top1"]
        checked = 0
        for i in range(len(keys)):
            if radii[i] <= 0:
                continue
            for _ in range(200):
                direction = unit(rng.normal(size=keys.shape[1]))
                delta = direction * radii[i] * rng.uniform(0.0, 0.999)
                x = q0[i] + delta  # need not be unit: the bound uses only ||delta||
                scores = keys @ x
                assert int(np.argmax(scores)) == i
                checked += 1
        assert checked > 0

    def test_certificate_is_sound_for_activation(self, store):
        keys, q0, rng = store
        theta = 0.5
        out = certified_radii(q0, keys, theta=theta)
        for i in np.where(out["activation"] > 0)[0]:
            for _ in range(100):
                delta = unit(rng.normal(size=keys.shape[1])) * out["activation"][i] * rng.uniform(0, 0.999)
                x = q0[i] + delta
                scores = keys @ x
                assert int(np.argmax(scores)) == i
                assert scores[i] >= theta - 1e-12

    def test_radius_is_tight_along_worst_direction(self, store):
        keys, q0, _ = store
        radii = certified_radii(q0, keys)["top1"]
        i = int(np.argmax(radii))
        gaps = (keys[i] - keys) @ q0[i]
        dists = np.linalg.norm(keys[i] - keys, axis=1)
        ratio = np.where(np.arange(len(keys)) == i, np.inf, gaps / np.maximum(dists, 1e-12))
        j = int(np.argmin(ratio))
        direction = (keys[j] - keys[i]) / dists[j]
        x = q0[i] + direction * radii[i] * 1.001
        assert keys[j] @ x > keys[i] @ x  # just past the radius, the competitor wins

    def test_non_top1_record_has_zero_radius(self):
        keys = unit(np.array([[1.0, 0.0], [0.0, 1.0]]))
        q0 = unit(np.array([[0.0, 1.0], [0.0, 1.0]]))  # record 0's canonical query prefers key 1
        r = certified_radii(q0, keys)["top1"]
        assert r[0] == 0.0 and r[1] > 0.0

    def test_displacement(self, store):
        keys, q0, _ = store
        d = query_displacements(q0[[3, 4]], q0, [3, 4])
        assert np.allclose(d, 0.0)


class TestNullAtom:
    def test_mass_formula_matches_explicit_read(self):
        rng = np.random.default_rng(1)
        s = rng.uniform(-1, 1, size=20)
        theta, tau = 0.4, 0.1
        kern = np.exp(s / tau)
        explicit = kern.sum() / (kern.sum() + np.exp(theta / tau))
        assert np.isclose(null_atom_real_mass(s, theta, tau), explicit)

    @pytest.mark.parametrize("tau", [0.2, 0.05, 0.01])
    def test_bounds_hold(self, tau):
        rng = np.random.default_rng(2)
        s = rng.uniform(-1, 1, size=(50, 30))
        theta = 0.5
        m = null_atom_real_mass(s, theta, tau)
        lo, hi = null_atom_mass_bounds(s.max(axis=1), s.shape[1], theta, tau)
        assert np.all(m >= lo - 1e-12) and np.all(m <= hi + 1e-12)

    def test_zero_temperature_limit_is_hard_threshold(self):
        rng = np.random.default_rng(3)
        s = rng.uniform(-1, 1, size=(200, 25))
        theta = 0.6
        top = s.max(axis=1)
        keep = np.abs(top - theta) > 0.02
        m = null_atom_real_mass(s[keep], theta, tau=1e-3)
        assert np.array_equal(m > 0.5, top[keep] > theta)


class TestAntiAtomVersusDeletion:
    @pytest.mark.parametrize("tau", [1.0, 0.1, 0.05])
    def test_shrinkage_identity(self, tau):
        rng = np.random.default_rng(4)
        n, e = 40, 16
        s = rng.uniform(-1, 1, size=n)
        v = unit(rng.normal(size=(n, e)))
        out = anti_atom_vs_deletion(s, v, target=int(np.argmax(s)), tau=tau)
        assert np.allclose(out["anti"], out["shrinkage"] * out["deletion"], atol=1e-12)

    def test_anti_atom_abstains_where_deletion_falls_through(self):
        rng = np.random.default_rng(5)
        n, e = 40, 16
        s = rng.uniform(-0.2, 0.3, size=n)
        s[7] = 0.95  # query sits on the record to be removed
        s[3] = 0.60  # a clear runner-up record
        v = unit(rng.normal(size=(n, e)))
        out = anti_atom_vs_deletion(s, v, target=7, tau=0.05)
        # deletion: a confident read of the runner-up record; anti-atom: near-zero read
        assert np.linalg.norm(out["deletion"]) > 0.5
        assert np.linalg.norm(out["anti"]) < 1e-3

    def test_far_queries_are_unaffected(self):
        rng = np.random.default_rng(6)
        n, e = 40, 16
        s = rng.uniform(0.3, 0.9, size=n)
        s[0] = -0.9  # record far from this query
        v = unit(rng.normal(size=(n, e)))
        out = anti_atom_vs_deletion(s, v, target=0, tau=0.05)
        assert out["shrinkage"] > 0.999999


class TestBucketsAndCalibration:
    def test_buckets_partition_and_rates(self):
        qt = ["paraphrase"] * 4 + ["neighborhood"] * 3
        exp = [0, 1, 2, 3, 0, 1, 2]
        got = [0, 1, 9, 9, 0, 5, 2]
        sim = [0.9, 0.4, 0.8, 0.3, 0.7, 0.8, 0.2]
        b = bucket_queries(qt, exp, got, sim, theta=0.55)
        assert b == [
            "activated_correct", "retrieved_rejected", "wrong_record_activated", "wrong_record_rejected",
            "false_activation_own_record", "false_activation_other_record", "silent",
        ]
        rates = bucket_rates(b, qt)
        assert np.isclose(sum(v for k, v in rates["paraphrase"].items() if k != "count"), 1.0)
        assert np.isclose(sum(v for k, v in rates["neighborhood"].items() if k != "count"), 1.0)

    def test_auc_and_youden(self):
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        labels = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        assert roc_auc(scores, labels) == 1.0
        t, j = youden_threshold(scores, labels, np.linspace(0, 1, 101))
        assert 0.3 < t <= 0.7 and j == 1.0
        assert roc_auc(np.array([0.5, 0.5]), np.array([True, False])) == 0.5

    def test_split_prefix(self):
        pre, core = split_prefix("Shayna does this and Yossel goes still and dies. Danielle Darrieux, a native")
        assert pre.startswith("Shayna") and core == "Danielle Darrieux, a native"
        assert split_prefix("The mother tongue of X is") == ("", "The mother tongue of X is")
