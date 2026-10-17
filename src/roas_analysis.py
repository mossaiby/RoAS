"""
src/roas_analysis.py

Pure-NumPy utilities added in the first revision. They have no model or torch
dependency so that the new propositions can be unit-tested in isolation
(tests/test_revision_theory.py) and reused by the benchmark scripts.

Contents
--------
1. Query-side margin certificate (Proposition "Query-side certified radius").
   Replaces the anchor-based Lipschitz condition of the original Eq. (1), which
   Reviewer 3 correctly noted is unsatisfiable for k = 1 by construction.
2. Null atoms and abstention (Proposition "Null atom and hard-threshold limit").
3. Anti-atom versus deletion (Proposition "Anti-atom = deletion + localized null mass").
4. Outcome bucketing for CounterFact queries (explicit counting rules for Table 4).
5. Light text features used in the failure analysis.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Query-side certified radius
# ---------------------------------------------------------------------------

def certified_radii(
    canonical_queries: np.ndarray,
    keys: np.ndarray,
    theta: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Certified query-side radii for every stored record.

    Record i has unit key k_i and a canonical query embedding q0_i (its rewrite
    prompt). Let s_ij = <q0_i, k_j> and g_ij = s_ii - s_ij. For any query x with
    delta = phi_q(x) - q0_i,

        s_i(x) - s_j(x) = g_ij + <delta, k_i - k_j> >= g_ij - ||delta|| ||k_i - k_j||,

    so record i stays the exact top-1 for x whenever
        ||delta|| < r_i := min_{j != i} g_ij / ||k_i - k_j||   (r_i = 0 if i is not top-1 at q0_i).
    With unit keys, s_i(x) >= s_ii - ||delta||, so activation at threshold theta is
    also guaranteed whenever ||delta|| <= s_ii - theta. The combined radius is
        r_i^act = min(r_i, max(s_ii - theta, 0)).

    No Lipschitz premise is used; everything is computable at insertion time from
    stored keys and one canonical query.

    Args:
        canonical_queries: [N, m] unit query embeddings, row i is record i's canonical query.
        keys: [N, m] unit key embeddings, row i is record i's key.
        theta: optional activation threshold.

    Returns:
        dict with "top1" radii [N], "self_score" s_ii [N], "is_top1_at_canonical" [N],
        and "activation" radii [N] when theta is given.
    """
    q0 = np.asarray(canonical_queries, dtype=np.float64)
    k = np.asarray(keys, dtype=np.float64)
    if q0.shape != k.shape:
        raise ValueError("canonical_queries and keys must both be [N, m] with matching rows")
    n = k.shape[0]
    scores = q0 @ k.T                                   # [N, N], s_ij
    self_score = np.diag(scores).copy()                 # s_ii
    gaps = self_score[:, None] - scores                 # g_ij
    sq = np.sum(k * k, axis=1)
    dist2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (k @ k.T), 0.0)
    dist = np.sqrt(dist2)                               # ||k_i - k_j||

    off = ~np.eye(n, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dist > 1e-12, gaps / dist, np.where(gaps > 0, np.inf, 0.0))
    ratio = np.where(off, ratio, np.inf)
    is_top1 = np.all((gaps > 0) | ~off, axis=1)
    r_top1 = np.where(is_top1, np.min(ratio, axis=1), 0.0)
    r_top1 = np.maximum(r_top1, 0.0)

    out = {"top1": r_top1, "self_score": self_score, "is_top1_at_canonical": is_top1}
    if theta is not None:
        r_act = np.minimum(r_top1, np.maximum(self_score - theta, 0.0))
        out["activation"] = r_act
    return out


def query_displacements(queries: np.ndarray, canonical_queries: np.ndarray, record_index: Sequence[int]) -> np.ndarray:
    """||phi_q(x) - q0_{i(x)}|| for each query x attached to record i(x)."""
    q = np.asarray(queries, dtype=np.float64)
    q0 = np.asarray(canonical_queries, dtype=np.float64)[np.asarray(record_index)]
    return np.linalg.norm(q - q0, axis=1)


# ---------------------------------------------------------------------------
# 2. Null atoms
# ---------------------------------------------------------------------------

def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def null_atom_real_mass(scores: np.ndarray, theta: float, tau: float, null_mass: float = 1.0) -> np.ndarray:
    """Mass that a read with one added null atom places on the real atoms.

    The null atom omega_0 has value 0, reference mass `null_mass`, and kernel
    K_x(omega_0) = exp(theta / tau) for every query, i.e. it behaves like a key at
    constant similarity theta. With Z_x = sum_i exp(s_i / tau), the real-atom mass is
        M_x = Z_x / (Z_x + null_mass * exp(theta / tau)) = sigmoid(logZ_x - theta/tau - log null_mass),
    and the read becomes M_x * R(x; Omega). As tau -> 0 with N fixed, M_x -> 1 if
    max_i s_i > theta and M_x -> 0 if max_i s_i < theta: the activation threshold of
    Section 6.4 is the zero-temperature limit of a null atom.

    Args:
        scores: [..., N] raw similarities.
    """
    log_z = _logsumexp(np.asarray(scores, dtype=np.float64) / tau, axis=-1)
    logit = log_z - theta / tau - np.log(null_mass)
    return 1.0 / (1.0 + np.exp(-logit))


def null_atom_mass_bounds(max_score: np.ndarray, n_atoms: int, theta: float, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form bounds on M_x using exp(s_(1)/tau) <= Z_x <= N exp(s_(1)/tau) (unit null mass)."""
    gap = (np.asarray(max_score, dtype=np.float64) - theta) / tau
    lower = 1.0 / (1.0 + np.exp(-gap))
    upper = 1.0 / (1.0 + np.exp(-gap) / n_atoms)
    return lower, upper


# ---------------------------------------------------------------------------
# 3. Anti-atom versus deletion
# ---------------------------------------------------------------------------

def signed_read(scores: np.ndarray, masses: np.ndarray, values: np.ndarray, tau: float) -> np.ndarray:
    """Total-variation-normalized signed read: sum_i K_i m_i v_i / sum_i K_i |m_i|."""
    s = np.asarray(scores, dtype=np.float64) / tau
    kern = np.exp(s - np.max(s))
    w = kern * np.asarray(masses, dtype=np.float64)
    z = np.sum(np.abs(w))
    return (w @ np.asarray(values, dtype=np.float64)) / z


def anti_atom_vs_deletion(
    scores: np.ndarray, values: np.ndarray, target: int, tau: float, mass: float = 1.0
) -> Dict[str, np.ndarray]:
    """Compare (a) adding a matched anti-atom and (b) deleting record `target`.

    With a = K_x(target) * mass and Z_o the kernel mass of the other records,
        R_anti(x) = Z_o / (Z_o + 2a) * R_del(x),
    i.e. the anti-atom read equals the deletion read shrunk toward the zero
    vector by the factor Z_o / (Z_o + 2a). Equivalently, an anti-atom is a deletion
    plus a null atom of mass 2*mass placed at the deleted key.
    """
    s = np.asarray(scores, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    n = s.shape[0]
    masses = np.ones(n)
    masses[target] = mass
    # (a) anti-atom: duplicate key with negative mass
    s_anti = np.concatenate([s, s[target:target + 1]])
    m_anti = np.concatenate([masses, [-mass]])
    v_anti = np.concatenate([v, v[target:target + 1]], axis=0)
    r_anti = signed_read(s_anti, m_anti, v_anti, tau)
    # (b) deletion
    keep = np.arange(n) != target
    r_del = signed_read(s[keep], masses[keep], v[keep], tau)
    # predicted shrinkage factor
    shift = np.max(s) / tau
    kern = np.exp(s / tau - shift)
    a = kern[target] * mass
    z_o = np.sum(kern[keep])
    factor = z_o / (z_o + 2.0 * a)
    return {"anti": r_anti, "deletion": r_del, "shrinkage": factor}


# ---------------------------------------------------------------------------
# 4. Outcome buckets (explicit counting rules for Table 4)
# ---------------------------------------------------------------------------

POSITIVE_BUCKETS = (
    "activated_correct",      # top-1 is the query's own record AND top-1 similarity >= theta  -> counted in activated recall
    "retrieved_rejected",     # top-1 is the own record but similarity < theta (gate rejects a correct retrieval)
    "wrong_record_activated", # top-1 is another record AND similarity >= theta (would inject a wrong edit)
    "wrong_record_rejected",  # top-1 is another record AND similarity < theta (retrieval miss, gate silent)
)
NEGATIVE_BUCKETS = (
    "silent",                       # top-1 similarity < theta: no read
    "false_activation_own_record",  # top-1 is the edited record whose neighborhood this is, and fires ("wrong-edit retrieval")
    "false_activation_other_record",# fires on some other record
)


def bucket_queries(
    query_types: Sequence[str],
    expected_record: Sequence[int],
    top1_record: Sequence[int],
    top1_similarity: Sequence[float],
    theta: float,
) -> List[str]:
    """Assign each query to exactly one outcome bucket (see POSITIVE/NEGATIVE_BUCKETS)."""
    out = []
    for qt, exp, got, sim in zip(query_types, expected_record, top1_record, top1_similarity):
        fired = sim >= theta
        correct = int(got) == int(exp)
        if qt in ("rewrite", "paraphrase"):
            if correct and fired:
                out.append("activated_correct")
            elif correct:
                out.append("retrieved_rejected")
            elif fired:
                out.append("wrong_record_activated")
            else:
                out.append("wrong_record_rejected")
        else:
            if not fired:
                out.append("silent")
            elif correct:
                out.append("false_activation_own_record")
            else:
                out.append("false_activation_other_record")
    return out


def bucket_rates(buckets: Sequence[str], query_types: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Fraction of each bucket per query type."""
    buckets = np.asarray(buckets)
    qts = np.asarray(query_types)
    result: Dict[str, Dict[str, float]] = {}
    for qt in sorted(set(qts.tolist())):
        sel = buckets[qts == qt]
        names = NEGATIVE_BUCKETS if qt == "neighborhood" else POSITIVE_BUCKETS
        result[qt] = {name: float(np.mean(sel == name)) if len(sel) else 0.0 for name in names}
        result[qt]["count"] = int(len(sel))
    return result


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC; ties count one half."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_of = np.empty(len(allv))
    rank_of[order] = ranks
    r_pos = rank_of[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def youden_threshold(scores: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> Tuple[float, float]:
    """Threshold on `grid` maximizing TPR - FPR (first maximizer). Returns (theta, J)."""
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=bool)
    best_t, best_j = float(grid[0]), -np.inf
    for t in grid:
        pred = scores >= t
        tpr = np.mean(pred[labels]) if labels.any() else 0.0
        fpr = np.mean(pred[~labels]) if (~labels).any() else 0.0
        if tpr - fpr > best_j:
            best_t, best_j = float(t), float(tpr - fpr)
    return best_t, best_j


# ---------------------------------------------------------------------------
# 5. Text features for the failure analysis
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def split_prefix(text: str) -> Tuple[str, str]:
    """Split a CounterFact paraphrase into (prefix, core).

    CounterFact paraphrase prompts are typically a model-generated distractor
    sentence followed by the actual paraphrased prompt. We treat the final
    sentence-like segment as the core; everything before it is the prefix.
    """
    parts = [p for p in _SENT_SPLIT.split(text.strip()) if p]
    if len(parts) <= 1:
        return "", text.strip()
    return " ".join(parts[:-1]), parts[-1]


def tokens(text: str) -> set:
    return {t.lower() for t in _TOKEN.findall(text)}


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(len(ta | tb), 1)


def contains_ci(haystack: str, needle: str) -> bool:
    return needle.strip().lower() in haystack.lower()
