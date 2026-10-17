"""
src/tri_space_abstention.py

Tri-space routing with an explicit "none of the sources is relevant" outcome
(Reviewer 1, comments 1 and 4).

Mechanism. Each space s in {ctx, corpus, params} is augmented with one null atom
omega_0^s: value 0, unit reference mass, kernel exp(theta / tau) for every query. By the
null-atom proposition (src/roas_analysis.null_atom_real_mass) the space's read becomes
M_s(x) R(x; Omega_s) with M_s(x) = Z_s(x) / (Z_s(x) + exp(theta/tau)). The combined read is
    z(x) = sum_s lambda_s(x) M_s(x) R(x; Omega_s),
and the system answers iff A(x) := sum_s lambda_s(x) M_s(x) >= 1/2, otherwise it abstains
(falls back to the base model / returns "no stored answer"). As tau -> 0 this reduces to the
hard rule "the routed space's best key clears theta", i.e. the Section 6.4 gate.

Nothing is retrained: the router, adapter and expert bank are trained exactly as in
src/atom_space_tri_gating_engine.py. One shared theta is calibrated (Youden's J) on a held-out
split -- the 'held-out formulation' in-scope queries versus half of the out-of-scope queries --
and evaluated on the surface-matched in-scope queries versus the other half. The CounterFact
theta* = 0.55 is NOT reused: a threshold is a property of the key geometry, and the tri-space
keys pass through a trained adapter.

Out-of-scope (OOS) queries come in two kinds, both under the shared wrapper
"Determine the output for: ...":
  * near-miss: same relation templates as stored items but unseen subjects (the analogue of
    CounterFact neighborhood prompts);
  * far: unrelated requests.

Writes results/tri_space_abstention.json and results/details/tri_gating_probe_outputs/.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from atom_space_tri_gating_engine import (
    CONTEXT_EPHEMERAL,
    CORPUS_FACTS,
    PARAM_EXPERT_TASKS,
    TriEngineConfig,
    TriSpaceLanguageModel,
)
from roas_analysis import roc_auc

OUT_DIR = os.path.join("results", "details", "tri_gating_probe_outputs")
SUMMARY_PATH = os.path.join("results", "tri_space_abstention.json")
SEEDS = [40, 41, 42, 43, 44]
WRAP = "Determine the output for: {}."
THETA_GRID = np.linspace(0.0, 1.0, 201)

OOS_NEAR_MISS = [
    # corpus-style relations, unseen subjects
    "Cobalt is a ferromagnetic transition metal used in", "Kotlin compiles to bytecode for the Java virtual",
    "Uranus rotates on its side relative to the", "Collagen provides tensile strength to connective",
    "Nginx serves static files as a reverse", "Argon is a noble gas used in welding",
    "Callisto is heavily cratered and orbits", "Keratin is a structural protein found in",
    "MongoDB stores documents in a binary JSON", "Phobos is a small moon orbiting",
    "Zinc is an essential trace element for", "Bluetooth transmits data over short-range",
    # context-style relations, unseen subjects
    "Agent Zulu is stationed at outpost", "Cipher Red unlocks vault door", "Project Atlas is supervised by director",
    "Vessel Comet is commanded by officer", "Beacon Delta monitors seismic tremors near", "Courier Ghost delivers sealed envelope to",
    # parameter-style tasks that no expert implements
    "Compute the factorial of state 5", "Format identifier using kebab-case convention",
    "Classify payload sensitivity for weather report", "Apply logical conjunction to two affirmative statements",
]
OOS_FAR = [
    "What is a good recipe for vegetable lasagna", "Recommend a novel for a long train journey",
    "How tall is the average giraffe", "Translate good morning into Portuguese",
    "Who won the football match yesterday", "Suggest a name for a pet turtle",
    "How do I remove a coffee stain from cotton", "What time does the bakery open",
    "Write a haiku about autumn leaves", "Plan a weekend trip to the mountains",
    "Which shoes are best for running a marathon", "Explain how to tie a bowline knot",
    "How long should I boil an egg", "Is it going to rain this afternoon",
    "What is the plot of a famous opera", "Suggest a gift for a ten-year-old",
    "How many cups are in a litre", "Describe the rules of table tennis",
    "Which houseplants need little sunlight", "Draft a polite reminder to pay rent",
    "What colour goes well with olive green", "How do I sharpen a kitchen knife",
]


def split_half(items: List[str]):
    return items[0::2], items[1::2]


def train(cfg: TriEngineConfig, data: Dict, device: str) -> TriSpaceLanguageModel:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    model = TriSpaceLanguageModel(cfg, data["in_dim"], data["vocab_size"], len(PARAM_EXPERT_TASKS)).to(device)
    with torch.no_grad():
        norm_e = model.get_normalized_embeddings()
        model.omega_corpus.set_atoms(data["corp_k"], norm_e[data["corp_t"]])
        model.omega_params.set_atoms(data["param_k"], torch.zeros_like(norm_e[data["param_t"]]))
        model.param_experts.expert_values.copy_(norm_e[data["param_t"]])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    best, patience = float("inf"), 0
    for _ in range(cfg.epochs):
        model.train()
        norm_e = model.get_normalized_embeddings()
        model.omega_corpus.values = norm_e[data["corp_t"]]
        l_corp, _, _ = model(data["corp_q"])
        l_ctx, _, _ = model(data["ctx_q"], ctx_keys=data["ctx_k"], ctx_values=norm_e[data["ctx_t"]])
        l_par, _, _ = model(data["param_q"])
        loss = (F.cross_entropy(l_corp, data["corp_t"]) + F.cross_entropy(l_ctx, data["ctx_t"])
                + F.cross_entropy(l_par, data["param_t"]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < best - 1e-4:
            best, patience = loss.item(), 0
        else:
            patience += 1
        if patience >= cfg.patience:
            break
    model.eval()
    with torch.no_grad():
        model.omega_corpus.values = model.get_normalized_embeddings()[data["corp_t"]]
    return model


@torch.no_grad()
def space_scores(model: TriSpaceLanguageModel, feats: torch.Tensor, ctx_k: torch.Tensor):
    """Raw cosine scores per space and the router distribution."""
    q = model.phi(feats)
    s_ctx = q @ model.phi(ctx_k).T
    s_corp = q @ model.phi(model.omega_corpus.keys).T
    s_par = q @ model.phi(model.omega_params.keys).T
    lam = F.softmax(model.gate_net(feats), dim=-1)
    return {"ctx": s_ctx, "corpus": s_corp, "params": s_par}, lam


@torch.no_grad()
def null_atom_forward(model, feats, ctx_k, ctx_v, theta: float):
    """Logits and answer statistic A(x) with one null atom per space at similarity theta."""
    tau = model.cfg.read_temperature
    scores, lam = space_scores(model, feats, ctx_k)
    norm_e = model.get_normalized_embeddings()
    reads, masses = [], []
    for name in ("ctx", "corpus", "params"):
        s = scores[name] / tau
        mu = F.softmax(s, dim=-1)
        if name == "ctx":
            r = mu @ ctx_v
        elif name == "corpus":
            r = mu @ model.omega_corpus.values
        else:
            r = model.param_experts(mu)
        log_z = torch.logsumexp(s, dim=-1)
        m = torch.sigmoid(log_z - theta / tau)
        reads.append(r)
        masses.append(m)
    m = torch.stack(masses, dim=-1)                       # [B, 3]
    z = sum(lam[:, i:i + 1] * m[:, i:i + 1] * reads[i] for i in range(3))
    logits = (z @ norm_e.T) / tau
    answer_stat = (lam * m).sum(dim=-1)                   # A(x)
    routed = lam.argmax(dim=-1)
    best_routed = torch.stack([scores[n].max(dim=-1).values for n in ("ctx", "corpus", "params")], dim=-1)
    hard_stat = best_routed.gather(1, routed[:, None]).squeeze(1)  # tau -> 0 limit statistic
    return logits, answer_stat, hard_stat


def encode_all(backbone, device):
    def enc(texts):
        return backbone.encode(texts, convert_to_tensor=True, device=device).clone()

    targets = sorted({o for _, _, o in CORPUS_FACTS} | {o for _, _, o in CONTEXT_EPHEMERAL} | {o for _, _, o in PARAM_EXPERT_TASKS})
    t2i = {t: i for i, t in enumerate(targets)}
    d = {"vocab_size": len(t2i)}
    d["corp_k"] = enc([f"{s} {r} {o}." for s, r, o in CORPUS_FACTS])
    d["corp_q"] = enc([f"What {r} {s}?" for s, r, o in CORPUS_FACTS])
    d["corp_heldout"] = enc([f"Retrieve the stored completion for this statement: {s} {r} ___." for s, r, o in CORPUS_FACTS])
    d["corp_matched"] = enc([WRAP.format(f"{s} {r}") for s, r, o in CORPUS_FACTS])
    d["corp_t"] = torch.tensor([t2i[o] for _, _, o in CORPUS_FACTS], device=device)
    d["ctx_k"] = enc([f"Premise: {s} {r} {o}." for s, r, o in CONTEXT_EPHEMERAL])
    d["ctx_q"] = enc([f"According to premise, what {r} {s}?" for s, r, o in CONTEXT_EPHEMERAL])
    d["ctx_heldout"] = enc([f"Using only the temporary statement, identify what {s} {r}." for s, r, o in CONTEXT_EPHEMERAL])
    d["ctx_matched"] = enc([WRAP.format(f"{s} {r}") for s, r, o in CONTEXT_EPHEMERAL])
    d["ctx_t"] = torch.tensor([t2i[o] for _, _, o in CONTEXT_EPHEMERAL], device=device)
    d["param_k"] = enc([f"Execute task: {desc}." for desc, _, _ in PARAM_EXPERT_TASKS])
    d["param_q"] = enc([desc for desc, _, _ in PARAM_EXPERT_TASKS])
    d["param_heldout"] = enc([f"Return the expert result for this operation: {desc}." for desc, _, _ in PARAM_EXPERT_TASKS])
    d["param_matched"] = enc([WRAP.format(desc) for desc, _, _ in PARAM_EXPERT_TASKS])
    d["param_t"] = torch.tensor([t2i[o] for _, _, o in PARAM_EXPERT_TASKS], device=device)
    near_cal, near_test = split_half(OOS_NEAR_MISS)
    far_cal, far_test = split_half(OOS_FAR)
    d["oos_cal"] = enc([WRAP.format(t) for t in near_cal + far_cal])
    d["oos_test_near"] = enc([WRAP.format(t) for t in near_test])
    d["oos_test_far"] = enc([WRAP.format(t) for t in far_test])
    d["in_dim"] = d["corp_k"].shape[-1]
    return d


def evaluate_seed(cfg, data, device) -> Dict:
    model = train(cfg, data, device)
    norm_e = model.get_normalized_embeddings()
    ctx_v = norm_e[data["ctx_t"]]
    ck = data["ctx_k"]

    def stat(feats, theta, which="soft"):
        logits, a, h = null_atom_forward(model, feats, ck, ctx_v, theta)
        return logits, (a if which == "soft" else h)

    # ---- calibration: held-out in-scope formulations vs half of the OOS queries ----
    cal_in = torch.cat([data["ctx_heldout"], data["corp_heldout"], data["param_heldout"]])
    best = {}
    for which in ("soft", "hard"):
        best_t, best_j = None, -np.inf
        for t in THETA_GRID:
            _, s_in = stat(cal_in, float(t), which)
            _, s_out = stat(data["oos_cal"], float(t), which)
            thr = 0.5 if which == "soft" else float(t)
            tpr = (s_in >= thr).float().mean().item()
            fpr = (s_out >= thr).float().mean().item()
            if tpr - fpr > best_j:
                best_t, best_j = float(t), tpr - fpr
        best[which] = (best_t, best_j)

    # ---- evaluation: surface-matched in-scope vs the other half of OOS ----
    res = {"theta_star": best["soft"][0], "youden_j_calibration": best["soft"][1],
           "theta_star_hard_limit": best["hard"][0]}
    theta = best["soft"][0]
    fam = {"context": ("ctx_matched", "ctx_t"), "corpus": ("corp_matched", "corp_t"), "parameter": ("param_matched", "param_t")}
    in_stats, out_stats = [], []
    for name, (fk, tk) in fam.items():
        logits_null, a = stat(data[fk], theta)
        answered = a >= 0.5
        correct = logits_null.argmax(-1) == data[tk]
        logits_plain, _, _ = model(data[fk], ctx_keys=ck, ctx_values=ctx_v)
        res[name] = {
            "coverage": answered.float().mean().item(),
            "selective_accuracy": (correct & answered).float().sum().item() / max(answered.sum().item(), 1),
            "accuracy_abstain_counts_wrong": (correct & answered).float().mean().item(),
            "accuracy_without_null_atoms": (logits_plain.argmax(-1) == data[tk]).float().mean().item(),
        }
        in_stats.append(a)
    for name in ("oos_test_near", "oos_test_far"):
        _, a = stat(data[name], theta)
        res[name] = {"abstention_rate": (a < 0.5).float().mean().item(), "count": int(a.numel())}
        out_stats.append(a)
    s_in = torch.cat(in_stats).cpu().numpy()
    s_out = torch.cat(out_stats).cpu().numpy()
    res["auc_answer_statistic_test"] = roc_auc(np.concatenate([s_in, s_out]),
                                               np.concatenate([np.ones_like(s_in, bool), np.zeros_like(s_out, bool)]))
    # Without null atoms the system cannot abstain: report what it would output on OOS
    for name in ("oos_test_near", "oos_test_far"):
        _, lam, _ = model(data[name], ctx_keys=ck, ctx_values=ctx_v)
        res[name]["router_mass_max_without_null_atoms_mean"] = lam.max(dim=-1).values.mean().item()
    return res


def aggregate(per_seed: List[Dict]) -> Dict:
    def walk(prefix, objs):
        out = {}
        for k, v in objs[0].items():
            vals = [o[k] for o in objs]
            if isinstance(v, dict):
                out[k] = walk(prefix + k + ".", vals)
            elif isinstance(v, (int, float)):
                arr = np.array(vals, dtype=float)
                out[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}
        return out
    return walk("", per_seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = p.parse_args()

    base_cfg = TriEngineConfig()
    device = base_cfg.device
    from sentence_transformers import SentenceTransformer
    backbone = SentenceTransformer(base_cfg.encoder_name, device=device)
    data = encode_all(backbone, device)

    per_seed = []
    for seed in args.seeds:
        cfg = TriEngineConfig(seed=seed)
        r = evaluate_seed(cfg, data, device)
        r["seed"] = seed
        per_seed.append(r)
        print(f"seed {seed}: theta*={r['theta_star']:.3f} | coverage ctx/corp/param = "
              f"{r['context']['coverage']:.2f}/{r['corpus']['coverage']:.2f}/{r['parameter']['coverage']:.2f} | "
              f"OOS abstain near/far = {r['oos_test_near']['abstention_rate']:.2f}/{r['oos_test_far']['abstention_rate']:.2f}")

    summary = {
        "mechanism": "one null atom per space at similarity theta; answer iff sum_s lambda_s M_s >= 1/2",
        "calibration": "Youden's J: held-out-formulation in-scope queries vs half of OOS queries",
        "evaluation": "surface-matched in-scope queries vs the other half of OOS queries",
        "oos_counts": {"near_miss_total": len(OOS_NEAR_MISS), "far_total": len(OOS_FAR)},
        "seeds": args.seeds,
        "aggregate": aggregate([{k: v for k, v in r.items() if k != "seed"} for r in per_seed]),
        "per_seed": per_seed,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    for path in (SUMMARY_PATH, os.path.join(OUT_DIR, "tri_space_abstention.json")):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
