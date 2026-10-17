"""Replace \\pending{LABEL} placeholders in the manuscript and response letter with values
read from the revision artifacts.

Usage (from the repository root, after running the revision scripts):
    python scripts/fill_pending.py            # rewrite paper/main.tex and paper/response_to_reviewers.tex
    python scripts/fill_pending.py --check    # only report which labels can / cannot be filled

Labels whose artifact is missing are left untouched and listed, so the script can be run
repeatedly as artifacts become available. Review the diff before resubmitting: the prose
around several numbers states an interpretation that should be checked against the values.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "paper" / "main.tex", ROOT / "paper" / "response_to_reviewers.tex"]


def load(rel):
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def num(x, d=3):
    return f"{x:.{d}f}"


def get(obj, *path):
    for p in path:
        obj = obj[p]
    return obj


def build():
    v = {}
    fa = load("results/counterfact_failure_analysis.json")
    if fa:
        sk = fa["submitted_keys"]
        prof = fa["paraphrase_failure_profile"]
        for th, key in (("60", "at_0.60"), ("55", "at_theta_star")):
            v[f"FA.para.wra.{th}"] = pct(sk[key]["paraphrase"]["wrong_record_activated"])
            v[f"FA.para.miss.{th}"] = pct(sk[key]["paraphrase"]["wrong_record_rejected"])
        for tag, name in (("rr", "retrieved_rejected"), ("ac", "activated_correct")):
            v[f"FA.{tag}.median_s1"] = num(prof[name]["top1_similarity"]["median"])
            v[f"FA.{tag}.has_prefix"] = pct(prof[name]["has_prefix_rate"])
            v[f"FA.{tag}.jaccard"] = num(prof[name]["token_jaccard_with_key"]["median"], 2)
        v["FA.rr.median_margin"] = num(prof["retrieved_rejected"]["rank_margin_s1_minus_s2"]["median"])
        v["FA.para.prefix_all"] = pct(fa["paraphrase_prefix_rate_all"])
        ps = fa["prefix_stripped_paraphrases"]
        v["FA.core.activated"] = pct(ps["at_theta_star"]["activated_correct"])
        v["FA.core.recovered"] = pct(ps["recovered_fraction_of_retrieved_rejected"])
        mg = fa["rank_margin_gate"]["at_gamma_star"]
        v["FA.margin.para"] = pct(mg["paraphrase"]["activated_correct"])
        v["FA.margin.neigh"] = pct(1 - mg["neighborhood"]["silent"])
        ka = fa["key_ablation_prompt_only"]
        v["FA.key.own_pt"] = num(ka["mean_own_similarity_paraphrase"]["prompt_target"])
        v["FA.key.own_po"] = num(ka["mean_own_similarity_paraphrase"]["prompt_only"])
        v["FA.key.theta"] = num(ka["theta_star"], 3)
        v["FA.key.para"] = pct(ka["at_theta_star"]["paraphrase"]["activated_correct"])
        v["FA.key.neigh"] = pct(1 - ka["at_theta_star"]["neighborhood"]["silent"])
        ex = fa["neighborhood_false_activations"]["additional_between_0.60_and_theta_star"]
        v["FA.extra.count"] = str(ex["count"])
        if ex["count"]:
            v["FA.extra.own"] = pct(ex["retrieved_own_record_rate"])
            v["FA.extra.samerel"] = pct(ex["retrieved_same_relation_rate"])
        v["FA.extra.base"] = pct(fa["neighborhood_false_activations"]["baseline_same_relation_rate_random_pair"])
        qc = fa["query_side_certificate"]
        v["FA.cert.top1canon"] = pct(qc["records_top1_at_canonical_query"])
        v["FA.cert.r_top1"] = num(qc["top1_radius"]["median"])
        v["FA.cert.r_act"] = num(qc["activation_radius"]["median"])
        cov = qc["coverage"]
        v["FA.cert.disp"] = num(cov["as_given/paraphrase"]["displacement"]["median"])
        v["FA.cert.para_top1"] = pct(cov["as_given/paraphrase"]["certified_top1_rate"])
        v["FA.cert.para_act"] = pct(cov["as_given/paraphrase"]["certified_activation_rate"])
        v["FA.cert.core_top1"] = pct(cov["prefix_stripped/paraphrase"]["certified_top1_rate"])
        v["FA.cert.core_act"] = pct(cov["prefix_stripped/paraphrase"]["certified_activation_rate"])

    ab = load("results/tri_space_abstention.json")
    if ab:
        ag = ab["aggregate"]

        def ms(node):
            return f"{100 * node['mean']:.1f}$\\pm${100 * node['std']:.1f}"

        for tag, fam in (("ctx", "context"), ("corp", "corpus"), ("par", "parameter")):
            v[f"AB.{tag}.cov"] = ms(ag[fam]["coverage"])
            v[f"AB.{tag}.sel"] = ms(ag[fam]["selective_accuracy"])
            v[f"AB.{tag}.plain"] = ms(ag[fam]["accuracy_without_null_atoms"])
        v["AB.oos.near"] = ms(ag["oos_test_near"]["abstention_rate"]) + "\\%"
        v["AB.oos.far"] = ms(ag["oos_test_far"]["abstention_rate"]) + "\\%"
        v["AB.cov.mean"] = pct(mean(ag[f]["coverage"]["mean"] for f in ("context", "corpus", "parameter"))) + "\\%"
        v["AB.theta"] = f"{ag['theta_star']['mean']:.3f}$\\pm${ag['theta_star']['std']:.3f}"
        v["AB.auc"] = f"{ag['auc_answer_statistic_test']['mean']:.3f}"

    for tag, rel in (("U", "results/counterfact_unified_gpt2.json"), ("XL", "results/counterfact_unified_gpt2-xl.json")):
        u = load(rel)
        if not u:
            continue
        for sysname in ("base", "read"):
            s = u["systems"][sysname]
            v[f"{tag}.{sysname}.eff"] = pct(s["efficacy_score"])
            v[f"{tag}.{sysname}.effarg"] = pct(s["canonical_argmax_accuracy"])
            v[f"{tag}.{sysname}.para"] = pct(s["paraphrase_score"])
            v[f"{tag}.{sysname}.pararg"] = pct(s["paraphrase_argmax_accuracy"])
            v[f"{tag}.{sysname}.neigh"] = pct(s["neighborhood_score"])
        v[f"{tag}.read.ctrl"] = pct(u["systems"]["read"]["control_text_activation_rate"], 0)

    r1c = load("results/counterfact_rome_results.json")
    if r1c and "paraphrase_argmax_accuracy" in r1c:
        v["R1C.pararg"] = pct(r1c["paraphrase_argmax_accuracy"])

    un = load("results/unlearning_results.json")
    if un and "anti_atom_vs_deletion" in un:
        rows = un["anti_atom_vs_deletion"]
        v["DEL.p_sens"] = pct(mean(r["deletion"]["p_sensitive"] for r in rows), 2) + "\\%"
        v["DEL.other"] = str(sum(r["deletion"]["argmax_is_other_stored_fact"] for r in rows))
        v["DEL.pmax"] = num(mean(r["deletion"]["max_probability"] for r in rows), 2)
        v["DEL.H"] = num(mean(r["deletion"]["normalized_entropy"] for r in rows), 2)
        v["DEL.spec"] = pct(mean(r["deletion"]["background_specificity"] for r in rows)) + "\\%"
        v["ANTI.norm"] = f"{mean(r['anti_atom']['read_norm'] for r in rows):.1e}"
        v["ANTI.H"] = num(mean(r["anti_atom"]["normalized_entropy"] for r in rows), 2)
    return v


PATTERN = re.compile(r"\\pending\{([^{}]+)\}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    values = build()
    for path in TARGETS:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        labels = sorted({m.group(1) for m in PATTERN.finditer(text)} - {"..."})
        filled = [l for l in labels if l.replace("\\_", "_") in values]
        missing = [l for l in labels if l not in filled]
        print(f"{path.relative_to(ROOT)}: {len(filled)} fillable, {len(missing)} still pending")
        for l in missing:
            print(f"   pending: {l}")
        if args.check:
            continue

        def sub(m):
            key = m.group(1).replace("\\_", "_")
            return values[key] if key in values else m.group(0)

        path.write_text(PATTERN.sub(sub, text), encoding="utf-8")


if __name__ == "__main__":
    main()
