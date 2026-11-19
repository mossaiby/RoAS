"""
src/counterfact_margin_scaling.py

CounterFact store-size sweep for the rank-margin gate, as cited in Section 6.4 of the manuscript.

The implementation now lives in src/margin_gate_scaling.py, which also runs zsRE. This wrapper
keeps the path cited in the paper working: it runs that script with --dataset counterfact and
writes the same files as before (results/counterfact_margin_scaling.json and
results/details/counterfact_margin_scaling_outputs/).

Usage (from the repository root):
    python src/counterfact_margin_scaling.py --orderings 3
"""

import sys

import margin_gate_scaling

if __name__ == "__main__":
    if "--dataset" in sys.argv:
        raise SystemExit("this wrapper always runs CounterFact; use src/margin_gate_scaling.py for other datasets")
    sys.argv[1:1] = ["--dataset", "counterfact"]
    margin_gate_scaling.main()
