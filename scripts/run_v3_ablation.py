#!/usr/bin/env python3
"""Run the predefined V3 ablations on identity-disjoint pair features."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BACKEND = os.path.join(ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from evaluation.ablation import load_jsonl, run_ablation, save_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_jsonl")
    parser.add_argument("output_json")
    parser.add_argument("--train-split", default="calibration")
    parser.add_argument("--eval-split", default="eval")
    args = parser.parse_args()

    rows = load_jsonl(args.feature_jsonl)
    results = run_ablation(rows, train_split=args.train_split, eval_split=args.eval_split)
    save_results(args.output_json, results)
    for result in results:
        print(
            f"{result.variant:28s} "
            f"EER={result.eer:.6f} "
            f"TAR@1e-3={result.tar_at_far_1e3:.6f} "
            f"TAR@1e-4={result.tar_at_far_1e4:.6f} "
            f"Brier={result.brier:.6f}"
        )


if __name__ == "__main__":
    main()
