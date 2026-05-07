"""
Simulate Decision Policies — Compare alternative safety rules and decision logic.

This script reads validation_results.jsonl (produced by validate_full_pipeline.py)
and re-applies different decision policies to the same raw scores, computing
a confusion matrix for each policy.

Policies simulated:
  1. current_hard_safety_rule: structural_sim < 0.40 → fused_score = 0 (current production)
  2. soft_cap: structural_sim < 0.40 → cap fused_score at 50, do not zero
  3. human_review_flag: structural_sim < 0.40 → flag for human review, keep score
  4. separate_channels: report face and mark scores independently, match if EITHER exceeds threshold
  5. strict_mark_override: veto only overridden if 3+ marks AND lr_marks >= 100

Outputs:
  - policy_comparison.json
  - policy_confusion_matrices.csv
  - recovered_true_positives.csv
  - new_false_positives.csv

Usage:
  python scripts/simulate_decision_policies.py \\
    --input validation/results/run1/validation_results.jsonl \\
    --output-dir validation/results/run1/policies
"""

import argparse
import csv
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate alternative safety rules against validation results."
    )
    parser.add_argument("--input", required=True, help="Path to validation_results.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for output files")
    parser.add_argument("--match-threshold", type=float, default=50.0,
                        help="Score threshold above which a pair is predicted as match (default: 50.0)")
    return parser.parse_args()


def load_results(path):
    """Load JSONL results file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def apply_current_hard_safety_rule(r, threshold):
    """Current production logic: structural_sim < 0.40 → score = 0 unless mark override."""
    score = r.get("fused_score", 0)
    veto = r.get("veto_triggered", False)
    override = r.get("veto_override_applied", False)
    if veto and not override:
        score = 0.0
    return score > threshold


def apply_soft_cap(r, threshold):
    """If veto triggered, cap score at 50 instead of zeroing."""
    score = r.get("fused_score", 0)
    veto = r.get("veto_triggered", False)
    if veto:
        score = min(score, 50.0)
    return score > threshold


def apply_human_review_flag(r, threshold):
    """Keep the Bayesian score intact regardless of veto. Flag for review but don't auto-reject."""
    score = r.get("fused_score", 0)
    return score > threshold


def apply_separate_channels(r, _threshold):
    """Match if face is strong OR marks are strong, independently."""
    face_strong = r.get("structural_sim", 0) >= 0.60
    marks_strong = r.get("lr_marks", 1.0) > 10.0
    return face_strong or marks_strong


def apply_strict_mark_override(r, threshold):
    """Veto can ONLY be overridden if 3+ matched marks AND lr_marks >= 100."""
    score = r.get("fused_score", 0)
    veto = r.get("veto_triggered", False)
    if veto:
        mark_count = r.get("accepted_correspondences_count", 0)
        lr_marks = r.get("lr_marks", 1.0)
        if mark_count >= 3 and lr_marks is not None and lr_marks >= 100.0:
            pass  # Override: keep the score
        else:
            score = 0.0
    return score > threshold


POLICIES = {
    "current_hard_safety_rule": {
        "description": "Current production: structural_sim < 0.40 → score = 0",
        "func": apply_current_hard_safety_rule,
    },
    "soft_cap": {
        "description": "Cap score at 50 instead of zeroing when veto triggers",
        "func": apply_soft_cap,
    },
    "human_review_flag": {
        "description": "Keep Bayesian score intact, flag for human review",
        "func": apply_human_review_flag,
    },
    "separate_channels": {
        "description": "Match if face >= 0.60 OR marks LR > 10, independently",
        "func": apply_separate_channels,
    },
    "strict_mark_override": {
        "description": "Veto overridden only if 3+ marks AND lr_marks >= 100",
        "func": apply_strict_mark_override,
    },
}


def evaluate_policy(results, policy_func, threshold):
    """Apply a policy function to all results and return confusion matrix + case lists."""
    tp, fp, tn, fn = 0, 0, 0, 0
    false_positives = []
    false_negatives = []

    for r in results:
        if r.get("error"):
            continue
        label = r["label_same_person"]
        predicted = policy_func(r, threshold)

        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
            false_positives.append(r)
        elif not predicted and not label:
            tn += 1
        else:
            fn += 1
            false_negatives.append(r)

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(tp / (tp + fp), 6) if (tp + fp) > 0 else 0,
        "recall": round(tp / (tp + fn), 6) if (tp + fn) > 0 else 0,
        "FAR": round(fp / (fp + tn), 6) if (fp + tn) > 0 else 0,
        "FRR": round(fn / (fn + tp), 6) if (fn + tp) > 0 else 0,
    }, false_positives, false_negatives


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    results = load_results(args.input)
    print(f"Loaded {len(results)} results from {args.input}")

    # Compute the baseline (current production) first
    baseline_func = POLICIES["current_hard_safety_rule"]["func"]
    baseline_matrix, baseline_fn_list, _ = evaluate_policy(results, baseline_func, args.match_threshold)
    baseline_tp_ids = set()
    baseline_fp_ids = set()
    for r in results:
        if r.get("error"):
            continue
        label = r["label_same_person"]
        predicted = baseline_func(r, args.match_threshold)
        if predicted and label:
            baseline_tp_ids.add(r["pair_id"])
        elif predicted and not label:
            baseline_fp_ids.add(r["pair_id"])

    comparison = {}
    all_matrices = []
    all_recovered_tp = []
    all_new_fp = []

    for policy_name, policy_info in POLICIES.items():
        print(f"Simulating: {policy_name}")
        matrix, fp_list, fn_list = evaluate_policy(results, policy_info["func"], args.match_threshold)

        # Determine which pairs changed state relative to baseline
        recovered_tp = []  # Was FN in baseline, now TP
        new_fp = []  # Was TN in baseline, now FP

        for r in results:
            if r.get("error"):
                continue
            label = r["label_same_person"]
            baseline_pred = baseline_func(r, args.match_threshold)
            policy_pred = policy_info["func"](r, args.match_threshold)

            if not baseline_pred and policy_pred and label:
                recovered_tp.append({
                    "pair_id": r["pair_id"],
                    "policy": policy_name,
                    "original_state": "FN",
                    "new_state": "TP",
                    "structural_sim": r.get("structural_sim"),
                    "fused_score": r.get("fused_score"),
                    "lr_marks": r.get("lr_marks"),
                })
            elif not baseline_pred and policy_pred and not label:
                new_fp.append({
                    "pair_id": r["pair_id"],
                    "policy": policy_name,
                    "original_state": "TN",
                    "new_state": "FP",
                    "structural_sim": r.get("structural_sim"),
                    "fused_score": r.get("fused_score"),
                    "lr_marks": r.get("lr_marks"),
                })

        comparison[policy_name] = {
            "description": policy_info["description"],
            **matrix,
            "recovered_true_positives": len(recovered_tp),
            "new_false_positives": len(new_fp),
        }

        all_matrices.append({"policy": policy_name, **matrix})
        all_recovered_tp.extend(recovered_tp)
        all_new_fp.extend(new_fp)

    # Write outputs
    with open(os.path.join(args.output_dir, "policy_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    with open(os.path.join(args.output_dir, "policy_confusion_matrices.csv"), "w", newline="") as f:
        fieldnames = ["policy", "TP", "FP", "TN", "FN", "precision", "recall", "FAR", "FRR"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_matrices)

    _write_dicts_csv(os.path.join(args.output_dir, "recovered_true_positives.csv"), all_recovered_tp)
    _write_dicts_csv(os.path.join(args.output_dir, "new_false_positives.csv"), all_new_fp)

    print(f"\n{'='*60}")
    print("Policy simulation complete.")
    for name, m in comparison.items():
        print(f"  {name}: TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']} "
              f"| recovered_TP={m['recovered_true_positives']} new_FP={m['new_false_positives']}")
    print(f"  Results: {args.output_dir}")
    print(f"{'='*60}")


def _write_dicts_csv(path, rows):
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("(no results)\n")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
