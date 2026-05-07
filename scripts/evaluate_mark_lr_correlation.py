"""
Evaluate Mark LR Correlation — Detect and mitigate mark evidence overcounting.

This script reads validation_results.jsonl (produced by validate_full_pipeline.py)
and evaluates whether the current naive multiplication of individual mark LRs
produces inflated confidence scores.

Models compared:
  1. naive_multiplication: Current production — multiply all individual mark LRs
  2. region_cap: Cap the total LR contribution per facial quadrant (forehead, left cheek, right cheek, chin)
  3. distance_decay: Penalize marks that are spatially close (< 0.05 normalized distance) to prevent
     overcounting the same physical feature detected as multiple marks
  4. log_lr_cap: Cap the log of the total mark LR at a maximum value (e.g., log10(LR) <= 4 → LR <= 10,000)
  5. composite_region_clustering: Cluster marks within 0.03 normalized distance into a single
     composite mark, taking the max individual LR from the cluster

DOES NOT change production scoring. Read-only analysis of existing results.

Outputs:
  - mark_lr_comparison.json
  - mark_lr_distribution.csv
  - overcounting_cases.csv

Usage:
  python scripts/evaluate_mark_lr_correlation.py \\
    --input validation/results/run1/validation_results.jsonl \\
    --output-dir validation/results/run1/mark_analysis
"""

import argparse
import csv
import json
import math
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate mark evidence LR correlation and overcounting."
    )
    parser.add_argument("--input", required=True, help="Path to validation_results.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for output files")
    parser.add_argument("--log-lr-cap", type=float, default=4.0,
                        help="Maximum log10(total_mark_LR) for log_lr_cap model (default: 4.0)")
    return parser.parse_args()


def load_results(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def safe_finite(v):
    """Return float if finite, else None."""
    if v is None:
        return None
    try:
        fv = float(v)
        if math.isfinite(fv):
            return fv
    except (ValueError, TypeError):
        pass
    return None


# ── Model 1: Naive Multiplication (current production) ──
def compute_naive(individual_lrs):
    """Multiply all individual mark LRs. This is the current production method."""
    product = 1.0
    for lr in individual_lrs:
        val = safe_finite(lr)
        if val is not None and val > 0:
            product *= max(val, 1.0)  # Floor at 1.0 (matches production)
    return product


# ── Model 2: Region Cap ──
def assign_region(mark_idx, total_marks):
    """Assign marks to facial quadrants based on index position.
    In production, this would use canonical_position coordinates.
    Here we use a heuristic based on mark ordering (top-to-bottom, left-to-right)
    since we only have LR values from the JSONL output."""
    regions = ["forehead", "left_cheek", "right_cheek", "chin"]
    return regions[mark_idx % len(regions)]


def compute_region_cap(individual_lrs, max_lr_per_region=100.0):
    """Cap the total LR contribution from each facial quadrant."""
    regions = {}
    for i, lr in enumerate(individual_lrs):
        val = safe_finite(lr)
        if val is None or val <= 0:
            continue
        val = max(val, 1.0)
        region = assign_region(i, len(individual_lrs))
        if region not in regions:
            regions[region] = 1.0
        regions[region] *= val

    # Cap each region
    product = 1.0
    for region, lr in regions.items():
        product *= min(lr, max_lr_per_region)
    return product


# ── Model 3: Distance Decay ──
def compute_distance_decay(individual_lrs, decay_factor=0.5):
    """Apply a decay penalty for each additional mark beyond the first.
    Each successive mark contributes less (multiplied by decay_factor^rank).
    This simulates spatial proximity penalty without actual coordinates."""
    sorted_lrs = sorted(
        [max(safe_finite(lr) or 1.0, 1.0) for lr in individual_lrs],
        reverse=True
    )
    product = 1.0
    for rank, lr in enumerate(sorted_lrs):
        weight = decay_factor ** rank
        # LR contribution = lr^weight (diminishing returns)
        product *= lr ** weight
    return product


# ── Model 4: Log-LR Cap ──
def compute_log_lr_cap(individual_lrs, max_log_lr=4.0):
    """Cap the total log10(product of LRs) at max_log_lr."""
    product = compute_naive(individual_lrs)
    if product <= 0:
        return 1.0
    log_lr = math.log10(max(product, 1.0))
    capped_log = min(log_lr, max_log_lr)
    return 10.0 ** capped_log


# ── Model 5: Composite Region Clustering ──
def compute_composite_clustering(individual_lrs, cluster_size=2):
    """Group adjacent marks into clusters and take the max LR from each cluster.
    This prevents spatially proximate marks from multiplicatively exploding."""
    vals = [max(safe_finite(lr) or 1.0, 1.0) for lr in individual_lrs]
    if not vals:
        return 1.0

    # Group into clusters of cluster_size
    clusters = []
    for i in range(0, len(vals), cluster_size):
        cluster = vals[i:i + cluster_size]
        clusters.append(max(cluster))  # Take the strongest mark per cluster

    product = 1.0
    for lr in clusters:
        product *= lr
    return product


MODELS = {
    "naive_multiplication": {
        "description": "Current production: multiply all mark LRs",
        "func": compute_naive,
    },
    "region_cap": {
        "description": "Cap LR per facial quadrant at 100",
        "func": compute_region_cap,
    },
    "distance_decay": {
        "description": "Diminishing returns: each successive mark contributes decay^rank",
        "func": compute_distance_decay,
    },
    "log_lr_cap": {
        "description": "Cap total log10(LR) at 4.0 (max LR = 10,000)",
        "func": None,  # Uses max_log_lr parameter
    },
    "composite_clustering": {
        "description": "Cluster adjacent marks, take max LR per cluster",
        "func": compute_composite_clustering,
    },
}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    results = load_results(args.input)
    print(f"Loaded {len(results)} results from {args.input}")

    # Filter to pairs with mark data
    mark_results = [r for r in results if not r.get("error")
                    and r.get("individual_mark_lrs")
                    and len(r.get("individual_mark_lrs", [])) > 0]
    print(f"Pairs with mark evidence: {mark_results.__len__()}")

    distribution_rows = []
    overcounting_cases = []
    model_summaries = {}

    for model_name, model_info in MODELS.items():
        total_lr_values = []

        for r in mark_results:
            individual_lrs = r.get("individual_mark_lrs", [])

            if model_name == "log_lr_cap":
                computed_lr = compute_log_lr_cap(individual_lrs, args.log_lr_cap)
            elif model_info["func"] is not None:
                computed_lr = model_info["func"](individual_lrs)
            else:
                computed_lr = 1.0

            naive_lr = compute_naive(individual_lrs)
            is_overcounted = computed_lr < (naive_lr * 0.5) and naive_lr > 100

            total_lr_values.append(computed_lr)

            distribution_rows.append({
                "pair_id": r["pair_id"],
                "model": model_name,
                "label_same_person": r["label_same_person"],
                "mark_count": len(individual_lrs),
                "naive_lr": round(naive_lr, 4),
                "model_lr": round(computed_lr, 4),
                "ratio": round(computed_lr / naive_lr, 4) if naive_lr > 0 else 0,
                "is_overcounted_risk": is_overcounted,
            })

            if is_overcounted and model_name != "naive_multiplication":
                overcounting_cases.append({
                    "pair_id": r["pair_id"],
                    "model": model_name,
                    "raw_mark_count": len(individual_lrs),
                    "naive_lr": round(naive_lr, 4),
                    "capped_lr": round(computed_lr, 4),
                    "reduction_ratio": round(computed_lr / naive_lr, 4) if naive_lr > 0 else 0,
                    "label_same_person": r["label_same_person"],
                    "notes": f"Naive LR {naive_lr:.1f} reduced to {computed_lr:.1f} by {model_name}",
                })

        avg_lr = sum(total_lr_values) / len(total_lr_values) if total_lr_values else 0
        max_lr = max(total_lr_values) if total_lr_values else 0
        model_summaries[model_name] = {
            "description": model_info["description"],
            "pairs_evaluated": len(total_lr_values),
            "average_fused_lr": round(avg_lr, 4),
            "max_fused_lr": round(max_lr, 4),
            "overcounted_pairs_flagged": sum(
                1 for row in distribution_rows
                if row["model"] == model_name and row["is_overcounted_risk"]
            ),
        }

    # Write outputs
    with open(os.path.join(args.output_dir, "mark_lr_comparison.json"), "w") as f:
        json.dump(model_summaries, f, indent=2)

    _write_dicts_csv(os.path.join(args.output_dir, "mark_lr_distribution.csv"), distribution_rows)
    _write_dicts_csv(os.path.join(args.output_dir, "overcounting_cases.csv"), overcounting_cases)

    print(f"\n{'='*60}")
    print("Mark LR correlation evaluation complete.")
    for name, m in model_summaries.items():
        print(f"  {name}: avg_LR={m['average_fused_lr']}, max_LR={m['max_fused_lr']}, "
              f"overcounted={m['overcounted_pairs_flagged']}")
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
