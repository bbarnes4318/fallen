"""Evaluate Strict Mark Matcher — Post-processing for strict V2 mark matcher validation.

Reads validation_results.jsonl and produces strict-matcher-specific reports including
display vs scoring correspondence separation, generic suppression stats, and cap analysis.

Usage:
  python scripts/evaluate_strict_mark_matcher.py \
    --input validation/results/run1/validation_results.jsonl \
    --output-dir validation/results/run1/strict_mark_analysis
"""

import argparse
import csv
import json
import math
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate strict V2 mark matcher results."
    )
    parser.add_argument("--input", required=True, help="Path to validation_results.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for output files")
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
    if v is None:
        return None
    try:
        fv = float(v)
        if math.isfinite(fv):
            return fv
    except (ValueError, TypeError):
        pass
    return None


def safe_median(lst):
    if not lst:
        return 0.0
    s = sorted(lst)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float((s[n // 2 - 1] + s[n // 2]) / 2.0)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    results = load_results(args.input)
    print(f"Loaded {len(results)} results from {args.input}")

    # Classify results
    tp, fp, tn, fn = 0, 0, 0, 0
    face_not_detected = 0
    false_positives = []
    false_negatives = []

    same_display_corr = []
    diff_display_corr = []
    same_scoring_corr = []
    diff_scoring_corr = []
    same_lr_marks = []
    diff_lr_marks = []
    all_lr_marks = []

    diff_lr_gt_100 = 0
    diff_lr_gt_1000 = 0
    total_generic_suppressed = 0
    total_distinctive_preserved = 0
    total_caps_applied = {}
    total_cluster_penalties = 0

    # Phase 1 constellation telemetry collectors
    same_bary_distances = []
    diff_bary_distances = []
    same_constellation_quality = []
    diff_constellation_quality = []
    same_region_diversity = []
    diff_region_diversity = []
    bary_available_count = 0
    bary_total_count = 0
    constellation_labels = {"NONE": 0, "WEAK": 0, "MODERATE": 0, "STRONG_REVIEW_SUPPORT": 0}
    # Comparison mode counters
    total_comparison_modes = {
        "same_triangle": 0,
        "same_anchor_set": 0,
        "different_triangle_fallback": 0,
        "unavailable": 0,
    }

    for r in results:
        if r.get("error") == "FACE_NOT_DETECTED":
            face_not_detected += 1
            continue
        if r.get("error"):
            continue

        label = r.get("label_same_person", False)
        score = r.get("fused_score", 0)
        veto = r.get("veto_triggered", False)
        override = r.get("veto_override_applied", False)
        predicted_match = score > 50.0 and not (veto and not override)

        if predicted_match and label:
            tp += 1
        elif predicted_match and not label:
            fp += 1
            false_positives.append(r.get("pair_id", "unknown"))
        elif not predicted_match and not label:
            tn += 1
        elif not predicted_match and label:
            fn += 1
            false_negatives.append(r.get("pair_id", "unknown"))

        lr_marks = safe_finite(r.get("lr_marks", 1.0)) or 1.0
        all_lr_marks.append(lr_marks)
        display_count = r.get("accepted_correspondences_count", 0)
        strict_data = r.get("strict_mark_data", {})
        scoring_count = strict_data.get("scoring_correspondences_count", display_count)

        if label:
            same_display_corr.append(display_count)
            same_scoring_corr.append(scoring_count)
            same_lr_marks.append(lr_marks)
        else:
            diff_display_corr.append(display_count)
            diff_scoring_corr.append(scoring_count)
            diff_lr_marks.append(lr_marks)
            if lr_marks > 100:
                diff_lr_gt_100 += 1
            if lr_marks > 1000:
                diff_lr_gt_1000 += 1

        total_generic_suppressed += strict_data.get("generic_suppressed", 0)
        total_distinctive_preserved += strict_data.get("distinctive_preserved", 0)

        for cap in strict_data.get("caps_applied", []):
            cap_key = cap.split(":")[0] if ":" in cap else cap
            total_caps_applied[cap_key] = total_caps_applied.get(cap_key, 0) + 1

        if strict_data.get("cluster_penalty_applied", False):
            total_cluster_penalties += 1

        # Phase 1 constellation telemetry
        bary_total_count += 1
        avg_bary_dist = strict_data.get("avg_barycentric_distance")
        bary_count = strict_data.get("barycentric_distance_count", 0)
        if bary_count > 0:
            bary_available_count += 1

        # Aggregate comparison mode counts
        mode_counts = strict_data.get("barycentric_comparison_mode_counts", {})
        for mode_key in total_comparison_modes:
            total_comparison_modes[mode_key] += mode_counts.get(mode_key, 0)

        # Only include valid bary distances (same_triangle or same_anchor_set)
        if avg_bary_dist is not None:
            if label:
                same_bary_distances.append(avg_bary_dist)
            else:
                diff_bary_distances.append(avg_bary_dist)

        constellation = r.get("constellation_telemetry", {})
        if constellation:
            cqs = constellation.get("constellation_quality_score")
            if cqs is not None:
                if label:
                    same_constellation_quality.append(cqs)
                else:
                    diff_constellation_quality.append(cqs)

            rd = constellation.get("region_diversity_count")
            if rd is not None:
                if label:
                    same_region_diversity.append(rd)
                else:
                    diff_region_diversity.append(rd)

            cql = constellation.get("constellation_quality_label", "NONE")
            if cql in constellation_labels:
                constellation_labels[cql] += 1

    # Build report
    def safe_avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    report = {
        "confusion_matrix": {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        },
        "face_not_detected_count": face_not_detected,
        "same_person_avg_display_correspondences": safe_avg(same_display_corr),
        "different_person_avg_display_correspondences": safe_avg(diff_display_corr),
        "same_person_avg_scoring_eligible_correspondences": safe_avg(same_scoring_corr),
        "different_person_avg_scoring_eligible_correspondences": safe_avg(diff_scoring_corr),
        "different_person_pairs_lr_marks_gt_100": diff_lr_gt_100,
        "different_person_pairs_lr_marks_gt_1000": diff_lr_gt_1000,
        "max_lr_marks": round(max(all_lr_marks), 4) if all_lr_marks else 0.0,
        "median_lr_marks": round(safe_median(all_lr_marks), 4),
        "generic_marks_suppressed_total": total_generic_suppressed,
        "distinctive_marks_preserved_total": total_distinctive_preserved,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "caps_applied_counts": total_caps_applied,
        "cluster_penalty_count": total_cluster_penalties,
        "total_pairs_evaluated": tp + fp + tn + fn,
        "same_person_pairs": len(same_lr_marks),
        "different_person_pairs": len(diff_lr_marks),
        "same_person_avg_lr_marks": safe_avg(same_lr_marks),
        "different_person_avg_lr_marks": safe_avg(diff_lr_marks),
        "same_person_median_lr_marks": round(safe_median(same_lr_marks), 4),
        "different_person_median_lr_marks": round(safe_median(diff_lr_marks), 4),
        # Phase 1 barycentric / constellation telemetry
        "barycentric_available_rate": round(bary_available_count / bary_total_count, 4) if bary_total_count > 0 else 0.0,
        "same_person_avg_barycentric_distance": safe_avg(same_bary_distances),
        "different_person_avg_barycentric_distance": safe_avg(diff_bary_distances),
        "same_person_avg_constellation_quality_score": safe_avg(same_constellation_quality),
        "different_person_avg_constellation_quality_score": safe_avg(diff_constellation_quality),
        "same_person_avg_region_diversity": safe_avg(same_region_diversity),
        "different_person_avg_region_diversity": safe_avg(diff_region_diversity),
        "constellation_quality_labels": constellation_labels,
        "barycentric_comparison_mode_counts": total_comparison_modes,
    }

    # Write JSON report
    report_path = os.path.join(args.output_dir, "strict_mark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Write CSV distribution
    csv_path = os.path.join(args.output_dir, "strict_mark_distribution.csv")
    dist_rows = []
    for r in results:
        if r.get("error"):
            continue
        strict_data = r.get("strict_mark_data", {})
        dist_rows.append({
            "pair_id": r.get("pair_id", ""),
            "label_same_person": r.get("label_same_person", ""),
            "lr_marks": safe_finite(r.get("lr_marks", 1.0)),
            "display_correspondences": r.get("accepted_correspondences_count", 0),
            "scoring_correspondences": strict_data.get("scoring_correspondences_count", 0),
            "generic_suppressed": strict_data.get("generic_suppressed", 0),
            "distinctive_preserved": strict_data.get("distinctive_preserved", 0),
            "lr_before_caps": safe_finite(strict_data.get("lr_before_caps")),
            "lr_after_all_caps": safe_finite(strict_data.get("lr_after_all_caps")),
            "cluster_penalty": strict_data.get("cluster_penalty_applied", False),
        })
    _write_dicts_csv(csv_path, dist_rows)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Strict Mark Matcher Evaluation")
    print(f"{'=' * 60}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  FACE_NOT_DETECTED: {face_not_detected}")
    print(f"  Same-person avg display corr: {safe_avg(same_display_corr)}")
    print(f"  Diff-person avg display corr: {safe_avg(diff_display_corr)}")
    print(f"  Same-person avg scoring corr: {safe_avg(same_scoring_corr)}")
    print(f"  Diff-person avg scoring corr: {safe_avg(diff_scoring_corr)}")
    print(f"  Diff-person lr_marks > 100: {diff_lr_gt_100}")
    print(f"  Diff-person lr_marks > 1000: {diff_lr_gt_1000}")
    print(f"  Max lr_marks: {max(all_lr_marks) if all_lr_marks else 0}")
    print(f"  Median lr_marks: {safe_median(all_lr_marks)}")
    print(f"  Generic suppressed: {total_generic_suppressed}")
    print(f"  Distinctive preserved: {total_distinctive_preserved}")
    print(f"  Caps applied: {total_caps_applied}")
    print(f"  Cluster penalties: {total_cluster_penalties}")
    print(f"  False positives: {false_positives}")
    print(f"  False negatives: {false_negatives}")
    print(f"  ── Phase 1 Constellation Telemetry ──")
    print(f"  Barycentric available rate: {round(bary_available_count / bary_total_count, 4) if bary_total_count > 0 else 0.0}")
    print(f"  Same-person avg bary distance: {safe_avg(same_bary_distances)}")
    print(f"  Diff-person avg bary distance: {safe_avg(diff_bary_distances)}")
    print(f"  Same-person avg constellation quality: {safe_avg(same_constellation_quality)}")
    print(f"  Diff-person avg constellation quality: {safe_avg(diff_constellation_quality)}")
    print(f"  Same-person avg region diversity: {safe_avg(same_region_diversity)}")
    print(f"  Diff-person avg region diversity: {safe_avg(diff_region_diversity)}")
    print(f"  Constellation labels: {constellation_labels}")
    print(f"  ── Comparison Mode Counts ──")
    for mode_key, mode_val in total_comparison_modes.items():
        print(f"    {mode_key}: {mode_val}")
    print(f"  Results: {args.output_dir}")
    print(f"{'=' * 60}")


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
