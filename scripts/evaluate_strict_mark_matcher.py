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
    
    same_fallback_landmark_deltas = []
    diff_fallback_landmark_deltas = []
    same_fallback_spatial_dists = []
    diff_fallback_spatial_dists = []
    
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
        "low_confidence": 0,
        "missing_anatomical_position": 0,
        "anchor_overlap_3": 0,
        "anchor_overlap_2": 0,
        "anchor_overlap_1": 0,
        "same_mesh_region": 0,
    }

    # Raw mark presence counters
    raw_debug_aggregate = {
        "total_marks": 0,
        "marks_with_anatomical_position": 0,
        "marks_with_barycentric_mode_2d": 0,
        "marks_with_nearest_landmark_fallback": 0,
        "marks_with_mesh_confidence_ge_070": 0,
        "marks_with_mesh_triangle_id": 0,
        "marks_with_nearest_landmark_indices": 0,
    }

    # Phase 2: Regional canonical coordinate collectors
    same_regional_uv_dists = []
    diff_regional_uv_dists = []
    same_regional_anchor_deltas = []
    diff_regional_anchor_deltas = []
    regional_available_count = 0
    regional_same_region_count = 0
    regional_same_subcell_count = 0
    regional_total_correspondences = 0
    same_regional_coord_qualities = []
    diff_regional_coord_qualities = []

    # Phase 2: Patch descriptor collectors
    same_patch_lbp_sims = []
    diff_patch_lbp_sims = []
    same_patch_combined_sims = []
    diff_patch_combined_sims = []
    same_patch_hu_dists = []
    diff_patch_hu_dists = []
    patch_available_count = 0
    patch_total_correspondences = 0

    # Phase 2B: Regional diagnostics
    regional_unavailable_reason_counts = {}
    regional_position_present_count = 0
    regional_different_canonical_count = 0
    regional_comparison_mode_counts = {}
    regional_related_compatible_count = 0
    regional_face_region_compatible_count = 0
    all_region_confidences = []
    all_coord_qualities = []
    canonical_region_distribution = {}

    # Phase 2B: Patch alternatives collectors
    same_patch_alt = {k: [] for k in ["v1_current", "hu_gradient", "hist_hu_gradient", "no_lbp"]}
    diff_patch_alt = {k: [] for k in ["v1_current", "hu_gradient", "hist_hu_gradient", "no_lbp"]}
    same_patch_comp = {k: [] for k in ["intensity", "hu", "gradient", "texture_energy", "edge_density_delta"]}
    diff_patch_comp = {k: [] for k in ["intensity", "hu", "gradient", "texture_energy", "edge_density_delta"]}

    # Phase 2B: Mark-type and region breakdown
    from collections import defaultdict
    marktype_stats = defaultdict(lambda: {"count": 0, "same_uv": [], "diff_uv": [], "same_patch": [], "diff_patch": [], "same_mq": [], "diff_mq": []})
    region_stats = defaultdict(lambda: {"count": 0, "same_uv": [], "diff_uv": [], "same_patch": [], "diff_patch": [], "same_mq": [], "diff_mq": []})

    # Phase 2B: Existing match quality collectors
    same_existing_mq = []
    diff_existing_mq = []
    same_cqs_v1 = []
    diff_cqs_v1 = []

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
        bary_count = strict_data.get("valid_barycentric_distance_count", 0)
        if bary_count > 0:
            bary_available_count += 1
            
        avg_fallback_delta = strict_data.get("avg_fallback_nearest_landmark_distance_delta")
        avg_fallback_spatial = strict_data.get("avg_normalized_spatial_distance")
        fallback_count = strict_data.get("fallback_mesh_telemetry_available_count", 0)

        # Aggregate comparison mode counts
        mode_counts = strict_data.get("barycentric_comparison_mode_counts", {})
        for mode_key in total_comparison_modes:
            total_comparison_modes[mode_key] += mode_counts.get(mode_key, 0)
        
        # Aggregate raw debug counts
        probe_bary_debug = r.get("barycentric_debug_probe", {})
        gallery_bary_debug = r.get("barycentric_debug_gallery", {})
        for debug_payload in [probe_bary_debug, gallery_bary_debug]:
            for key in raw_debug_aggregate.keys():
                raw_debug_aggregate[key] += debug_payload.get(key, 0)


        # Only include valid bary distances (same_triangle or same_anchor_set)
        if avg_bary_dist is not None:
            if label:
                same_bary_distances.append(avg_bary_dist)
            else:
                diff_bary_distances.append(avg_bary_dist)
                
        # Collect fallback metrics
        if avg_fallback_delta is not None:
            if label: same_fallback_landmark_deltas.append(avg_fallback_delta)
            else: diff_fallback_landmark_deltas.append(avg_fallback_delta)
            
        if avg_fallback_spatial is not None:
            if label: same_fallback_spatial_dists.append(avg_fallback_spatial)
            else: diff_fallback_spatial_dists.append(avg_fallback_spatial)

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

        # Phase 2: Regional + Patch per-correspondence telemetry
        # Iterate scoring correspondences to collect regional/patch metrics
        scoring_corrs = []
        # Check for scoring_correspondences inside strict_mark_data or at top level
        if "strict_mark_data" in r:
            # The correspondences come from the pipeline trace, check validation JSONL
            pass
        # Regional and patch data is embedded in accepted_correspondences_detail
        for c in r.get("accepted_correspondences_detail", []):
            regional_total_correspondences += 1
            patch_total_correspondences += 1

            # Regional metrics
            if c.get("regional_available"):
                regional_available_count += 1
                if c.get("regional_same_region"):
                    regional_same_region_count += 1
                if c.get("regional_same_subcell"):
                    regional_same_subcell_count += 1
                ruv = c.get("regional_uv_distance")
                if ruv is not None:
                    if label: same_regional_uv_dists.append(ruv)
                    else: diff_regional_uv_dists.append(ruv)
                rad = c.get("regional_anchor_distance_delta")
                if rad is not None:
                    if label: same_regional_anchor_deltas.append(rad)
                    else: diff_regional_anchor_deltas.append(rad)
                for qkey in ["regional_coordinate_quality_gallery", "regional_coordinate_quality_probe"]:
                    qv = c.get(qkey)
                    if qv is not None:
                        if label: same_regional_coord_qualities.append(qv)
                        else: diff_regional_coord_qualities.append(qv)

            # Phase 2B: regional diagnostics
            if c.get("regional_position_present_gallery") and c.get("regional_position_present_probe"):
                regional_position_present_count += 1
            reason = c.get("regional_unavailable_reason")
            if reason:
                regional_unavailable_reason_counts[reason] = regional_unavailable_reason_counts.get(reason, 0) + 1
            if not c.get("regional_same_region", True):
                regional_different_canonical_count += 1
            cmode = c.get("regional_comparison_mode", "unavailable")
            regional_comparison_mode_counts[cmode] = regional_comparison_mode_counts.get(cmode, 0) + 1
            if cmode == "related_region_compatible":
                regional_related_compatible_count += 1
            elif cmode == "face_region_compatible_cross_canonical":
                regional_face_region_compatible_count += 1
            for conf_key in ["regional_region_confidence_gallery", "regional_region_confidence_probe"]:
                cv = c.get(conf_key)
                if cv is not None: all_region_confidences.append(cv)
            for cq_key in ["regional_coordinate_quality_gallery", "regional_coordinate_quality_probe"]:
                cqv = c.get(cq_key)
                if cqv is not None: all_coord_qualities.append(cqv)
            for cr_key in ["regional_canonical_region_gallery", "regional_canonical_region_probe"]:
                crv = c.get(cr_key)
                if crv and crv != "unknown":
                    canonical_region_distribution[crv] = canonical_region_distribution.get(crv, 0) + 1

            # Patch metrics
            if c.get("patch_available"):
                patch_available_count += 1
                lbp = c.get("patch_lbp_similarity")
                if lbp is not None:
                    if label: same_patch_lbp_sims.append(lbp)
                    else: diff_patch_lbp_sims.append(lbp)
                comb = c.get("patch_combined_similarity")
                if comb is not None:
                    if label: same_patch_combined_sims.append(comb)
                    else: diff_patch_combined_sims.append(comb)
                hu = c.get("patch_hu_moment_distance")
                if hu is not None:
                    if label: same_patch_hu_dists.append(hu)
                    else: diff_patch_hu_dists.append(hu)

                # Phase 2B: patch alternatives
                for alt_key, field in [("v1_current", "patch_combined_similarity_v1_current"),
                                       ("hu_gradient", "patch_combined_similarity_hu_gradient"),
                                       ("hist_hu_gradient", "patch_combined_similarity_hist_hu_gradient"),
                                       ("no_lbp", "patch_combined_similarity_no_lbp")]:
                    v = c.get(field)
                    if v is not None:
                        if label: same_patch_alt[alt_key].append(v)
                        else: diff_patch_alt[alt_key].append(v)
                for comp_key, field in [("intensity", "patch_intensity_similarity"),
                                        ("hu", "patch_hu_similarity"),
                                        ("gradient", "patch_gradient_similarity"),
                                        ("texture_energy", "patch_texture_energy_similarity"),
                                        ("edge_density_delta", "patch_edge_density_delta")]:
                    v = c.get(field)
                    if v is not None:
                        if label: same_patch_comp[comp_key].append(v)
                        else: diff_patch_comp[comp_key].append(v)

            # Phase 2B: existing match quality and quality score v1
            emq = c.get("existing_match_quality")
            if emq is not None:
                if label: same_existing_mq.append(emq)
                else: diff_existing_mq.append(emq)
            cqsv1 = c.get("correspondence_quality_score_v1")
            if cqsv1 is not None:
                if label: same_cqs_v1.append(cqsv1)
                else: diff_cqs_v1.append(cqsv1)

            # Phase 2B: mark-type and region breakdown
            mtype = c.get("mark_type") or "unknown"
            cregion = c.get("regional_canonical_region_gallery") or "unknown"
            best_patch = c.get("patch_combined_similarity")
            mq_val = c.get("existing_match_quality")
            uv_val = c.get("regional_uv_distance")
            for stats_dict, key_val in [(marktype_stats, mtype), (region_stats, cregion)]:
                stats_dict[key_val]["count"] += 1
                if uv_val is not None:
                    if label: stats_dict[key_val]["same_uv"].append(uv_val)
                    else: stats_dict[key_val]["diff_uv"].append(uv_val)
                if best_patch is not None:
                    if label: stats_dict[key_val]["same_patch"].append(best_patch)
                    else: stats_dict[key_val]["diff_patch"].append(best_patch)
                if mq_val is not None:
                    if label: stats_dict[key_val]["same_mq"].append(mq_val)
                    else: stats_dict[key_val]["diff_mq"].append(mq_val)

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
        "barycentric_cost_enabled": False,
        "barycentric_telemetry_only": True,
        "fallback_mesh_telemetry_only": True,
        "does_not_affect_scoring": True,
        "same_person_avg_fallback_nearest_landmark_distance_delta": safe_avg(same_fallback_landmark_deltas),
        "different_person_avg_fallback_nearest_landmark_distance_delta": safe_avg(diff_fallback_landmark_deltas),
        "same_person_avg_fallback_normalized_spatial_distance": safe_avg(same_fallback_spatial_dists),
        "different_person_avg_fallback_normalized_spatial_distance": safe_avg(diff_fallback_spatial_dists),
        "same_person_avg_constellation_quality_score": safe_avg(same_constellation_quality),
        "different_person_avg_constellation_quality_score": safe_avg(diff_constellation_quality),
        "same_person_avg_region_diversity": safe_avg(same_region_diversity),
        "different_person_avg_region_diversity": safe_avg(diff_region_diversity),
        "constellation_quality_labels": constellation_labels,
        "barycentric_comparison_mode_counts": total_comparison_modes,
        "raw_mark_presence_totals": raw_debug_aggregate,
        # Phase 2: Regional canonical coordinate telemetry
        "regional_coordinate_version": "2.0.0-regional-canonical",
        "regional_available_rate": round(regional_available_count / regional_total_correspondences, 4) if regional_total_correspondences > 0 else 0.0,
        "regional_same_region_rate": round(regional_same_region_count / regional_total_correspondences, 4) if regional_total_correspondences > 0 else 0.0,
        "regional_same_subcell_rate": round(regional_same_subcell_count / regional_total_correspondences, 4) if regional_total_correspondences > 0 else 0.0,
        "same_person_avg_regional_uv_distance": safe_avg(same_regional_uv_dists),
        "different_person_avg_regional_uv_distance": safe_avg(diff_regional_uv_dists),
        "same_person_avg_regional_anchor_delta": safe_avg(same_regional_anchor_deltas),
        "different_person_avg_regional_anchor_delta": safe_avg(diff_regional_anchor_deltas),
        "same_person_avg_regional_coord_quality": safe_avg(same_regional_coord_qualities),
        "different_person_avg_regional_coord_quality": safe_avg(diff_regional_coord_qualities),
        "regional_cost_enabled": False,
        "regional_telemetry_only": True,
        # Phase 2: Patch descriptor telemetry
        "patch_descriptor_version": "1.0.0",
        "patch_available_rate": round(patch_available_count / patch_total_correspondences, 4) if patch_total_correspondences > 0 else 0.0,
        "same_person_avg_patch_lbp_similarity": safe_avg(same_patch_lbp_sims),
        "different_person_avg_patch_lbp_similarity": safe_avg(diff_patch_lbp_sims),
        "same_person_avg_patch_combined_similarity": safe_avg(same_patch_combined_sims),
        "different_person_avg_patch_combined_similarity": safe_avg(diff_patch_combined_sims),
        "same_person_avg_patch_hu_moment_distance": safe_avg(same_patch_hu_dists),
        "different_person_avg_patch_hu_moment_distance": safe_avg(diff_patch_hu_dists),
        "patch_cost_enabled": False,
        "patch_telemetry_only": True,
        # Phase 2B: Regional diagnostics
        "regional_position_present_count": regional_position_present_count,
        "regional_unavailable_reason_counts": regional_unavailable_reason_counts,
        "regional_different_canonical_region_count": regional_different_canonical_count,
        "regional_comparison_mode_counts": regional_comparison_mode_counts,
        "regional_related_region_compatible_count": regional_related_compatible_count,
        "regional_face_region_compatible_cross_canonical_count": regional_face_region_compatible_count,
        "region_confidence_min": round(min(all_region_confidences), 4) if all_region_confidences else None,
        "region_confidence_median": round(safe_median(all_region_confidences), 4) if all_region_confidences else None,
        "region_confidence_max": round(max(all_region_confidences), 4) if all_region_confidences else None,
        "coordinate_quality_min": round(min(all_coord_qualities), 4) if all_coord_qualities else None,
        "coordinate_quality_median": round(safe_median(all_coord_qualities), 4) if all_coord_qualities else None,
        "coordinate_quality_max": round(max(all_coord_qualities), 4) if all_coord_qualities else None,
        "canonical_region_distribution": canonical_region_distribution,
        # Phase 2B: Core separation — existing_match_quality and quality_score_v1
        "same_person_avg_existing_match_quality": safe_avg(same_existing_mq),
        "different_person_avg_existing_match_quality": safe_avg(diff_existing_mq),
        "same_person_avg_correspondence_quality_score_v1": safe_avg(same_cqs_v1),
        "different_person_avg_correspondence_quality_score_v1": safe_avg(diff_cqs_v1),
        # Phase 2B: Patch alternatives
        "patch_alternatives": {
            alt: {
                "same_person_avg": safe_avg(same_patch_alt[alt]),
                "different_person_avg": safe_avg(diff_patch_alt[alt]),
                "same_count": len(same_patch_alt[alt]),
                "diff_count": len(diff_patch_alt[alt]),
            } for alt in same_patch_alt
        },
        "patch_components": {
            comp: {
                "same_person_avg": safe_avg(same_patch_comp[comp]),
                "different_person_avg": safe_avg(diff_patch_comp[comp]),
                "same_count": len(same_patch_comp[comp]),
                "diff_count": len(diff_patch_comp[comp]),
            } for comp in same_patch_comp
        },
        # Phase 2B: Mark-type breakdown
        "mark_type_breakdown": {
            mt: {
                "count": s["count"],
                "same_avg_uv_distance": safe_avg(s["same_uv"]),
                "diff_avg_uv_distance": safe_avg(s["diff_uv"]),
                "same_avg_patch_similarity": safe_avg(s["same_patch"]),
                "diff_avg_patch_similarity": safe_avg(s["diff_patch"]),
                "same_avg_existing_match_quality": safe_avg(s["same_mq"]),
                "diff_avg_existing_match_quality": safe_avg(s["diff_mq"]),
            } for mt, s in sorted(marktype_stats.items(), key=lambda x: -x[1]["count"])
        },
        # Phase 2B: Region breakdown
        "canonical_region_breakdown": {
            rg: {
                "count": s["count"],
                "same_avg_uv_distance": safe_avg(s["same_uv"]),
                "diff_avg_uv_distance": safe_avg(s["diff_uv"]),
                "same_avg_patch_similarity": safe_avg(s["same_patch"]),
                "diff_avg_patch_similarity": safe_avg(s["diff_patch"]),
                "same_avg_existing_match_quality": safe_avg(s["same_mq"]),
                "diff_avg_existing_match_quality": safe_avg(s["diff_mq"]),
            } for rg, s in sorted(region_stats.items(), key=lambda x: -x[1]["count"])
        },
    }

    # ── Phase 3A: Signal purification, region-normalized UV, availability, hard impostors ──
    # TELEMETRY ONLY — does not affect scoring, Bayesian fusion, or production logic.
    from phase3a_analysis import (
        compute_signal_purification_filters,
        compute_region_normalized_uv,
        compute_availability_diagnostics,
        compute_hard_impostor_mining,
    )
    report["signal_purification_filters"] = compute_signal_purification_filters(results)
    report["region_normalized_uv"] = compute_region_normalized_uv(results)
    report["availability_diagnostics"] = compute_availability_diagnostics(results)
    report["hard_impostor_mining"] = compute_hard_impostor_mining(results)

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
    
    print("\n--- Constellation & Barycentric Phase 1 ---")
    print(f"  Valid Barycentric Availability Rate: {round(bary_available_count / bary_total_count * 100, 2) if bary_total_count > 0 else 0}%")
    print(f"  Same-person Avg Barycentric Distance: {safe_avg(same_bary_distances)}")
    print(f"  Diff-person Avg Barycentric Distance: {safe_avg(diff_bary_distances)}")
    print(f"  Same-person Avg Fallback Landmark Delta: {safe_avg(same_fallback_landmark_deltas)}")
    print(f"  Diff-person Avg Fallback Landmark Delta: {safe_avg(diff_fallback_landmark_deltas)}")
    print(f"  Same-person Avg Fallback Spatial Distance: {safe_avg(same_fallback_spatial_dists)}")
    print(f"  Diff-person Avg Fallback Spatial Distance: {safe_avg(diff_fallback_spatial_dists)}")
    print(f"  Same-person Avg Constellation Quality: {safe_avg(same_constellation_quality)}")
    print(f"  Diff-person Avg Constellation Quality: {safe_avg(diff_constellation_quality)}")
    print(f"  Constellation Labels: {constellation_labels}")
    print(f"  Barycentric Modes: {total_comparison_modes}")
    print(f"  Raw Mark Presence: {raw_debug_aggregate}")

    print("\n--- Phase 2: Regional Canonical Coordinates ---")
    print(f"  Regional Available Rate: {round(regional_available_count / regional_total_correspondences * 100, 2) if regional_total_correspondences > 0 else 0}%")
    print(f"  Same-Region Rate: {round(regional_same_region_count / regional_total_correspondences * 100, 2) if regional_total_correspondences > 0 else 0}%")
    print(f"  Same-Subcell Rate: {round(regional_same_subcell_count / regional_total_correspondences * 100, 2) if regional_total_correspondences > 0 else 0}%")
    print(f"  Same-person Avg UV Distance: {safe_avg(same_regional_uv_dists)}")
    print(f"  Diff-person Avg UV Distance: {safe_avg(diff_regional_uv_dists)}")
    print(f"  Same-person Avg Anchor Delta: {safe_avg(same_regional_anchor_deltas)}")
    print(f"  Diff-person Avg Anchor Delta: {safe_avg(diff_regional_anchor_deltas)}")
    print(f"  Same-person Avg Coord Quality: {safe_avg(same_regional_coord_qualities)}")
    print(f"  Diff-person Avg Coord Quality: {safe_avg(diff_regional_coord_qualities)}")

    print("\n--- Phase 2: Patch Descriptors ---")
    print(f"  Patch Available Rate: {round(patch_available_count / patch_total_correspondences * 100, 2) if patch_total_correspondences > 0 else 0}%")
    print(f"  Same-person Avg LBP Similarity: {safe_avg(same_patch_lbp_sims)}")
    print(f"  Diff-person Avg LBP Similarity: {safe_avg(diff_patch_lbp_sims)}")
    print(f"  Same-person Avg Combined Similarity: {safe_avg(same_patch_combined_sims)}")
    print(f"  Diff-person Avg Combined Similarity: {safe_avg(diff_patch_combined_sims)}")
    print(f"  Same-person Avg Hu Moment Distance: {safe_avg(same_patch_hu_dists)}")
    print(f"  Diff-person Avg Hu Moment Distance: {safe_avg(diff_patch_hu_dists)}")

    print("\n--- Phase 2B: Regional Diagnostics ---")
    print(f"  Position Present (both): {regional_position_present_count}")
    print(f"  Unavailable Reasons: {regional_unavailable_reason_counts}")
    print(f"  Comparison Modes: {regional_comparison_mode_counts}")
    print(f"  Related Region Compatible: {regional_related_compatible_count}")
    print(f"  Face Region Compatible (cross-canonical): {regional_face_region_compatible_count}")
    print(f"  Region Confidence: min={round(min(all_region_confidences), 4) if all_region_confidences else 'N/A'} med={round(safe_median(all_region_confidences), 4) if all_region_confidences else 'N/A'} max={round(max(all_region_confidences), 4) if all_region_confidences else 'N/A'}")
    print(f"  Coordinate Quality: min={round(min(all_coord_qualities), 4) if all_coord_qualities else 'N/A'} med={round(safe_median(all_coord_qualities), 4) if all_coord_qualities else 'N/A'} max={round(max(all_coord_qualities), 4) if all_coord_qualities else 'N/A'}")
    print(f"  Same-person Avg Existing Match Quality: {safe_avg(same_existing_mq)}")
    print(f"  Diff-person Avg Existing Match Quality: {safe_avg(diff_existing_mq)}")
    print(f"  Same-person Avg CQS v1: {safe_avg(same_cqs_v1)}")
    print(f"  Diff-person Avg CQS v1: {safe_avg(diff_cqs_v1)}")

    print("\n--- Phase 2B: Patch Alternatives ---")
    for alt in ["v1_current", "hu_gradient", "hist_hu_gradient", "no_lbp"]:
        print(f"  {alt}: same={safe_avg(same_patch_alt[alt])} diff={safe_avg(diff_patch_alt[alt])}")
    for comp in ["intensity", "hu", "gradient", "texture_energy", "edge_density_delta"]:
        print(f"  {comp}: same={safe_avg(same_patch_comp[comp])} diff={safe_avg(diff_patch_comp[comp])}")

    print("\n--- Phase 2B: Mark-Type Breakdown ---")
    for mt, s in sorted(marktype_stats.items(), key=lambda x: -x[1]["count"])[:10]:
        print(f"  {mt}: n={s['count']} same_uv={safe_avg(s['same_uv'])} diff_uv={safe_avg(s['diff_uv'])} same_patch={safe_avg(s['same_patch'])} diff_patch={safe_avg(s['diff_patch'])} same_mq={safe_avg(s['same_mq'])} diff_mq={safe_avg(s['diff_mq'])}")

    print("\n--- Phase 2B: Region Breakdown ---")
    for rg, s in sorted(region_stats.items(), key=lambda x: -x[1]["count"])[:10]:
        print(f"  {rg}: n={s['count']} same_uv={safe_avg(s['same_uv'])} diff_uv={safe_avg(s['diff_uv'])} same_patch={safe_avg(s['same_patch'])} diff_patch={safe_avg(s['diff_patch'])} same_mq={safe_avg(s['same_mq'])} diff_mq={safe_avg(s['diff_mq'])}")

    # ── Phase 3A Console Output ──
    print("\n--- Phase 3A: Signal Purification Filters ---")
    spf = report.get("signal_purification_filters", {})
    for fname in ["include_all_current", "suppress_light_scar", "distinctive_only",
                   "high_value_types_only", "remove_unknown_region", "useful_regions_only"]:
        fd = spf.get(fname, {})
        print(f"  {fname}: n={fd.get('correspondences_retained', 0)} "
              f"same_uv={fd.get('same_avg_uv_distance', 0)} "
              f"diff_uv={fd.get('diff_avg_uv_distance', 0)} "
              f"delta={fd.get('uv_separation_delta', 0)} "
              f"fp_risk={fd.get('fp_risk_proxy', 0)}")

    print("\n--- Phase 3A: Region-Normalized UV ---")
    rnuv = report.get("region_normalized_uv", {}).get("aggregated", {})
    for metric in ["raw", "zscore_by_region", "percentile_by_region", "quality_weighted", "region_calibrated"]:
        m = rnuv.get(metric, {})
        print(f"  {metric}: same={m.get('same_avg', 0)} diff={m.get('diff_avg', 0)} delta={m.get('delta', 0)}")

    print("\n--- Phase 3A: Availability Diagnostics ---")
    avail = report.get("availability_diagnostics", {})
    print(f"  Fully available (same_region): {avail.get('fully_available_same_region', 0)}")
    fc = avail.get("face_region_compatible", {})
    print(f"  Face region compatible: n={fc.get('count', 0)} delta={fc.get('separation_delta', 0)} preserves={fc.get('preserves_separation', False)}")
    rc = avail.get("related_region_compatible", {})
    print(f"  Related region compatible: n={rc.get('count', 0)} delta={rc.get('separation_delta', 0)} preserves={rc.get('preserves_separation', False)}")
    print(f"  Unavailable (no fallback): {avail.get('unavailable_no_fallback', {}).get('count', 0)}")
    print(f"  Potential rescue count: {avail.get('potential_rescue_count', 0)}")
    print(f"  Rescue preserves separation: {avail.get('rescue_preserves_separation', False)}")
    print(f"  Noise risk: {avail.get('noise_risk', 'unknown')}")

    print("\n--- Phase 3A: Hard Impostor Mining ---")
    him = report.get("hard_impostor_mining", {})
    print(f"  Total different-person pairs: {him.get('total_different_person_pairs', 0)}")
    print(f"  Light scar drove signal: {him.get('light_scar_drove_signal', False)}")
    print(f"  Unknown region drove signal: {him.get('unknown_region_drove_signal', False)}")
    print(f"  Recommended suppressions: {him.get('recommended_suppressions', [])}")
    top5 = him.get("top_25_hardest", [])[:5]
    for p in top5:
        print(f"    {p.get('pair_id')}: danger={p.get('danger_score')} lr={p.get('lr_marks')} top_type={p.get('top_mark_type')} top_region={p.get('top_region')}")

    print(f"{'=' * 60}\n")
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
