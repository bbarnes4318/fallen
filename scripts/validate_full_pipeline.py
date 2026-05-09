"""
Validate Full Pipeline — Production-equivalent biometric verification validation.

This script calls the exact same core backend logic used by /verify/fuse:
  - fetch_image_from_url (GCS image loading)
  - apply_clahe (CLAHE preprocessing)
  - align_face_crop (MediaPipe alignment to 256x256 canonical crop)
  - extract_ensemble_embeddings (ArcFace + Facenet512)
  - compute_ensemble_similarity (60/40 weighted cosine)
  - _run_mark_evidence_pipeline (mark detection + matching + LR)
  - score_to_lr_ensemble (calibrated LR from ensemble score)
  - evaluate_mark_veto_override (mark override eligibility)
  - Bayesian posterior fusion (LR_total -> posterior)
  - Veto protocol (structural_sim < 0.40)

It does NOT:
  - Write to the database
  - Create payment jobs
  - Call the frontend
  - Mutate any production data
  - Change any scoring logic

Usage:
  # Dry-run (validate manifest only):
  python scripts/validate_full_pipeline.py --manifest validation/validation_pairs.csv --output-dir validation/results/run1 --dry-run

  # Full run (requires backend container environment with models loaded):
  python scripts/validate_full_pipeline.py --manifest validation/validation_pairs.csv --output-dir validation/results/run1
"""

import argparse
import csv
import json
import os
import sys
import time
import math
import traceback

# Ensure the backend package is importable
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run production-equivalent verification pipeline against a validation manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to validation_pairs.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to write results")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest only, do not process images")
    parser.add_argument("--shard-index", type=int, default=0, help="Index of the shard to process (0-based)")
    parser.add_argument("--shard-count", type=int, default=1, help="Total number of shards")
    return parser.parse_args()


REQUIRED_COLUMNS = [
    "pair_id", "image1_url_or_gcs_path", "image2_url_or_gcs_path",
    "label_same_person", "category",
]


def validate_manifest(manifest_path):
    """Load and validate the CSV manifest structure."""
    if not os.path.exists(manifest_path):
        print(f"ERROR: Manifest not found: {manifest_path}")
        sys.exit(1)

    pairs = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            print(f"ERROR: Manifest missing required columns: {missing}")
            sys.exit(1)
        for i, row in enumerate(reader):
            row["_row_num"] = i + 2  # 1-indexed, skip header
            pairs.append(row)

    print(f"Manifest validated: {len(pairs)} pairs, columns={header}")
    return pairs


def run_single_pair(pair, pipeline_modules):
    """Run the full production-equivalent pipeline for a single image pair.

    This replicates the exact sequence inside verify_pipeline() from main.py
    lines ~2260-2590, without any DB writes or HTTP response wrapping.
    """
    fetch_image_from_url = pipeline_modules["fetch_image_from_url"]
    apply_clahe = pipeline_modules["apply_clahe"]
    align_face_crop = pipeline_modules["align_face_crop"]
    extract_ensemble_embeddings = pipeline_modules["extract_ensemble_embeddings"]
    compute_ensemble_similarity = pipeline_modules["compute_ensemble_similarity"]
    score_to_lr_ensemble = pipeline_modules["score_to_lr_ensemble"]
    evaluate_mark_veto_override = pipeline_modules["evaluate_mark_veto_override"]
    _run_mark_evidence_pipeline = pipeline_modules["_run_mark_evidence_pipeline"]
    estimate_age = pipeline_modules["estimate_age"]
    cross_spectral_normalize = pipeline_modules["cross_spectral_normalize"]
    compute_image_hash = pipeline_modules["compute_image_hash"]
    calculate_cosine_similarity = pipeline_modules["calculate_cosine_similarity"]
    finite_or_none = pipeline_modules["finite_or_none"]
    CALIBRATION = pipeline_modules["CALIBRATION"]
    TIER4_CALIBRATION = pipeline_modules["TIER4_CALIBRATION"]

    t0 = time.time()

    # 1. Fetch images from GCS (same as verify_pipeline line 2272-2274)
    gallery_img, gallery_file_hash = fetch_image_from_url(pair["image2_url_or_gcs_path"])
    probe_img, probe_file_hash = fetch_image_from_url(pair["image1_url_or_gcs_path"])

    # 2. CLAHE preprocessing (same as verify_pipeline line 2337-2338)
    gallery_clahe = apply_clahe(gallery_img)
    probe_clahe = apply_clahe(probe_img)

    # 3. Face alignment & crop to 256x256 (same as verify_pipeline line 2342-2343)
    gallery_aligned, gallery_landmarks = align_face_crop(gallery_clahe)
    probe_aligned, probe_landmarks = align_face_crop(probe_clahe)

    if gallery_landmarks is None or probe_landmarks is None:
        return {
            "error": "FACE_NOT_DETECTED",
            "timing_ms": int((time.time() - t0) * 1000),
        }

    # 3.5 Temporal Invariance (same as verify_pipeline line 2357-2362)
    gallery_age = estimate_age(gallery_aligned)
    probe_age = estimate_age(probe_aligned)
    temporal_delta = abs(probe_age - gallery_age)
    gallery_aligned, probe_aligned, spectral_correction = cross_spectral_normalize(
        gallery_aligned, probe_aligned
    )

    # 4. TIER 1: Structural Identity — Neural Ensemble (same as line 2366-2369)
    ensemble_gallery = extract_ensemble_embeddings(gallery_aligned)
    ensemble_probe = extract_ensemble_embeddings(probe_aligned)
    structural_sim, arcface_sim, secondary_sim = compute_ensemble_similarity(
        ensemble_gallery, ensemble_probe
    )

    # 5. Veto Protocol (same as line 2422-2423)
    veto_triggered = structural_sim < 0.40

    # 6. Mark Evidence Pipeline (same as line 2428-2471)
    mark_payload = _run_mark_evidence_pipeline(
        probe_img=probe_img,
        gallery_img=gallery_img,
        probe_file_hash=probe_file_hash,
        gallery_file_hash=gallery_file_hash,
        mode="production",
        target_size=1024,
    )
    raw_probe_marks = mark_payload.get("raw_probe_marks", [])
    raw_gallery_marks = mark_payload.get("raw_gallery_marks", [])
    accepted_correspondences = mark_payload.get("accepted_correspondences", [])
    lr_marks = mark_payload.get("lr_marks") if mark_payload.get("lr_marks") is not None else 1.0
    individual_mark_lrs = mark_payload.get("individual_mark_lrs", [])

    mark_result = {
        "matched": len(accepted_correspondences),
        "total_gallery": len(raw_gallery_marks),
        "total_probe": len(raw_probe_marks),
        "lr_marks": lr_marks,
        "mark_lrs": individual_mark_lrs,
    }

    # 7. Bayesian Evidence Fusion (same as line 2539-2555)
    lr_ensemble = score_to_lr_ensemble(structural_sim, temporal_delta=temporal_delta)
    lr_total = lr_ensemble * lr_marks
    PRIOR = 0.5
    posterior = (PRIOR * lr_total) / ((PRIOR * lr_total) + (1.0 - PRIOR))
    fused_score = posterior * 100.0
    bayesian_fused_score = fused_score

    # 8. Mark Override Protocol (same as line 2563-2585)
    mark_override_eval = evaluate_mark_veto_override(mark_result, lr_marks)
    veto_reason = None
    veto_override_applied = False

    if veto_triggered:
        if mark_override_eval["eligible"]:
            veto_reason = "ARCFACE_VETO_MARK_OVERRIDE"
            veto_override_applied = True
        else:
            fused_score = 0.0
            veto_reason = "ARCFACE_VETO"

    # 9. Determine conclusion (same as line 2586-2591)
    if veto_triggered and not veto_override_applied:
        conclusion = "Inconclusive — Limited by Face-Model Threshold"
    elif veto_triggered and veto_override_applied:
        conclusion = "Supports Common Source — Face-Model Veto Overridden by Mark Correspondence"
    elif fused_score > 90.0:
        conclusion = "Strongly Supports Common Source"
    elif fused_score > 75.0:
        conclusion = "Supports Common Source"
    else:
        conclusion = "Inconclusive — Insufficient Evidence"

    elapsed_ms = int((time.time() - t0) * 1000)

    # Build evidence trace
    raw_probe = mark_payload.get("raw_probe_marks", [])
    raw_gallery = mark_payload.get("raw_gallery_marks", [])
    accepted = mark_payload.get("accepted_correspondences", [])
    rejected = mark_payload.get("rejected_correspondences", [])
    
    import collections
    def _extract_barycentric_debug(marks):
        total = len(marks)
        has_anat = 0
        mode_2d = 0
        fallback = 0
        conf_ge_070 = 0
        has_triangle = 0
        has_indices = 0
        mesh_regions = []
        bary_modes = []
        confs = []

        for m in marks:
            anat = m.get("anatomical_position")
            if anat:
                has_anat += 1
                mode = anat.get("barycentric_mode")
                bary_modes.append(mode)
                if mode == "2d_mesh_approximation":
                    mode_2d += 1
                if mode == "nearest_landmark_fallback":
                    fallback += 1
                
                conf = anat.get("mesh_confidence", 0.0)
                confs.append(conf)
                if conf >= 0.70:
                    conf_ge_070 += 1
                
                if anat.get("mesh_triangle_id") is not None:
                    has_triangle += 1
                if anat.get("nearest_landmark_indices"):
                    has_indices += 1
                
                mesh_regions.append(anat.get("mesh_region", "unknown"))

        s_confs = sorted(confs)
        conf_min = s_confs[0] if s_confs else 0.0
        conf_max = s_confs[-1] if s_confs else 0.0
        conf_med = s_confs[len(s_confs)//2] if s_confs else 0.0

        return {
            "total_marks": total,
            "marks_with_anatomical_position": has_anat,
            "marks_with_barycentric_mode_2d": mode_2d,
            "marks_with_nearest_landmark_fallback": fallback,
            "marks_with_mesh_confidence_ge_070": conf_ge_070,
            "marks_with_mesh_triangle_id": has_triangle,
            "marks_with_nearest_landmark_indices": has_indices,
            "mesh_region_distribution": dict(collections.Counter(mesh_regions)),
            "barycentric_mode_distribution": dict(collections.Counter(bary_modes)),
            "mesh_confidence_min": conf_min,
            "mesh_confidence_median": conf_med,
            "mesh_confidence_max": conf_max,
        }

    probe_summary = [
        {
            "mark_type": m.get("mark_type"),
            "channel": m.get("channel"),
            "face_region": m.get("face_region"),
            "centroid": m.get("centroid"),
            "area": m.get("area"),
            "confidence": m.get("confidence"),
            "salience": m.get("salience")
        } for m in raw_probe
    ]
    
    gallery_summary = [
        {
            "mark_type": m.get("mark_type"),
            "channel": m.get("channel"),
            "face_region": m.get("face_region"),
            "centroid": m.get("centroid"),
            "area": m.get("area"),
            "confidence": m.get("confidence"),
            "salience": m.get("salience")
        } for m in raw_gallery
    ]
    
    probe_bary_debug = _extract_barycentric_debug(raw_probe)
    gallery_bary_debug = _extract_barycentric_debug(raw_gallery)
    
    accepted_detail = [
        {
            "gallery_idx": c.get("gallery_idx"),
            "probe_idx": c.get("probe_idx"),
            "mark_type": c.get("mark_type") or c.get("type"),
            "channel": c.get("channel"),
            "face_region": c.get("face_region") or c.get("region"),
            "gallery_centroid": c.get("gallery_centroid"),
            "probe_centroid": c.get("probe_centroid"),
            "position_distance": c.get("position_distance"),
            "area_ratio": c.get("area_ratio"),
            "match_quality": c.get("match_quality"),
            "match_cost": c.get("match_cost") or c.get("cost"),
            "lr": finite_or_none(c.get("lr", 1.0))
        } for c in accepted
    ]
    
    rejected_sorted = sorted(rejected, key=lambda x: x.get("cost", 9999))[:25]
    rejected_summary = [
        {
            "reason": c.get("reason"),
            "gallery_idx": c.get("gallery_idx"),
            "probe_idx": c.get("probe_idx"),
            "cost": c.get("cost"),
            "distance": c.get("distance"),
            "type_mismatch": c.get("type_mismatch"),
            "region_mismatch": c.get("region_mismatch")
        } for c in rejected_sorted
    ]
    
    distances = [c.get("position_distance", 0) for c in accepted_detail if c.get("position_distance") is not None]
    regions = [c.get("face_region") for c in accepted_detail if c.get("face_region") is not None]
    types = [c.get("mark_type") for c in accepted_detail if c.get("mark_type") is not None]
    channels = [c.get("channel") for c in accepted_detail if c.get("channel") is not None]
    qualities = [c.get("match_quality", 0) for c in accepted_detail if c.get("match_quality") is not None]
    lrs = [c.get("lr", 1.0) for c in accepted_detail]
    
    import collections
    region_counts = collections.Counter(regions)
    type_counts = collections.Counter(types)
    channel_counts = collections.Counter(channels)
    
    def safe_median(lst):
        if not lst: return 0.0
        s = sorted(lst)
        n = len(s)
        if n % 2 == 1:
            return float(s[n//2])
        else:
            return float((s[n//2 - 1] + s[n//2]) / 2.0)
            
    distinctive_types_set = {"dark_mole", "mole", "light_scar", "scar", "linear_scar", "structural_crater", "depression_scar"}
    distinctive_types_found = [t for t in types if t in distinctive_types_set]
    generic_types_found = [t for t in types if t not in distinctive_types_set]
    distinctive_regions_found = [r for r, t in zip(regions, types) if t in distinctive_types_set]
    generic_regions_found = [r for r, t in zip(regions, types) if t not in distinctive_types_set]

    evidence_aggregate = {
        "accepted_correspondences_count": len(accepted),
        "distinct_face_regions_count": len(set(regions)),
        "distinct_mark_types_count": len(set(types)),
        "distinct_channels_count": len(set(channels)),
        "distinctive_mark_count": len(distinctive_types_found),
        "generic_mark_count": len(generic_types_found),
        "distinctive_mark_types": list(set(distinctive_types_found)),
        "generic_mark_types": list(set(generic_types_found)),
        "distinctive_region_count": len(set(distinctive_regions_found)),
        "generic_region_count": len(set(generic_regions_found)),
        "generic_only_match": len(distinctive_types_found) == 0 and len(generic_types_found) > 0,
        "largest_single_mark_type_correspondence_count": max(type_counts.values()) if type_counts else 0,
        "average_position_distance": round(sum(distances) / len(distances), 4) if distances else 0.0,
        "median_position_distance": round(safe_median(distances), 4),
        "max_position_distance": round(max(distances), 4) if distances else 0.0,
        "average_match_quality": round(sum(qualities) / len(qualities), 4) if qualities else 0.0,
        "top_5_individual_mark_lrs": [finite_or_none(x) for x in sorted(lrs, reverse=True)[:5]],
        "top_5_mark_types": [t[0] for t in type_counts.most_common(5)],
        "same_region_cluster_count": sum(1 for c in region_counts.values() if c > 1),
        "largest_single_region_correspondence_count": max(region_counts.values()) if region_counts else 0,
        "largest_single_channel_correspondence_count": max(channel_counts.values()) if channel_counts else 0,
    }

    result = {
        "calibration_loaded": CALIBRATION is not None,
        "calibration_source": "gcs_or_local" if CALIBRATION else "missing",
        "calibration_status": "loaded" if CALIBRATION else "failed",
        "calibration_keys": list(CALIBRATION.keys()) if CALIBRATION else [],
        "ensemble_threshold_count": len(CALIBRATION.get("ensemble", {}).get("thresholds", {})) if CALIBRATION else 0,
        "arcface_threshold_count": len(CALIBRATION.get("arcface", {}).get("thresholds", {})) if CALIBRATION else 0,
        "structural_sim": round(structural_sim, 6),
        "arcface_sim": round(arcface_sim, 6),
        "facenet_sim": round(secondary_sim, 6),
        "fused_score": round(fused_score, 4),
        "bayesian_fused_score": round(bayesian_fused_score, 4),
        "lr_ensemble": finite_or_none(lr_ensemble),
        "lr_marks": finite_or_none(lr_marks),
        "lr_total": finite_or_none(lr_total),
        "raw_probe_marks_count": len(raw_probe_marks),
        "raw_gallery_marks_count": len(raw_gallery_marks),
        "accepted_correspondences_count": len(accepted_correspondences),
        "individual_mark_lrs": [finite_or_none(x) for x in individual_mark_lrs],
        "veto_triggered": veto_triggered,
        "veto_reason": veto_reason,
        "veto_override_applied": veto_override_applied,
        "conclusion": conclusion,
        "temporal_delta": round(temporal_delta, 1),
        "spectral_correction": spectral_correction,
        "error": None,
        "timing_ms": elapsed_ms,
        "accepted_correspondences_detail": accepted_detail,
        "raw_probe_marks_summary": probe_summary,
        "raw_gallery_marks_summary": gallery_summary,
        "rejected_correspondences_summary": rejected_summary,
        "evidence_aggregate": evidence_aggregate,
        "validation_gates": mark_payload.get("validation_gates", {}),
        "barycentric_debug_probe": probe_bary_debug,
        "barycentric_debug_gallery": gallery_bary_debug,
    }

    # ── Strict V2 telemetry (only when USE_STRICT_MARK_MATCHER_V2=true) ──
    if mark_payload.get("strict_mode"):
        result["strict_mark_data"] = {
            "display_correspondences_count": mark_payload.get("displayed_marks_count", 0),
            "scoring_correspondences_count": mark_payload.get("scoring_eligible_marks_count", 0),
            "suppressed_count": len(mark_payload.get("suppressed_correspondences", [])),
            "generic_suppressed": mark_payload.get("generic_marks_suppressed_count", 0),
            "distinctive_preserved": mark_payload.get("distinctive_marks_preserved_count", 0),
            "lr_before_caps": finite_or_none(mark_payload.get("lr_before_caps")),
            "lr_after_all_caps": finite_or_none(mark_payload.get("lr_after_all_caps")),
            "caps_applied": mark_payload.get("caps_applied", []),
            "cluster_penalty_applied": mark_payload.get("cluster_penalty_applied", False),
        }

        # Phase 1 constellation telemetry (DOES NOT AFFECT SCORING)
        constellation_telem = mark_payload.get("constellation_telemetry")
        if constellation_telem is not None:
            result["constellation_telemetry"] = constellation_telem

        # Collect per-correspondence barycentric comparison mode stats
        bary_dists = []
        fallback_landmark_deltas = []
        fallback_spatial_dists = []
        comparison_mode_counts = {
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
        for c in mark_payload.get("scoring_correspondences", []):
            mode = c.get("barycentric_comparison_mode", "unavailable")
            if mode in comparison_mode_counts:
                comparison_mode_counts[mode] += 1
            
            # Check overlaps
            overlap = c.get("mesh_anchor_overlap_count", 0)
            if overlap == 3: comparison_mode_counts["anchor_overlap_3"] += 1
            elif overlap == 2: comparison_mode_counts["anchor_overlap_2"] += 1
            elif overlap == 1: comparison_mode_counts["anchor_overlap_1"] += 1
            
            # Check mesh region match
            reg_g = c.get("mesh_region_gallery")
            reg_p = c.get("mesh_region_probe")
            if reg_g and reg_p and reg_g != "unknown" and reg_g == reg_p:
                comparison_mode_counts["same_mesh_region"] += 1

            # Only include valid comparisons (same_triangle or same_anchor_set)
            if c.get("valid_barycentric_distance_available", False):
                bd = c.get("barycentric_distance")
                if bd is not None:
                    bary_dists.append(bd)
                    
            if c.get("fallback_mesh_telemetry_available", False):
                nld = c.get("nearest_landmark_distance_delta")
                nsd = c.get("normalized_spatial_distance")
                if nld is not None: fallback_landmark_deltas.append(nld)
                if nsd is not None: fallback_spatial_dists.append(nsd)

        result["strict_mark_data"]["barycentric_comparison_mode_counts"] = comparison_mode_counts
        
        result["strict_mark_data"]["valid_barycentric_distance_count"] = len(bary_dists)
        result["strict_mark_data"]["avg_barycentric_distance"] = round(sum(bary_dists) / len(bary_dists), 6) if bary_dists else None

        result["strict_mark_data"]["fallback_mesh_telemetry_available_count"] = len(fallback_landmark_deltas)
        result["strict_mark_data"]["avg_fallback_nearest_landmark_distance_delta"] = round(sum(fallback_landmark_deltas) / len(fallback_landmark_deltas), 6) if fallback_landmark_deltas else None
        result["strict_mark_data"]["avg_normalized_spatial_distance"] = round(sum(fallback_spatial_dists) / len(fallback_spatial_dists), 6) if fallback_spatial_dists else None
        
        result["strict_mark_data"]["barycentric_cost_enabled"] = False
        result["strict_mark_data"]["barycentric_telemetry_only"] = True
        result["strict_mark_data"]["fallback_mesh_telemetry_only"] = True
        result["strict_mark_data"]["does_not_affect_scoring"] = True


    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pairs = validate_manifest(args.manifest)
    
    total_pairs = len(pairs)
    if args.shard_count > 1:
        pairs = [p for i, p in enumerate(pairs) if i % args.shard_count == args.shard_index]
        print(f"Total manifest pairs: {total_pairs}")
        print(f"Shard index: {args.shard_index}")
        print(f"Shard count: {args.shard_count}")
        print(f"Shard pair count: {len(pairs)}")

    if args.dry_run:
        print("DRY-RUN complete. Manifest is valid. No images were processed.")
        # Write a summary for CI verification
        summary = {
            "dry_run": True,
            "manifest_path": args.manifest,
            "pair_count": len(pairs),
            "categories": list(set(p.get("category", "") for p in pairs)),
        }
        with open(os.path.join(args.output_dir, "dry_run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return

    # ── Import production pipeline modules ──
    # These are the exact same functions used by the /verify/fuse endpoint.
    # Importing here so dry-run works without the full backend environment.
    try:
        from pipeline_core import (
            fetch_image_from_url,
            apply_clahe,
            align_face_crop,
            extract_ensemble_embeddings,
            compute_ensemble_similarity,
            score_to_lr_ensemble,
            evaluate_mark_veto_override,
            _run_mark_evidence_pipeline,
            estimate_age,
            cross_spectral_normalize,
            compute_image_hash,
            calculate_cosine_similarity,
            finite_or_none,
            CALIBRATION,
            TIER4_CALIBRATION,
        )
    except ImportError as e:
        print(f"FATAL: Cannot import backend pipeline modules. "
              f"This script must run inside the backend container environment.\n{e}")
        sys.exit(1)

    pipeline_modules = {
        "fetch_image_from_url": fetch_image_from_url,
        "apply_clahe": apply_clahe,
        "align_face_crop": align_face_crop,
        "extract_ensemble_embeddings": extract_ensemble_embeddings,
        "compute_ensemble_similarity": compute_ensemble_similarity,
        "score_to_lr_ensemble": score_to_lr_ensemble,
        "evaluate_mark_veto_override": evaluate_mark_veto_override,
        "_run_mark_evidence_pipeline": _run_mark_evidence_pipeline,
        "estimate_age": estimate_age,
        "cross_spectral_normalize": cross_spectral_normalize,
        "compute_image_hash": compute_image_hash,
        "calculate_cosine_similarity": calculate_cosine_similarity,
        "finite_or_none": finite_or_none,
        "CALIBRATION": CALIBRATION,
        "TIER4_CALIBRATION": TIER4_CALIBRATION,
    }

    results_path = os.path.join(args.output_dir, "validation_results.jsonl")
    fp_path = os.path.join(args.output_dir, "false_positives.csv")
    fn_path = os.path.join(args.output_dir, "false_negatives.csv")
    conflict_path = os.path.join(args.output_dir, "conflicting_evidence_cases.csv")
    summary_path = os.path.join(args.output_dir, "summary_metrics.json")

    all_results = []

    with open(results_path, "w", encoding="utf-8") as f:
        for idx, pair in enumerate(pairs):
            pair_id = pair["pair_id"]
            label = pair["label_same_person"].strip().lower() == "true"
            category = pair.get("category", "unknown")
            print(f"[{idx+1}/{len(pairs)}] Processing {pair_id} ({category})...", end=" ", flush=True)

            row = {
                "pair_id": pair_id,
                "label_same_person": label,
                "category": category,
            }

            try:
                result = run_single_pair(pair, pipeline_modules)
                row.update(result)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e)}"
                row["timing_ms"] = 0
                traceback.print_exc()

            all_results.append(row)
            f.write(json.dumps(row) + "\n")
            status = row.get("error") or row.get("conclusion", "?")
            print(f"{row.get('timing_ms', 0)}ms — {status}")

    # ── Post-processing: generate summary outputs ──
    tp, fp, tn, fn = 0, 0, 0, 0
    false_positives = []
    false_negatives = []
    conflicting_cases = []

    for r in all_results:
        if r.get("error"):
            continue
        label = r["label_same_person"]
        score = r.get("fused_score", 0)
        veto = r.get("veto_triggered", False)
        predicted_match = score > 50.0 and not (veto and not r.get("veto_override_applied", False))

        if predicted_match and label:
            tp += 1
        elif predicted_match and not label:
            fp += 1
            false_positives.append(r)
        elif not predicted_match and not label:
            tn += 1
        elif not predicted_match and label:
            fn += 1
            false_negatives.append(r)

        # Conflicting evidence: face weak / marks strong, or face strong / marks weak
        face_strong = r.get("structural_sim", 0) >= 0.60
        face_weak = r.get("structural_sim", 0) < 0.40
        marks_strong = r.get("lr_marks", 1.0) > 10.0
        marks_weak = r.get("lr_marks", 1.0) <= 1.0

        if (face_weak and marks_strong) or (face_strong and marks_weak):
            conflict_type = "face_weak_marks_strong" if face_weak else "face_strong_marks_weak"
            conflicting_cases.append({**r, "conflict_type": conflict_type})

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    summary = {
        "total_pairs": len(all_results),
        "errors": sum(1 for r in all_results if r.get("error")),
        "evaluated": total,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "FAR": round(far, 4),
        "FRR": round(frr, 4),
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "conflicting_evidence_count": len(conflicting_cases),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _write_csv(fp_path, false_positives)
    _write_csv(fn_path, false_negatives)
    _write_csv(conflict_path, conflicting_cases)

    print(f"\n{'='*60}")
    print(f"Validation complete.")
    print(f"  Total: {len(all_results)}, Evaluated: {total}, Errors: {summary['errors']}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Precision={precision:.4f} Recall={recall:.4f} FAR={far:.4f} FRR={frr:.4f}")
    print(f"  Results: {args.output_dir}")
    print(f"{'='*60}")


def _write_csv(path, rows):
    """Write a list of dicts to CSV."""
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
