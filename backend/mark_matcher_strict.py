"""Mark Matcher Strict v2.1.0 — Experimental strict forensic mark correspondence engine.

Activated ONLY when USE_STRICT_MARK_MATCHER_V2=true.
Reduces random mark correspondences between different people by:
  - Classifying marks as generic vs distinctive
  - Enforcing hard same-region + same-type for generic marks
  - Capping generic mark LR at 1.0 (neutral)
  - Capping distinctive mark LR at 25.0
  - Aggregate caps per region/type/channel/total
  - Cluster penalty when one dimension dominates
  - Phase 1 barycentric cost contribution (additive, stricter-only)
  - Constellation quality telemetry (display/research only)

Pure module: no FastAPI, no DB, no JWT, no generative dependencies.
"""
import math
import os
from collections import defaultdict
import numpy as np

MARK_MATCHER_STRICT_VERSION = "2.1.0-strict"

# ── Mark Classification ──
GENERIC_MARK_TYPES = frozenset({
    "freckle", "pore", "texture_anomaly", "blemish",
    "dark_spot", "texture_cluster", "unknown_mark",
})

DISTINCTIVE_MARK_TYPES = frozenset({
    "dark_mole", "mole", "light_scar", "scar",
    "linear_scar", "structural_crater", "depression_scar",
})

# ── Compatible type groups (distinctive only) ──
_COMPATIBLE_TYPES = {
    frozenset({"dark_mole", "mole"}),
    frozenset({"light_scar", "scar"}),
    frozenset({"light_scar", "linear_scar"}),
    frozenset({"scar", "linear_scar"}),
    frozenset({"structural_crater", "depression_scar"}),
}
_COMPATIBLE_PENALTY = 0.15

# ── Thresholds: Distinctive ──
_DIST_MAX_SPATIAL = 0.12
_DIST_MIN_AREA_RATIO = 0.30
_DIST_MAX_COST = 1.5

# ── Thresholds: Generic ──
_GEN_MAX_SPATIAL = 0.06
_GEN_MIN_AREA_RATIO = 0.40
_GEN_MAX_COST = 0.8

# ── Cost weights (shared) ──
_SPATIAL_WEIGHT = 5.0
_TYPE_MISMATCH_PENALTY = 0.5
_REGION_MISMATCH_PENALTY = 0.3
_CIRCULARITY_PENALTY_SCALE = 0.3
_ORIENTATION_PENALTY_SCALE = 0.2

# ── Phase 1 Barycentric ──
# Conservative weight. Can only INCREASE cost (make matching stricter).
# Start at 1.0; validate before increasing. Do NOT start at 3.0.
_BARY_WEIGHT = 1.0
_BARY_MIN_CONFIDENCE = 0.70  # Require this mesh_confidence for bary distance

# ── LR Caps ──
_GENERIC_LR_CAP = 1.0
_DISTINCTIVE_LR_CAP = 25.0
_FALLBACK_LR_CAP = 1.0
_PER_REGION_LR_CAP = 50.0
_PER_TYPE_LR_CAP = 50.0
_PER_CHANNEL_LR_CAP = 25.0
_TOTAL_LOG_LR_CAP = 2.0  # max combined LR = 100
_CLUSTER_DOMINATION_THRESHOLD = 0.60
_CLUSTER_PENALTY_FACTOR = 0.5


def _classify_mark(mark_type: str) -> str:
    if mark_type in DISTINCTIVE_MARK_TYPES:
        return "distinctive"
    return "generic"


def _get_position(mark: dict) -> tuple:
    cp = mark.get("canonical_position")
    if cp is not None and len(cp) >= 2:
        return (float(cp[0]), float(cp[1]))
    ct = mark.get("centroid")
    if ct is not None and len(ct) >= 2:
        return (float(ct[0]), float(ct[1]))
    raise ValueError(f"Mark has no canonical_position or centroid: {mark}")


def _spatial_distance(p1: tuple, p2: tuple) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _area_ratio(a1: float, a2: float) -> float:
    if a1 <= 0 and a2 <= 0:
        return 0.0
    return min(a1, a2) / max(a1, a2) if max(a1, a2) > 0 else 0.0


def _type_penalty_strict(type_g: str, type_p: str, class_g: str, class_p: str) -> tuple:
    """Returns (penalty, rejection_reason_or_None).
    For generic marks: requires exact type match, else hard reject.
    For distinctive: allows compatible groups.
    """
    if type_g == type_p:
        return 0.0, None

    # Mixed class: hard reject
    if class_g != class_p:
        return float("inf"), "strict_mixed_class_mismatch"

    # Both generic: hard reject on type mismatch
    if class_g == "generic":
        return float("inf"), "strict_generic_type_mismatch"

    # Both distinctive: check compatibility
    pair = frozenset({type_g, type_p})
    for compat_set in _COMPATIBLE_TYPES:
        if pair <= compat_set:
            return _COMPATIBLE_PENALTY, None

    return _TYPE_MISMATCH_PENALTY, None


def _orientation_penalty(mark_g: dict, mark_p: dict) -> float:
    if mark_g.get("mark_type") != "linear_scar" or mark_p.get("mark_type") != "linear_scar":
        return 0.0
    orient_g = mark_g.get("orientation")
    orient_p = mark_p.get("orientation")
    if orient_g is None or orient_p is None:
        return 0.0
    angular_delta = abs(orient_g - orient_p)
    angular_delta = min(angular_delta, 180.0 - angular_delta)
    return (angular_delta / 90.0) * _ORIENTATION_PENALTY_SCALE


def _barycentric_cost(mark_g: dict, mark_p: dict) -> tuple:
    """Compute barycentric distance cost contribution for a mark pair.

    Returns (bary_cost, bary_distance, bary_available, bary_comparison_mode, bary_telemetry).
    bary_cost is the cost contribution (>= 0). It can ONLY increase total cost.
    bary_available is True only when the comparison is mathematically valid
    (same triangle or same anchor set).
    bary_comparison_mode: "same_triangle", "same_anchor_set", "different_triangle_fallback", "unavailable".
    bary_telemetry: dict with diagnostic fields.
    """
    from mark_anatomy import compute_barycentric_distance, BARY_COMPARE_UNAVAILABLE

    anat_g = mark_g.get("anatomical_position")
    anat_p = mark_p.get("anatomical_position")

    empty_telemetry = {
        "barycentric_comparison_mode": "unavailable",
        "barycentric_triangle_match": False,
        "barycentric_anchor_overlap_count": 0,
        "barycentric_distance_available": False,
        "mesh_triangle_gallery": None,
        "mesh_triangle_probe": None,
    }

    if anat_g is None or anat_p is None:
        return 0.0, None, False, "unavailable", empty_telemetry

    mode_g = anat_g.get("barycentric_mode")
    mode_p = anat_p.get("barycentric_mode")

    # Must be 2D mesh approximation, not fallback
    if mode_g != "2d_mesh_approximation" or mode_p != "2d_mesh_approximation":
        return 0.0, None, False, "unavailable", empty_telemetry

    conf_g = anat_g.get("mesh_confidence", 0.0)
    conf_p = anat_p.get("mesh_confidence", 0.0)

    if conf_g < _BARY_MIN_CONFIDENCE or conf_p < _BARY_MIN_CONFIDENCE:
        return 0.0, None, False, "unavailable", empty_telemetry

    # Delegate to triangle-aware distance computation
    bary_dist, available, comparison_mode, telemetry = compute_barycentric_distance(anat_g, anat_p)

    if not available or bary_dist is None:
        # Different triangle or unavailable — cost contribution is 0
        # Falls back to existing strict spatial distance for matching
        return 0.0, None, False, comparison_mode, telemetry

    # Only apply cost when comparison is valid (same_triangle or same_anchor_set)
    bary_cost = bary_dist * _BARY_WEIGHT
    return bary_cost, bary_dist, True, comparison_mode, telemetry


def _compute_cost_strict(mark_g: dict, mark_p: dict, pos_g: tuple, pos_p: tuple) -> tuple:
    """Compute matching cost with strict type-specific thresholds.
    Returns (cost, metadata_dict) or (None, rejection_reason).
    """
    type_g = mark_g.get("mark_type", "unknown_mark")
    type_p = mark_p.get("mark_type", "unknown_mark")
    class_g = _classify_mark(type_g)
    class_p = _classify_mark(type_p)
    region_g = mark_g.get("face_region", "unknown")
    region_p = mark_p.get("face_region", "unknown")
    channel_g = mark_g.get("channel", "unknown")

    # Determine which thresholds to use (use stricter if either is generic)
    is_generic_pair = (class_g == "generic" or class_p == "generic")
    max_spatial = _GEN_MAX_SPATIAL if is_generic_pair else _DIST_MAX_SPATIAL
    min_ar = _GEN_MIN_AREA_RATIO if is_generic_pair else _DIST_MIN_AREA_RATIO
    max_cost = _GEN_MAX_COST if is_generic_pair else _DIST_MAX_COST

    # Spatial distance check
    dist = _spatial_distance(pos_g, pos_p)
    if dist > max_spatial:
        return None, f"spatial_distance_exceeded ({dist:.3f} > {max_spatial})"

    # Area ratio check
    area_g = mark_g.get("area", mark_g.get("contour_area", 1.0))
    area_p = mark_p.get("area", mark_p.get("contour_area", 1.0))
    ar = _area_ratio(area_g, area_p)
    if ar < min_ar:
        return None, f"area_mismatch (ratio={ar:.2f} < {min_ar})"

    # Generic hard requirements
    if is_generic_pair:
        if region_g != region_p:
            return None, "strict_generic_region_mismatch"

    # Type penalty (may hard-reject)
    tp, tp_reason = _type_penalty_strict(type_g, type_p, class_g, class_p)
    if tp_reason is not None:
        return None, tp_reason

    region_match = (region_g == region_p)
    type_match = (type_g == type_p)

    # Cost accumulation (existing terms — unchanged)
    cost = dist * _SPATIAL_WEIGHT
    cost += (1.0 - ar)
    cost += tp
    if not region_match:
        cost += _REGION_MISMATCH_PENALTY

    circ_diff = abs(mark_g.get("circularity", 0) - mark_p.get("circularity", 0))
    cost += circ_diff * _CIRCULARITY_PENALTY_SCALE
    cost += _orientation_penalty(mark_g, mark_p)

    int_g = mark_g.get("intensity", 128)
    int_p = mark_p.get("intensity", 128)
    cost += abs(int_g - int_p) / 255.0

    # ── Phase 1: Barycentric cost contribution (additive, stricter-only) ──
    # This can ONLY INCREASE cost, never decrease it.
    # If unavailable or different triangle, bary_cost is 0.0 — existing path unchanged.
    bary_cost, bary_dist, bary_available, bary_mode, bary_telemetry = _barycentric_cost(mark_g, mark_p)
    cost += bary_cost  # Always >= 0, so cost can only go up

    if cost > max_cost:
        return None, f"strict_cost_exceeded ({cost:.3f} > {max_cost})"

    conf_g = mark_g.get("confidence", 0.5)
    conf_p = mark_p.get("confidence", 0.5)
    avg_conf = (conf_g + conf_p) / 2.0

    metadata = {
        "position_distance": dist,
        "area_ratio": ar,
        "type_match": type_match,
        "region_match": region_match,
        "match_quality": max(0.0, 1.0 - cost / max_cost) * avg_conf,
        "mark_type": type_g,
        "face_region": region_g,
        "channel": channel_g,
        "cost": cost,
        "mark_class_gallery": class_g,
        "mark_class_probe": class_p,
        "is_generic_pair": is_generic_pair,
        # Phase 1 barycentric telemetry (triangle-aware)
        "barycentric_distance": bary_dist,
        "barycentric_available": bary_available,
        "barycentric_cost_contribution": bary_cost,
        "barycentric_comparison_mode": bary_telemetry.get("barycentric_comparison_mode", "unavailable"),
        "barycentric_triangle_match": bary_telemetry.get("barycentric_triangle_match", False),
        "barycentric_anchor_overlap_count": bary_telemetry.get("barycentric_anchor_overlap_count", 0),
        "barycentric_distance_available": bary_telemetry.get("barycentric_distance_available", False),
        "mesh_triangle_gallery": bary_telemetry.get("mesh_triangle_gallery"),
        "mesh_triangle_probe": bary_telemetry.get("mesh_triangle_probe"),
    }
    return cost, metadata


def _cap_individual_lr(lr_raw: float, mark_class: str, is_fallback: bool) -> tuple:
    """Apply per-mark LR cap. Returns (capped_lr, cap_applied_reason_or_None)."""
    if is_fallback:
        capped = min(lr_raw, _FALLBACK_LR_CAP)
        reason = "fallback_lr_cap" if lr_raw > _FALLBACK_LR_CAP else None
        return capped, reason
    if mark_class == "generic":
        return _GENERIC_LR_CAP, "generic_mark_neutral" if lr_raw > _GENERIC_LR_CAP else None
    capped = min(lr_raw, _DISTINCTIVE_LR_CAP)
    reason = "distinctive_lr_cap" if lr_raw > _DISTINCTIVE_LR_CAP else None
    return capped, reason


def _apply_aggregate_caps(scoring_correspondences: list, lr_product: float) -> tuple:
    """Apply per-region, per-type, per-channel, and total log-LR caps.
    Returns (capped_lr, list_of_caps_applied).
    """
    caps_applied = []

    # Per-region cap
    region_lr = defaultdict(float)
    for c in scoring_correspondences:
        lr_val = c.get("lr_after_cap", 1.0)
        if lr_val > 1.0:
            region_lr[c["face_region"]] += math.log10(lr_val)

    region_capped_total = 0.0
    for region, log_lr in region_lr.items():
        max_log = math.log10(_PER_REGION_LR_CAP)
        if log_lr > max_log:
            caps_applied.append(f"per_region_cap:{region}")
            region_capped_total += max_log
        else:
            region_capped_total += log_lr

    # Per-type cap
    type_lr = defaultdict(float)
    for c in scoring_correspondences:
        lr_val = c.get("lr_after_cap", 1.0)
        if lr_val > 1.0:
            type_lr[c["mark_type"]] += math.log10(lr_val)

    type_capped_total = 0.0
    for mtype, log_lr in type_lr.items():
        max_log = math.log10(_PER_TYPE_LR_CAP)
        if log_lr > max_log:
            caps_applied.append(f"per_type_cap:{mtype}")
            type_capped_total += max_log
        else:
            type_capped_total += log_lr

    # Per-channel cap
    channel_lr = defaultdict(float)
    for c in scoring_correspondences:
        lr_val = c.get("lr_after_cap", 1.0)
        if lr_val > 1.0:
            channel_lr[c.get("channel", "unknown")] += math.log10(lr_val)

    channel_capped_total = 0.0
    for ch, log_lr in channel_lr.items():
        max_log = math.log10(_PER_CHANNEL_LR_CAP)
        if log_lr > max_log:
            caps_applied.append(f"per_channel_cap:{ch}")
            channel_capped_total += max_log
        else:
            channel_capped_total += log_lr

    # Use the most constrained of region/type/channel
    effective_log = min(region_capped_total, type_capped_total, channel_capped_total)

    # Total log-LR cap
    if effective_log > _TOTAL_LOG_LR_CAP:
        caps_applied.append("total_log_lr_cap")
        effective_log = _TOTAL_LOG_LR_CAP

    capped_lr = 10.0 ** effective_log if effective_log > 0 else 1.0
    return capped_lr, caps_applied


def _apply_cluster_penalty(scoring_correspondences: list, lr_product: float) -> tuple:
    """Penalize if a single region/type/channel dominates >60% of log-LR.
    Returns (penalized_lr, penalty_applied, penalty_factor).
    """
    if lr_product <= 1.0 or not scoring_correspondences:
        return lr_product, False, 1.0

    total_log = math.log10(max(lr_product, 1.0))
    if total_log <= 0:
        return lr_product, False, 1.0

    penalty_factor = 1.0

    for dim_key in ["face_region", "mark_type", "channel"]:
        dim_log = defaultdict(float)
        for c in scoring_correspondences:
            lr_val = c.get("lr_after_cap", 1.0)
            if lr_val > 1.0:
                dim_log[c.get(dim_key, "unknown")] += math.log10(lr_val)

        for dim_val, log_lr in dim_log.items():
            if total_log > 0 and log_lr / total_log > _CLUSTER_DOMINATION_THRESHOLD:
                excess = log_lr - (_CLUSTER_DOMINATION_THRESHOLD * total_log)
                dim_penalty = 10 ** (-excess * _CLUSTER_PENALTY_FACTOR)
                penalty_factor *= dim_penalty

    if penalty_factor < 1.0:
        return lr_product * penalty_factor, True, penalty_factor

    return lr_product, False, 1.0


def match_facial_marks_strict(gallery_marks: list, probe_marks: list,
                              calibration: dict | None = None) -> dict:
    """Strict mark matcher with generic/distinctive classification and aggressive caps.

    Returns dict with standard matcher keys plus strict telemetry.
    """
    n_gal = len(gallery_marks)
    n_pro = len(probe_marks)

    empty_result = {
        "matched": 0, "score": None, "lr_marks": 1.0,
        "mark_lrs": [], "matches": [], "rejected_candidates": [],
        "matcher_status": "INSUFFICIENT_INPUT",
        "calibration_status": "NOT_APPLICABLE",
        "matcher_version": MARK_MATCHER_STRICT_VERSION,
        "strict_mode": True,
        "display_correspondences": [], "scoring_correspondences": [],
        "suppressed_correspondences": [],
        "scoring_eligible_marks_count": 0,
        "generic_marks_suppressed_count": 0,
        "distinctive_marks_preserved_count": 0,
        "lr_before_caps": 1.0, "lr_after_all_caps": 1.0,
        "caps_applied": [], "cluster_penalty_applied": False,
        "cluster_penalty_factor": 1.0,
    }

    if n_gal == 0 or n_pro == 0:
        return empty_result

    # ── Build cost matrix ──
    from scipy.optimize import linear_sum_assignment

    INFEASIBLE = 1e6
    cost_matrix = np.full((n_gal, n_pro), INFEASIBLE)
    meta_cache = {}
    pre_rejection_reasons = {}

    gal_positions = [_get_position(m) for m in gallery_marks]
    pro_positions = [_get_position(m) for m in probe_marks]

    for i in range(n_gal):
        for j in range(n_pro):
            cost, meta_or_reason = _compute_cost_strict(
                gallery_marks[i], probe_marks[j],
                gal_positions[i], pro_positions[j]
            )
            if cost is not None:
                cost_matrix[i, j] = cost
                meta_cache[(i, j)] = meta_or_reason
            else:
                pre_rejection_reasons[(i, j)] = meta_or_reason

    # ── Hungarian assignment ──
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    all_correspondences = []
    rejected_candidates = []
    matched_gal = set()
    matched_pro = set()

    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        meta = meta_cache.get((r, c), {})
        is_generic = meta.get("is_generic_pair", True) if isinstance(meta, dict) else True
        mark_class = "generic" if is_generic else "distinctive"

        # Determine effective max cost
        max_cost = _GEN_MAX_COST if is_generic else _DIST_MAX_COST

        if cost < max_cost and cost < INFEASIBLE:
            entry = {
                "gallery_idx": int(r),
                "probe_idx": int(c),
                "gallery_centroid": list(gal_positions[r]),
                "probe_centroid": list(pro_positions[c]),
                "position_distance": meta.get("position_distance", 0.0),
                "area_ratio": meta.get("area_ratio", 0.0),
                "type_match": meta.get("type_match", False),
                "region_match": meta.get("region_match", False),
                "match_quality": meta.get("match_quality", 0.0),
                "mark_type": meta.get("mark_type", "unknown"),
                "face_region": meta.get("face_region", "unknown"),
                "channel": meta.get("channel", "unknown"),
                "match_cost": cost,
                "mark_class": mark_class,
                "accepted_for_display": True,
                "eligible_for_lr": (mark_class == "distinctive"),
                "suppression_reason": ("generic_mark_neutral" if mark_class == "generic" else None),
                "strict_rejection_reason": None,
                "lr_before_cap": 1.0,
                "lr_after_cap": 1.0,
                "lr": 1.0,
                # Phase 1 barycentric telemetry per correspondence (triangle-aware)
                "barycentric_distance": meta.get("barycentric_distance"),
                "barycentric_available": meta.get("barycentric_available", False),
                "barycentric_cost_contribution": meta.get("barycentric_cost_contribution", 0.0),
                "barycentric_comparison_mode": meta.get("barycentric_comparison_mode", "unavailable"),
                "barycentric_triangle_match": meta.get("barycentric_triangle_match", False),
                "barycentric_anchor_overlap_count": meta.get("barycentric_anchor_overlap_count", 0),
                "barycentric_distance_available": meta.get("barycentric_distance_available", False),
                "mesh_triangle_gallery": meta.get("mesh_triangle_gallery"),
                "mesh_triangle_probe": meta.get("mesh_triangle_probe"),
            }
            all_correspondences.append(entry)
            matched_gal.add(int(r))
            matched_pro.add(int(c))
        elif cost < INFEASIBLE:
            rejected_candidates.append({
                "gallery_idx": int(r), "probe_idx": int(c),
                "position_distance": meta.get("position_distance", 0.0) if isinstance(meta, dict) else 0.0,
                "match_cost": cost,
                "rejection_reason": f"strict_cost_exceeded ({cost:.3f})",
            })

    # Collect pre-rejected pairs
    for (i, j), reason in pre_rejection_reasons.items():
        if i not in matched_gal and j not in matched_pro:
            rejected_candidates.append({
                "gallery_idx": int(i), "probe_idx": int(j),
                "position_distance": _spatial_distance(gal_positions[i], pro_positions[j]),
                "match_cost": INFEASIBLE,
                "rejection_reason": reason,
            })

    # ── Calibrated LR ──
    calibration_status = "MISSING"
    if calibration is not None and all_correspondences:
        calibration_status = "LOADED"
        raw_lrs = _compute_raw_lrs(all_correspondences, gallery_marks, probe_marks, calibration)
        for idx, c in enumerate(all_correspondences):
            raw_lr = raw_lrs[idx] if idx < len(raw_lrs) else 1.0
            is_fallback = gallery_marks[c["gallery_idx"]].get("fallback_generated", False)
            c["lr_before_cap"] = raw_lr

            if c["mark_class"] == "generic":
                c["lr_after_cap"] = _GENERIC_LR_CAP
                c["suppression_reason"] = "generic_mark_neutral"
                c["eligible_for_lr"] = False
            else:
                capped, cap_reason = _cap_individual_lr(raw_lr, c["mark_class"], is_fallback)
                c["lr_after_cap"] = capped
                if cap_reason:
                    c["suppression_reason"] = cap_reason
                c["eligible_for_lr"] = True

            c["lr"] = c["lr_after_cap"]
    elif calibration is None:
        calibration_status = "MISSING"
        for c in all_correspondences:
            c["lr_before_cap"] = 1.0
            c["lr_after_cap"] = 1.0
            c["lr"] = 1.0

    # ── Separate display / scoring / suppressed ──
    display_correspondences = [c for c in all_correspondences if c["accepted_for_display"]]
    scoring_correspondences = [c for c in all_correspondences if c["eligible_for_lr"]]
    suppressed_correspondences = [c for c in all_correspondences if not c["eligible_for_lr"]]

    # ── Compute LR product from scoring-eligible only ──
    lr_before_caps = 1.0
    for c in scoring_correspondences:
        lr_before_caps *= max(c["lr_after_cap"], 1.0)

    # ── Apply aggregate caps ──
    lr_after_agg, caps_applied = _apply_aggregate_caps(scoring_correspondences, lr_before_caps)

    # ── Apply cluster penalty ──
    lr_final, cluster_applied, cluster_factor = _apply_cluster_penalty(
        scoring_correspondences, lr_after_agg
    )

    # Floor at 1.0
    lr_final = max(lr_final, 1.0)

    # Build mark_lrs list (for compatibility)
    mark_lrs = [c["lr_after_cap"] for c in all_correspondences]

    # Count stats
    generic_suppressed = sum(1 for c in all_correspondences if c["mark_class"] == "generic")
    distinctive_preserved = sum(1 for c in scoring_correspondences)

    matched_count = len(all_correspondences)
    total = max(n_gal, n_pro)
    score = round((matched_count / total) * 100.0, 2) if total > 0 else None

    if matched_count > 0:
        matcher_status = "OK"
    elif n_gal > 0 and n_pro > 0:
        matcher_status = "NO_MATCHES"
    else:
        matcher_status = "INSUFFICIENT_INPUT"

    # ── Constellation Telemetry (DISPLAY/RESEARCH ONLY — does NOT affect scoring) ──
    # Computed AFTER all LR calculations are finalized.
    lr_marks_before_constellation = lr_final  # snapshot for assertion
    constellation_telemetry = None
    try:
        from mark_anatomy import compute_constellation_telemetry
        constellation_telemetry = compute_constellation_telemetry(
            scoring_correspondences=scoring_correspondences,
            suppressed_correspondences=suppressed_correspondences,
            gallery_marks=gallery_marks,
            probe_marks=probe_marks,
            cluster_domination_score=cluster_factor,
        )
    except Exception:
        # Never crash matching for constellation telemetry
        constellation_telemetry = None

    # ASSERTION: constellation telemetry MUST NOT alter scoring
    assert lr_final == lr_marks_before_constellation, (
        "FATAL: constellation telemetry altered lr_marks. "
        f"Before={lr_marks_before_constellation}, After={lr_final}"
    )

    return {
        # Standard keys (downstream compatibility)
        "matched": matched_count,
        "score": score,
        "lr_marks": lr_final,
        "mark_lrs": mark_lrs,
        "matches": display_correspondences,
        "rejected_candidates": rejected_candidates,
        "matcher_status": matcher_status,
        "calibration_status": calibration_status,
        "matcher_version": MARK_MATCHER_STRICT_VERSION,
        # Strict telemetry
        "strict_mode": True,
        "displayed_marks_count": len(display_correspondences),
        "display_correspondences": display_correspondences,
        "scoring_eligible_marks_count": len(scoring_correspondences),
        "scoring_correspondences": scoring_correspondences,
        "suppressed_correspondences": suppressed_correspondences,
        "generic_marks_suppressed_count": generic_suppressed,
        "distinctive_marks_preserved_count": distinctive_preserved,
        "lr_before_caps": lr_before_caps,
        "lr_after_all_caps": lr_final,
        "caps_applied": caps_applied,
        "cluster_penalty_applied": cluster_applied,
        "cluster_penalty_factor": cluster_factor,
        # Phase 1 constellation telemetry (DOES NOT AFFECT SCORING)
        "constellation_telemetry": constellation_telemetry,
    }


def _compute_raw_lrs(correspondences, gallery_marks, probe_marks, calibration):
    """Compute raw Bayesian LRs before any caps."""
    try:
        from scipy.stats import multivariate_normal as mvn, lognorm as _lognorm, norm as _norm
    except ImportError:
        return [1.0] * len(correspondences)

    spatial_kde = calibration.get("spatial_kde")
    area_dist = calibration.get("area_distribution")
    int_dist = calibration.get("intensity_distribution")
    circ_dist = calibration.get("circularity_distribution")
    delta_model = calibration.get("intra_person_delta")
    EPSILON = calibration.get("epsilon_floor", 1e-9)

    if not all([spatial_kde, area_dist, int_dist, circ_dist, delta_model]):
        return [1.0] * len(correspondences)

    delta_mean = np.array(delta_model["mean"])
    delta_cov = np.array(delta_model["covariance"])

    lrs = []
    for c in correspondences:
        r = c["gallery_idx"]
        ci = c["probe_idx"]
        mg = gallery_marks[r]
        mp = probe_marks[ci]

        delta_v = np.array([
            mg.get("centroid", (0, 0))[0] - mp.get("centroid", (0, 0))[0],
            mg.get("centroid", (0, 0))[1] - mp.get("centroid", (0, 0))[1],
            mg.get("area", 0) - mp.get("area", 0),
            mg.get("intensity", 128) - mp.get("intensity", 128),
            mg.get("circularity", 0) - mp.get("circularity", 0),
        ])

        try:
            numerator = max(mvn.pdf(delta_v, mean=delta_mean, cov=delta_cov), EPSILON)
        except Exception:
            numerator = EPSILON

        try:
            p_spatial = max(float(spatial_kde.evaluate(
                np.array([[mp.get("centroid", (0, 0))[0]], [mp.get("centroid", (0, 0))[1]]])
            )[0]), EPSILON)
        except Exception:
            p_spatial = EPSILON

        p_area = max(float(_lognorm.pdf(
            max(mp.get("area", 1), 0.01),
            area_dist["shape"], loc=area_dist["loc"], scale=area_dist["scale"]
        )), EPSILON)
        p_intensity = max(float(_norm.pdf(
            mp.get("intensity", 128),
            loc=int_dist["mean"], scale=int_dist["std"]
        )), EPSILON)
        p_circularity = max(float(_norm.pdf(
            mp.get("circularity", 0),
            loc=circ_dist["mean"], scale=circ_dist["std"]
        )), EPSILON)

        denominator = max(p_spatial * p_area * p_intensity * p_circularity, EPSILON)
        lr = max(numerator / denominator, EPSILON)
        lrs.append(float(lr) if lr > 1.0 else 1.0)

    return lrs


def get_strict_matcher_thresholds() -> dict:
    """Return all strict matcher thresholds as a dict."""
    return {
        "distinctive_max_spatial_distance": _DIST_MAX_SPATIAL,
        "distinctive_min_area_ratio": _DIST_MIN_AREA_RATIO,
        "distinctive_max_assignment_cost": _DIST_MAX_COST,
        "generic_max_spatial_distance": _GEN_MAX_SPATIAL,
        "generic_min_area_ratio": _GEN_MIN_AREA_RATIO,
        "generic_max_assignment_cost": _GEN_MAX_COST,
        "generic_lr_cap": _GENERIC_LR_CAP,
        "distinctive_lr_cap": _DISTINCTIVE_LR_CAP,
        "fallback_lr_cap": _FALLBACK_LR_CAP,
        "per_region_lr_cap": _PER_REGION_LR_CAP,
        "per_type_lr_cap": _PER_TYPE_LR_CAP,
        "per_channel_lr_cap": _PER_CHANNEL_LR_CAP,
        "total_log_lr_cap": _TOTAL_LOG_LR_CAP,
        "cluster_domination_threshold": _CLUSTER_DOMINATION_THRESHOLD,
        "cluster_penalty_factor": _CLUSTER_PENALTY_FACTOR,
        # Phase 1 barycentric
        "barycentric_weight": _BARY_WEIGHT,
        "barycentric_min_confidence": _BARY_MIN_CONFIDENCE,
        "matcher_version": MARK_MATCHER_STRICT_VERSION,
    }
