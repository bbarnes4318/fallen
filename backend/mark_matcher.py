"""Mark Matcher v1.0.0 — Pure forensic facial mark correspondence engine.

Matches gallery marks to probe marks using normalized canonical coordinates,
one-to-one Hungarian assignment, and multi-factor cost scoring.

Pure module: no FastAPI, no DB, no JWT, no generative dependencies.
"""
import math
import numpy as np

MARK_MATCHER_VERSION = "1.0.0"

# ── Matching thresholds ──
_MAX_SPATIAL_DISTANCE = 0.20       # Normalized coord distance threshold
_MIN_AREA_RATIO = 0.20            # Minimum area ratio to consider
_MAX_ASSIGNMENT_COST = 2.0        # Max cost for accepted match
_TYPE_MISMATCH_PENALTY = 0.5      # Cost penalty for different mark types
_REGION_MISMATCH_PENALTY = 0.3    # Cost penalty for different face regions
_CIRCULARITY_PENALTY_SCALE = 0.3  # Multiplied by abs circularity difference
_ORIENTATION_PENALTY_SCALE = 0.2  # For linear scars: angular_delta/90 * scale
_SPATIAL_WEIGHT = 5.0             # Weight for spatial distance in cost

# ── Compatible mark type groups ──
# Types within the same group receive a reduced penalty instead of full mismatch
_COMPATIBLE_TYPES = {
    frozenset({"dark_mole", "dark_spot"}),
    frozenset({"light_scar", "linear_scar"}),
    frozenset({"blemish", "dark_spot"}),
    frozenset({"blemish", "texture_cluster"}),
}
_COMPATIBLE_PENALTY = 0.15  # Reduced penalty for compatible but not identical types


def _get_position(mark: dict) -> tuple:
    """Extract normalized (x, y) position from a mark descriptor.

    Prefers canonical_position, falls back to centroid.
    """
    cp = mark.get("canonical_position")
    if cp is not None and len(cp) >= 2:
        return (float(cp[0]), float(cp[1]))
    ct = mark.get("centroid")
    if ct is not None and len(ct) >= 2:
        return (float(ct[0]), float(ct[1]))
    raise ValueError(f"Mark has no canonical_position or centroid: {mark}")


def _spatial_distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two normalized (x, y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _area_ratio(a1: float, a2: float) -> float:
    """Ratio of smaller to larger area. Returns 0 if both are zero."""
    if a1 <= 0 and a2 <= 0:
        return 0.0
    return min(a1, a2) / max(a1, a2) if max(a1, a2) > 0 else 0.0


def _type_penalty(type_g: str, type_p: str) -> float:
    """Cost penalty for mark type mismatch.

    Returns 0 for identical, reduced penalty for compatible types,
    full penalty for incompatible types.
    """
    if type_g == type_p:
        return 0.0
    pair = frozenset({type_g, type_p})
    for compat_set in _COMPATIBLE_TYPES:
        if pair <= compat_set:
            return _COMPATIBLE_PENALTY
    return _TYPE_MISMATCH_PENALTY


def _orientation_penalty(mark_g: dict, mark_p: dict) -> float:
    """Angular orientation penalty for linear scars."""
    if mark_g.get("mark_type") != "linear_scar" or mark_p.get("mark_type") != "linear_scar":
        return 0.0
    orient_g = mark_g.get("orientation")
    orient_p = mark_p.get("orientation")
    if orient_g is None or orient_p is None:
        return 0.0
    angular_delta = abs(orient_g - orient_p)
    angular_delta = min(angular_delta, 180.0 - angular_delta)
    return (angular_delta / 90.0) * _ORIENTATION_PENALTY_SCALE


def _compute_cost(mark_g: dict, mark_p: dict, pos_g: tuple, pos_p: tuple) -> tuple:
    """Compute matching cost and metadata for a gallery-probe mark pair.

    Returns (cost, metadata_dict) or (None, rejection_reason) if pre-rejected.
    """
    dist = _spatial_distance(pos_g, pos_p)
    if dist > _MAX_SPATIAL_DISTANCE:
        return None, f"spatial_distance_exceeded ({dist:.3f} > {_MAX_SPATIAL_DISTANCE})"

    area_g = mark_g.get("area", mark_g.get("contour_area", 1.0))
    area_p = mark_p.get("area", mark_p.get("contour_area", 1.0))
    ar = _area_ratio(area_g, area_p)
    if ar < _MIN_AREA_RATIO:
        return None, f"area_mismatch (ratio={ar:.2f} < {_MIN_AREA_RATIO})"

    type_g = mark_g.get("mark_type", "unknown")
    type_p = mark_p.get("mark_type", "unknown")
    region_g = mark_g.get("face_region", "unknown")
    region_p = mark_p.get("face_region", "unknown")

    type_match = (type_g == type_p)
    region_match = (region_g == region_p)

    # Cost accumulation
    cost = dist * _SPATIAL_WEIGHT
    cost += (1.0 - ar)
    cost += _type_penalty(type_g, type_p)
    if not region_match:
        cost += _REGION_MISMATCH_PENALTY

    # Circularity difference
    circ_diff = abs(mark_g.get("circularity", 0) - mark_p.get("circularity", 0))
    cost += circ_diff * _CIRCULARITY_PENALTY_SCALE

    # Orientation penalty for linear scars
    cost += _orientation_penalty(mark_g, mark_p)

    # Intensity difference (normalized)
    int_g = mark_g.get("intensity", 128)
    int_p = mark_p.get("intensity", 128)
    cost += abs(int_g - int_p) / 255.0

    # Salience/confidence weighting — slightly favour high-confidence matches
    conf_g = mark_g.get("confidence", 0.5)
    conf_p = mark_p.get("confidence", 0.5)
    avg_conf = (conf_g + conf_p) / 2.0

    metadata = {
        "position_distance": dist,
        "area_ratio": ar,
        "type_match": type_match,
        "region_match": region_match,
        "match_quality": max(0.0, 1.0 - cost / _MAX_ASSIGNMENT_COST) * avg_conf,
        "mark_type": type_g,
        "face_region": region_g,
        "cost": cost,
    }
    return cost, metadata


def match_facial_marks(gallery_marks: list, probe_marks: list,
                       calibration: dict | None = None) -> dict:
    """Match gallery marks to probe marks using Hungarian one-to-one assignment.

    Args:
        gallery_marks: List of mark descriptors from gallery image.
        probe_marks: List of mark descriptors from probe image.
        calibration: Optional Bayesian calibration dict. If None, LR stays neutral (1.0).

    Returns:
        dict with keys: matched, score, lr_marks, mark_lrs, matches,
        rejected_candidates, matcher_status, calibration_status,
        matcher_version.
    """
    n_gal = len(gallery_marks)
    n_pro = len(probe_marks)

    # ── Empty input guard ──
    if n_gal == 0 or n_pro == 0:
        return {
            "matched": 0,
            "score": None,
            "lr_marks": 1.0,
            "mark_lrs": [],
            "matches": [],
            "rejected_candidates": [],
            "matcher_status": "INSUFFICIENT_INPUT",
            "calibration_status": "NOT_APPLICABLE",
            "matcher_version": MARK_MATCHER_VERSION,
        }

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
            cost, meta_or_reason = _compute_cost(
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

    matches = []
    rejected_candidates = []

    matched_gal = set()
    matched_pro = set()

    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost < _MAX_ASSIGNMENT_COST:
            meta = meta_cache.get((r, c), {})
            match_entry = {
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
                "match_cost": cost,
                "lr": 1.0,  # Default neutral; overwritten by calibration below
            }
            matches.append(match_entry)
            matched_gal.add(int(r))
            matched_pro.add(int(c))
        elif cost < INFEASIBLE:
            meta = meta_cache.get((r, c), {})
            rejected_candidates.append({
                "gallery_idx": int(r),
                "probe_idx": int(c),
                "position_distance": meta.get("position_distance", 0.0),
                "match_cost": cost,
                "rejection_reason": f"cost_exceeded ({cost:.3f} > {_MAX_ASSIGNMENT_COST})",
            })

    # ── Collect pre-rejected pairs that were never assigned ──
    for (i, j), reason in pre_rejection_reasons.items():
        # Only report if neither mark was matched (to avoid noise)
        if i not in matched_gal and j not in matched_pro:
            rejected_candidates.append({
                "gallery_idx": int(i),
                "probe_idx": int(j),
                "position_distance": _spatial_distance(gal_positions[i], pro_positions[j]),
                "match_cost": INFEASIBLE,
                "rejection_reason": reason,
            })

    # ── Calibrated LR (only if calibration provided) ──
    calibration_status = "MISSING"
    lr_marks = 1.0
    mark_lrs = []

    if calibration is not None:
        calibration_status = "LOADED"
        lr_marks, mark_lrs = _apply_calibration(matches, gallery_marks, probe_marks, calibration)
        for idx, m in enumerate(matches):
            m["lr"] = mark_lrs[idx] if idx < len(mark_lrs) else 1.0
    else:
        # No calibration — all LRs stay at neutral 1.0
        mark_lrs = [1.0] * len(matches)
        for m in matches:
            m["lr"] = 1.0

    # ── Score ──
    matched_count = len(matches)
    total = max(n_gal, n_pro)
    score = round((matched_count / total) * 100.0, 2) if total > 0 else None

    # ── Status ──
    if matched_count > 0:
        matcher_status = "OK"
    elif n_gal > 0 and n_pro > 0:
        matcher_status = "NO_MATCHES"
    else:
        matcher_status = "INSUFFICIENT_INPUT"

    return {
        "matched": matched_count,
        "score": score,
        "lr_marks": lr_marks,
        "mark_lrs": mark_lrs,
        "matches": matches,
        "rejected_candidates": rejected_candidates,
        "matcher_status": matcher_status,
        "calibration_status": calibration_status,
        "matcher_version": MARK_MATCHER_VERSION,
    }


def _apply_calibration(matches: list, gallery_marks: list, probe_marks: list,
                       calibration: dict) -> tuple:
    """Apply Bayesian calibration to matched pairs.

    Returns (combined_lr, [individual_lrs]).

    If calibration data is incomplete, returns neutral LRs.
    """
    try:
        from scipy.stats import multivariate_normal as mvn, lognorm as _lognorm, norm as _norm
    except ImportError:
        return 1.0, [1.0] * len(matches)

    spatial_kde = calibration.get("spatial_kde")
    area_dist = calibration.get("area_distribution")
    int_dist = calibration.get("intensity_distribution")
    circ_dist = calibration.get("circularity_distribution")
    delta_model = calibration.get("intra_person_delta")
    EPSILON = calibration.get("epsilon_floor", 1e-9)

    if not all([spatial_kde, area_dist, int_dist, circ_dist, delta_model]):
        return 1.0, [1.0] * len(matches)

    delta_mean = np.array(delta_model["mean"])
    delta_cov = np.array(delta_model["covariance"])

    mark_lrs = []
    for match in matches:
        r = match["gallery_idx"]
        c = match["probe_idx"]
        mg = gallery_marks[r]
        mp = probe_marks[c]

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

        if lr > 1.0:
            mark_lrs.append(float(lr))
        else:
            mark_lrs.append(1.0)

    combined_lr = 1.0
    for lr_val in mark_lrs:
        combined_lr *= lr_val

    return combined_lr, mark_lrs


def get_matcher_thresholds() -> dict:
    """Return the current matcher thresholds as a dict (single source of truth)."""
    return {
        "max_spatial_distance": _MAX_SPATIAL_DISTANCE,
        "min_area_ratio": _MIN_AREA_RATIO,
        "max_assignment_cost": _MAX_ASSIGNMENT_COST,
        "type_mismatch_penalty": _TYPE_MISMATCH_PENALTY,
        "region_mismatch_penalty": _REGION_MISMATCH_PENALTY,
        "compatible_type_penalty": _COMPATIBLE_PENALTY,
        "spatial_weight": _SPATIAL_WEIGHT,
        "matcher_version": MARK_MATCHER_VERSION,
    }
