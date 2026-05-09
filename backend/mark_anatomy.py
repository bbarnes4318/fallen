"""Mark Anatomy v1.0.0 — Anatomical mesh-relative coordinate mapping + constellation telemetry.

Phase 1: 2D mesh approximation using nearest-3-landmarks barycentric projection
from the MediaPipe Face Mesh (478 landmarks).

This is NOT true 3D barycentric projection onto a canonical face mesh.
It is an approximation that projects marks onto the nearest triangle formed
by 3 MediaPipe landmarks in normalized 2D space.

Pure module: no FastAPI, no DB, no JWT dependencies.
"""
import math
from collections import Counter
from itertools import combinations

MARK_ANATOMY_VERSION = "2.0.0"
COORDINATE_SYSTEM_VERSION = "1.0.0-phase1"
REGIONAL_COORDINATE_VERSION = "2.0.0-regional-canonical"
BARYCENTRIC_MODE_2D = "2d_mesh_approximation"
BARYCENTRIC_MODE_FALLBACK = "nearest_landmark_fallback"
TRIANGLE_SOURCE = "nearest_3_mediapipe_landmarks"

# ── Mesh confidence thresholds ──
_MIN_TRIANGLE_AREA = 1e-8  # Below this, triangle is degenerate
_BARY_TELEMETRY_MIN_CONFIDENCE = 0.50  # Below this, barycentric coords are totally ignored even for telemetry
_BARY_COST_MIN_CONFIDENCE = 0.70  # Below this, barycentric distance doesn't contribute to match cost

# ── Expanded ~15-region landmark mapping ──
# This is TELEMETRY ONLY. The existing 8-region face_region in mark_detector.py
# is unchanged and still used for production matching, caps, and UI.
#
# Each landmark index maps to an expanded anatomical region.
# MediaPipe Face Mesh has 478 landmarks (468 base + 10 iris refinement).

# Landmark index groups (from mark_detector.py, extended)
_LEFT_EYE_IDX = frozenset({33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246})
_RIGHT_EYE_IDX = frozenset({362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398})
_LEFT_BROW_IDX = frozenset({70, 63, 105, 66, 107, 55, 65, 52, 53, 46})
_RIGHT_BROW_IDX = frozenset({300, 293, 334, 296, 336, 285, 295, 282, 283, 276})
_NOSE_IDX = frozenset({1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 220, 115, 48, 64, 102,
                       49, 131, 134, 236, 196, 3, 51, 281, 275, 440, 344, 278, 294, 331,
                       279, 360, 363, 456, 420, 399, 412, 351})
_LIPS_IDX = frozenset({61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
                       402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270,
                       409, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78})

# Forehead landmarks (upper face, above brows)
_FOREHEAD_IDX = frozenset({10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109,
                           69, 104, 68, 71, 139, 162, 127, 234, 93})
# Left temple
_LEFT_TEMPLE_IDX = frozenset({132, 58, 172, 136, 150, 149, 176, 148})
# Right temple
_RIGHT_TEMPLE_IDX = frozenset({389, 356, 454, 323, 361, 288, 397, 365})
# Glabella (between brows)
_GLABELLA_IDX = frozenset({9, 151, 108, 69, 104, 68, 337, 299, 333})
# Nose bridge (upper nose)
_NOSE_BRIDGE_IDX = frozenset({168, 6, 197, 195, 5, 4, 1})
# Philtrum (between nose and upper lip)
_PHILTRUM_IDX = frozenset({164, 167, 165, 92, 186, 57, 43, 106, 182, 83, 18, 313, 406,
                           335, 273, 287, 410, 322, 391, 393})
# Left cheek
_LEFT_CHEEK_IDX = frozenset({116, 117, 118, 119, 120, 121, 122, 123, 187, 205, 206,
                             207, 213, 192, 214, 210, 211, 212, 135, 138, 170, 169,
                             140, 171, 175})
# Right cheek
_RIGHT_CHEEK_IDX = frozenset({345, 346, 347, 348, 349, 350, 351, 352, 411, 425, 426,
                              427, 433, 416, 434, 430, 431, 432, 364, 367, 395, 394,
                              369, 396, 400})
# Left nasolabial
_LEFT_NASOLABIAL_IDX = frozenset({49, 48, 131, 134, 198, 126, 209, 217, 174, 196})
# Right nasolabial
_RIGHT_NASOLABIAL_IDX = frozenset({279, 278, 360, 363, 420, 355, 429, 437, 399, 419})
# Chin/jaw
_CHIN_JAW_IDX = frozenset({152, 377, 378, 379, 400, 148, 176, 149, 150, 136, 172,
                           58, 132, 93, 234, 127, 162})
# Mentolabial (between lower lip and chin)
_MENTOLABIAL_IDX = frozenset({152, 377, 400, 148, 176, 149, 150, 17, 84, 181, 314})


def _build_landmark_region_map():
    """Build landmark index -> expanded region name mapping.

    Priority order matters: more specific regions override broader ones.
    """
    region_map = {}
    # Broad regions first (will be overridden by specific)
    for idx in _FOREHEAD_IDX:
        region_map[idx] = "forehead"
    for idx in _LEFT_CHEEK_IDX:
        region_map[idx] = "left_cheek"
    for idx in _RIGHT_CHEEK_IDX:
        region_map[idx] = "right_cheek"
    for idx in _CHIN_JAW_IDX:
        region_map[idx] = "chin_jaw"
    for idx in _NOSE_IDX:
        region_map[idx] = "nose"
    for idx in _LIPS_IDX:
        region_map[idx] = "mouth"
    # Specific regions override broad
    for idx in _LEFT_TEMPLE_IDX:
        region_map[idx] = "left_temple"
    for idx in _RIGHT_TEMPLE_IDX:
        region_map[idx] = "right_temple"
    for idx in _GLABELLA_IDX:
        region_map[idx] = "glabella"
    for idx in _NOSE_BRIDGE_IDX:
        region_map[idx] = "nose_bridge"
    for idx in _PHILTRUM_IDX:
        region_map[idx] = "philtrum"
    for idx in _LEFT_NASOLABIAL_IDX:
        region_map[idx] = "left_nasolabial"
    for idx in _RIGHT_NASOLABIAL_IDX:
        region_map[idx] = "right_nasolabial"
    for idx in _MENTOLABIAL_IDX:
        region_map[idx] = "mentolabial"
    for idx in _LEFT_EYE_IDX | _LEFT_BROW_IDX:
        region_map[idx] = "left_periocular"
    for idx in _RIGHT_EYE_IDX | _RIGHT_BROW_IDX:
        region_map[idx] = "right_periocular"
    return region_map


_LANDMARK_REGION_MAP = _build_landmark_region_map()


# ── Pure math helpers ──

def _triangle_area_2d(a, b, c):
    """Compute area of triangle with vertices a, b, c (each is (x, y) tuple).

    Uses the cross-product formula: area = 0.5 * |det([[bx-ax, cx-ax],[by-ay, cy-ay]])|
    """
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def _solve_barycentric_2d(p, a, b, c):
    """Compute barycentric coordinates (u, v, w) of point P in triangle ABC.

    Uses the standard linear algebra method:
      v0 = C - A, v1 = B - A, v2 = P - A
      u = (dot11*dot02 - dot01*dot12) / denom
      v = (dot00*dot12 - dot01*dot02) / denom
      w = 1 - u - v

    Returns (u, v, w) or None if triangle is degenerate.
    The coordinates satisfy: P ≈ u*A + v*B + w*C when (u,v,w) sum to 1.
    If P is inside the triangle, all of u,v,w are in [0,1].
    """
    v0 = (c[0] - a[0], c[1] - a[1])
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (p[0] - a[0], p[1] - a[1])

    dot00 = v0[0] * v0[0] + v0[1] * v0[1]
    dot01 = v0[0] * v1[0] + v0[1] * v1[1]
    dot02 = v0[0] * v2[0] + v0[1] * v2[1]
    dot11 = v1[0] * v1[0] + v1[1] * v1[1]
    dot12 = v1[0] * v2[0] + v1[1] * v2[1]

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return None  # Degenerate triangle

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1.0 - u - v

    return (u, v, w)


def _find_nearest_3_landmarks(cx, cy, landmarks, w, h):
    """Find the 3 nearest MediaPipe landmarks to point (cx, cy) in normalized coords.

    Args:
        cx, cy: Mark centroid in normalized [0,1] coordinates.
        landmarks: MediaPipe landmark list (must have .x, .y attributes).
        w, h: Image width and height (used only for landmark access pattern).

    Returns:
        List of (index, normalized_x, normalized_y, distance) sorted by distance,
        length 3. Returns fewer if landmarks list is too short.
    """
    if landmarks is None:
        return []

    distances = []
    for i, lm in enumerate(landmarks):
        lx = lm.x
        ly = lm.y
        d = math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)
        distances.append((i, lx, ly, d))

    distances.sort(key=lambda x: x[3])
    return distances[:3]


def compute_expanded_mesh_region(landmark_indices):
    """Map landmark indices to expanded ~15-region anatomical region.

    Uses majority vote from the landmark indices. Falls back to "unknown"
    if no landmarks are in the region map.

    This is TELEMETRY ONLY — does not replace the 8-region face_region
    used in production matching.

    Args:
        landmark_indices: List of landmark index integers.

    Returns:
        String region name from the expanded ~15-region map.
    """
    if not landmark_indices:
        return "unknown"

    regions = []
    for idx in landmark_indices:
        region = _LANDMARK_REGION_MAP.get(idx)
        if region is not None:
            regions.append(region)

    if not regions:
        return "unknown"

    # Majority vote
    counter = Counter(regions)
    return counter.most_common(1)[0][0]


def compute_mesh_confidence(bary_coords, triangle_area, nearest_distance):
    """Compute confidence score for the barycentric coordinate mapping.

    Factors:
    1. Whether the point is inside the triangle (all bary coords in [0,1])
    2. Triangle area (very small = low confidence, degenerate)
    3. Distance from mark to nearest landmark (closer = more reliable)

    Args:
        bary_coords: Tuple (u, v, w) or None.
        triangle_area: Float area of the triangle in normalized coords.
        nearest_distance: Float distance from mark to nearest landmark.

    Returns:
        Float in [0, 1]. Higher = more reliable.
    """
    if bary_coords is None:
        return 0.0

    u, v, w = bary_coords
    score = 1.0

    # Factor 1: Inside triangle check
    inside = (u >= -0.05 and v >= -0.05 and w >= -0.05 and
              u <= 1.05 and v <= 1.05 and w <= 1.05)
    if not inside:
        score *= 0.5  # Point is outside the triangle — less reliable

    # Factor 2: Triangle area (in normalized coords, typical face triangle ~0.001-0.01)
    if triangle_area < _MIN_TRIANGLE_AREA:
        return 0.0  # Degenerate
    # Larger triangles = landmarks are farther apart = less precise mapping
    # Sweet spot is ~0.0005 to 0.005
    if triangle_area > 0.01:
        score *= 0.7
    elif triangle_area < 0.0001:
        score *= 0.8  # Very tight triangle, might be unstable

    # Factor 3: Nearest landmark distance
    # In normalized coords, 0.01 is close, 0.10 is far
    if nearest_distance > 0.10:
        score *= 0.5
    elif nearest_distance > 0.05:
        score *= 0.75

    return max(0.0, min(1.0, score))


def compute_barycentric_position(mark_centroid, landmarks, image_shape):
    """Compute the full barycentric/anatomical position for a single mark.

    Args:
        mark_centroid: Tuple (cx, cy) in NORMALIZED [0,1] coordinates.
        landmarks: MediaPipe landmark list with .x, .y attributes, or None.
        image_shape: Tuple (height, width) of the image.

    Returns:
        Dict with all barycentric position fields, or a fallback dict
        if landmarks are unavailable or triangle is degenerate.
    """
    if landmarks is None or len(landmarks) < 3:
        return {
            "barycentric_u": None,
            "barycentric_v": None,
            "barycentric_w": None,
            "barycentric_mode": BARYCENTRIC_MODE_FALLBACK,
            "triangle_source": TRIANGLE_SOURCE,
            "mesh_triangle_id": None,
            "nearest_landmark_indices": [],
            "nearest_landmark_distance": None,
            "mesh_region": "unknown",
            "mesh_confidence": 0.0,
            "coordinate_system_version": COORDINATE_SYSTEM_VERSION,
        }

    cx, cy = mark_centroid
    h, w = image_shape[:2]

    # Find 3 nearest landmarks
    nearest_3 = _find_nearest_3_landmarks(cx, cy, landmarks, w, h)

    if len(nearest_3) < 3:
        return {
            "barycentric_u": None,
            "barycentric_v": None,
            "barycentric_w": None,
            "barycentric_mode": BARYCENTRIC_MODE_FALLBACK,
            "triangle_source": TRIANGLE_SOURCE,
            "mesh_triangle_id": None,
            "nearest_landmark_indices": [n[0] for n in nearest_3],
            "nearest_landmark_distance": nearest_3[0][3] if nearest_3 else None,
            "mesh_region": compute_expanded_mesh_region([n[0] for n in nearest_3]),
            "mesh_confidence": 0.0,
            "coordinate_system_version": COORDINATE_SYSTEM_VERSION,
        }

    idx_a, ax, ay, _ = nearest_3[0]
    idx_b, bx, by, _ = nearest_3[1]
    idx_c, cx_l, cy_l, _ = nearest_3[2]
    nearest_dist = nearest_3[0][3]

    landmark_indices = [idx_a, idx_b, idx_c]
    triangle_id = f"tri_{idx_a}_{idx_b}_{idx_c}"

    # Compute triangle area
    tri_area = _triangle_area_2d((ax, ay), (bx, by), (cx_l, cy_l))

    # Compute barycentric coordinates
    bary = _solve_barycentric_2d((cx, cy), (ax, ay), (bx, by), (cx_l, cy_l))

    if bary is None or tri_area < _MIN_TRIANGLE_AREA:
        # Degenerate triangle — fallback
        return {
            "barycentric_u": None,
            "barycentric_v": None,
            "barycentric_w": None,
            "barycentric_mode": BARYCENTRIC_MODE_FALLBACK,
            "triangle_source": TRIANGLE_SOURCE,
            "mesh_triangle_id": triangle_id,
            "nearest_landmark_indices": landmark_indices,
            "nearest_landmark_distance": float(nearest_dist),
            "mesh_region": compute_expanded_mesh_region(landmark_indices),
            "mesh_confidence": 0.0,
            "coordinate_system_version": COORDINATE_SYSTEM_VERSION,
        }

    u, v, w_val = bary
    confidence = compute_mesh_confidence(bary, tri_area, nearest_dist)

    return {
        "barycentric_u": float(u),
        "barycentric_v": float(v),
        "barycentric_w": float(w_val),
        "barycentric_mode": BARYCENTRIC_MODE_2D,
        "triangle_source": TRIANGLE_SOURCE,
        "mesh_triangle_id": triangle_id,
        "nearest_landmark_indices": landmark_indices,
        "nearest_landmark_distance": float(nearest_dist),
        "mesh_region": compute_expanded_mesh_region(landmark_indices),
        "mesh_confidence": float(confidence),
        "coordinate_system_version": COORDINATE_SYSTEM_VERSION,
    }


def _normalize_anchor_set(indices):
    """Return a frozenset of landmark indices for set-based comparison.

    This normalizes the anchor set so that two triangles with the same
    landmarks in different order can be recognized as compatible.
    """
    if not indices:
        return frozenset()
    return frozenset(int(i) for i in indices)


def _align_bary_coords_to_sorted_anchors(bary_coords, anchor_indices, target_sorted_indices):
    """Re-order barycentric coordinates to match a canonical (sorted) vertex order.

    Barycentric coordinates (u, v, w) correspond to vertices (A, B, C) in the
    order stored in nearest_landmark_indices. If two marks share the same anchor
    set but different vertex ordering, we must align them before comparison.

    Args:
        bary_coords: Tuple (u, v, w) as stored.
        anchor_indices: List [idx_a, idx_b, idx_c] corresponding to (u, v, w).
        target_sorted_indices: The canonical sorted list of indices.

    Returns:
        Tuple (u', v', w') aligned to target_sorted_indices order,
        or None if alignment fails.
    """
    if bary_coords is None or len(anchor_indices) != 3 or len(target_sorted_indices) != 3:
        return None

    # Build mapping from original vertex position to bary coord
    try:
        # anchor_indices[i] was vertex i, with bary_coords[i]
        # We need to reorder so index target_sorted_indices[j] maps to position j
        index_to_bary = {}
        for i, idx in enumerate(anchor_indices):
            index_to_bary[int(idx)] = bary_coords[i]

        aligned = tuple(index_to_bary[int(t)] for t in target_sorted_indices)
        return aligned
    except (KeyError, TypeError):
        return None


# Comparison mode constants
BARY_COMPARE_SAME_TRIANGLE = "same_triangle"
BARY_COMPARE_SAME_ANCHOR_SET = "same_anchor_set"
BARY_COMPARE_DIFFERENT_TRIANGLE = "different_triangle_fallback"
BARY_COMPARE_UNAVAILABLE = "unavailable"


def compute_barycentric_distance(anat_a, anat_b):
    """Compute distance between two marks in barycentric space.

    CRITICAL MATH RULE: Barycentric coordinates are only directly comparable
    when both marks share the same or compatible mesh triangle (anchor set).

    Comparison modes:
    - same_triangle: mesh_triangle_id matches exactly → direct u/v/w comparison
    - same_anchor_set: same landmark indices (possibly different order) → aligned comparison
    - different_triangle_fallback: different anchors → NOT comparable, returns unavailable
    - unavailable: missing data, fallback mode, or low confidence

    Args:
        anat_a: anatomical_position dict from mark A.
        anat_b: anatomical_position dict from mark B.

    Returns:
        Tuple (distance, available, comparison_mode, telemetry_dict).
        distance is float or None.
        available is True only when comparison is mathematically valid.
        comparison_mode is one of the BARY_COMPARE_* constants.
        telemetry_dict contains diagnostic fields for reporting.
    """
    empty_telemetry = {
        "barycentric_comparison_mode": BARY_COMPARE_UNAVAILABLE,
        "barycentric_triangle_match": False,
        "barycentric_anchor_overlap_count": 0,
        "barycentric_distance_available": False,
        "mesh_triangle_gallery": None,
        "mesh_triangle_probe": None,
        "fallback_mesh_telemetry_available": False,
    }

    if anat_a is None or anat_b is None:
        return None, False, BARY_COMPARE_UNAVAILABLE, empty_telemetry

    mode_a = anat_a.get("barycentric_mode")
    mode_b = anat_b.get("barycentric_mode")

    if mode_a != BARYCENTRIC_MODE_2D or mode_b != BARYCENTRIC_MODE_2D:
        return None, False, BARY_COMPARE_UNAVAILABLE, empty_telemetry

    conf_a = anat_a.get("mesh_confidence", 0.0)
    conf_b = anat_b.get("mesh_confidence", 0.0)

    if conf_a < _BARY_TELEMETRY_MIN_CONFIDENCE or conf_b < _BARY_TELEMETRY_MIN_CONFIDENCE:
        return None, False, BARY_COMPARE_UNAVAILABLE, empty_telemetry

    ua = anat_a.get("barycentric_u")
    va = anat_a.get("barycentric_v")
    wa = anat_a.get("barycentric_w")
    ub = anat_b.get("barycentric_u")
    vb = anat_b.get("barycentric_v")
    wb = anat_b.get("barycentric_w")

    if any(x is None for x in [ua, va, wa, ub, vb, wb]):
        return None, False, BARY_COMPARE_UNAVAILABLE, empty_telemetry

    tri_id_a = anat_a.get("mesh_triangle_id")
    tri_id_b = anat_b.get("mesh_triangle_id")
    anchors_a = anat_a.get("nearest_landmark_indices", [])
    anchors_b = anat_b.get("nearest_landmark_indices", [])

    anchor_set_a = _normalize_anchor_set(anchors_a)
    anchor_set_b = _normalize_anchor_set(anchors_b)
    anchor_overlap = len(anchor_set_a & anchor_set_b)

    base_telemetry = {
        "barycentric_triangle_match": False,
        "barycentric_anchor_overlap_count": anchor_overlap,
        "barycentric_distance_available": False,
        "mesh_triangle_gallery": tri_id_a,
        "mesh_triangle_probe": tri_id_b,
        "fallback_mesh_telemetry_available": False,
    }

    # ── Case A: Same triangle (exact match) ──
    if tri_id_a is not None and tri_id_b is not None and tri_id_a == tri_id_b:
        dist = math.sqrt((ua - ub) ** 2 + (va - vb) ** 2 + (wa - wb) ** 2)
        base_telemetry["barycentric_comparison_mode"] = BARY_COMPARE_SAME_TRIANGLE
        base_telemetry["barycentric_triangle_match"] = True
        base_telemetry["barycentric_distance_available"] = True
        return dist, True, BARY_COMPARE_SAME_TRIANGLE, base_telemetry

    # ── Case B: Same anchor set, different vertex ordering ──
    if anchor_overlap == 3:
        target_sorted_indices = sorted(anchor_set_a)
        
        bary_a = (ua, va, wa)
        aligned_a = _align_bary_coords_to_sorted_anchors(bary_a, anchors_a, target_sorted_indices)
        
        bary_b = (ub, vb, wb)
        aligned_b = _align_bary_coords_to_sorted_anchors(bary_b, anchors_b, target_sorted_indices)

        if aligned_a is not None and aligned_b is not None:
            dist = math.sqrt((aligned_a[0] - aligned_b[0]) ** 2 + 
                             (aligned_a[1] - aligned_b[1]) ** 2 + 
                             (aligned_a[2] - aligned_b[2]) ** 2)
            base_telemetry["barycentric_comparison_mode"] = BARY_COMPARE_SAME_ANCHOR_SET
            base_telemetry["barycentric_distance_available"] = True
            return dist, True, BARY_COMPARE_SAME_ANCHOR_SET, base_telemetry

    # ── Case C: Different triangle — NOT comparable ──
    base_telemetry["barycentric_comparison_mode"] = BARY_COMPARE_DIFFERENT_TRIANGLE
    base_telemetry["barycentric_distance_available"] = False
    
    if anchor_overlap > 0:
        base_telemetry["fallback_mesh_telemetry_available"] = True
        base_telemetry["nearest_landmark_distance_delta"] = round(abs(anat_a.get("nearest_landmark_distance", 0) - anat_b.get("nearest_landmark_distance", 0)), 6)
        
    return None, False, BARY_COMPARE_DIFFERENT_TRIANGLE, base_telemetry


def compute_constellation_telemetry(scoring_correspondences, suppressed_correspondences,
                                     gallery_marks, probe_marks,
                                     cluster_domination_score=1.0):
    """Compute constellation quality telemetry from matched correspondences.

    This is DISPLAY/RESEARCH ONLY. It MUST NOT change final score.

    Args:
        scoring_correspondences: List of scoring-eligible correspondence dicts.
        suppressed_correspondences: List of suppressed (generic) correspondence dicts.
        gallery_marks: Full list of gallery mark descriptors.
        probe_marks: Full list of probe mark descriptors.
        cluster_domination_score: Float from cluster penalty calculation (1.0 = no domination).

    Returns:
        Dict with all constellation telemetry fields.
    """
    n_scoring = len(scoring_correspondences)
    n_suppressed = len(suppressed_correspondences)

    # Node counts
    distinctive_node_count = n_scoring
    generic_node_count = n_suppressed
    scoring_eligible_node_count = n_scoring

    # Region and type diversity (from scoring correspondences only)
    scoring_regions = [c.get("face_region", "unknown") for c in scoring_correspondences]
    scoring_types = [c.get("mark_type", "unknown") for c in scoring_correspondences]
    region_diversity_count = len(set(scoring_regions))
    mark_type_diversity_count = len(set(scoring_types))

    # Graph metrics
    graph_edge_count = n_scoring * (n_scoring - 1) // 2 if n_scoring >= 2 else 0

    # Pairwise distance consistency
    pairwise_errors = []
    if n_scoring >= 2:
        for (c1, c2) in combinations(scoring_correspondences, 2):
            # Gallery-side pairwise distance
            gc1 = c1.get("gallery_centroid", [0, 0])
            gc2 = c2.get("gallery_centroid", [0, 0])
            d_gallery = math.sqrt((gc1[0] - gc2[0]) ** 2 + (gc1[1] - gc2[1]) ** 2)

            # Probe-side pairwise distance
            pc1 = c1.get("probe_centroid", [0, 0])
            pc2 = c2.get("probe_centroid", [0, 0])
            d_probe = math.sqrt((pc1[0] - pc2[0]) ** 2 + (pc1[1] - pc2[1]) ** 2)

            pairwise_errors.append(abs(d_gallery - d_probe))

    if pairwise_errors:
        avg_pairwise_error = sum(pairwise_errors) / len(pairwise_errors)
        sorted_errors = sorted(pairwise_errors)
        n_err = len(sorted_errors)
        if n_err % 2 == 1:
            median_pairwise_error = sorted_errors[n_err // 2]
        else:
            median_pairwise_error = (sorted_errors[n_err // 2 - 1] + sorted_errors[n_err // 2]) / 2.0
        max_pairwise_error = max(pairwise_errors)
    else:
        avg_pairwise_error = 0.0
        median_pairwise_error = 0.0
        max_pairwise_error = 0.0

    # Graph edge consistency score: 1 - mean_error / max_possible
    # max_possible in normalized coords is sqrt(2) ≈ 1.414
    max_possible = math.sqrt(2.0)
    graph_edge_consistency_score = max(0.0, 1.0 - avg_pairwise_error / max_possible) if max_possible > 0 else 0.0

    # Cluster domination (already computed upstream, passed in)
    cluster_dom_fraction = 1.0 - cluster_domination_score  # higher = more dominated

    # Constellation quality score
    if scoring_eligible_node_count == 0:
        constellation_quality_score = 0.0
    else:
        node_factor = min(scoring_eligible_node_count / 3.0, 1.0)
        region_factor = min(region_diversity_count / 3.0, 1.0)
        type_factor = min(mark_type_diversity_count / 2.0, 1.0)
        anti_cluster = 1.0 - cluster_dom_fraction

        constellation_quality_score = (
            0.25 * node_factor +
            0.25 * region_factor +
            0.25 * graph_edge_consistency_score +
            0.15 * type_factor +
            0.10 * anti_cluster
        )

    # Quality label
    if constellation_quality_score < 0.15:
        quality_label = "NONE"
    elif constellation_quality_score < 0.40:
        quality_label = "WEAK"
    elif constellation_quality_score < 0.70:
        quality_label = "MODERATE"
    else:
        quality_label = "STRONG_REVIEW_SUPPORT"

    return {
        "distinctive_node_count": distinctive_node_count,
        "generic_node_count": generic_node_count,
        "scoring_eligible_node_count": scoring_eligible_node_count,
        "region_diversity_count": region_diversity_count,
        "mark_type_diversity_count": mark_type_diversity_count,
        "graph_edge_count": graph_edge_count,
        "average_pairwise_distance_error": round(avg_pairwise_error, 6),
        "median_pairwise_distance_error": round(median_pairwise_error, 6),
        "max_pairwise_distance_error": round(max_pairwise_error, 6),
        "graph_edge_consistency_score": round(graph_edge_consistency_score, 4),
        "cluster_domination_score": round(cluster_domination_score, 4),
        "constellation_quality_score": round(constellation_quality_score, 4),
        "constellation_quality_label": quality_label,
        "telemetry_only": True,
        "does_not_affect_scoring": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Regional Canonical Coordinate System v2.0.0
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each region is defined by an origin landmark + u-axis landmark + v-axis landmark.
# This creates a local coordinate basis anchored to anatomically stable points.
#
# For a mark at centroid P:
#   O = origin anchor position
#   U_vec = (u_axis_anchor - O)  → defines the positive-u direction
#   V_vec = (v_axis_anchor - O)  → defines the positive-v direction
#   P_rel = P - O
#   region_u = dot(P_rel, U_vec) / dot(U_vec, U_vec)
#   region_v = dot(P_rel, V_vec) / dot(V_vec, V_vec)
#
# Raw region_u/region_v are NEVER silently clamped.
# If mark falls outside the region frame, region_coordinate_in_bounds = False.
#
# TELEMETRY ONLY — does NOT affect scoring.
# ═══════════════════════════════════════════════════════════════════════════════

_REGION_ANCHOR_TABLE = {
    # ── Forehead ──
    "forehead": {
        "region_name": "forehead",
        "origin_anchor": 10,      # Top-center of forehead
        "u_axis_anchor": 338,     # Right forehead (defines +u = rightward)
        "v_axis_anchor": 151,     # Below center (defines +v = downward)
        "optional_boundary_anchors": [109, 284],
        "allowed_region_landmarks": list(_FOREHEAD_IDX),
        "notes": "Upper face, above brow line. Origin at hairline center.",
    },
    # ── Left periocular ──
    "left_periocular": {
        "region_name": "left_periocular",
        "origin_anchor": 70,      # Left brow medial
        "u_axis_anchor": 46,      # Left brow lateral
        "v_axis_anchor": 145,     # Left eye lower
        "optional_boundary_anchors": [33, 133],
        "allowed_region_landmarks": list(_LEFT_EYE_IDX | _LEFT_BROW_IDX),
        "notes": "Left eye + brow region.",
    },
    # ── Right periocular ──
    "right_periocular": {
        "region_name": "right_periocular",
        "origin_anchor": 300,     # Right brow medial
        "u_axis_anchor": 276,     # Right brow lateral
        "v_axis_anchor": 374,     # Right eye lower
        "optional_boundary_anchors": [362, 263],
        "allowed_region_landmarks": list(_RIGHT_EYE_IDX | _RIGHT_BROW_IDX),
        "notes": "Right eye + brow region.",
    },
    # ── Glabella ──
    "glabella": {
        "region_name": "glabella",
        "origin_anchor": 9,       # Center of glabella
        "u_axis_anchor": 337,     # Right side
        "v_axis_anchor": 168,     # Nose bridge top
        "optional_boundary_anchors": [108, 299],
        "allowed_region_landmarks": list(_GLABELLA_IDX),
        "notes": "Between the brows. Small region, high stability.",
    },
    # ── Nose bridge ──
    "nose_bridge": {
        "region_name": "nose_bridge",
        "origin_anchor": 6,       # Upper nose bridge
        "u_axis_anchor": 197,     # Nose right side
        "v_axis_anchor": 4,       # Nose tip direction
        "optional_boundary_anchors": [168, 195],
        "allowed_region_landmarks": list(_NOSE_BRIDGE_IDX),
        "notes": "Upper nose structure. Narrow region.",
    },
    # ── Nose (full) ──
    "nose": {
        "region_name": "nose",
        "origin_anchor": 6,       # Upper nose bridge center
        "u_axis_anchor": 327,     # Right alar
        "v_axis_anchor": 4,       # Nose tip
        "optional_boundary_anchors": [98, 2],
        "allowed_region_landmarks": list(_NOSE_IDX),
        "notes": "Full nose. Overlaps nose_bridge; nose_bridge takes priority.",
    },
    # ── Left cheek ──
    "left_cheek": {
        "region_name": "left_cheek",
        "origin_anchor": 116,     # Upper left cheek (below eye)
        "u_axis_anchor": 123,     # Mid-left cheek lateral
        "v_axis_anchor": 187,     # Lower left cheek
        "optional_boundary_anchors": [205, 135],
        "allowed_region_landmarks": list(_LEFT_CHEEK_IDX),
        "notes": "Left cheek body. Largest region. Key for mole mapping.",
    },
    # ── Right cheek ──
    "right_cheek": {
        "region_name": "right_cheek",
        "origin_anchor": 345,     # Upper right cheek (below eye)
        "u_axis_anchor": 352,     # Mid-right cheek lateral
        "v_axis_anchor": 411,     # Lower right cheek
        "optional_boundary_anchors": [425, 364],
        "allowed_region_landmarks": list(_RIGHT_CHEEK_IDX),
        "notes": "Right cheek body. Mirror of left_cheek.",
    },
    # ── Left nasolabial ──
    "left_nasolabial": {
        "region_name": "left_nasolabial",
        "origin_anchor": 49,      # Upper nasolabial
        "u_axis_anchor": 131,     # Lateral
        "v_axis_anchor": 198,     # Lower nasolabial
        "optional_boundary_anchors": [48, 134],
        "allowed_region_landmarks": list(_LEFT_NASOLABIAL_IDX),
        "notes": "Left nasolabial fold region.",
    },
    # ── Right nasolabial ──
    "right_nasolabial": {
        "region_name": "right_nasolabial",
        "origin_anchor": 279,     # Upper nasolabial
        "u_axis_anchor": 360,     # Lateral
        "v_axis_anchor": 420,     # Lower nasolabial
        "optional_boundary_anchors": [278, 363],
        "allowed_region_landmarks": list(_RIGHT_NASOLABIAL_IDX),
        "notes": "Right nasolabial fold region.",
    },
    # ── Philtrum ──
    "philtrum": {
        "region_name": "philtrum",
        "origin_anchor": 164,     # Upper philtrum (below nose)
        "u_axis_anchor": 186,     # Left edge
        "v_axis_anchor": 18,      # Lower lip edge
        "optional_boundary_anchors": [167, 165],
        "allowed_region_landmarks": list(_PHILTRUM_IDX),
        "notes": "Between nose and upper lip.",
    },
    # ── Mouth (lips) ──
    "mouth": {
        "region_name": "mouth",
        "origin_anchor": 0,       # Upper lip center
        "u_axis_anchor": 291,     # Right mouth corner
        "v_axis_anchor": 17,      # Lower lip center
        "optional_boundary_anchors": [61, 78],
        "allowed_region_landmarks": list(_LIPS_IDX),
        "notes": "Lip region. Low mark density expected.",
    },
    # ── Left temple ──
    "left_temple": {
        "region_name": "left_temple",
        "origin_anchor": 132,     # Upper temple
        "u_axis_anchor": 58,      # Lateral temple
        "v_axis_anchor": 172,     # Lower temple
        "optional_boundary_anchors": [136, 150],
        "allowed_region_landmarks": list(_LEFT_TEMPLE_IDX),
        "notes": "Left temporal region. Sparse landmarks.",
    },
    # ── Right temple ──
    "right_temple": {
        "region_name": "right_temple",
        "origin_anchor": 389,     # Upper temple
        "u_axis_anchor": 356,     # Lateral temple
        "v_axis_anchor": 454,     # Lower temple
        "optional_boundary_anchors": [323, 361],
        "allowed_region_landmarks": list(_RIGHT_TEMPLE_IDX),
        "notes": "Right temporal region. Sparse landmarks.",
    },
    # ── Chin / jaw ──
    "chin_jaw": {
        "region_name": "chin_jaw",
        "origin_anchor": 152,     # Chin tip
        "u_axis_anchor": 377,     # Right jaw
        "v_axis_anchor": 400,     # Upper chin
        "optional_boundary_anchors": [148, 176],
        "allowed_region_landmarks": list(_CHIN_JAW_IDX),
        "notes": "Chin and jawline.",
    },
    # ── Mentolabial ──
    "mentolabial": {
        "region_name": "mentolabial",
        "origin_anchor": 17,      # Lower lip center
        "u_axis_anchor": 314,     # Right lower lip edge
        "v_axis_anchor": 152,     # Chin
        "optional_boundary_anchors": [84, 181],
        "allowed_region_landmarks": list(_MENTOLABIAL_IDX),
        "notes": "Between lower lip and chin.",
    },
}

# Fallback for marks whose region is "unknown" or not in the table
_REGION_ANCHOR_FALLBACK = {
    "region_name": "unknown",
    "origin_anchor": 1,       # Nose tip
    "u_axis_anchor": 454,     # Right ear
    "v_axis_anchor": 152,     # Chin
    "optional_boundary_anchors": [],
    "allowed_region_landmarks": [],
    "notes": "Global fallback using whole-face anchors. Low precision.",
}


def _get_landmark_xy(landmarks, idx):
    """Get (x, y) normalized coordinates from a MediaPipe landmark by index.

    Returns (x, y) float tuple, or None if the index is out of range or
    the landmark list is missing.
    """
    if landmarks is None or idx < 0 or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    if hasattr(lm, "x"):
        return (float(lm.x), float(lm.y))
    if isinstance(lm, (list, tuple)) and len(lm) >= 2:
        return (float(lm[0]), float(lm[1]))
    return None


def compute_regional_canonical_position(mark_centroid, landmarks, image_shape):
    """Compute the regional canonical coordinate position for a single mark.

    This maps a mark into a region-local (u, v) coordinate frame defined by
    three anatomically stable anchor landmarks: origin, u-axis, v-axis.

    TELEMETRY ONLY — does NOT affect production scoring.

    Args:
        mark_centroid: Tuple (cx, cy) in NORMALIZED [0,1] coordinates.
        landmarks: MediaPipe landmark list with .x, .y attributes, or None.
        image_shape: Tuple (height, width) of the image.

    Returns:
        Dict with regional canonical coordinate fields, or a fallback dict
        if landmarks are unavailable.
    """
    empty_result = {
        "coordinate_model": "regional_canonical",
        "coordinate_system_version": REGIONAL_COORDINATE_VERSION,
        "canonical_region": "unknown",
        "canonical_region_subcell": None,
        "region_u": None,
        "region_v": None,
        "region_coordinate_in_bounds": False,
        "distance_to_primary_anchor": None,
        "distance_to_secondary_anchor": None,
        "anchor_pair_id": None,
        "anchor_distances": [],
        "region_confidence": 0.0,
        "coordinate_quality": 0.0,
        "telemetry_only": True,
        "does_not_affect_scoring": True,
    }

    if landmarks is None or len(landmarks) < 3:
        return empty_result

    cx, cy = mark_centroid

    # Step 1: Determine the expanded mesh region for this mark
    # Use existing barycentric nearest-3 to classify region via majority vote
    nearest_3 = _find_nearest_3_landmarks(cx, cy, landmarks, 1, 1)
    if len(nearest_3) < 3:
        return empty_result

    region_indices = [n[0] for n in nearest_3]
    canonical_region = compute_expanded_mesh_region(region_indices)

    # Step 2: Look up anchor table entry
    anchor_entry = _REGION_ANCHOR_TABLE.get(canonical_region, _REGION_ANCHOR_FALLBACK)
    origin_idx = anchor_entry["origin_anchor"]
    u_axis_idx = anchor_entry["u_axis_anchor"]
    v_axis_idx = anchor_entry["v_axis_anchor"]

    # Step 3: Get anchor positions
    origin_pt = _get_landmark_xy(landmarks, origin_idx)
    u_axis_pt = _get_landmark_xy(landmarks, u_axis_idx)
    v_axis_pt = _get_landmark_xy(landmarks, v_axis_idx)

    if origin_pt is None or u_axis_pt is None or v_axis_pt is None:
        return empty_result

    # Step 4: Compute coordinate basis vectors
    u_vec = (u_axis_pt[0] - origin_pt[0], u_axis_pt[1] - origin_pt[1])
    v_vec = (v_axis_pt[0] - origin_pt[0], v_axis_pt[1] - origin_pt[1])

    # Check for degenerate basis (parallel or zero-length vectors)
    u_dot_u = u_vec[0] * u_vec[0] + u_vec[1] * u_vec[1]
    v_dot_v = v_vec[0] * v_vec[0] + v_vec[1] * v_vec[1]

    if u_dot_u < 1e-12 or v_dot_v < 1e-12:
        return empty_result

    # Step 5: Project mark centroid into the coordinate frame
    p_rel = (cx - origin_pt[0], cy - origin_pt[1])
    region_u = (p_rel[0] * u_vec[0] + p_rel[1] * u_vec[1]) / u_dot_u
    region_v = (p_rel[0] * v_vec[0] + p_rel[1] * v_vec[1]) / v_dot_v

    # Step 6: Determine if in-bounds (raw values preserved, NEVER clamped)
    in_bounds = (0.0 <= region_u <= 1.0 and 0.0 <= region_v <= 1.0)

    # Step 7: Compute subcell (3x3 grid)
    if in_bounds:
        subcell_col = min(int(region_u * 3), 2)
        subcell_row = min(int(region_v * 3), 2)
        canonical_region_subcell = f"{canonical_region}_{subcell_row}_{subcell_col}"
    else:
        # Out of bounds — still compute a subcell for telemetry,
        # but clamp only for the subcell label, not the raw coordinates
        clamped_u = max(0.0, min(1.0, region_u))
        clamped_v = max(0.0, min(1.0, region_v))
        subcell_col = min(int(clamped_u * 3), 2)
        subcell_row = min(int(clamped_v * 3), 2)
        canonical_region_subcell = f"{canonical_region}_{subcell_row}_{subcell_col}_oob"

    # Step 8: Compute anchor distances
    dist_to_origin = math.sqrt((cx - origin_pt[0]) ** 2 + (cy - origin_pt[1]) ** 2)
    dist_to_u_axis = math.sqrt((cx - u_axis_pt[0]) ** 2 + (cy - u_axis_pt[1]) ** 2)
    dist_to_v_axis = math.sqrt((cx - v_axis_pt[0]) ** 2 + (cy - v_axis_pt[1]) ** 2)

    # Include optional boundary anchors
    anchor_distances = [
        round(dist_to_origin, 6),
        round(dist_to_u_axis, 6),
        round(dist_to_v_axis, 6),
    ]
    for bnd_idx in anchor_entry.get("optional_boundary_anchors", []):
        bnd_pt = _get_landmark_xy(landmarks, bnd_idx)
        if bnd_pt is not None:
            d = math.sqrt((cx - bnd_pt[0]) ** 2 + (cy - bnd_pt[1]) ** 2)
            anchor_distances.append(round(d, 6))

    anchor_pair_id = f"anc_{origin_idx}_{u_axis_idx}_{v_axis_idx}"

    # Step 9: Compute coordinate quality and region confidence
    # Factors: in-bounds, basis orthogonality, distance to origin
    cross_product = abs(u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0])
    basis_area = math.sqrt(u_dot_u * v_dot_v)
    orthogonality = cross_product / basis_area if basis_area > 1e-12 else 0.0

    quality = 1.0
    if not in_bounds:
        quality *= 0.5
    # Penalize near-parallel basis vectors (orthogonality close to 0)
    quality *= max(0.2, min(1.0, orthogonality / 0.5))
    # Penalize marks very far from origin
    if dist_to_origin > 0.3:
        quality *= 0.6
    elif dist_to_origin > 0.2:
        quality *= 0.8

    # Region confidence: based on whether nearest landmarks actually belong
    # to the assigned region
    allowed = set(anchor_entry.get("allowed_region_landmarks", []))
    if allowed:
        in_region_count = sum(1 for idx in region_indices if idx in allowed)
        region_confidence = in_region_count / len(region_indices)
    else:
        region_confidence = 0.3  # Unknown fallback

    coordinate_quality = round(quality * region_confidence, 4)

    return {
        "coordinate_model": "regional_canonical",
        "coordinate_system_version": REGIONAL_COORDINATE_VERSION,
        "canonical_region": canonical_region,
        "canonical_region_subcell": canonical_region_subcell,
        "region_u": round(float(region_u), 6),
        "region_v": round(float(region_v), 6),
        "region_coordinate_in_bounds": in_bounds,
        "distance_to_primary_anchor": round(dist_to_origin, 6),
        "distance_to_secondary_anchor": round(dist_to_u_axis, 6),
        "anchor_pair_id": anchor_pair_id,
        "anchor_distances": anchor_distances,
        "region_confidence": round(float(region_confidence), 4),
        "coordinate_quality": coordinate_quality,
        "telemetry_only": True,
        "does_not_affect_scoring": True,
    }


def compute_regional_coordinate_distance(pos_a, pos_b):
    """Compute distance between two marks in regional canonical coordinate space.

    CRITICAL RULE: Regional coordinates are only directly comparable when both
    marks share the same canonical_region. Cross-region comparison is NOT valid.

    TELEMETRY ONLY — does NOT affect production scoring.

    Args:
        pos_a: regional_position dict from mark A (output of compute_regional_canonical_position).
        pos_b: regional_position dict from mark B (output of compute_regional_canonical_position).

    Returns:
        Tuple (distance, available, telemetry_dict).
        distance is float or None.
        available is True only when comparison is valid.
        telemetry_dict contains diagnostic fields.
    """
    empty_telemetry = {
        "regional_comparison_available": False,
        "same_canonical_region": False,
        "same_region_subcell": False,
        "region_uv_distance": None,
        "anchor_distance_delta": None,
        "regional_coordinate_version": REGIONAL_COORDINATE_VERSION,
        "telemetry_only": True,
        "does_not_affect_scoring": True,
    }

    if pos_a is None or pos_b is None:
        return None, False, empty_telemetry

    region_a = pos_a.get("canonical_region", "unknown")
    region_b = pos_b.get("canonical_region", "unknown")
    same_region = (region_a == region_b and region_a != "unknown")

    subcell_a = pos_a.get("canonical_region_subcell")
    subcell_b = pos_b.get("canonical_region_subcell")
    same_subcell = (subcell_a is not None and subcell_b is not None and subcell_a == subcell_b)

    u_a = pos_a.get("region_u")
    v_a = pos_a.get("region_v")
    u_b = pos_b.get("region_u")
    v_b = pos_b.get("region_v")

    # Anchor distance delta (always computable if both have anchor_distances)
    anc_a = pos_a.get("anchor_distances", [])
    anc_b = pos_b.get("anchor_distances", [])
    anchor_distance_delta = None
    if anc_a and anc_b:
        min_len = min(len(anc_a), len(anc_b))
        if min_len > 0:
            delta_sq = sum((anc_a[i] - anc_b[i]) ** 2 for i in range(min_len))
            anchor_distance_delta = round(math.sqrt(delta_sq), 6)

    telemetry = {
        "regional_comparison_available": False,
        "same_canonical_region": same_region,
        "same_region_subcell": same_subcell,
        "region_uv_distance": None,
        "anchor_distance_delta": anchor_distance_delta,
        "regional_coordinate_version": REGIONAL_COORDINATE_VERSION,
        "telemetry_only": True,
        "does_not_affect_scoring": True,
    }

    if not same_region:
        return None, False, telemetry

    if any(x is None for x in [u_a, v_a, u_b, v_b]):
        return None, False, telemetry

    # Euclidean distance in (u, v) space
    dist = math.sqrt((u_a - u_b) ** 2 + (v_a - v_b) ** 2)

    telemetry["regional_comparison_available"] = True
    telemetry["region_uv_distance"] = round(dist, 6)

    return dist, True, telemetry
