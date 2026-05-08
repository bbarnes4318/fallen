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

MARK_ANATOMY_VERSION = "1.0.0"
COORDINATE_SYSTEM_VERSION = "1.0.0-phase1"
BARYCENTRIC_MODE_2D = "2d_mesh_approximation"
BARYCENTRIC_MODE_FALLBACK = "nearest_landmark_fallback"
TRIANGLE_SOURCE = "nearest_3_mediapipe_landmarks"

# ── Mesh confidence thresholds ──
_MIN_TRIANGLE_AREA = 1e-8  # Below this, triangle is degenerate
_MIN_BARY_CONFIDENCE = 0.70  # Below this, barycentric coords are not used for matching

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


def compute_barycentric_distance(anat_a, anat_b):
    """Compute distance between two marks in barycentric space.

    Only valid when both marks have mode = "2d_mesh_approximation"
    and mesh_confidence >= _MIN_BARY_CONFIDENCE.

    Args:
        anat_a: anatomical_position dict from mark A.
        anat_b: anatomical_position dict from mark B.

    Returns:
        Tuple (distance, available). distance is float or None.
        available is True only if both marks have valid barycentric coords.
    """
    if anat_a is None or anat_b is None:
        return None, False

    mode_a = anat_a.get("barycentric_mode")
    mode_b = anat_b.get("barycentric_mode")

    if mode_a != BARYCENTRIC_MODE_2D or mode_b != BARYCENTRIC_MODE_2D:
        return None, False

    conf_a = anat_a.get("mesh_confidence", 0.0)
    conf_b = anat_b.get("mesh_confidence", 0.0)

    if conf_a < _MIN_BARY_CONFIDENCE or conf_b < _MIN_BARY_CONFIDENCE:
        return None, False

    ua = anat_a.get("barycentric_u")
    va = anat_a.get("barycentric_v")
    wa = anat_a.get("barycentric_w")
    ub = anat_b.get("barycentric_u")
    vb = anat_b.get("barycentric_v")
    wb = anat_b.get("barycentric_w")

    if any(x is None for x in [ua, va, wa, ub, vb, wb]):
        return None, False

    # Euclidean distance in 3D barycentric space
    dist = math.sqrt((ua - ub) ** 2 + (va - vb) ** 2 + (wa - wb) ** 2)
    return dist, True


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
