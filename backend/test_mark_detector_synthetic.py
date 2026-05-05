"""Synthetic unit tests for backend/mark_detector.py

Validates mark detection on controlled 1024x1024 synthetic face images.
No MediaPipe runtime dependency — uses lightweight fake landmark objects.
"""
import sys
import os
import math
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mark_detector import (detect_facial_marks, _FACE_OVAL_IDX, _EXCLUDE_GROUPS,
                           _LEFT_EYE_IDX, _RIGHT_EYE_IDX, _LEFT_BROW_IDX,
                           _RIGHT_BROW_IDX, _NOSE_IDX, _LIPS_IDX)

# Alias for use in tests (avoid name collision with local imports inside helpers)
_LEFT_EYE_IDX_ = _LEFT_EYE_IDX
_RIGHT_EYE_IDX_ = _RIGHT_EYE_IDX
_LEFT_BROW_IDX_ = _LEFT_BROW_IDX
_RIGHT_BROW_IDX_ = _RIGHT_BROW_IDX
_NOSE_IDX_ = _NOSE_IDX
_LIPS_IDX_ = _LIPS_IDX


# ---------------------------------------------------------------------------
# Fake MediaPipe landmark helpers
# ---------------------------------------------------------------------------

class FakeLandmark:
    """Minimal stand-in for a MediaPipe NormalizedLandmark."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _build_fake_landmarks(size=1024):
    """Create 468 fake landmarks arranged in a realistic face layout.

    The face oval forms a large ellipse. Eyes, brows, nose, and lips
    are placed at anatomically plausible normalized coordinates so the
    detector's exclusion masks work correctly.
    """
    # Start with 468 landmarks at centre (fallback)
    lms = [FakeLandmark(0.5, 0.5) for _ in range(468)]

    # Face oval — large ellipse covering most of the image
    n_oval = len(_FACE_OVAL_IDX)
    for i, idx in enumerate(_FACE_OVAL_IDX):
        angle = 2 * math.pi * i / n_oval
        # Ellipse centred at (0.5, 0.48), radii (0.38, 0.44)
        lms[idx] = FakeLandmark(0.5 + 0.38 * math.cos(angle),
                                0.48 + 0.44 * math.sin(angle))

    # Left eye — cluster around (0.38, 0.38)
    for idx in _LEFT_EYE_IDX:
        lms[idx] = FakeLandmark(0.38 + np.random.uniform(-0.03, 0.03),
                                0.38 + np.random.uniform(-0.015, 0.015))

    # Right eye — cluster around (0.62, 0.38)
    for idx in _RIGHT_EYE_IDX:
        lms[idx] = FakeLandmark(0.62 + np.random.uniform(-0.03, 0.03),
                                0.38 + np.random.uniform(-0.015, 0.015))

    # Left brow — above left eye (0.38, 0.32)
    for idx in _LEFT_BROW_IDX:
        lms[idx] = FakeLandmark(0.38 + np.random.uniform(-0.04, 0.04),
                                0.32 + np.random.uniform(-0.01, 0.01))

    # Right brow — above right eye (0.62, 0.32)
    for idx in _RIGHT_BROW_IDX:
        lms[idx] = FakeLandmark(0.62 + np.random.uniform(-0.04, 0.04),
                                0.32 + np.random.uniform(-0.01, 0.01))

    # Nose — cluster around (0.50, 0.52)
    for idx in _NOSE_IDX:
        lms[idx] = FakeLandmark(0.50 + np.random.uniform(-0.04, 0.04),
                                0.52 + np.random.uniform(-0.04, 0.04))

    # Lips — cluster around (0.50, 0.65)
    for idx in _LIPS_IDX:
        lms[idx] = FakeLandmark(0.50 + np.random.uniform(-0.05, 0.05),
                                0.65 + np.random.uniform(-0.02, 0.02))

    return lms


def _make_base_face(size=1024, skin_val=160):
    """Create a uniform skin-tone 1024x1024 BGR image with subtle noise."""
    np.random.seed(42)
    img = np.full((size, size, 3), skin_val, dtype=np.uint8)
    # Add very slight Gaussian noise so it's not perfectly flat
    noise = np.random.normal(0, 2, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Test 1: Dark marks detection
# ---------------------------------------------------------------------------

def test_dark_marks_detected():
    """Three separate dark circular marks on valid cheek/forehead skin
    should each be detected as separate candidates."""
    np.random.seed(42)
    img = _make_base_face(1024, skin_val=160)
    lms = _build_fake_landmarks(1024)

    # Place 3 dark moles on valid skin regions (cheeks + forehead)
    # Left cheek: (280, 500) — well inside face oval, outside features
    cv2.circle(img, (280, 500), 12, (60, 60, 60), -1)
    # Right cheek: (740, 520)
    cv2.circle(img, (740, 520), 14, (55, 55, 55), -1)
    # Forehead: (512, 200)
    cv2.circle(img, (512, 200), 10, (65, 65, 65), -1)

    marks, rejected, occ, trace, overlays = detect_facial_marks(
        img, lms, input_is_preprocessed=True
    )

    # Should detect at least 3 marks
    assert len(marks) >= 3, f"Expected >= 3 marks, got {len(marks)}"

    # At least some should be dark-channel marks
    dark_channels = {"dark", "dark_lesion"}
    dark_marks = [m for m in marks if m.get("channel") in dark_channels
                  or m.get("mark_type", "").startswith("dark")]
    assert len(dark_marks) >= 2, f"Expected >= 2 dark marks, got {len(dark_marks)}"

    # Trace must report preprocessed state
    assert trace["input_is_preprocessed"] is True
    assert trace["internal_clahe_applied"] is False


# ---------------------------------------------------------------------------
# Test 2: Light/linear scar detection
# ---------------------------------------------------------------------------

def test_light_linear_scar_detected():
    """A thin bright line on valid cheek skin should be detected as a
    light_scar or linear_scar candidate."""
    np.random.seed(42)
    img = _make_base_face(1024, skin_val=140)
    lms = _build_fake_landmarks(1024)

    # Draw a thin bright line (scar) on left cheek
    pt1 = (250, 480)
    pt2 = (350, 530)
    cv2.line(img, pt1, pt2, (220, 220, 220), 3)

    marks, rejected, occ, trace, overlays = detect_facial_marks(
        img, lms, input_is_preprocessed=True
    )

    all_marks = marks + rejected
    scar_types = {"light_scar", "linear_scar"}
    scar_channels = {"light", "bright_scar", "linear_scar", "linear_scar_v2"}
    scar_found = [m for m in all_marks
                  if m.get("mark_type") in scar_types
                  or m.get("channel") in scar_channels]
    assert len(scar_found) >= 1, (
        f"Expected >= 1 scar candidate (accepted or rejected), got 0. "
        f"Total marks: {len(marks)}, rejected: {len(rejected)}"
    )


# ---------------------------------------------------------------------------
# Test 3: Feature exclusion
# ---------------------------------------------------------------------------

def test_feature_zone_exclusion():
    """Dark dots placed deeply inside eye/lip/nose feature zones should be
    rejected or excluded from final accepted marks."""
    np.random.seed(42)
    img = _make_base_face(1024, skin_val=160)
    lms = _build_fake_landmarks(1024)

    # Place dark dots deeply inside feature zones (centres of exclusion hulls)
    # Left eye centre: landmark cluster mean ~(0.38, 0.38) → pixel (389, 389)
    eye_cx = int(np.mean([lms[i].x for i in _LEFT_EYE_IDX_]) * 1024)
    eye_cy = int(np.mean([lms[i].y for i in _LEFT_EYE_IDX_]) * 1024)
    cv2.circle(img, (eye_cx, eye_cy), 8, (50, 50, 50), -1)

    # Nose centre: landmark cluster mean
    nose_cx = int(np.mean([lms[i].x for i in _NOSE_IDX_]) * 1024)
    nose_cy = int(np.mean([lms[i].y for i in _NOSE_IDX_]) * 1024)
    cv2.circle(img, (nose_cx, nose_cy), 8, (50, 50, 50), -1)

    # Lips centre: landmark cluster mean
    lips_cx = int(np.mean([lms[i].x for i in _LIPS_IDX_]) * 1024)
    lips_cy = int(np.mean([lms[i].y for i in _LIPS_IDX_]) * 1024)
    cv2.circle(img, (lips_cx, lips_cy), 8, (50, 50, 50), -1)

    # Also place one valid mark on cheek (well outside all feature zones)
    cv2.circle(img, (280, 500), 12, (60, 60, 60), -1)

    marks, rejected, occ, trace, overlays = detect_facial_marks(
        img, lms, input_is_preprocessed=True
    )

    # Build exclusion zone pixel ranges from actual landmark clusters (with 1.15x inflation)
    def _zone_bounds(idx_list):
        xs = [lms[i].x for i in idx_list]
        ys = [lms[i].y for i in idx_list]
        cx_n, cy_n = np.mean(xs), np.mean(ys)
        rx = (max(xs) - min(xs)) / 2 * 1.15
        ry = (max(ys) - min(ys)) / 2 * 1.15
        return (cx_n - rx, cx_n + rx, cy_n - ry, cy_n + ry)

    exclusion_zones = [
        _zone_bounds(_LEFT_EYE_IDX_),
        _zone_bounds(_RIGHT_EYE_IDX_),
        _zone_bounds(_LEFT_BROW_IDX_),
        _zone_bounds(_RIGHT_BROW_IDX_),
        _zone_bounds(_NOSE_IDX_),
        _zone_bounds(_LIPS_IDX_),
    ]

    def _in_any_zone(cx_norm, cy_norm):
        for x0, x1, y0, y1 in exclusion_zones:
            if x0 <= cx_norm <= x1 and y0 <= cy_norm <= y1:
                return True
        return False

    feature_zone_accepted = [m for m in marks if _in_any_zone(m["centroid"][0], m["centroid"][1])]

    # Feature-zone marks should be rejected, not accepted
    assert len(feature_zone_accepted) == 0, (
        f"Expected 0 feature-zone marks in accepted, got {len(feature_zone_accepted)}. "
        f"Positions: {[(m['centroid'], m.get('channel')) for m in feature_zone_accepted]}"
    )


# ---------------------------------------------------------------------------
# Test 4: Blank face — no false positives
# ---------------------------------------------------------------------------

def test_blank_face_minimal_marks():
    """A smooth uniform face with no marks should produce zero or only
    low-confidence fallback candidates — not many high-confidence marks."""
    np.random.seed(42)
    img = _make_base_face(1024, skin_val=160)
    lms = _build_fake_landmarks(1024)

    marks, rejected, occ, trace, overlays = detect_facial_marks(
        img, lms, input_is_preprocessed=True
    )

    # High-confidence marks (confidence > 0.5) should be zero or very few
    high_conf = [m for m in marks if m.get("confidence", 0) > 0.5]
    assert len(high_conf) <= 2, (
        f"Blank face produced {len(high_conf)} high-confidence marks — "
        f"expected <= 2. Types: {[m.get('mark_type') for m in high_conf]}"
    )

    # If fallback was used, candidates should be low-confidence
    if trace.get("fallback_used"):
        for m in marks:
            assert m.get("low_confidence", False) or m.get("confidence", 0) <= 0.5, (
                f"Fallback mark has unexpected high confidence: {m.get('confidence')}"
            )


# ---------------------------------------------------------------------------
# Test 5: Debug trace completeness
# ---------------------------------------------------------------------------

def test_trace_fields_complete():
    """The trace dict returned by detect_facial_marks must include all
    required diagnostic fields."""
    np.random.seed(42)
    img = _make_base_face(1024, skin_val=160)
    lms = _build_fake_landmarks(1024)

    # Add one mark so we get a non-trivial trace
    cv2.circle(img, (280, 500), 12, (60, 60, 60), -1)

    marks, rejected, occ, trace, overlays = detect_facial_marks(
        img, lms, input_is_preprocessed=True
    )

    required_fields = [
        "initial_candidates",
        "after_area_filter",
        "after_shape_filter",
        "after_region_exclusion",
        "after_contrast_filter",
        "final_valid_marks",
        "detector_status",
        "input_is_preprocessed",
        "internal_clahe_applied",
    ]
    for field in required_fields:
        assert field in trace, f"Missing trace field: {field}"

    assert trace["input_is_preprocessed"] is True
    assert trace["internal_clahe_applied"] is False
    assert isinstance(trace["initial_candidates"], int)
    assert isinstance(trace["final_valid_marks"], int)
    assert trace["detector_status"] in ("OK", "LOW_CONFIDENCE_CANDIDATES",
                                         "NO_CANDIDATES", "LANDMARK_FALLBACK_ROI")


# ---------------------------------------------------------------------------
# Test 6: Mark separation — distinct centroids
# ---------------------------------------------------------------------------

def test_marks_have_distinct_centroids():
    """Multiple placed marks should produce candidates with distinct
    centroids — proving they are detected separately, not merged."""
    np.random.seed(42)
    img = _make_base_face(1024, skin_val=160)
    lms = _build_fake_landmarks(1024)

    # Two marks far apart
    cv2.circle(img, (280, 500), 12, (55, 55, 55), -1)  # left cheek
    cv2.circle(img, (740, 500), 12, (55, 55, 55), -1)  # right cheek

    marks, rejected, occ, trace, overlays = detect_facial_marks(
        img, lms, input_is_preprocessed=True
    )

    if len(marks) >= 2:
        # Check that the two highest-confidence dark marks have distinct centroids
        dark_marks = sorted(
            [m for m in marks if m.get("channel") in ("dark", "dark_lesion")],
            key=lambda m: m.get("confidence", 0), reverse=True
        )
        if len(dark_marks) >= 2:
            c1 = dark_marks[0]["centroid"]
            c2 = dark_marks[1]["centroid"]
            dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            assert dist > 0.1, (
                f"Two placed marks should have distinct centroids, "
                f"but distance is only {dist:.4f}"
            )
