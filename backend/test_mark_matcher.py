"""Unit tests for backend/mark_matcher.py

Tests the isolated forensic mark matcher with controlled synthetic mark descriptors.
No MediaPipe, no image processing — pure geometric + descriptor matching.
"""
import sys
import os
import math
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mark_matcher import match_facial_marks, MARK_MATCHER_VERSION, get_matcher_thresholds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mark(centroid, area=50.0, intensity=120.0, circularity=0.7,
               mark_type="dark_mole", face_region="left_cheek",
               confidence=0.8, orientation=None):
    """Create a minimal mark descriptor dict for testing."""
    m = {
        "centroid": centroid,
        "canonical_position": centroid,
        "area": area,
        "contour_area": area,
        "intensity": intensity,
        "circularity": circularity,
        "mark_type": mark_type,
        "face_region": face_region,
        "confidence": confidence,
    }
    if orientation is not None:
        m["orientation"] = orientation
    return m


# ---------------------------------------------------------------------------
# Test 1: Three matching marks with slight offsets
# ---------------------------------------------------------------------------

def test_three_matching_marks():
    """Same three marks with slight coordinate offsets should all match."""
    gallery = [
        _make_mark((0.30, 0.50), area=50, mark_type="dark_mole", face_region="left_cheek"),
        _make_mark((0.70, 0.50), area=45, mark_type="dark_mole", face_region="right_cheek"),
        _make_mark((0.50, 0.25), area=40, mark_type="dark_spot", face_region="forehead"),
    ]
    probe = [
        _make_mark((0.31, 0.51), area=48, mark_type="dark_mole", face_region="left_cheek"),
        _make_mark((0.69, 0.49), area=47, mark_type="dark_mole", face_region="right_cheek"),
        _make_mark((0.51, 0.26), area=38, mark_type="dark_spot", face_region="forehead"),
    ]

    result = match_facial_marks(gallery, probe)

    assert result["matched"] == 3, f"Expected 3 matches, got {result['matched']}"
    assert result["matcher_status"] == "OK"
    assert len(result["matches"]) == 3

    # Each match should link correct gallery/probe indices
    matched_pairs = {(m["gallery_idx"], m["probe_idx"]) for m in result["matches"]}
    assert (0, 0) in matched_pairs
    assert (1, 1) in matched_pairs
    assert (2, 2) in matched_pairs


# ---------------------------------------------------------------------------
# Test 2: Marks shifted too far → no matches
# ---------------------------------------------------------------------------

def test_marks_too_far_no_matches():
    """Marks shifted beyond spatial threshold should produce no matches."""
    gallery = [
        _make_mark((0.10, 0.10), mark_type="dark_mole"),
        _make_mark((0.20, 0.20), mark_type="dark_mole"),
    ]
    probe = [
        _make_mark((0.80, 0.80), mark_type="dark_mole"),
        _make_mark((0.90, 0.90), mark_type="dark_mole"),
    ]

    result = match_facial_marks(gallery, probe)

    assert result["matched"] == 0
    assert result["matcher_status"] == "NO_MATCHES"
    assert len(result["rejected_candidates"]) > 0

    # Check rejection reasons exist
    for rej in result["rejected_candidates"]:
        assert "rejection_reason" in rej
        assert "gallery_idx" in rej
        assert "probe_idx" in rej
        assert "position_distance" in rej


# ---------------------------------------------------------------------------
# Test 3: One-to-one assignment enforcement
# ---------------------------------------------------------------------------

def test_one_to_one_assignment():
    """One gallery mark cannot match two probe marks. Hungarian ensures 1:1."""
    gallery = [
        _make_mark((0.50, 0.50), area=50, mark_type="dark_mole"),
    ]
    probe = [
        _make_mark((0.50, 0.50), area=50, mark_type="dark_mole"),
        _make_mark((0.51, 0.51), area=48, mark_type="dark_mole"),
    ]

    result = match_facial_marks(gallery, probe)

    # Should match exactly 1 (gallery has only 1 mark)
    assert result["matched"] == 1, f"Expected 1 match, got {result['matched']}"
    assert len(result["matches"]) == 1

    # The matched gallery idx should be 0 (only gallery mark)
    assert result["matches"][0]["gallery_idx"] == 0

    # Two probe marks for one gallery → only one gets matched
    matched_probe_indices = {m["probe_idx"] for m in result["matches"]}
    assert len(matched_probe_indices) == 1


# ---------------------------------------------------------------------------
# Test 4: Type mismatch reduces quality
# ---------------------------------------------------------------------------

def test_type_mismatch_reduces_quality():
    """Identical position but different type should either match with lower
    quality or be rejected due to higher cost."""
    # Same-type pair
    gallery_same = [_make_mark((0.50, 0.50), mark_type="dark_mole")]
    probe_same = [_make_mark((0.51, 0.51), mark_type="dark_mole")]
    result_same = match_facial_marks(gallery_same, probe_same)

    # Different-type pair at same positions
    gallery_diff = [_make_mark((0.50, 0.50), mark_type="dark_mole")]
    probe_diff = [_make_mark((0.51, 0.51), mark_type="light_scar")]
    result_diff = match_facial_marks(gallery_diff, probe_diff)

    # Both should match (distance is small)
    assert result_same["matched"] >= 1
    assert result_diff["matched"] >= 1

    # Type-mismatched pair should have lower quality or higher cost
    q_same = result_same["matches"][0]["match_quality"]
    q_diff = result_diff["matches"][0]["match_quality"]
    cost_same = result_same["matches"][0]["match_cost"]
    cost_diff = result_diff["matches"][0]["match_cost"]

    assert cost_diff > cost_same, (
        f"Type mismatch should increase cost: same={cost_same:.3f}, diff={cost_diff:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 5: Region mismatch reduces quality
# ---------------------------------------------------------------------------

def test_region_mismatch_reduces_quality():
    """Same type but different face region should have higher cost."""
    gallery_same = [_make_mark((0.50, 0.50), face_region="left_cheek")]
    probe_same = [_make_mark((0.51, 0.51), face_region="left_cheek")]
    result_same = match_facial_marks(gallery_same, probe_same)

    gallery_diff = [_make_mark((0.50, 0.50), face_region="left_cheek")]
    probe_diff = [_make_mark((0.51, 0.51), face_region="forehead")]
    result_diff = match_facial_marks(gallery_diff, probe_diff)

    assert result_same["matched"] >= 1
    assert result_diff["matched"] >= 1

    cost_same = result_same["matches"][0]["match_cost"]
    cost_diff = result_diff["matches"][0]["match_cost"]

    assert cost_diff > cost_same, (
        f"Region mismatch should increase cost: same={cost_same:.3f}, diff={cost_diff:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 6: Missing calibration → neutral LR, no crash
# ---------------------------------------------------------------------------

def test_missing_calibration_neutral_lr():
    """Without calibration, matcher should return LR = 1.0 and not crash."""
    gallery = [_make_mark((0.50, 0.50))]
    probe = [_make_mark((0.51, 0.51))]

    result = match_facial_marks(gallery, probe, calibration=None)

    assert result["matched"] >= 1
    assert result["lr_marks"] == 1.0, f"Expected lr_marks=1.0, got {result['lr_marks']}"

    for lr in result["mark_lrs"]:
        assert lr == 1.0, f"Individual LR should be 1.0 without calibration, got {lr}"

    assert result["calibration_status"] == "MISSING"

    # LR must never be > 1.0 without calibration
    assert result["lr_marks"] <= 1.0


# ---------------------------------------------------------------------------
# Test 7: Empty inputs → INSUFFICIENT_INPUT
# ---------------------------------------------------------------------------

def test_empty_inputs_insufficient():
    """Empty gallery or probe should return INSUFFICIENT_INPUT status."""
    # Empty gallery
    result = match_facial_marks([], [_make_mark((0.5, 0.5))])
    assert result["matcher_status"] == "INSUFFICIENT_INPUT"
    assert result["matched"] == 0
    assert result["score"] is None

    # Empty probe
    result = match_facial_marks([_make_mark((0.5, 0.5))], [])
    assert result["matcher_status"] == "INSUFFICIENT_INPUT"
    assert result["matched"] == 0

    # Both empty
    result = match_facial_marks([], [])
    assert result["matcher_status"] == "INSUFFICIENT_INPUT"
    assert result["matched"] == 0


# ---------------------------------------------------------------------------
# Test 8: Match objects include all required fields
# ---------------------------------------------------------------------------

def test_match_objects_complete():
    """Every accepted match must include all required forensic fields."""
    gallery = [_make_mark((0.30, 0.50), mark_type="dark_mole", face_region="left_cheek")]
    probe = [_make_mark((0.31, 0.51), mark_type="dark_mole", face_region="left_cheek")]

    result = match_facial_marks(gallery, probe)
    assert result["matched"] >= 1

    required_match_fields = [
        "gallery_idx", "probe_idx",
        "gallery_centroid", "probe_centroid",
        "position_distance", "area_ratio",
        "type_match", "region_match",
        "match_quality", "mark_type", "face_region", "lr",
    ]

    for match in result["matches"]:
        for field in required_match_fields:
            assert field in match, f"Missing field '{field}' in match object"

    # Also check top-level result fields
    required_result_fields = [
        "matched", "score", "lr_marks", "mark_lrs",
        "matches", "rejected_candidates", "matcher_status",
    ]
    for field in required_result_fields:
        assert field in result, f"Missing field '{field}' in result"


# ---------------------------------------------------------------------------
# Test 9: Rejected candidates include required fields
# ---------------------------------------------------------------------------

def test_rejected_candidates_complete():
    """Rejected candidates must include gallery_idx, probe_idx,
    position_distance, match_cost, and rejection_reason."""
    gallery = [_make_mark((0.10, 0.10))]
    probe = [_make_mark((0.80, 0.80))]

    result = match_facial_marks(gallery, probe)
    assert result["matched"] == 0
    assert len(result["rejected_candidates"]) > 0

    required_reject_fields = [
        "gallery_idx", "probe_idx", "position_distance",
        "match_cost", "rejection_reason",
    ]
    for rej in result["rejected_candidates"]:
        for field in required_reject_fields:
            assert field in rej, f"Missing field '{field}' in rejected candidate"


# ---------------------------------------------------------------------------
# Test 10: canonical_position preferred over centroid
# ---------------------------------------------------------------------------

def test_canonical_position_preferred():
    """If canonical_position differs from centroid, matcher should use
    canonical_position."""
    gallery = [{"centroid": (0.10, 0.10), "canonical_position": (0.50, 0.50),
                "area": 50, "intensity": 120, "circularity": 0.7,
                "mark_type": "dark_mole", "face_region": "left_cheek",
                "confidence": 0.8}]
    probe = [{"centroid": (0.90, 0.90), "canonical_position": (0.51, 0.51),
              "area": 48, "intensity": 118, "circularity": 0.7,
              "mark_type": "dark_mole", "face_region": "left_cheek",
              "confidence": 0.8}]

    result = match_facial_marks(gallery, probe)

    # centroid distance is 0.8+, canonical_position distance is ~0.014
    # Should match using canonical_position
    assert result["matched"] == 1
    assert result["matches"][0]["position_distance"] < 0.05


# ---------------------------------------------------------------------------
# Test 11: Version and thresholds accessible
# ---------------------------------------------------------------------------

def test_version_and_thresholds():
    """MARK_MATCHER_VERSION and get_matcher_thresholds must be accessible."""
    assert MARK_MATCHER_VERSION == "1.0.0"

    thresholds = get_matcher_thresholds()
    assert "max_spatial_distance" in thresholds
    assert "max_assignment_cost" in thresholds
    assert "matcher_version" in thresholds
    assert thresholds["matcher_version"] == MARK_MATCHER_VERSION
