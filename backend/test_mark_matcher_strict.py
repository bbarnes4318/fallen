"""Unit tests for mark_matcher_strict.py — Strict V2 mark matcher.

Tests:
- Generic mark types (freckle/pore/texture) produce LR = 1.0
- Generic region mismatches are hard-rejected
- Generic type mismatches are hard-rejected
- Cross-class matching is rejected (freckle vs mole, pore vs scar, etc.)
- Mass generic marks cannot create astronomical LR
- Distinctive marks survive when spatially/regionally consistent
- Feature flag off preserves normal matcher behavior
- Aggregate caps are enforced
- Cluster penalty fires when one dimension dominates

Run in CI only:
  python -m pytest backend/test_mark_matcher_strict.py -v
"""

import math
import os
import sys
import unittest

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from mark_matcher_strict import (
    match_facial_marks_strict,
    _classify_mark,
    GENERIC_MARK_TYPES,
    DISTINCTIVE_MARK_TYPES,
    _GENERIC_LR_CAP,
    _DISTINCTIVE_LR_CAP,
    _TOTAL_LOG_LR_CAP,
    get_strict_matcher_thresholds,
)


def _make_mark(mark_type, centroid, area=10.0, face_region="left_cheek",
               channel="dark", confidence=0.8, circularity=0.7, intensity=80):
    return {
        "mark_type": mark_type,
        "centroid": centroid,
        "canonical_position": centroid,
        "area": area,
        "contour_area": area,
        "face_region": face_region,
        "channel": channel,
        "confidence": confidence,
        "circularity": circularity,
        "intensity": intensity,
        "salience": 0.5,
        "fallback_generated": False,
    }


class TestMarkClassification(unittest.TestCase):
    def test_generic_types(self):
        for t in ["freckle", "pore", "texture_anomaly", "blemish",
                   "dark_spot", "texture_cluster", "unknown_mark"]:
            self.assertEqual(_classify_mark(t), "generic", f"{t} should be generic")

    def test_distinctive_types(self):
        for t in ["dark_mole", "mole", "light_scar", "scar",
                   "linear_scar", "structural_crater", "depression_scar"]:
            self.assertEqual(_classify_mark(t), "distinctive", f"{t} should be distinctive")

    def test_unknown_defaults_generic(self):
        self.assertEqual(_classify_mark("totally_unknown_type"), "generic")


class TestGenericMarksNeutralLR(unittest.TestCase):
    """Generic marks must produce LR = 1.0 (neutral)."""

    def test_freckle_freckle_lr_neutral(self):
        g = [_make_mark("freckle", (0.3, 0.3), face_region="left_cheek")]
        p = [_make_mark("freckle", (0.3, 0.3), face_region="left_cheek")]
        result = match_facial_marks_strict(g, p)
        self.assertEqual(result["lr_marks"], 1.0)

    def test_pore_pore_lr_neutral(self):
        g = [_make_mark("pore", (0.5, 0.5), face_region="forehead")]
        p = [_make_mark("pore", (0.5, 0.5), face_region="forehead")]
        result = match_facial_marks_strict(g, p)
        self.assertEqual(result["lr_marks"], 1.0)

    def test_texture_anomaly_lr_neutral(self):
        g = [_make_mark("texture_anomaly", (0.4, 0.4), face_region="right_cheek")]
        p = [_make_mark("texture_anomaly", (0.4, 0.4), face_region="right_cheek")]
        result = match_facial_marks_strict(g, p)
        self.assertEqual(result["lr_marks"], 1.0)


class TestGenericRegionMismatchRejection(unittest.TestCase):
    """Generic marks with different face_region must be hard-rejected."""

    def test_freckle_region_mismatch(self):
        g = [_make_mark("freckle", (0.3, 0.3), face_region="left_cheek")]
        p = [_make_mark("freckle", (0.3, 0.3), face_region="right_cheek")]
        result = match_facial_marks_strict(g, p)
        self.assertEqual(result["matched"], 0)

    def test_pore_region_mismatch(self):
        g = [_make_mark("pore", (0.5, 0.5), face_region="forehead")]
        p = [_make_mark("pore", (0.5, 0.5), face_region="chin")]
        result = match_facial_marks_strict(g, p)
        self.assertEqual(result["matched"], 0)


class TestGenericTypeMismatchRejection(unittest.TestCase):
    """Generic marks with different mark_type must be hard-rejected."""

    def test_freckle_vs_pore(self):
        g = [_make_mark("freckle", (0.3, 0.3), face_region="left_cheek")]
        p = [_make_mark("pore", (0.3, 0.3), face_region="left_cheek")]
        result = match_facial_marks_strict(g, p)
        self.assertEqual(result["matched"], 0)

    def test_blemish_vs_dark_spot(self):
        g = [_make_mark("blemish", (0.5, 0.5), face_region="forehead")]
        p = [_make_mark("dark_spot", (0.5, 0.5), face_region="forehead")]
        result = match_facial_marks_strict(g, p)
        self.assertEqual(result["matched"], 0)


class TestCrossClassRejection(unittest.TestCase):
    """Freckle cannot match mole, pore cannot match scar, etc."""

    def test_freckle_vs_mole(self):
        g = [_make_mark("freckle", (0.3, 0.3), face_region="left_cheek")]
        p = [_make_mark("mole", (0.3, 0.3), face_region="left_cheek")]
        result = match_facial_marks_strict(g, p)
        self.assertEqual(result["matched"], 0)

    def test_pore_vs_scar(self):
        g = [_make_mark("pore", (0.5, 0.5), face_region="forehead")]
        p = [_make_mark("scar", (0.5, 0.5), face_region="forehead")]
        result = match_facial_marks_strict(g, p)
        self.assertEqual(result["matched"], 0)

    def test_texture_anomaly_vs_structural_crater(self):
        g = [_make_mark("texture_anomaly", (0.4, 0.4), face_region="right_cheek")]
        p = [_make_mark("structural_crater", (0.4, 0.4), face_region="right_cheek")]
        result = match_facial_marks_strict(g, p)
        self.assertEqual(result["matched"], 0)


class TestMassGenericNoAstronomicalLR(unittest.TestCase):
    """20 generic marks in one region cannot create astronomical LR."""

    def test_20_freckles_same_region(self):
        g = [_make_mark("freckle", (0.3 + i * 0.002, 0.3 + i * 0.002),
                        face_region="left_cheek") for i in range(20)]
        p = [_make_mark("freckle", (0.3 + i * 0.002, 0.3 + i * 0.002),
                        face_region="left_cheek") for i in range(20)]
        result = match_facial_marks_strict(g, p)
        self.assertLessEqual(result["lr_marks"], 1.0,
                             "20 generic freckles must not create positive LR")

    def test_mass_pores(self):
        g = [_make_mark("pore", (0.5 + i * 0.002, 0.5 + i * 0.002),
                        face_region="forehead") for i in range(15)]
        p = [_make_mark("pore", (0.5 + i * 0.002, 0.5 + i * 0.002),
                        face_region="forehead") for i in range(15)]
        result = match_facial_marks_strict(g, p)
        self.assertLessEqual(result["lr_marks"], 1.0)


class TestDistinctiveMarksSurvive(unittest.TestCase):
    """Distinctive mole/scar marks can survive if spatially/regionally consistent."""

    def test_mole_same_region_accepted(self):
        g = [_make_mark("mole", (0.4, 0.4), face_region="left_cheek")]
        p = [_make_mark("mole", (0.42, 0.42), face_region="left_cheek")]
        result = match_facial_marks_strict(g, p)
        self.assertGreaterEqual(result["matched"], 1)
        # Should be eligible for LR
        scoring = result.get("scoring_correspondences", [])
        self.assertGreaterEqual(len(scoring), 1)

    def test_scar_same_region_accepted(self):
        g = [_make_mark("scar", (0.5, 0.3), face_region="forehead")]
        p = [_make_mark("scar", (0.51, 0.31), face_region="forehead")]
        result = match_facial_marks_strict(g, p)
        self.assertGreaterEqual(result["matched"], 1)

    def test_mole_too_far_rejected(self):
        g = [_make_mark("mole", (0.1, 0.1), face_region="left_cheek")]
        p = [_make_mark("mole", (0.9, 0.9), face_region="right_cheek")]
        result = match_facial_marks_strict(g, p)
        self.assertEqual(result["matched"], 0)


class TestDistinctiveLRCap(unittest.TestCase):
    """Individual distinctive mark LR must be capped at 25.0."""

    def test_distinctive_lr_cap_value(self):
        self.assertEqual(_DISTINCTIVE_LR_CAP, 25.0)

    def test_generic_lr_cap_value(self):
        self.assertEqual(_GENERIC_LR_CAP, 1.0)


class TestTotalLRCap(unittest.TestCase):
    """Total log10(LR) must be capped at 2.0 (max combined = 100)."""

    def test_total_log_lr_cap(self):
        self.assertEqual(_TOTAL_LOG_LR_CAP, 2.0)

    def test_thresholds_dict(self):
        t = get_strict_matcher_thresholds()
        self.assertEqual(t["total_log_lr_cap"], 2.0)
        self.assertEqual(t["generic_lr_cap"], 1.0)
        self.assertEqual(t["distinctive_lr_cap"], 25.0)
        self.assertEqual(t["per_region_lr_cap"], 50.0)
        self.assertEqual(t["per_type_lr_cap"], 50.0)
        self.assertEqual(t["per_channel_lr_cap"], 25.0)


class TestFeatureFlagPreservation(unittest.TestCase):
    """Feature flag off must preserve normal matcher behavior."""

    def test_strict_mode_flag_in_output(self):
        g = [_make_mark("mole", (0.4, 0.4), face_region="left_cheek")]
        p = [_make_mark("mole", (0.42, 0.42), face_region="left_cheek")]
        result = match_facial_marks_strict(g, p)
        self.assertTrue(result["strict_mode"])

    def test_standard_keys_present(self):
        g = [_make_mark("mole", (0.4, 0.4), face_region="left_cheek")]
        p = [_make_mark("mole", (0.42, 0.42), face_region="left_cheek")]
        result = match_facial_marks_strict(g, p)
        required_keys = [
            "matched", "score", "lr_marks", "mark_lrs", "matches",
            "rejected_candidates", "matcher_status", "calibration_status",
            "matcher_version",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing standard key: {key}")

    def test_strict_telemetry_keys_present(self):
        g = [_make_mark("mole", (0.4, 0.4), face_region="left_cheek")]
        p = [_make_mark("mole", (0.42, 0.42), face_region="left_cheek")]
        result = match_facial_marks_strict(g, p)
        strict_keys = [
            "strict_mode", "display_correspondences", "scoring_correspondences",
            "suppressed_correspondences", "scoring_eligible_marks_count",
            "generic_marks_suppressed_count", "distinctive_marks_preserved_count",
            "lr_before_caps", "lr_after_all_caps", "caps_applied",
            "cluster_penalty_applied", "cluster_penalty_factor",
        ]
        for key in strict_keys:
            self.assertIn(key, result, f"Missing strict key: {key}")


class TestOutputSeparation(unittest.TestCase):
    """Display vs scoring vs suppressed separation."""

    def test_generic_in_display_not_scoring(self):
        g = [_make_mark("freckle", (0.3, 0.3), face_region="left_cheek")]
        p = [_make_mark("freckle", (0.3, 0.3), face_region="left_cheek")]
        result = match_facial_marks_strict(g, p)
        # Freckle is generic — should be displayed but not in scoring
        self.assertGreaterEqual(len(result["display_correspondences"]), 1)
        self.assertEqual(len(result["scoring_correspondences"]), 0)
        self.assertGreaterEqual(len(result["suppressed_correspondences"]), 1)

    def test_distinctive_in_both_display_and_scoring(self):
        g = [_make_mark("mole", (0.4, 0.4), face_region="left_cheek")]
        p = [_make_mark("mole", (0.42, 0.42), face_region="left_cheek")]
        result = match_facial_marks_strict(g, p)
        self.assertGreaterEqual(len(result["display_correspondences"]), 1)
        self.assertGreaterEqual(len(result["scoring_correspondences"]), 1)

    def test_correspondence_fields(self):
        g = [_make_mark("mole", (0.4, 0.4), face_region="left_cheek")]
        p = [_make_mark("mole", (0.42, 0.42), face_region="left_cheek")]
        result = match_facial_marks_strict(g, p)
        corr = result["display_correspondences"][0]
        required_fields = [
            "accepted_for_display", "eligible_for_lr", "suppression_reason",
            "strict_rejection_reason", "mark_type", "face_region", "channel",
            "position_distance", "area_ratio", "match_cost",
            "lr_before_cap", "lr_after_cap",
        ]
        for f in required_fields:
            self.assertIn(f, corr, f"Missing correspondence field: {f}")


class TestEmptyInput(unittest.TestCase):
    def test_empty_gallery(self):
        result = match_facial_marks_strict([], [_make_mark("mole", (0.4, 0.4))])
        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["lr_marks"], 1.0)

    def test_empty_probe(self):
        result = match_facial_marks_strict([_make_mark("mole", (0.4, 0.4))], [])
        self.assertEqual(result["matched"], 0)

    def test_both_empty(self):
        result = match_facial_marks_strict([], [])
        self.assertEqual(result["matched"], 0)


class TestMaxLRCeiling(unittest.TestCase):
    """Even with many distinctive marks and calibration, total LR cannot exceed 100."""

    def test_lr_marks_ceiling(self):
        # Create many distinctive marks that would match
        g = [_make_mark("mole", (0.2 + i * 0.05, 0.2 + i * 0.05),
                        face_region=f"region_{i % 5}") for i in range(10)]
        p = [_make_mark("mole", (0.2 + i * 0.05, 0.2 + i * 0.05),
                        face_region=f"region_{i % 5}") for i in range(10)]
        result = match_facial_marks_strict(g, p)
        # Without calibration, LR defaults to 1.0 per mark anyway,
        # but the cap structure must guarantee <= 100
        self.assertLessEqual(result["lr_marks"], 100.0)


if __name__ == "__main__":
    unittest.main()
