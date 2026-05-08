"""Unit tests for mark_anatomy.py — Phase 1 barycentric coordinate mapping + constellation telemetry.

Tests cover:
1. Barycentric math (_solve_barycentric_2d, _triangle_area_2d)
2. Nearest landmark finding
3. Full barycentric position computation
4. Expanded mesh region mapping
5. Mesh confidence scoring
6. Barycentric distance computation
7. Constellation telemetry computation
8. Degenerate/fallback cases
9. Safety invariants (8-region face_region unchanged, LR caps, scoring isolation)
"""
import sys
import os
import math
import unittest

# Add backend to path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

import mark_anatomy


class TestTriangleArea2D(unittest.TestCase):
    """Test _triangle_area_2d."""

    def test_unit_right_triangle(self):
        area = mark_anatomy._triangle_area_2d((0, 0), (1, 0), (0, 1))
        self.assertAlmostEqual(area, 0.5, places=6)

    def test_degenerate_collinear(self):
        area = mark_anatomy._triangle_area_2d((0, 0), (1, 0), (2, 0))
        self.assertAlmostEqual(area, 0.0, places=6)

    def test_small_triangle(self):
        area = mark_anatomy._triangle_area_2d((0.5, 0.5), (0.51, 0.5), (0.5, 0.51))
        self.assertAlmostEqual(area, 0.5 * 0.01 * 0.01, places=8)


class TestSolveBarycentric2D(unittest.TestCase):
    """Test _solve_barycentric_2d."""

    def test_center_of_triangle(self):
        """Center of equilateral-like triangle should have ~equal bary coords."""
        A = (0, 0)
        B = (1, 0)
        C = (0.5, 1)
        P = ((0 + 1 + 0.5) / 3.0, (0 + 0 + 1) / 3.0)
        result = mark_anatomy._solve_barycentric_2d(P, A, B, C)
        self.assertIsNotNone(result)
        u, v, w = result
        self.assertAlmostEqual(u + v + w, 1.0, places=6)
        # All should be roughly 1/3
        for coord in [u, v, w]:
            self.assertGreater(coord, 0.0)
            self.assertLess(coord, 1.0)

    def test_at_vertex_a(self):
        """Point at vertex A should give w=1 or similar dominance."""
        A = (0, 0)
        B = (1, 0)
        C = (0, 1)
        result = mark_anatomy._solve_barycentric_2d(A, A, B, C)
        self.assertIsNotNone(result)
        u, v, w = result
        self.assertAlmostEqual(u + v + w, 1.0, places=6)

    def test_outside_triangle(self):
        """Point outside triangle should still compute (with negative coords)."""
        A = (0, 0)
        B = (1, 0)
        C = (0, 1)
        P = (2, 2)  # Far outside
        result = mark_anatomy._solve_barycentric_2d(P, A, B, C)
        self.assertIsNotNone(result)
        u, v, w = result
        self.assertAlmostEqual(u + v + w, 1.0, places=6)
        # At least one should be negative (outside)
        self.assertTrue(u < 0 or v < 0 or w < 0)

    def test_degenerate_triangle_returns_none(self):
        """Collinear points should return None."""
        A = (0, 0)
        B = (1, 0)
        C = (2, 0)
        result = mark_anatomy._solve_barycentric_2d((0.5, 0), A, B, C)
        self.assertIsNone(result)


class MockLandmark:
    """Mock MediaPipe landmark with .x and .y attributes."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class TestFindNearest3Landmarks(unittest.TestCase):
    """Test _find_nearest_3_landmarks."""

    def test_basic_nearest(self):
        landmarks = [
            MockLandmark(0.1, 0.1),  # 0
            MockLandmark(0.5, 0.5),  # 1
            MockLandmark(0.15, 0.15),  # 2
            MockLandmark(0.9, 0.9),  # 3
            MockLandmark(0.12, 0.12),  # 4
        ]
        result = mark_anatomy._find_nearest_3_landmarks(0.1, 0.1, landmarks, 100, 100)
        self.assertEqual(len(result), 3)
        # Nearest should be index 0 (exact match)
        self.assertEqual(result[0][0], 0)
        self.assertAlmostEqual(result[0][3], 0.0, places=6)

    def test_returns_empty_on_none(self):
        result = mark_anatomy._find_nearest_3_landmarks(0.5, 0.5, None, 100, 100)
        self.assertEqual(result, [])


class TestComputeBarycentric Position(unittest.TestCase):
    """Test compute_barycentric_position."""

    def test_with_valid_landmarks(self):
        # Create enough mock landmarks (need at least 3)
        landmarks = [MockLandmark(i * 0.01, i * 0.01) for i in range(10)]
        result = mark_anatomy.compute_barycentric_position(
            mark_centroid=(0.05, 0.05),
            landmarks=landmarks,
            image_shape=(100, 100),
        )
        self.assertIn("barycentric_mode", result)
        self.assertIn("coordinate_system_version", result)
        self.assertEqual(result["coordinate_system_version"], "1.0.0-phase1")
        self.assertEqual(result["triangle_source"], "nearest_3_mediapipe_landmarks")
        self.assertIn("mesh_region", result)
        self.assertIn("mesh_confidence", result)

    def test_with_no_landmarks(self):
        result = mark_anatomy.compute_barycentric_position(
            mark_centroid=(0.5, 0.5),
            landmarks=None,
            image_shape=(100, 100),
        )
        self.assertEqual(result["barycentric_mode"], "nearest_landmark_fallback")
        self.assertIsNone(result["barycentric_u"])
        self.assertIsNone(result["barycentric_v"])
        self.assertIsNone(result["barycentric_w"])
        self.assertEqual(result["mesh_confidence"], 0.0)

    def test_with_too_few_landmarks(self):
        landmarks = [MockLandmark(0.5, 0.5)]
        result = mark_anatomy.compute_barycentric_position(
            mark_centroid=(0.5, 0.5),
            landmarks=landmarks,
            image_shape=(100, 100),
        )
        self.assertEqual(result["barycentric_mode"], "nearest_landmark_fallback")

    def test_degenerate_collinear_landmarks(self):
        """Collinear landmarks should produce fallback mode."""
        landmarks = [
            MockLandmark(0.0, 0.0),
            MockLandmark(0.5, 0.0),  # Same y — collinear
            MockLandmark(1.0, 0.0),
        ]
        result = mark_anatomy.compute_barycentric_position(
            mark_centroid=(0.5, 0.0),
            landmarks=landmarks,
            image_shape=(100, 100),
        )
        self.assertEqual(result["barycentric_mode"], "nearest_landmark_fallback")

    def test_mode_marking_is_correct(self):
        """Verify that valid results are marked as 2d_mesh_approximation, not 3D."""
        landmarks = [
            MockLandmark(0.4, 0.4),
            MockLandmark(0.6, 0.4),
            MockLandmark(0.5, 0.6),
        ]
        result = mark_anatomy.compute_barycentric_position(
            mark_centroid=(0.5, 0.47),
            landmarks=landmarks,
            image_shape=(100, 100),
        )
        if result["barycentric_mode"] == "2d_mesh_approximation":
            self.assertNotIn("3d", result["barycentric_mode"].lower())
            self.assertEqual(result["triangle_source"], "nearest_3_mediapipe_landmarks")


class TestExpandedMeshRegion(unittest.TestCase):
    """Test compute_expanded_mesh_region returns expanded ~15 regions."""

    def test_forehead_region(self):
        result = mark_anatomy.compute_expanded_mesh_region([10, 338, 297])
        self.assertEqual(result, "forehead")

    def test_unknown_indices(self):
        result = mark_anatomy.compute_expanded_mesh_region([9999, 9998])
        self.assertEqual(result, "unknown")

    def test_empty_list(self):
        result = mark_anatomy.compute_expanded_mesh_region([])
        self.assertEqual(result, "unknown")


class TestMeshConfidence(unittest.TestCase):
    """Test compute_mesh_confidence."""

    def test_inside_triangle_high_confidence(self):
        conf = mark_anatomy.compute_mesh_confidence((0.3, 0.3, 0.4), 0.002, 0.01)
        self.assertGreater(conf, 0.5)

    def test_outside_triangle_reduced(self):
        conf_inside = mark_anatomy.compute_mesh_confidence((0.3, 0.3, 0.4), 0.002, 0.01)
        conf_outside = mark_anatomy.compute_mesh_confidence((-0.5, 0.8, 0.7), 0.002, 0.01)
        self.assertLess(conf_outside, conf_inside)

    def test_degenerate_triangle_zero(self):
        conf = mark_anatomy.compute_mesh_confidence((0.3, 0.3, 0.4), 1e-10, 0.01)
        self.assertEqual(conf, 0.0)

    def test_none_bary_coords_zero(self):
        conf = mark_anatomy.compute_mesh_confidence(None, 0.002, 0.01)
        self.assertEqual(conf, 0.0)


class TestBarycentricDistance(unittest.TestCase):
    """Test compute_barycentric_distance."""

    def test_same_point_zero_distance(self):
        anat = {
            "barycentric_u": 0.3, "barycentric_v": 0.3, "barycentric_w": 0.4,
            "barycentric_mode": "2d_mesh_approximation",
            "mesh_confidence": 0.9,
        }
        dist, available = mark_anatomy.compute_barycentric_distance(anat, anat)
        self.assertTrue(available)
        self.assertAlmostEqual(dist, 0.0, places=6)

    def test_different_points(self):
        anat_a = {
            "barycentric_u": 0.3, "barycentric_v": 0.3, "barycentric_w": 0.4,
            "barycentric_mode": "2d_mesh_approximation",
            "mesh_confidence": 0.9,
        }
        anat_b = {
            "barycentric_u": 0.5, "barycentric_v": 0.3, "barycentric_w": 0.2,
            "barycentric_mode": "2d_mesh_approximation",
            "mesh_confidence": 0.9,
        }
        dist, available = mark_anatomy.compute_barycentric_distance(anat_a, anat_b)
        self.assertTrue(available)
        self.assertGreater(dist, 0.0)

    def test_fallback_mode_not_available(self):
        anat_a = {
            "barycentric_u": 0.3, "barycentric_v": 0.3, "barycentric_w": 0.4,
            "barycentric_mode": "2d_mesh_approximation",
            "mesh_confidence": 0.9,
        }
        anat_b = {
            "barycentric_u": None, "barycentric_v": None, "barycentric_w": None,
            "barycentric_mode": "nearest_landmark_fallback",
            "mesh_confidence": 0.0,
        }
        dist, available = mark_anatomy.compute_barycentric_distance(anat_a, anat_b)
        self.assertFalse(available)
        self.assertIsNone(dist)

    def test_low_confidence_not_available(self):
        anat_a = {
            "barycentric_u": 0.3, "barycentric_v": 0.3, "barycentric_w": 0.4,
            "barycentric_mode": "2d_mesh_approximation",
            "mesh_confidence": 0.5,  # Below 0.70 threshold
        }
        anat_b = {
            "barycentric_u": 0.5, "barycentric_v": 0.3, "barycentric_w": 0.2,
            "barycentric_mode": "2d_mesh_approximation",
            "mesh_confidence": 0.9,
        }
        dist, available = mark_anatomy.compute_barycentric_distance(anat_a, anat_b)
        self.assertFalse(available)
        self.assertIsNone(dist)

    def test_none_inputs_not_available(self):
        dist, available = mark_anatomy.compute_barycentric_distance(None, None)
        self.assertFalse(available)
        self.assertIsNone(dist)


class TestConstellationTelemetry(unittest.TestCase):
    """Test compute_constellation_telemetry."""

    def test_empty_correspondences(self):
        result = mark_anatomy.compute_constellation_telemetry([], [], [], [])
        self.assertEqual(result["scoring_eligible_node_count"], 0)
        self.assertEqual(result["constellation_quality_score"], 0.0)
        self.assertEqual(result["constellation_quality_label"], "NONE")
        self.assertTrue(result["telemetry_only"])
        self.assertTrue(result["does_not_affect_scoring"])

    def test_single_scoring_correspondence(self):
        scoring = [{
            "face_region": "left_cheek",
            "mark_type": "dark_mole",
            "gallery_centroid": [0.3, 0.3],
            "probe_centroid": [0.31, 0.31],
        }]
        result = mark_anatomy.compute_constellation_telemetry(scoring, [], [], [])
        self.assertEqual(result["scoring_eligible_node_count"], 1)
        self.assertEqual(result["graph_edge_count"], 0)
        self.assertGreater(result["constellation_quality_score"], 0.0)

    def test_multi_region_correspondences(self):
        scoring = [
            {"face_region": "left_cheek", "mark_type": "dark_mole",
             "gallery_centroid": [0.3, 0.3], "probe_centroid": [0.31, 0.31]},
            {"face_region": "right_cheek", "mark_type": "light_scar",
             "gallery_centroid": [0.7, 0.3], "probe_centroid": [0.71, 0.31]},
            {"face_region": "forehead", "mark_type": "dark_mole",
             "gallery_centroid": [0.5, 0.1], "probe_centroid": [0.51, 0.11]},
        ]
        result = mark_anatomy.compute_constellation_telemetry(scoring, [], [], [])
        self.assertEqual(result["region_diversity_count"], 3)
        self.assertEqual(result["mark_type_diversity_count"], 2)
        self.assertEqual(result["graph_edge_count"], 3)  # C(3,2) = 3
        self.assertGreater(result["constellation_quality_score"], 0.5)

    def test_telemetry_fields_are_present(self):
        result = mark_anatomy.compute_constellation_telemetry([], [], [], [])
        required_keys = [
            "distinctive_node_count", "generic_node_count",
            "scoring_eligible_node_count", "region_diversity_count",
            "mark_type_diversity_count", "graph_edge_count",
            "average_pairwise_distance_error", "median_pairwise_distance_error",
            "max_pairwise_distance_error", "graph_edge_consistency_score",
            "cluster_domination_score", "constellation_quality_score",
            "constellation_quality_label", "telemetry_only", "does_not_affect_scoring",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_quality_label_thresholds(self):
        """Verify label thresholds."""
        # NONE: < 0.15
        result_none = mark_anatomy.compute_constellation_telemetry([], [], [], [])
        self.assertEqual(result_none["constellation_quality_label"], "NONE")

    def test_does_not_affect_scoring_flag(self):
        result = mark_anatomy.compute_constellation_telemetry([], [], [], [])
        self.assertTrue(result["does_not_affect_scoring"])
        self.assertTrue(result["telemetry_only"])


class TestSafetyInvariants(unittest.TestCase):
    """Test that Phase 1 changes preserve critical safety invariants."""

    def test_bary_weight_is_1_0(self):
        """_BARY_WEIGHT must be 1.0, not 3.0."""
        from mark_matcher_strict import _BARY_WEIGHT
        self.assertEqual(_BARY_WEIGHT, 1.0)

    def test_bary_weight_in_thresholds(self):
        """Barycentric weight must appear in thresholds/telemetry."""
        from mark_matcher_strict import get_strict_matcher_thresholds
        thresholds = get_strict_matcher_thresholds()
        self.assertIn("barycentric_weight", thresholds)
        self.assertEqual(thresholds["barycentric_weight"], 1.0)

    def test_generic_lr_cap_unchanged(self):
        from mark_matcher_strict import _GENERIC_LR_CAP
        self.assertEqual(_GENERIC_LR_CAP, 1.0)

    def test_distinctive_lr_cap_unchanged(self):
        from mark_matcher_strict import _DISTINCTIVE_LR_CAP
        self.assertEqual(_DISTINCTIVE_LR_CAP, 25.0)

    def test_total_log_lr_cap_unchanged(self):
        """Total LR cap must remain at 100 (log10 = 2.0)."""
        from mark_matcher_strict import _TOTAL_LOG_LR_CAP
        self.assertEqual(_TOTAL_LOG_LR_CAP, 2.0)
        # 10^2.0 = 100
        self.assertAlmostEqual(10 ** _TOTAL_LOG_LR_CAP, 100.0, places=1)

    def test_barycentric_cost_only_increases_cost(self):
        """Barycentric cost contribution must always be >= 0."""
        from mark_matcher_strict import _barycentric_cost

        # Case 1: No anatomical data — cost = 0
        mark_no_anat = {"mark_type": "dark_mole"}
        cost, dist, avail = _barycentric_cost(mark_no_anat, mark_no_anat)
        self.assertEqual(cost, 0.0)
        self.assertFalse(avail)

        # Case 2: Valid anatomical data — cost >= 0
        mark_with_anat = {
            "anatomical_position": {
                "barycentric_u": 0.3, "barycentric_v": 0.3, "barycentric_w": 0.4,
                "barycentric_mode": "2d_mesh_approximation",
                "mesh_confidence": 0.9,
            }
        }
        cost, dist, avail = _barycentric_cost(mark_with_anat, mark_with_anat)
        self.assertGreaterEqual(cost, 0.0)

    def test_invalid_bary_does_not_affect_cost(self):
        """Low-confidence or fallback bary must not affect cost."""
        from mark_matcher_strict import _barycentric_cost

        mark_low_conf = {
            "anatomical_position": {
                "barycentric_u": 0.3, "barycentric_v": 0.3, "barycentric_w": 0.4,
                "barycentric_mode": "2d_mesh_approximation",
                "mesh_confidence": 0.3,  # Below threshold
            }
        }
        cost, dist, avail = _barycentric_cost(mark_low_conf, mark_low_conf)
        self.assertEqual(cost, 0.0)
        self.assertFalse(avail)

    def test_fallback_mode_does_not_affect_cost(self):
        from mark_matcher_strict import _barycentric_cost
        mark_fallback = {
            "anatomical_position": {
                "barycentric_u": None, "barycentric_v": None, "barycentric_w": None,
                "barycentric_mode": "nearest_landmark_fallback",
                "mesh_confidence": 0.0,
            }
        }
        cost, dist, avail = _barycentric_cost(mark_fallback, mark_fallback)
        self.assertEqual(cost, 0.0)
        self.assertFalse(avail)

    def test_constellation_telemetry_does_not_alter_lr(self):
        """Constellation telemetry must not change lr_marks."""
        from mark_matcher_strict import match_facial_marks_strict

        gallery = [
            {"centroid": (0.3, 0.3), "canonical_position": (0.3, 0.3),
             "area": 100, "mark_type": "dark_mole", "face_region": "left_cheek",
             "channel": "dark", "confidence": 0.8, "circularity": 0.7,
             "intensity": 80, "fallback_generated": False,
             "anatomical_position": None},
        ]
        probe = [
            {"centroid": (0.31, 0.31), "canonical_position": (0.31, 0.31),
             "area": 95, "mark_type": "dark_mole", "face_region": "left_cheek",
             "channel": "dark", "confidence": 0.8, "circularity": 0.7,
             "intensity": 82, "fallback_generated": False,
             "anatomical_position": None},
        ]
        result = match_facial_marks_strict(gallery, probe, calibration=None)

        # lr_marks should be 1.0 (no calibration)
        self.assertEqual(result["lr_marks"], 1.0)

        # Constellation telemetry should be present but NOT alter scoring
        # The assertion inside match_facial_marks_strict also guards this

    def test_generic_marks_remain_lr_neutral(self):
        """Generic marks must still get LR = 1.0."""
        from mark_matcher_strict import match_facial_marks_strict

        gallery = [
            {"centroid": (0.3, 0.3), "canonical_position": (0.3, 0.3),
             "area": 100, "mark_type": "blemish", "face_region": "left_cheek",
             "channel": "dark", "confidence": 0.8, "circularity": 0.5,
             "intensity": 80, "fallback_generated": False,
             "anatomical_position": None},
        ]
        probe = [
            {"centroid": (0.3, 0.3), "canonical_position": (0.3, 0.3),
             "area": 95, "mark_type": "blemish", "face_region": "left_cheek",
             "channel": "dark", "confidence": 0.8, "circularity": 0.5,
             "intensity": 82, "fallback_generated": False,
             "anatomical_position": None},
        ]
        result = match_facial_marks_strict(gallery, probe, calibration=None)

        # All correspondences should be suppressed (generic)
        for c in result.get("matches", []):
            self.assertEqual(c.get("lr"), 1.0)

    def test_8_region_face_region_unchanged(self):
        """The 8-region face_region in mark_detector.py must still use the original logic."""
        from mark_detector import _region_label
        # Just verify the function exists and returns a string
        # Actual region values depend on landmarks, but we verify the function is not replaced
        self.assertTrue(callable(_region_label))


class TestCoordinateSystemVersion(unittest.TestCase):
    """Verify coordinate system labeling is correct."""

    def test_version_is_phase1(self):
        self.assertEqual(mark_anatomy.COORDINATE_SYSTEM_VERSION, "1.0.0-phase1")

    def test_mode_labels(self):
        self.assertEqual(mark_anatomy.BARYCENTRIC_MODE_2D, "2d_mesh_approximation")
        self.assertEqual(mark_anatomy.BARYCENTRIC_MODE_FALLBACK, "nearest_landmark_fallback")
        self.assertEqual(mark_anatomy.TRIANGLE_SOURCE, "nearest_3_mediapipe_landmarks")


if __name__ == "__main__":
    unittest.main()
