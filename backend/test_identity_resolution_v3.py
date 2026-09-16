import numpy as np

from identity_resolution_v3 import (
    CalibratedFusionModel,
    Decision,
    IdentityProfile,
    MarkConstellation,
    MarkObservation,
    QualityEvidence,
    Visibility,
    build_fusion_features,
    cosine_similarity,
    mark_distinctiveness_information,
)


def test_cosine_similarity_is_raw_similarity_not_percentage():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_mark_constellation_produces_interpretable_features():
    a = MarkConstellation(nodes=(
        MarkObservation("a", "left_cheek", (0.25, 0.50), 0.03),
        MarkObservation("b", "forehead", (0.50, 0.25), 0.02),
    ))
    b = MarkConstellation(nodes=(
        MarkObservation("c", "left_cheek", (0.25, 0.50), 0.03),
        MarkObservation("d", "forehead", (0.50, 0.25), 0.02),
    ))
    result = a.compare(b)
    assert result["node_overlap"] == 1.0
    assert 0.0 <= result["score"] <= 1.0


def test_unobservable_marks_are_not_added_as_present_evidence():
    profile = IdentityProfile("person-1")
    mark = MarkObservation(
        "m1", "left_cheek", (0.3, 0.5), 0.02,
        visibility=Visibility.UNOBSERVABLE,
    )
    updated = profile.update(
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        quality=0.9,
        marks=[mark],
        temporal_value=2.0,
        admission_confidence=0.999,
    )
    assert updated is True
    assert "m1" not in profile.persistent_marks


def test_profile_rejects_low_confidence_admission():
    profile = IdentityProfile("person-1")
    updated = profile.update(
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        quality=0.9,
        marks=[],
        temporal_value=0.0,
        admission_confidence=0.90,
    )
    assert updated is False
    assert profile.observation_count == 0


def test_quality_is_an_evidence_channel():
    q = QualityEvidence(overall=0.2)
    features = build_fusion_features(
        global_similarity=0.8,
        local_similarity=0.6,
        geometry_similarity=0.7,
        asymmetry_similarity=0.5,
        quality=q,
        mark_correspondences=[],
        constellation_score=None,
        temporal_gap=10.0,
    )
    assert features["quality_overall"] == 0.2
    assert features["temporal_gap"] == 10.0


def test_missing_calibration_is_indeterminate():
    fusion = CalibratedFusionModel()
    assert fusion.predict_probability({"global_similarity": 0.9}) is None
    assert fusion.decision(None) is Decision.INDETERMINATE


def test_distinctiveness_information_never_invents_population_prevalence():
    assert mark_distinctiveness_information(None) is None
    assert mark_distinctiveness_information(0.01) > mark_distinctiveness_information(0.5)
