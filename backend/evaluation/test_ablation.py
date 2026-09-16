import pytest

from evaluation.ablation import run_ablation


def _rows():
    rows = []
    for i in range(12):
        same = i % 2 == 0
        rows.append({
            "subject_a": f"cal-a-{i}",
            "subject_b": f"cal-b-{i}",
            "same_identity": same,
            "split": "calibration",
            "global_similarity": 0.85 if same else 0.20,
            "geometry_similarity": 0.80 if same else 0.30,
            "legacy_mark_score": 0.70 if same else 0.10,
            "local_similarity": 0.90 if same else 0.20,
            "mark_similarity": 0.85 if same else 0.10,
            "constellation_score": 0.88 if same else 0.12,
            "quality_overall": 0.9,
            "pose_delta": 0.1,
            "temporal_gap": 0.0,
            "occlusion_delta": 0.0,
        })
    for i in range(12):
        same = i % 2 == 0
        rows.append({
            "subject_a": f"eval-a-{i}",
            "subject_b": f"eval-b-{i}",
            "same_identity": same,
            "split": "eval",
            "global_similarity": 0.82 if same else 0.22,
            "geometry_similarity": 0.78 if same else 0.31,
            "legacy_mark_score": 0.68 if same else 0.11,
            "local_similarity": 0.89 if same else 0.21,
            "mark_similarity": 0.83 if same else 0.11,
            "constellation_score": 0.86 if same else 0.13,
            "quality_overall": 0.9,
            "pose_delta": 0.1,
            "temporal_gap": 0.0,
            "occlusion_delta": 0.0,
        })
    return rows


def test_ablation_runs_without_identity_leakage():
    results = run_ablation(_rows())
    assert len(results) == 7
    assert {r.variant for r in results} == {
        "global_only",
        "global_geometry",
        "global_legacy_marks",
        "global_local",
        "global_local_marks",
        "global_local_constellation",
        "full_v3",
    }


def test_ablation_rejects_identity_overlap():
    rows = _rows()
    rows[-1]["subject_a"] = "cal-a-0"
    with pytest.raises(ValueError, match="identity leakage"):
        run_ablation(rows)
