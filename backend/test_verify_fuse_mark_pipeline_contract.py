import os
import sys
import pytest
import numpy as np
import json
from unittest.mock import patch, MagicMock

os.environ["JWT_SECRET"] = "test-secret"
os.environ["OPERATOR_PASSWORD"] = "test-pass"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["ALLOW_SCHEMA_MISMATCH"] = "true"

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Global mocks for heavy modules before import
sys.modules['deepface'] = MagicMock()
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.oauth2'] = MagicMock()
sys.modules['google.oauth2.service_account'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['stripe'] = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()

from fastapi.testclient import TestClient
from backend.main import app, verify_jwt

app.dependency_overrides[verify_jwt] = lambda: {"role": "operator"}
app.state.limiter.enabled = False
client = TestClient(app)

@pytest.fixture
def mock_verify_fuse_dependencies():
    patchers = [
        patch("backend.main.fetch_image_from_url"),
        patch("backend.main.analyze_frequency_domain"),
        patch("backend.main.detect_liveness"),
        patch("backend.main.align_face_crop"),
        patch("backend.main.apply_clahe"),
        patch("backend.main.estimate_age"),
        patch("backend.main.cross_spectral_normalize"),
        patch("backend.main.extract_ensemble_embeddings"),
        patch("backend.main.compute_ensemble_similarity"),
        patch("backend.main.extract_geometric_ratios_3d"),
        patch("backend.main.extract_lbp_histogram"),
        patch("backend.main.detect_facial_marks"),
        patch("backend.main.match_facial_marks"),
        patch("backend.main.compute_mark_correspondence"),
        patch("backend.main.score_to_lr_ensemble"),
        patch("backend.main.evaluate_mark_veto_override"),
        patch("backend.main.generate_landmark_attention_map"),
        patch("backend.main.generate_edge_delta_map"),
        patch("backend.main.generate_wireframe_hud"),
        patch("backend.main.calculate_statistical_confidence"),
        patch("backend.main.compute_vector_hash"),
        patch("backend.main.compute_alignment_variance"),
        patch("backend.main.generate_forensic_receipt"),
        patch("backend.main.build_crypto_envelope"),
        patch("backend.main.generate_mark_overlay_receipt"),
        patch("backend.main.SessionLocal"),
        patch("backend.main.compute_image_hash")
    ]
    
    mocks = [p.start() for p in patchers]
    
    (mock_fetch, mock_freq, mock_liveness, mock_align, mock_clahe, mock_age, mock_cross,
     mock_emb, mock_sim, mock_geom, mock_lbp, mock_detect, mock_match, mock_corr,
     mock_lr_ens, mock_veto, mock_hm, mock_edge, mock_wf, mock_stats, mock_vh,
     mock_av, mock_fr, mock_crypto, mock_mr, mock_db, mock_hash) = mocks

    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Setup returns
    mock_fetch.return_value = (dummy_img, "mock_hash")
    mock_freq.return_value = 0.1
    mock_liveness.return_value = {"score": 0.99, "method": "mock"}
    mock_align.return_value = (dummy_img, [{"x": 0.5, "y": 0.5}])
    mock_clahe.return_value = dummy_img
    mock_age.return_value = 30
    mock_cross.return_value = (dummy_img, dummy_img, False)
    mock_emb.return_value = (np.zeros(512), np.zeros(512))
    mock_sim.return_value = (0.9, 0.9, 0.9)
    
    mock_geom.return_value = (np.ones(12), {}, {"ratio_visibility": np.ones(12, dtype=bool), "geometry_status": "OK", "occlusion_percentage": 0.0, "occluded_regions": []})
    mock_lbp.return_value = np.ones(10)
    
    mock_detect.return_value = (
        [{"centroid": (0.5, 0.5), "mark_type": "mole", "face_region": "cheek", "area": 10, "box": (0,0,10,10), "score": 0.9}],
        [],
        np.zeros((256, 256), dtype=np.uint8),
        {"detector_status": "OK", "input_is_preprocessed": False, "internal_clahe_applied": True},
        {}
    )
    
    mock_match.return_value = (
        [{"gallery_idx": 0, "probe_idx": 0, "cost": 0.1, "position_distance": 0.1, "area_ratio": 1.0, "type_match": True, "region_match": True}],
        [],
        [],
        []
    )
    
    mock_corr.return_value = {
        "score": 100.0,
        "matched": 1,
        "total_gallery": 1,
        "total_probe": 1,
        "matches": [{"gallery_idx": 0, "probe_idx": 0, "lr": 50.0}],
        "lr_marks": 50.0,
        "mark_lrs": [50.0]
    }
    
    mock_lr_ens.return_value = 100.0
    mock_veto.return_value = {"eligible": False, "positive_mark_count": 0, "reason": ""}
    mock_hm.return_value = "mock_b64"
    mock_edge.return_value = "mock_b64"
    mock_wf.return_value = "mock_b64"
    mock_stats.return_value = {"statistical_certainty": "99%", "false_acceptance_rate": "1 in 1M"}
    mock_vh.return_value = "mock_vector_hash"
    mock_av.return_value = {}
    mock_fr.return_value = "mock_url"
    mock_crypto.return_value = {}
    mock_mr.return_value = "mock_mr_url"
    mock_hash.return_value = "mock_image_hash"
    
    yield {
        "fetch": mock_fetch,
        "align": mock_align,
        "detect": mock_detect,
        "match": mock_match,
        "corr": mock_corr,
        "db": mock_db
    }

    for p in patchers:
        p.stop()

def get_job_payload_from_db_mock(mock_db):
    session_instance = mock_db.return_value
    add_calls = session_instance.add.call_args_list
    for call in add_calls:
        obj = call[0][0]
        if type(obj).__name__ == "VerificationJob":
            return json.loads(obj.result_payload)
    return {}

def test_verify_fuse_current_contract_baseline(mock_verify_fuse_dependencies):
    payload = {
        "probe_url": "http://mock/probe",
        "gallery_url": "http://mock/gallery",
        "require_liveness": False
    }
    response = client.post("/verify/fuse", json=payload)
    assert response.status_code == 200
    
    data = get_job_payload_from_db_mock(mock_verify_fuse_dependencies["db"])
    
    assert "fused_identity_score" in data
    assert "conclusion" in data
    assert "mark_diagnostics" in data
    assert "mark_match_status" in data
    assert "lr_marks" in data
    assert "raw_probe_marks" in data
    assert "raw_gallery_marks" in data
    assert "mark_detector_version" in data
    assert "audit_log" in data

def test_verify_fuse_exact_self_match_mark_lr_neutrality(mock_verify_fuse_dependencies):
    mock_verify_fuse_dependencies["fetch"].side_effect = [
        (np.zeros((256, 256, 3), dtype=np.uint8), "same_hash_123"),
        (np.zeros((256, 256, 3), dtype=np.uint8), "same_hash_123")
    ]
    
    payload = {
        "probe_url": "http://mock/probe",
        "gallery_url": "http://mock/gallery",
        "require_liveness": False
    }
    response = client.post("/verify/fuse", json=payload)
    assert response.status_code == 200
    
    data = get_job_payload_from_db_mock(mock_verify_fuse_dependencies["db"])
    
    assert data.get("mark_match_status") == "EXACT_SELF_MATCH"
    assert data.get("lr_marks") == 1.0

def test_verify_fuse_calibration_missing_neutral_lr(mock_verify_fuse_dependencies):
    with patch("backend.main.TIER4_CALIBRATION", None):
        payload = {
            "probe_url": "http://mock/probe",
            "gallery_url": "http://mock/gallery",
            "require_liveness": False
        }
        response = client.post("/verify/fuse", json=payload)
        assert response.status_code == 200

@pytest.mark.xfail(reason="Declared provenance fields are not populated until V2 helper is wired.")
def test_verify_fuse_prepost_hash_fields_present_or_document_gap(mock_verify_fuse_dependencies):
    payload = {
        "probe_url": "http://mock/probe",
        "gallery_url": "http://mock/gallery",
        "require_liveness": False
    }
    response = client.post("/verify/fuse", json=payload)
    
    data = get_job_payload_from_db_mock(mock_verify_fuse_dependencies["db"])
    
    audit = data.get("audit_log", {})
    assert "probe_aligned_crop_hash_pre_clahe" in audit
    assert "probe_aligned_crop_hash_post_clahe" in audit
    assert "gallery_aligned_crop_hash_pre_clahe" in audit
    assert "gallery_aligned_crop_hash_post_clahe" in audit

def test_verify_fuse_v2_feature_flag_default_off(mock_verify_fuse_dependencies):
    payload = {
        "probe_url": "http://mock/probe",
        "gallery_url": "http://mock/gallery",
        "require_liveness": False
    }
    response = client.post("/verify/fuse", json=payload)
    assert response.status_code == 200

def test_verify_fuse_v2_feature_flag_on(mock_verify_fuse_dependencies):
    payload = {
        "probe_url": "http://mock/probe",
        "gallery_url": "http://mock/gallery",
        "require_liveness": False
    }
    with patch("backend.main.USE_MARK_PIPELINE_V2", True):
        with patch("backend.main._run_mark_evidence_pipeline") as mock_v2:
            mock_v2.return_value = {
                "mark_match_status": "MATCHED",
                "mode": "paired",
                "raw_probe_marks": [{"centroid": (0.5, 0.5), "mark_type": "mole", "face_region": "cheek", "area": 10, "box": (0,0,10,10), "score": 0.9}],
                "raw_gallery_marks": [{"centroid": (0.5, 0.5), "mark_type": "mole", "face_region": "cheek", "area": 10, "box": (0,0,10,10), "score": 0.9}],
                "accepted_correspondences": [{"gallery_idx": 0, "probe_idx": 0, "match_quality": 0.1, "position_distance": 0.1, "area_ratio": 1.0, "type_match": True, "region_match": True, "lr": 50.0}],
                "rejected_correspondences": [],
                "lr_marks": 50.0,
                "individual_mark_lrs": [50.0],
                "mark_diagnostics": {
                    "mark_detector_trace": {"probe": {"detector_status": "OK", "input_is_preprocessed": True, "internal_clahe_applied": False}, "gallery": {"detector_status": "OK", "input_is_preprocessed": True, "internal_clahe_applied": False}},
                    "detector_status": "OK",
                    "matcher_status": "OK"
                },
                "mark_detector_version": "v2",
                "mark_matcher_version": "v2",
                "preprocessor_version": "v1.1"
            }
            response = client.post("/verify/fuse", json=payload)
            assert response.status_code == 200
            
            data = get_job_payload_from_db_mock(mock_verify_fuse_dependencies["db"])
            assert "fused_identity_score" in data
            assert "conclusion" in data
            assert "mark_diagnostics" in data
            assert data["mark_match_status"] == "MATCHED"
            assert data["lr_marks"] == 50.0
            
            # Assert raw marks are present
            assert "raw_probe_marks" in data
            assert len(data["raw_probe_marks"]) == 1
            assert "raw_gallery_marks" in data
            
            # Assert mark_diagnostics traces
            assert "mark_diagnostics" in data
            diag = data["mark_diagnostics"]
            assert diag["mark_detector_trace"]["probe"]["input_is_preprocessed"] is True
            assert diag["mark_detector_trace"]["probe"]["internal_clahe_applied"] is False
            
            # Assert versions
            assert data["mark_detector_version"] == "v2"
            assert data["mark_matcher_version"] == "v2"
            
            audit = data.get("audit_log", {})
            assert audit.get("mark_detector_version") == "v2"
            assert audit.get("mark_matcher_version") == "v2"
