import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# 1. Setup environment to bypass heavy initializations and secrets
os.environ["JWT_SECRET"] = "test-secret"
os.environ["OPERATOR_PASSWORD"] = "test-pass"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["ALLOW_SCHEMA_MISMATCH"] = "true"

# Ensure backend directory is first in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 2. Mock heavy dependencies before importing main
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

# Override auth dependency
app.dependency_overrides[verify_jwt] = lambda: {"role": "operator"}

client = TestClient(app)

@pytest.fixture
def mock_pipeline():
    # Patch the heavy functions to prevent actual computer vision logic from running
    with patch("backend.main._resolve_mark_image") as mock_resolve, \
         patch("backend.main.align_face_crop") as mock_align, \
         patch("backend.image_preprocessor.preprocess_for_mark_detection") as mock_pp, \
         patch("backend.main.detect_facial_marks") as mock_detect, \
         patch("backend.main.match_marks_v2") as mock_match:
         
        dummy_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        
        # Default mock behaviors
        mock_resolve.side_effect = lambda b64, url, label: (dummy_img, f"hash_{label}")
        
        mock_align.return_value = (dummy_img, [{"x": 0.5, "y": 0.5}])
        
        mock_pp.return_value = {
            "images": {"mark_detector_input_bgr": dummy_img},
            "debug_b64": {"aligned_b64": "data:image/png;base64,dummy"},
            "decoded_hash": "mock_decoded_hash"
        }
        
        mock_detect.return_value = (
            [{"centroid": (0.5, 0.5), "mark_type": "mole", "face_region": "cheek", "area": 10, "box": (0,0,10,10), "score": 0.9}],
            [],
            np.zeros((1024, 1024), dtype=np.uint8),
            {"detector_status": "OK", "input_is_preprocessed": True, "internal_clahe_applied": False},
            {}
        )
        
        mock_match.return_value = {
            "matched": 1,
            "score": 100.0,
            "lr_marks": 50.0,
            "mark_lrs": [50.0],
            "matches": [{"gallery_idx": 0, "probe_idx": 0, "lr": 50.0}],
            "rejected_candidates": [],
            "matcher_status": "OK",
            "calibration_status": "LOADED",
            "matcher_version": "v2.0.0"
        }
        
        yield {
            "resolve": mock_resolve,
            "align": mock_align,
            "pp": mock_pp,
            "detect": mock_detect,
            "match": mock_match
        }

def test_marks_analyze_probe_only(mock_pipeline):
    payload = {
        "probe_b64": "dummy_b64"
    }
    
    response = client.post("/marks/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Assertions
    assert data["mode"] == "probe_only"
    assert data["mark_match_status"] == "NOT_RUN_SINGLE_IMAGE"
    assert data["mark_diagnostics"]["matcher_status"] == "NOT_RUN_SINGLE_IMAGE"
    assert data["lr_marks"] is None
    assert "probe_preprocessing" in data and data["probe_preprocessing"] is not None
    assert data.get("gallery_preprocessing") is None
    assert "detector_thresholds" in data
    assert "mark_detector_version" in data
    assert "mark_matcher_version" in data
    assert len(data["raw_probe_marks"]) == 1
    assert data["rejected_probe_marks"] == []
    assert data["accepted_correspondences"] == []

def test_marks_analyze_paired_mode(mock_pipeline):
    payload = {
        "probe_b64": "dummy_b64",
        "gallery_b64": "dummy_b64_gallery"
    }
    
    response = client.post("/marks/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Assertions
    assert data["mode"] == "paired"
    assert "accepted_correspondences" in data
    assert isinstance(data["accepted_correspondences"], list)
    assert "rejected_correspondences" in data
    assert isinstance(data["rejected_correspondences"], list)
    
    assert data["mark_diagnostics"]["accepted_correspondences_count"] == mock_pipeline["match"].return_value["matched"]
    assert data["mark_diagnostics"]["matcher_status"] == mock_pipeline["match"].return_value["matcher_status"]
    assert "calibration_status" in data["mark_diagnostics"]
    assert "matcher_thresholds" in data["lr_calculation_trace"]
    
    probe_trace = data["mark_diagnostics"]["mark_detector_trace"]["probe"]
    assert probe_trace["input_is_preprocessed"] is True
    assert probe_trace["internal_clahe_applied"] is False

def test_marks_analyze_exact_self_match(mock_pipeline):
    # Setup exactly the same hash for probe and gallery
    mock_pipeline["resolve"].side_effect = lambda b64, url, label: (np.zeros((10,10,3)), "same_hash_123")
    
    payload = {
        "probe_b64": "dummy_b64",
        "gallery_b64": "dummy_b64_gallery"
    }
    
    response = client.post("/marks/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["mark_match_status"] == "EXACT_SELF_MATCH"
    assert data["lr_marks"] == 1.0
    
    for match in data["accepted_correspondences"]:
        assert match["lr"] == 1.0
        
    summary = data["mark_diagnostics"]["rejection_summary"]
    assert "self-match" in summary.lower() or "neutralized" in summary.lower()

def test_marks_analyze_face_detection_failure(mock_pipeline):
    # Mock align_face_crop to return None for landmarks (failure)
    mock_pipeline["align"].return_value = (None, None)
    
    payload = {
        "probe_b64": "dummy_b64"
    }
    
    response = client.post("/marks/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["mark_match_status"] == "FACE_NOT_DETECTED"
    assert data["mark_diagnostics"]["matcher_status"] == "NOT_RUN_FACE_DETECTION_FAILED"
    assert "detector_thresholds" in data
    assert "mark_detector_version" in data
    assert "mark_matcher_version" in data
