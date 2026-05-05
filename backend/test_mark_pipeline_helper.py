import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
import os

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["JWT_SECRET_KEY"] = "mock"
os.environ["JWT_SECRET"] = "mock"
os.environ["OPERATOR_PASSWORD"] = "mock"
os.environ["STRIPE_API_KEY"] = "mock"

# Mock heavy dependencies before importing main
sys.modules['deepface'] = MagicMock()
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.oauth2'] = MagicMock()
sys.modules['google.oauth2.service_account'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['stripe'] = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()

from backend.main import _run_mark_evidence_pipeline

@pytest.fixture
def mock_pipeline_deps():
    with patch("backend.main.align_face_crop") as mock_align, \
         patch("backend.image_preprocessor.preprocess_for_mark_detection") as mock_prep, \
         patch("backend.main.detect_facial_marks") as mock_detect, \
         patch("backend.main.match_marks_v2") as mock_match:
        
        # Default successful alignments
        mock_align.return_value = (np.zeros((256, 256, 3), dtype=np.uint8), [{"x": 0.5, "y": 0.5}])
        
        # Default preprocessing
        mock_prep.return_value = {
            "images": {"mark_detector_input_bgr": np.zeros((1024, 1024, 3), dtype=np.uint8)},
            "decoded_hash": "mockhash",
            "aligned_pre_clahe_hash": "mockhash2",
            "aligned_post_clahe_hash": "mockhash3",
            "original_dimensions": (500, 500),
            "decoded_dimensions": (500, 500),
            "aligned_dimensions": (256, 256),
            "quality": {},
            "preprocessing_steps": [],
            "preprocessor_version": "1.0.0",
            "debug_b64": {
                "aligned_b64": "data:image/png;base64,mock",
                "lab_clahe_b64": "data:image/png;base64,mock",
                "illumination_normalized_b64": "data:image/png;base64,mock",
                "mark_detector_input_b64": "data:image/png;base64,mock",
                "skin_mask_b64": "data:image/png;base64,mock"
            }
        }
        
        # Default marks
        mock_detect.return_value = (
            [{"centroid": (0.5, 0.5), "mark_type": "mole", "face_region": "cheek", "area": 10, "box": (0,0,10,10), "score": 0.9}],
            [],
            np.zeros((1024, 1024), dtype=np.uint8),
            {"detector_status": "OK", "input_is_preprocessed": True, "internal_clahe_applied": False},
            {}
        )
        
        # Default match
        mock_match.return_value = {
            "matched": 1,
            "score": 100.0,
            "lr_marks": 50.0,
            "mark_lrs": [50.0],
            "matches": [{"gallery_idx": 0, "probe_idx": 0, "lr": 50.0}],
            "rejected_candidates": [],
            "matcher_status": "OK",
            "calibration_status": "APPLIED"
        }
        
        yield {
            "align": mock_align,
            "prep": mock_prep,
            "detect": mock_detect,
            "match": mock_match
        }


def test_helper_probe_only(mock_pipeline_deps):
    probe_img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    result = _run_mark_evidence_pipeline(
        probe_img=probe_img,
        gallery_img=None,
        probe_file_hash="hash1",
        gallery_file_hash=None,
        mode="diagnostic"
    )
    
    assert result["mode"] == "probe_only"
    assert result["mark_match_status"] == "NOT_RUN_SINGLE_IMAGE"
    assert result["matcher_status"] == "NOT_RUN_SINGLE_IMAGE"
    assert result["lr_marks"] is None
    assert result["gallery_preprocessing"] is None
    assert len(result["accepted_correspondences"]) == 0
    assert result["mark_diagnostics"]["probe_detector_status"] == "OK"
    assert result["mark_diagnostics"]["gallery_detector_status"] == "NOT_PROVIDED"

    # thresholds and versions Check
    assert "detector_thresholds" in result
    assert "matcher_thresholds" in result
    assert "mark_detector_version" in result
    assert "mark_matcher_version" in result
    assert "preprocessor_version" in result

def test_helper_paired_mode(mock_pipeline_deps):
    probe_img = np.zeros((500, 500, 3), dtype=np.uint8)
    gallery_img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    result = _run_mark_evidence_pipeline(
        probe_img=probe_img,
        gallery_img=gallery_img,
        probe_file_hash="hash1",
        gallery_file_hash="hash2",
        mode="diagnostic"
    )
    
    assert result["mode"] == "paired"
    assert result["mark_match_status"] == "INSUFFICIENT_MARKS" # Since mock only returns 1 mark each
    assert result["matcher_status"] == "OK"
    assert result["lr_marks"] == 50.0
    assert result["individual_mark_lrs"] == [50.0]
    assert len(result["accepted_correspondences"]) == 1
    assert result["mark_diagnostics"]["probe_detector_status"] == "OK"
    assert result["mark_diagnostics"]["gallery_detector_status"] == "OK"
    assert result["probe_preprocessing"] is not None
    assert result["gallery_preprocessing"] is not None

def test_helper_exact_self_match(mock_pipeline_deps):
    probe_img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    result = _run_mark_evidence_pipeline(
        probe_img=probe_img,
        gallery_img=probe_img,
        probe_file_hash="same_hash",
        gallery_file_hash="same_hash",
        mode="diagnostic"
    )
    
    assert result["mode"] == "paired"
    assert result["mark_match_status"] == "EXACT_SELF_MATCH"
    assert result["matcher_status"] == "EXACT_SELF_MATCH"
    assert result["lr_marks"] == 1.0
    assert result["individual_mark_lrs"] == [1.0]
    for c in result["accepted_correspondences"]:
        assert c["lr"] == 1.0
    
    summary = result["mark_diagnostics"]["rejection_summary"]
    assert "self-match" in summary.lower() or "neutralized" in summary.lower() or "circular" in summary.lower()

def test_helper_face_detection_failure(mock_pipeline_deps):
    probe_img = np.zeros((500, 500, 3), dtype=np.uint8)
    # Force align to fail
    mock_pipeline_deps["align"].return_value = (None, None)
    
    result = _run_mark_evidence_pipeline(
        probe_img=probe_img,
        gallery_img=probe_img,
        probe_file_hash="hash1",
        gallery_file_hash="hash2",
        mode="diagnostic"
    )
    
    assert result["mark_match_status"] == "FACE_NOT_DETECTED"
    assert result["matcher_status"] == "NOT_RUN_FACE_DETECTION_FAILED"
    assert result["lr_marks"] is None
    assert "detector_thresholds" in result
    assert "mark_detector_version" in result

def test_helper_detector_trace_and_dynamic_dimensions(mock_pipeline_deps):
    probe_img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    result = _run_mark_evidence_pipeline(
        probe_img=probe_img,
        gallery_img=None,
        probe_file_hash="hash1",
        gallery_file_hash=None,
        mode="diagnostic",
        target_size=1024 # Testing target size plumbing
    )
    
    trace = result["mark_diagnostics"]["mark_detector_trace"]["probe"]
    assert trace["input_is_preprocessed"] is True
    assert trace["internal_clahe_applied"] is False
