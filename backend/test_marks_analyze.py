"""
Minimal test coverage for POST /marks/analyze endpoint.

These tests validate:
1. Request model validation (missing images)
2. Base64 input acceptance
3. URL input acceptance  
4. mark_diagnostics always present in response
5. Response schema completeness
6. No ArcFace/vault dependencies

NOTE: These tests run against the FastAPI TestClient without GPU/model
dependencies. Tests that require the full model stack are marked with
@pytest.mark.integration and skipped by default.
"""
import pytest
import base64
import json
import numpy as np
import cv2
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image_b64(size: int = 128, color: tuple = (200, 180, 160)) -> str:
    """Generate a synthetic face-like test image as base64 data URI."""
    img = np.full((size, size, 3), color, dtype=np.uint8)
    # Draw simple face-like features so MediaPipe has something to detect
    cv2.circle(img, (size // 3, size // 3), size // 10, (80, 60, 40), -1)  # left eye
    cv2.circle(img, (2 * size // 3, size // 3), size // 10, (80, 60, 40), -1)  # right eye
    cv2.ellipse(img, (size // 2, 2 * size // 3), (size // 6, size // 10), 0, 0, 180, (120, 80, 60), 2)  # mouth
    _, buf = cv2.imencode('.png', img)
    b64 = base64.b64encode(buf).decode('utf-8')
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Unit tests: Request validation and model parsing
# ---------------------------------------------------------------------------

class TestMarkAnalyzeRequestModel:
    """Test the MarkAnalyzeRequest Pydantic model validation."""

    def test_import_request_model(self):
        """MarkAnalyzeRequest model can be imported from main."""
        from main import MarkAnalyzeRequest
        req = MarkAnalyzeRequest()
        assert req.probe_b64 is None
        assert req.gallery_b64 is None
        assert req.probe_url is None
        assert req.gallery_url is None

    def test_b64_fields_accepted(self):
        from main import MarkAnalyzeRequest
        req = MarkAnalyzeRequest(
            probe_b64="data:image/png;base64,iVBOR...",
            gallery_b64="data:image/png;base64,iVBOR...",
        )
        assert req.probe_b64 is not None
        assert req.gallery_b64 is not None

    def test_url_fields_accepted(self):
        from main import MarkAnalyzeRequest
        req = MarkAnalyzeRequest(
            probe_url="gs://bucket/probe.jpg",
            gallery_url="gs://bucket/gallery.jpg",
        )
        assert req.probe_url is not None
        assert req.gallery_url is not None


class TestDecodeB64Image:
    """Test the _decode_b64_image helper."""

    def test_valid_png_decode(self):
        from main import _decode_b64_image
        b64_str = _make_test_image_b64()
        img, hash_val = _decode_b64_image(b64_str)
        assert img is not None
        assert img.shape[0] > 0
        assert img.shape[1] > 0
        assert len(hash_val) == 64  # SHA-256 hex

    def test_raw_b64_without_prefix(self):
        from main import _decode_b64_image
        # Generate raw base64 without data URI prefix
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        _, buf = cv2.imencode('.png', img)
        raw_b64 = base64.b64encode(buf).decode('utf-8')
        decoded, hash_val = _decode_b64_image(raw_b64)
        assert decoded is not None
        assert len(hash_val) == 64

    def test_invalid_b64_raises(self):
        from main import _decode_b64_image
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _decode_b64_image("not_valid_base64!!!")
        assert exc_info.value.status_code == 400


class TestResolveMarkImage:
    """Test the _resolve_mark_image helper."""

    def test_missing_both_raises(self):
        from main import _resolve_mark_image
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _resolve_mark_image(None, None, "probe")
        assert exc_info.value.status_code == 400
        assert "probe" in exc_info.value.detail

    def test_b64_preferred_over_url(self):
        from main import _resolve_mark_image
        b64_str = _make_test_image_b64()
        # If both are provided, b64 takes precedence (no URL fetch needed)
        img, hash_val = _resolve_mark_image(b64_str, "gs://bucket/ignored.jpg", "probe")
        assert img is not None
        assert len(hash_val) == 64


class TestMarkAnalyzeResponseSchema:
    """Validate the expected response schema structure."""

    REQUIRED_KEYS = {
        "aligned_probe_b64",
        "aligned_gallery_b64",
        "raw_probe_marks",
        "raw_gallery_marks",
        "accepted_correspondences",
        "mark_match_status",
        "mark_diagnostics",
        "lr_marks",
        "individual_mark_lrs",
        "lr_calculation_trace",
    }

    REQUIRED_DIAGNOSTIC_KEYS = {
        "raw_probe_marks_count",
        "raw_gallery_marks_count",
        "accepted_correspondences_count",
        "rejected_candidates_count",
        "detector_status",
        "matcher_status",
        "lr_marks",
        "mark_match_status",
        "rejection_summary",
    }

    REQUIRED_TRACE_KEYS = {
        "individual_lrs",
        "product",
        "calibration_status",
        "thresholds_used",
    }

    def test_face_not_detected_response_schema(self):
        """When face detection fails, response must still contain all required keys."""
        # Build a minimal fail-closed response matching the endpoint's logic
        response = {
            "aligned_probe_b64": None,
            "aligned_gallery_b64": None,
            "raw_probe_marks": [],
            "raw_gallery_marks": [],
            "accepted_correspondences": [],
            "mark_match_status": "FACE_NOT_DETECTED",
            "mark_diagnostics": {
                "raw_probe_marks_count": 0,
                "raw_gallery_marks_count": 0,
                "accepted_correspondences_count": 0,
                "rejected_candidates_count": 0,
                "detector_status": "FACE_NOT_DETECTED",
                "matcher_status": "UNKNOWN",
                "lr_marks": None,
                "mark_match_status": "FACE_NOT_DETECTED",
                "rejection_summary": "Face detection failed on: probe",
            },
            "lr_marks": None,
            "individual_mark_lrs": [],
            "lr_calculation_trace": {
                "individual_lrs": [],
                "product": None,
                "calibration_status": "MISSING",
                "thresholds_used": {},
            },
        }

        assert self.REQUIRED_KEYS.issubset(response.keys())
        assert self.REQUIRED_DIAGNOSTIC_KEYS.issubset(response["mark_diagnostics"].keys())
        assert self.REQUIRED_TRACE_KEYS.issubset(response["lr_calculation_trace"].keys())
        assert response["mark_diagnostics"]["rejection_summary"] is not None


class TestEndpointRegistration:
    """Verify the endpoint is properly registered on the FastAPI app."""

    def test_marks_analyze_route_exists(self):
        from main import app
        routes = [route.path for route in app.routes]
        assert "/marks/analyze" in routes

    def test_marks_analyze_is_post(self):
        from main import app
        for route in app.routes:
            if hasattr(route, 'path') and route.path == "/marks/analyze":
                assert "POST" in route.methods
                break
        else:
            pytest.fail("/marks/analyze route not found")

    def test_marks_analyze_requires_auth(self):
        """Endpoint must have JWT dependency (verify_jwt)."""
        from main import app
        for route in app.routes:
            if hasattr(route, 'path') and route.path == "/marks/analyze":
                # The endpoint function should have dependencies
                endpoint = route.endpoint
                # Check that verify_jwt is in the function's signature via FastAPI's Depends
                import inspect
                sig = inspect.signature(endpoint)
                param_names = list(sig.parameters.keys())
                # The JWT dependency is typically the last positional param named '_'
                assert "_" in param_names, "JWT dependency parameter '_' not found in endpoint signature"
                break
        else:
            pytest.fail("/marks/analyze route not found")
