"""Tests for backend/image_preprocessor.py

Validates the deterministic forensic preprocessor against synthetic images.
No external dependencies beyond opencv-python and numpy.
"""
import sys
import os
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_preprocessor import (
    sha256_image,
    encode_debug_png_b64,
    compute_quality_metrics,
    apply_lab_clahe,
    apply_retinex_lighting_normalization,
    build_skin_mask_from_landmarks,
    preprocess_for_mark_detection,
)


def _make_synthetic_face(size=256):
    """Create a low-contrast synthetic face-like BGR image for testing."""
    img = np.full((size, size, 3), 120, dtype=np.uint8)
    # Add a subtle gradient to simulate lighting variation
    for y in range(size):
        img[y, :, :] = np.clip(img[y, :, :].astype(np.int32) + (y - size // 2) // 4, 0, 255).astype(np.uint8)
    # Add a dark circle (simulated mole)
    cv2.circle(img, (size // 3, size // 3), 5, (60, 60, 60), -1)
    # Add a bright spot
    cv2.circle(img, (2 * size // 3, size // 3), 4, (200, 200, 200), -1)
    return img


# --- Test 1: preprocess_for_mark_detection returns all required keys ---

def test_preprocess_returns_all_keys():
    img = _make_synthetic_face(256)
    result = preprocess_for_mark_detection(img, landmarks=None, target_size=1024)

    required_top = [
        "decoded_hash", "aligned_pre_clahe_hash", "aligned_post_clahe_hash",
        "original_dimensions", "decoded_dimensions", "aligned_dimensions",
        "quality", "images", "debug_b64", "preprocessing_steps",
    ]
    for key in required_top:
        assert key in result, f"Missing top-level key: {key}"

    required_images = [
        "aligned_bgr", "lab_clahe_bgr", "illumination_normalized_bgr",
        "mark_detector_input_bgr", "skin_mask",
    ]
    for key in required_images:
        assert key in result["images"], f"Missing images key: {key}"

    required_debug = [
        "aligned_b64", "lab_clahe_b64", "illumination_normalized_b64",
        "mark_detector_input_b64", "skin_mask_b64",
    ]
    for key in required_debug:
        assert key in result["debug_b64"], f"Missing debug_b64 key: {key}"


# --- Test 2: output image dimensions are 1024x1024 ---

def test_output_dimensions_1024():
    img = _make_synthetic_face(256)
    result = preprocess_for_mark_detection(img, target_size=1024)
    aligned = result["images"]["aligned_bgr"]
    assert aligned.shape == (1024, 1024, 3), f"Expected (1024,1024,3), got {aligned.shape}"
    assert result["aligned_dimensions"] == "1024x1024"


# --- Test 3: hashes are non-empty strings ---

def test_hashes_are_nonempty():
    img = _make_synthetic_face(256)
    result = preprocess_for_mark_detection(img, target_size=1024)
    assert isinstance(result["decoded_hash"], str) and len(result["decoded_hash"]) == 64
    assert isinstance(result["aligned_pre_clahe_hash"], str) and len(result["aligned_pre_clahe_hash"]) == 64
    assert isinstance(result["aligned_post_clahe_hash"], str) and len(result["aligned_post_clahe_hash"]) == 64


# --- Test 4: lab_clahe_bgr differs from aligned_bgr on low-contrast image ---

def test_clahe_changes_image():
    img = _make_synthetic_face(256)
    result = preprocess_for_mark_detection(img, target_size=1024)
    aligned = result["images"]["aligned_bgr"]
    clahe = result["images"]["lab_clahe_bgr"]
    # They must not be identical on a low-contrast input
    assert not np.array_equal(aligned, clahe), "CLAHE should change a low-contrast image"
    # Pre and post hashes must differ
    assert result["aligned_pre_clahe_hash"] != result["aligned_post_clahe_hash"]


# --- Test 5: mark_detector_input_bgr is uint8 and same shape ---

def test_mark_input_shape_and_dtype():
    img = _make_synthetic_face(256)
    result = preprocess_for_mark_detection(img, target_size=1024)
    mark_input = result["images"]["mark_detector_input_bgr"]
    aligned = result["images"]["aligned_bgr"]
    assert mark_input.dtype == np.uint8
    assert mark_input.shape == aligned.shape


# --- Test 6: compute_quality_metrics returns all expected fields ---

def test_quality_metrics_fields():
    img = _make_synthetic_face(256)
    metrics = compute_quality_metrics(img)
    required = [
        "blur_laplacian", "exposure_mean", "exposure_std",
        "underexposed_fraction", "overexposed_fraction", "width", "height",
    ]
    for key in required:
        assert key in metrics, f"Missing quality metric: {key}"
    assert metrics["width"] == 256
    assert metrics["height"] == 256
    assert isinstance(metrics["blur_laplacian"], float)


# --- Test 7: encode_debug_png_b64 returns correct data URI ---

def test_encode_debug_png_b64():
    img = _make_synthetic_face(64)
    result = encode_debug_png_b64(img)
    assert result.startswith("data:image/png;base64,")
    # Verify it decodes back
    b64_part = result.split(",", 1)[1]
    raw = np.frombuffer(__import__("base64").b64decode(b64_part), dtype=np.uint8)
    decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (64, 64)


# --- Test 8: no generative dependencies ---

def test_no_generative_dependencies():
    """Verify that image_preprocessor does not import any generative
    restoration libraries (GFPGAN, CodeFormer, etc.)."""
    import image_preprocessor as mod
    source_lines = open(mod.__file__, "r", encoding="utf-8").readlines()
    banned = ["gfpgan", "codeformer", "basicsr", "realesrgan"]
    for line in source_lines:
        stripped = line.strip().lower()
        # Only check actual import statements
        if stripped.startswith("import ") or stripped.startswith("from "):
            for term in banned:
                assert term not in stripped, f"Banned generative import found: {term} in '{line.strip()}'"


# --- Additional: sha256_image determinism ---

def test_sha256_determinism():
    img = _make_synthetic_face(128)
    h1 = sha256_image(img)
    h2 = sha256_image(img)
    assert h1 == h2, "sha256_image must be deterministic"
    # Different shape same bytes should differ
    flat = img.reshape(-1, 3)
    h3 = sha256_image(flat)
    assert h1 != h3, "Different shapes must produce different hashes"


# --- Additional: apply_retinex is deterministic and preserves dtype ---

def test_retinex_deterministic_and_uint8():
    img = _make_synthetic_face(128)
    r1 = apply_retinex_lighting_normalization(img)
    r2 = apply_retinex_lighting_normalization(img)
    assert np.array_equal(r1, r2), "Retinex must be deterministic"
    assert r1.dtype == np.uint8
    assert r1.shape == img.shape


# --- Additional: build_skin_mask_from_landmarks returns None when no landmarks ---

def test_skin_mask_none_without_landmarks():
    result = build_skin_mask_from_landmarks((256, 256), None)
    assert result is None


# --- Additional: preprocess with target_size already matching ---

def test_preprocess_no_resize_needed():
    img = _make_synthetic_face(1024)
    result = preprocess_for_mark_detection(img, target_size=1024)
    assert result["images"]["aligned_bgr"].shape == (1024, 1024, 3)
    assert "no resize needed" in result["preprocessing_steps"]
