#!/usr/bin/env python3
"""Mark Detector v2.1.0 — Unit Tests.
Imports directly from backend.mark_detector (pure module, no FastAPI/DB).
"""
import sys
import os
import json
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Add backend to path so we can import the pure module
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from mark_detector import detect_facial_marks, serialize_mark_descriptor, MARK_DETECTOR_VERSION

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "scripts" / "face_landmarker.task"

# ═══════════════════════════════════════════════════════════
# MOCK LANDMARKS (simple face geometry for synthetic tests)
# ═══════════════════════════════════════════════════════════

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = 1.0

def _make_mock_landmarks(n=478):
    """Generate plausible mock landmarks for a 256x256 face crop.
    Places key landmarks at anatomically reasonable positions."""
    lms = []
    for i in range(n):
        # Default: spread across face area
        lms.append(MockLandmark(0.5 + 0.15 * math.sin(i * 0.1), 0.5 + 0.15 * math.cos(i * 0.1)))

    # Face oval — large elliptical boundary covering most of the crop
    oval_idx = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
                152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
    for i, idx in enumerate(oval_idx):
        angle = 2 * math.pi * i / len(oval_idx) - math.pi / 2
        rx, ry = 0.42, 0.46  # Wide oval
        if idx < n:
            lms[idx] = MockLandmark(0.5 + rx * math.cos(angle), 0.50 + ry * math.sin(angle))

    # Eyes — small regions to exclude
    left_eye = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
    for i, idx in enumerate(left_eye):
        a = 2 * math.pi * i / len(left_eye)
        if idx < n:
            lms[idx] = MockLandmark(0.37 + 0.03 * math.cos(a), 0.38 + 0.015 * math.sin(a))

    right_eye = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
    for i, idx in enumerate(right_eye):
        a = 2 * math.pi * i / len(right_eye)
        if idx < n:
            lms[idx] = MockLandmark(0.63 + 0.03 * math.cos(a), 0.38 + 0.015 * math.sin(a))

    # Brows
    left_brow = [70,63,105,66,107,55,65,52,53,46]
    for i, idx in enumerate(left_brow):
        if idx < n:
            lms[idx] = MockLandmark(0.37 + 0.06 * i / len(left_brow), 0.33)

    right_brow = [300,293,334,296,336,285,295,282,283,276]
    for i, idx in enumerate(right_brow):
        if idx < n:
            lms[idx] = MockLandmark(0.63 + 0.06 * i / len(right_brow), 0.33)

    # Nose — ALL 38 indices must be overridden to keep exclusion zone small
    nose_idx = [1,2,98,327,168,6,197,195,5,4,45,220,115,48,64,102,49,131,134,236,
                196,3,51,281,275,440,344,278,294,331,279,360,363,456,420,399,412,351]
    for i, idx in enumerate(nose_idx):
        a = 2 * math.pi * i / len(nose_idx)
        if idx < n:
            lms[idx] = MockLandmark(0.50 + 0.025 * math.cos(a), 0.50 + 0.03 * math.sin(a))

    # Lips — ALL 40 indices must be overridden
    lips_idx = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,
                14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,
                13,82,81,80,191,78]
    for i, idx in enumerate(lips_idx):
        a = 2 * math.pi * i / len(lips_idx)
        if idx < n:
            lms[idx] = MockLandmark(0.50 + 0.05 * math.cos(a), 0.67 + 0.02 * math.sin(a))

    return lms


def _make_synthetic_face(size=256):
    """Create a synthetic face-like image: skin-toned gradient with basic features."""
    img = np.ones((size, size, 3), dtype=np.uint8)
    # Skin tone base
    img[:, :] = [180, 160, 140]  # BGR skin tone
    # Add slight gradient for realism
    for y in range(size):
        factor = 0.9 + 0.2 * (y / size)
        img[y, :] = np.clip(img[y, :] * factor, 0, 255).astype(np.uint8)
    # Gaussian blur for smoothness
    img = cv2.GaussianBlur(img, (5, 5), 2)
    return img


# ═══════════════════════════════════════════════════════════
# TEST A: Synthetic Dark Mole
# ═══════════════════════════════════════════════════════════

def test_synthetic_dark_mole():
    print("\n  TEST A: Synthetic dark mole detection")
    img = _make_synthetic_face()
    landmarks = _make_mock_landmarks()

    # Draw a dark mole on left cheek at (100, 140) — inside face oval
    mole_x, mole_y = 100, 140
    cv2.circle(img, (mole_x, mole_y), 6, (30, 25, 20), -1)  # Very dark, radius 6

    marks, rejected, occ, trace, overlays = detect_facial_marks(img, landmarks)

    # Find dark_lesion candidates near the mole
    found = False
    for m in marks:
        px, py = m["centroid_px"]
        dist = math.sqrt((px - mole_x)**2 + (py - mole_y)**2)
        if dist < 20 and m["channel"] in ("dark_lesion", "dark"):
            found = True
            break

    assert len(marks) > 0, f"Expected marks > 0, got {len(marks)}"
    assert found, f"No dark_lesion/dark candidate found near ({mole_x},{mole_y}). Marks: {[(m['channel'], m['centroid_px']) for m in marks[:5]]}"
    assert trace["final_valid_marks"] > 0
    print(f"    [PASS] Found {len(marks)} marks, dark mole detected near ({mole_x},{mole_y})")
    print(f"    Trace: {json.dumps({k:v for k,v in trace.items() if not k.startswith('roi')}, indent=6)}")


# ═══════════════════════════════════════════════════════════
# TEST B: Synthetic Bright/Linear Scar
# ═══════════════════════════════════════════════════════════

def test_synthetic_bright_scar():
    print("\n  TEST B: Synthetic bright/linear scar detection")
    img = _make_synthetic_face()
    landmarks = _make_mock_landmarks()

    # Draw a bright line scar on cheek area at y=140
    scar_y = 140
    scar_x_start, scar_x_end = 150, 200
    cv2.line(img, (scar_x_start, scar_y), (scar_x_end, scar_y), (240, 235, 230), 3)  # Very bright, thick

    marks, rejected, occ, trace, overlays = detect_facial_marks(img, landmarks)

    # Find bright_scar or linear_scar_v2 near the scar
    found = False
    scar_cx = (scar_x_start + scar_x_end) / 2
    for m in marks:
        px, py = m["centroid_px"]
        dist = math.sqrt((px - scar_cx)**2 + (py - scar_y)**2)
        if dist < 30 and m["channel"] in ("bright_scar", "linear_scar_v2", "light", "linear_scar"):
            found = True
            break

    assert len(marks) > 0, f"Expected marks > 0, got {len(marks)}"
    assert found, f"No bright/linear scar candidate near scar. Marks: {[(m['channel'], m['centroid_px']) for m in marks[:5]]}"
    print(f"    [PASS] Found {len(marks)} marks, scar detected near ({scar_cx},{scar_y})")


# ═══════════════════════════════════════════════════════════
# TEST C: JSON Serialization
# ═══════════════════════════════════════════════════════════

def test_json_serialization():
    print("\n  TEST C: JSON serialization safety")
    img = _make_synthetic_face()
    landmarks = _make_mock_landmarks()
    # Add a mark to ensure we have something to serialize
    cv2.circle(img, (100, 140), 6, (30, 25, 20), -1)

    marks, rejected, occ, trace, overlays = detect_facial_marks(img, landmarks)
    assert len(marks) > 0, "Need at least one mark for serialization test"

    for m in marks:
        s = serialize_mark_descriptor(m)
        # Must not contain 'contour'
        assert "contour" not in s, f"Serialized mark still contains 'contour' key"
        # Must not contain numpy arrays
        for k, v in s.items():
            assert not isinstance(v, np.ndarray), f"Key '{k}' is still a numpy array"
            assert not isinstance(v, (np.floating, np.integer)), f"Key '{k}' is numpy scalar: {type(v)}"
        # Must be JSON-serializable
        try:
            json.dumps(s)
        except (TypeError, ValueError) as e:
            assert False, f"Serialized mark not JSON-safe: {e}"

    print(f"    [PASS] {len(marks)} marks serialized cleanly, no numpy/contour leaks")


# ═══════════════════════════════════════════════════════════
# TEST D: Fallback Mode
# ═══════════════════════════════════════════════════════════

def test_fallback_mode():
    print("\n  TEST D: Fallback candidate mode")
    # Create a nearly uniform image (no real marks)
    img = np.ones((256, 256, 3), dtype=np.uint8) * 170
    img = cv2.GaussianBlur(img, (15, 15), 5)
    landmarks = _make_mock_landmarks()

    marks, rejected, occ, trace, overlays = detect_facial_marks(img, landmarks)

    assert "fallback_used" in trace, "Trace must include fallback_used"
    assert "fallback_candidates" in trace, "Trace must include fallback_candidates"
    assert "detector_status" in trace, "Trace must include detector_status"
    assert "fallback_lr_cap" in trace, "Trace must include fallback_lr_cap"
    assert "fallback_penalty_applied" in trace, "Trace must include fallback_penalty_applied"

    if trace["fallback_used"]:
        for m in marks:
            assert m.get("low_confidence") == True, f"Fallback mark missing low_confidence=True"
            assert m.get("fallback_generated") == True, f"Fallback mark missing fallback_generated=True"
        if len(marks) > 0:
            assert trace["detector_status"] == "LOW_CONFIDENCE_CANDIDATES"
            assert trace["fallback_lr_cap"] is not None
            assert trace["fallback_penalty_applied"] == True
            print(f"    [PASS] Fallback triggered: {len(marks)} low-confidence candidates")
        else:
            assert trace["detector_status"] == "NO_CANDIDATES"
            print(f"    [PASS] Fallback triggered but no candidates found — NO_CANDIDATES correct")
    else:
        # If strict pass found marks on uniform image, that's fine
        print(f"    [PASS] Strict pass found {len(marks)} marks (no fallback needed)")


# ═══════════════════════════════════════════════════════════
# TEST E: Known Visible Marks (conditional)
# ═══════════════════════════════════════════════════════════

_DETECT_SCRIPT = r"""
import sys, json, cv2, numpy as np
from mediapipe.tasks.python import vision, BaseOptions
import mediapipe as mp
model_path = sys.argv[1]
image_path = sys.argv[2]
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=False, num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(options)
img = cv2.imread(image_path)
if img.shape[0] < 128 or img.shape[1] < 128:
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = landmarker.detect(mp_img)
landmarker.close()
if not result.face_landmarks:
    print(json.dumps({"landmarks": None}))
else:
    lms = [{"x": l.x, "y": l.y, "z": l.z} for l in result.face_landmarks[0]]
    print(json.dumps({"landmarks": lms}))
"""


def test_known_visible_marks():
    print("\n  TEST E: Known visible marks (arnold_test.jpg)")
    img_path = PROJECT_ROOT / "arnold_test.jpg"
    if not img_path.exists():
        print("    [SKIP] arnold_test.jpg not found")
        return

    if not MODEL_PATH.exists():
        print("    [SKIP] face_landmarker.task not found")
        return

    # Get landmarks via subprocess (MediaPipe isolation)
    result = subprocess.run(
        [sys.executable, "-c", _DETECT_SCRIPT, str(MODEL_PATH), str(img_path)],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"    [SKIP] Landmark detection failed: {result.stderr[:200]}")
        return

    data = json.loads(result.stdout)
    if data["landmarks"] is None:
        print("    [SKIP] No landmarks detected")
        return

    lms = [MockLandmark(d["x"], d["y"], d["z"]) for d in data["landmarks"]]

    img = cv2.imread(str(img_path))
    aligned = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

    marks, rejected, occ, trace, overlays = detect_facial_marks(aligned, lms)

    assert len(marks) > 0, f"Expected raw marks > 0 on real face, got 0. Trace: {trace}"

    # Verify expanded trace fields
    for field in ["dark_lesion_initial_candidates", "bright_scar_initial_candidates",
                  "linear_scar_initial_candidates", "texture_anomaly_initial_candidates",
                  "strict_final_valid_marks", "fallback_used", "detector_status",
                  "fallback_lr_cap", "fallback_penalty_applied"]:
        assert field in trace, f"Trace missing expanded field: {field}"

    channels_found = set(m["channel"] for m in marks)
    print(f"    [PASS] {len(marks)} marks detected on real face")
    print(f"    Channels: {channels_found}")
    print(f"    Trace status: {trace['detector_status']}")
    print(f"    Strict: {trace['strict_final_valid_marks']}, Fallback: {trace['fallback_used']}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"  Mark Detector v{MARK_DETECTOR_VERSION} — Unit Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    tests = [
        ("A", test_synthetic_dark_mole),
        ("B", test_synthetic_bright_scar),
        ("C", test_json_serialization),
        ("D", test_fallback_mode),
        ("E", test_known_visible_marks),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"    [FAIL] Test {name}: {e}")
            failed += 1

    print(f"\n  RESULT: {passed}/{passed+failed} tests passed.")
    if failed > 0:
        print("  *** FAILURES DETECTED ***")
        sys.exit(1)
    else:
        print("  *** ALL TESTS PASSED ***")
    return 0


if __name__ == "__main__":
    main()
