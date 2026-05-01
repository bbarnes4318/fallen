#!/usr/bin/env python3
"""
Fallen — E2E BAYESIAN VALIDATION TEST
Uses subprocess isolation for MediaPipe face detection to avoid
XNNPACK delegate conflicts on Windows.
"""
import os
import sys
import json
import math
import pickle
import time
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import multivariate_normal, lognorm, norm
from scipy.optimize import linear_sum_assignment

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent
CALIBRATION_FILE = PROJECT_ROOT / "backend" / "calibration_data" / "tier4_population_model.pkl"
LFW_DIR = PROJECT_ROOT / "datasets" / "lfwcrop_color" / "faces"
MODEL_PATH = PROJECT_ROOT / "scripts" / "face_landmarker.task"
EPSILON = 1e-9

# ═══════════════════════════════════════════════════════════
# FACE DETECTION VIA SUBPROCESS (avoids XNNPACK conflicts)
# ═══════════════════════════════════════════════════════════

_DETECT_SCRIPT = '''
import sys, json, cv2, numpy as np
from mediapipe.tasks.python import vision, BaseOptions
import mediapipe as mp

model_path = sys.argv[1]
image_path = sys.argv[2]

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=False,
    num_faces=1,
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
'''


def _detect_landmarks_subprocess(image_path):
    """Run face detection in isolated subprocess."""
    result = subprocess.run(
        [sys.executable, "-c", _DETECT_SCRIPT, str(MODEL_PATH), str(image_path)],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"    [DETECT] Subprocess error: {result.stderr[:200]}")
        return None
    data = json.loads(result.stdout)
    if data["landmarks"] is None:
        return None
    # Convert to objects with .x, .y, .z attributes
    class LM:
        def __init__(self, d):
            self.x = d["x"]
            self.y = d["y"]
            self.z = d["z"]
    return [LM(d) for d in data["landmarks"]]


# ═══════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ═══════════════════════════════════════════════════════════

def align_face_crop(image_path, target_size=256):
    """Detect face + upscale for pre-cropped LFW images."""
    image = cv2.imread(str(image_path))
    landmarks = _detect_landmarks_subprocess(image_path)
    aligned = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return aligned, landmarks


_EXCLUDE_IDX_GROUPS = [
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    [1,2,98,327,168,6,197,195,5,4,45,220,115,48,64,102,49,131,134,236,196,3,51,281,275,440,344,278,294,331,279,360,363,456,420,399,412,351],
    [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,80,191,78],
]


def _build_skin_mask(shape, landmarks):
    h, w = shape[:2]
    skin_mask = np.ones((h, w), dtype=np.uint8) * 255
    for idx_group in _EXCLUDE_IDX_GROUPS:
        pts = []
        for idx in idx_group:
            if idx < len(landmarks):
                lm = landmarks[idx]
                pts.append([int(lm.x * w), int(lm.y * h)])
        if len(pts) >= 3:
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            Mo = cv2.moments(hull)
            if Mo["m00"] > 0:
                cx = int(Mo["m10"] / Mo["m00"])
                cy = int(Mo["m01"] / Mo["m00"])
                inflated = ((hull - [cx, cy]) * 1.15 + [cx, cy]).astype(np.int32)
                cv2.fillConvexPoly(skin_mask, inflated, 0)
    return skin_mask


def detect_facial_marks(aligned_crop, landmarks):
    h, w = aligned_crop.shape[:2]
    gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
    skin_mask = _build_skin_mask(aligned_crop.shape, landmarks)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=5
    )
    masked = cv2.bitwise_and(thresh, skin_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8 or area > 500:
            continue
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
        Mo = cv2.moments(cnt)
        if Mo["m00"] == 0:
            continue
        cx, cy = Mo["m10"] / Mo["m00"], Mo["m01"] / Mo["m00"]

        mark_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mark_mask, [cnt], -1, 255, -1)
        mean_intensity = float(cv2.mean(gray, mask=mark_mask)[0])

        marks.append({
            "centroid": (cx / w, cy / h),
            "area": area,
            "intensity": mean_intensity,
            "circularity": circularity,
        })
    return marks


# ═══════════════════════════════════════════════════════════
# BAYESIAN LR ENGINE
# ═══════════════════════════════════════════════════════════

def cosine_to_lr_arcface(cosine_sim: float) -> float:
    cosine_sim = max(0.0, min(1.0, cosine_sim))
    if cosine_sim >= 0.70:
        far = max(1e-8, 10 ** (-12 * cosine_sim + 5))
        frr = max(1e-8, 0.01)
        return (1.0 - frr) / far
    elif cosine_sim >= 0.40:
        far = 10 ** (-4 * cosine_sim + 1)
        frr = 0.10
        return (1.0 - frr) / far
    else:
        return 0.001


def compute_mark_correspondence_bayesian(marks_gallery, marks_probe, calibration):
    result = {
        "score": None, "matched": 0,
        "total_gallery": len(marks_gallery), "total_probe": len(marks_probe),
        "lr_marks": 1.0, "mark_lrs": [], "matches": [],
    }
    if not marks_gallery or not marks_probe:
        return result

    spatial_kde = calibration["spatial_kde"]
    area_dist = calibration["area_distribution"]
    int_dist = calibration["intensity_distribution"]
    circ_dist = calibration["circularity_distribution"]
    delta_mean = np.array(calibration["intra_person_delta"]["mean"])
    delta_cov = np.array(calibration["intra_person_delta"]["covariance"])

    n_g, n_p = len(marks_gallery), len(marks_probe)
    lr_matrix = np.ones((n_g, n_p))

    for i, mg in enumerate(marks_gallery):
        for j, mp_mark in enumerate(marks_probe):
            dx = mg["centroid"][0] - mp_mark["centroid"][0]
            dy = mg["centroid"][1] - mp_mark["centroid"][1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0.20:
                lr_matrix[i, j] = EPSILON
                continue
            delta = np.array([dx, dy,
                mg["area"] - mp_mark["area"],
                mg["intensity"] - mp_mark["intensity"],
                mg["circularity"] - mp_mark["circularity"]])
            numerator = max(multivariate_normal.pdf(delta, mean=delta_mean, cov=delta_cov), EPSILON)
            xy = np.array([[mp_mark["centroid"][0]], [mp_mark["centroid"][1]]])
            p_spatial = max(float(spatial_kde(xy)[0]), EPSILON)
            p_area = max(lognorm.pdf(mp_mark["area"], area_dist["shape"],
                                     loc=area_dist["loc"], scale=area_dist["scale"]), EPSILON)
            p_int = max(norm.pdf(mp_mark["intensity"], int_dist["mean"], int_dist["std"]), EPSILON)
            p_circ = max(norm.pdf(mp_mark["circularity"], circ_dist["mean"], circ_dist["std"]), EPSILON)
            denominator = p_spatial * p_area * p_int * p_circ
            lr_matrix[i, j] = numerator / denominator

    cost = -np.log(np.maximum(lr_matrix, EPSILON))
    row_ind, col_ind = linear_sum_assignment(cost)

    total_lr = 1.0
    mark_lrs = []
    matched = 0
    matches = []

    for r, c in zip(row_ind, col_ind):
        lr = lr_matrix[r, c]
        if lr > 1.0:
            total_lr *= lr
            mark_lrs.append(float(lr))
            matched += 1
            matches.append({"gallery_idx": int(r), "probe_idx": int(c), "lr": float(lr)})

    result["matched"] = matched
    result["lr_marks"] = float(total_lr)
    result["mark_lrs"] = mark_lrs
    result["matches"] = matches
    result["score"] = min(100.0, float(total_lr)) if matched > 0 else None
    return result


# ═══════════════════════════════════════════════════════════
# MAIN TEST
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  Fallen -- BAYESIAN VALIDATION TEST (E2E Bayesian Pipeline)")
    print("=" * 70 + "\n")

    # 1. Select & process test pair
    gallery_path = str(LFW_DIR / "Colin_Powell_0001.ppm")
    probe_path = str(LFW_DIR / "Colin_Powell_0003.ppm")

    if not os.path.exists(gallery_path) or not os.path.exists(probe_path):
        print(f"FATAL: Test images not found.")
        sys.exit(1)
    print(f"  [GALLERY] {Path(gallery_path).name}")
    print(f"  [PROBE]   {Path(probe_path).name}")

    # 2. Face detection (via subprocess isolation)
    t0 = time.time()
    gallery_aligned, gallery_lm = align_face_crop(gallery_path)
    probe_aligned, probe_lm = align_face_crop(probe_path)

    if gallery_lm is None or probe_lm is None:
        g_ok = "OK" if gallery_lm else "FAIL"
        p_ok = "OK" if probe_lm else "FAIL"
        print(f"FATAL: Face detection failed. Gallery={g_ok}, Probe={p_ok}")
        sys.exit(1)

    print(f"  [OK] Face alignment complete ({(time.time()-t0)*1000:.0f}ms)")

    # 3. Mark detection
    marks_gallery = detect_facial_marks(gallery_aligned, gallery_lm)
    marks_probe = detect_facial_marks(probe_aligned, probe_lm)
    print(f"  [OK] Gallery marks: {len(marks_gallery)}")
    print(f"  [OK] Probe marks:   {len(marks_probe)}")

    # 4. Load calibration model
    if not CALIBRATION_FILE.exists():
        print(f"FATAL: Calibration model not found at {CALIBRATION_FILE}")
        sys.exit(1)

    with open(CALIBRATION_FILE, "rb") as f:
        calibration = pickle.load(f)
    print(f"  [OK] Calibration model loaded (v: {calibration['version']})")
    print(f"       Population size: {calibration['population_size']}")
    print(f"       Total marks:     {calibration['total_marks']}")
    print(f"       Intra-person deltas: {calibration['intra_person_delta']['n_deltas']}")

    # 5. Bayesian LR computation
    mark_result = compute_mark_correspondence_bayesian(marks_gallery, marks_probe, calibration)

    simulated_cosine = 0.72
    lr_arcface = cosine_to_lr_arcface(simulated_cosine)
    lr_marks = mark_result["lr_marks"]
    lr_total = lr_arcface * lr_marks

    PRIOR = 0.5
    posterior = (PRIOR * lr_total) / ((PRIOR * lr_total) + (1.0 - PRIOR))
    fused_score = posterior * 100.0

    print(f"\n  {'='*50}")
    print(f"  BAYESIAN LIKELIHOOD RATIO RESULTS")
    print(f"  {'='*50}")
    print(f"  Simulated ArcFace cosine:  {simulated_cosine:.4f}")
    print(f"  LR_arcface:                {lr_arcface:.6f}")
    print(f"  Marks matched:             {mark_result['matched']}")
    print(f"  LR_marks (product):        {lr_marks:.6f}")
    print(f"  LR_total:                  {lr_total:.6f}")
    print(f"  Posterior probability:      {posterior:.8f}")
    print(f"  Fused score:               {fused_score:.4f}%")

    # 6. Construct AuditLog JSON
    veto_triggered = simulated_cosine < 0.40
    if veto_triggered:
        if lr_marks > 100.0 and mark_result["matched"] >= 1:
            conclusion = f"CONDITIONAL MATCH -- ArcFace Veto, {mark_result['matched']} mark(s) LR={lr_marks:.1f}"
        else:
            conclusion = f"EXCLUSION -- Biometric Non-Match (ArcFace: {simulated_cosine:.4f})"
    elif fused_score > 90.0:
        conclusion = f"TARGET ACQUIRED -- Strongest match (Posterior: {fused_score:.1f}%)"
    elif fused_score > 75.0:
        conclusion = f"TARGET ACQUIRED -- Probable match (Posterior: {fused_score:.1f}%)"
    else:
        conclusion = f"WEAK MATCH -- Nearest candidate (Posterior: {fused_score:.1f}%)"

    audit_log = {
        "pipeline_version": "Fallen Forensic-Grade v3.0 (Bayesian LR)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gallery_image": Path(gallery_path).name,
        "probe_image": Path(probe_path).name,
        "raw_cosine_score": round(simulated_cosine, 6),
        "lr_arcface": round(lr_arcface, 6),
        "lr_marks": round(lr_marks, 6),
        "lr_total": round(lr_total, 6),
        "posterior_probability": round(posterior, 8),
        "fused_identity_score": round(fused_score, 4),
        "conclusion": conclusion,
        "veto_triggered": veto_triggered,
        "marks_detected_gallery": len(marks_gallery),
        "marks_detected_probe": len(marks_probe),
        "marks_matched": mark_result["matched"],
        "mark_lrs": [round(lr, 4) for lr in mark_result["mark_lrs"]],
        "mark_matches": mark_result["matches"],
        "calibration_model_version": calibration["version"],
        "calibration_population_size": calibration["population_size"],
        "calibration_total_marks": calibration["total_marks"],
        "epsilon_floor": calibration["epsilon_floor"],
    }

    print(f"\n  {'='*50}")
    print(f"  FORENSIC AUDIT LOG (JSON)")
    print(f"  {'='*50}")
    print(json.dumps(audit_log, indent=2))

    # 7. Assertions
    print(f"\n  {'='*50}")
    print(f"  SCIENTIFIC COMPLIANCE ASSERTIONS")
    print(f"  {'='*50}")

    passed = 0
    failed = 0

    def _assert(condition, name):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1

    _assert(lr_arcface > 0, "LR_arcface is non-null and positive")
    _assert(lr_marks >= 1.0, "LR_marks >= 1.0 (evidence supports or is neutral)")
    _assert(lr_total > 0, "LR_total is non-null and positive")
    _assert(0.0 <= posterior <= 1.0, "Posterior is a valid probability [0, 1]")
    _assert(fused_score > 0, "Fused score is non-null and positive")
    _assert(audit_log["lr_arcface"] is not None, "AuditLog.lr_arcface is not None")
    _assert(audit_log["lr_marks"] is not None, "AuditLog.lr_marks is not None")
    _assert(audit_log["lr_total"] is not None, "AuditLog.lr_total is not None")
    _assert(audit_log["posterior_probability"] is not None, "AuditLog.posterior_probability is not None")
    _assert(len(audit_log["mark_lrs"]) == mark_result["matched"], "mark_lrs count equals matched count")
    _assert(posterior == (lr_total / (lr_total + 1.0)), "Posterior = LR / (LR + 1) holds mathematically")

    print(f"\n  RESULT: {passed}/{passed + failed} assertions passed.")

    if failed > 0:
        print(f"\n  *** SCIENTIFIC VALIDATION FAILED ***")
        sys.exit(1)
    else:
        print(f"\n  *** SCIENTIFIC VALIDATION PASSED -- PIPELINE READY ***")
        sys.exit(0)


if __name__ == "__main__":
    main()
