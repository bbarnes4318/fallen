#!/usr/bin/env python3
"""
=============================================================================
  AURUMSHIELD — TIER 4 BAYESIAN POPULATION CALIBRATION ENGINE
  Daubert-Compliant Background Model for Likelihood Ratio Framework
=============================================================================

Processes the LFW dataset (~9,000+ images) to build:

  Phase 1 — Population Background Model P(E|Hd):
    - 2D Gaussian KDE over normalized (x, y) mark coordinates
    - Log-Normal distribution for mark area
    - Gaussian distributions for intensity and circularity

  Phase 2 — Intra-Person Variability Model P(E|Hp):
    - Multivariate Gaussian over 5-D delta vectors from matched pairs

Output:
  backend/calibration_data/tier4_population_model.pkl

Usage:
  python calibrate_tier4_population.py
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import time
import zipfile
import pickle
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
from scipy.stats import gaussian_kde, lognorm, norm
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    from google.cloud import storage as gcs
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

# ── MediaPipe Compatibility Layer ──
# Production uses mediapipe==0.10.11 (solutions API).
# Local dev (Python 3.13) uses mediapipe>=0.10.30 (tasks API only).
_USE_TASKS_API = False
face_mesh = None

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    )
except AttributeError:
    # MediaPipe 0.10.30+ dropped solutions API
    _USE_TASKS_API = True
    from mediapipe.tasks.python import vision, BaseOptions
    import mediapipe as mp

    _MODEL_PATH = os.path.join(Path(__file__).resolve().parent, "face_landmarker.task")
    if not os.path.exists(_MODEL_PATH):
        print(f"FATAL: face_landmarker.task not found at {_MODEL_PATH}")
        sys.exit(1)

    _landmarker_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        output_face_blendshapes=False,
        num_faces=1,
    )
    _face_landmarker = vision.FaceLandmarker.create_from_options(_landmarker_options)


def _process_face_mesh(rgb_image):
    """
    Compatibility wrapper: returns face landmarks in the same format
    regardless of which MediaPipe API is available.
    Returns list of landmark objects with .x, .y, .z attributes, or None.
    """
    if _USE_TASKS_API:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = _face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]  # List of NormalizedLandmark
    else:
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if os.getenv("K_SERVICE") or os.getenv("CLOUD_RUN_JOB"):
    DATASETS_DIR = "/tmp/datasets"
else:
    DATASETS_DIR = str(PROJECT_ROOT / "datasets")

GCS_BUCKET = "hoppwhistle-facial-uploads"
GCS_BLOB_PATH = "datasets/lfw_color.zip"
LFW_DIR = os.path.join(DATASETS_DIR, "lfwcrop_color", "faces")
OUTPUT_DIR = str(BACKEND_DIR / "calibration_data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tier4_population_model.pkl")



# ── Terminal colors ──
class _C:
    GOLD = "\033[38;2;212;175;55m"
    GREEN = "\033[38;2;0;255;128m"
    RED = "\033[38;2;255;60;60m"
    CYAN = "\033[38;2;80;200;255m"
    DIM = "\033[2m"
    RESET = "\033[0m"

def _ts():
    return datetime.now().strftime("%H:%M:%S")


# ═══════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS (replicated from main.py for standalone use)
# ═══════════════════════════════════════════════════════════

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def align_face_crop(image, target_size=256):
    """Canonical face alignment — mirrors main.py."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = _process_face_mesh(rgb)
    if landmarks is None:
        return cv2.resize(image, (target_size, target_size)), None

    h, w = image.shape[:2]

    le_idx = [33, 133, 160, 159, 158, 144, 145, 153]
    re_idx = [263, 362, 387, 386, 385, 373, 374, 380]
    le_c = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in le_idx], axis=0)
    re_c = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in re_idx], axis=0)

    angle = math.degrees(math.atan2(re_c[1] - le_c[1], re_c[0] - le_c[0]))
    mid = ((le_c[0] + re_c[0]) / 2, (le_c[1] + re_c[1]) / 2)
    M = cv2.getRotationMatrix2D(mid, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    rgb_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    lm_rot = _process_face_mesh(rgb_rot)
    lm = lm_rot if lm_rot is not None else landmarks

    xs = [l.x * w for l in lm]
    ys = [l.y * h for l in lm]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    fw, fh = x_max - x_min, y_max - y_min
    px, py = fw * 0.25, fh * 0.25
    x_min, x_max = max(0, int(x_min - px)), min(w, int(x_max + px))
    y_min, y_max = max(0, int(y_min - py)), min(h, int(y_max + py))

    cw, ch = x_max - x_min, y_max - y_min
    if cw > ch:
        d = cw - ch
        y_min, y_max = max(0, y_min - d // 2), min(h, y_max + (d - d // 2))
    elif ch > cw:
        d = ch - cw
        x_min, x_max = max(0, x_min - d // 2), min(w, x_max + (d - d // 2))

    cropped = rotated[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        cropped = rotated
    aligned = cv2.resize(cropped, (target_size, target_size))

    rgb_a = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    final_lm = _process_face_mesh(rgb_a)
    return aligned, final_lm


# ── Mark Detection (mirrors main.py Tier 4) ──
_EXCLUDE_IDX_GROUPS = [
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],  # L eye
    [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],  # R eye
    [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],  # L brow
    [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],  # R brow
    [1,2,98,327,168,6,197,195,5,4,45,220,115,48,64,102,49,131,134,236,196,3,51,281,275,440,344,278,294,331,279,360,363,456,420,399,412,351],  # Nose
    [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,80,191,78],  # Lips
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
            M = cv2.moments(hull)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                inflated = ((hull - [cx, cy]) * 1.15 + [cx, cy]).astype(np.int32)
                cv2.fillConvexPoly(skin_mask, inflated, 0)
    return skin_mask


def detect_facial_marks(aligned_crop, landmarks):
    """Detect facial marks — mirrors main.py detect_facial_marks()."""
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
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

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
# DATASET MANAGEMENT
# ═══════════════════════════════════════════════════════════

def download_and_extract_lfw():
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)
    archive_path = os.path.join(DATASETS_DIR, "lfw_color.zip")
    if not os.path.exists(LFW_DIR):
        if not os.path.exists(archive_path):
            if not GCS_AVAILABLE:
                print(f"  {_C.RED}FATAL: google-cloud-storage required.{_C.RESET}")
                sys.exit(1)
            print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.GOLD}DOWNLOADING LFW FROM GCS...{_C.RESET}")
            client = gcs.Client()
            bucket = client.bucket(GCS_BUCKET)
            bucket.blob(GCS_BLOB_PATH).download_to_filename(archive_path)
            print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.GREEN}DOWNLOAD COMPLETE.{_C.RESET}")
        print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.GOLD}EXTRACTING...{_C.RESET}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=DATASETS_DIR)
        os.remove(archive_path)
        print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.GREEN}EXTRACTION COMPLETE.{_C.RESET}")
    else:
        print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.CYAN}LFW DATASET FOUND.{_C.RESET}")


def discover_images():
    """Groups all images by person identity. Returns {name: [paths]}."""
    person_images = defaultdict(list)
    for root, _, files in os.walk(LFW_DIR):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".ppm", ".png")):
                fp = os.path.join(root, f)
                stem = Path(fp).stem
                person = "_".join(stem.split("_")[:-1])
                person_images[person].append(fp)
    return dict(person_images)


# ═══════════════════════════════════════════════════════════
# PHASE 1: POPULATION BACKGROUND MODEL — P(E|Hd)
# ═══════════════════════════════════════════════════════════

def extract_all_marks(person_images):
    """
    Run mark detection on every image. Returns:
      - all_marks: list of [x, y, area, intensity, circularity]
      - per_image_marks: {filepath: [mark_dicts]} for Phase 2 reuse
    """
    all_marks = []
    per_image_marks = {}
    all_paths = []
    for paths in person_images.values():
        all_paths.extend(paths)

    faces = 0
    errors = 0

    for filepath in tqdm(all_paths, desc="  Phase 1 — Extracting marks"):
        try:
            image = cv2.imread(filepath)
            if image is None:
                errors += 1
                continue
            aligned, landmarks = align_face_crop(apply_clahe(image))
            if landmarks is None:
                per_image_marks[filepath] = []
                continue
            faces += 1
            marks = detect_facial_marks(aligned, landmarks)
            per_image_marks[filepath] = marks
            for m in marks:
                all_marks.append([
                    m["centroid"][0], m["centroid"][1],
                    m["area"], m["intensity"], m["circularity"],
                ])
        except Exception:
            errors += 1
            per_image_marks[filepath] = []

    print(f"\n  {_C.GREEN}✓ Extraction complete{_C.RESET}")
    print(f"    Faces detected:  {faces}")
    print(f"    Marks collected: {len(all_marks)}")
    print(f"    Errors:          {errors}")
    return all_marks, per_image_marks


def fit_population_model(all_marks):
    """Fits KDE and parametric distributions to the population mark data."""
    if len(all_marks) < 100:
        print(f"  {_C.RED}FATAL: Insufficient marks ({len(all_marks)}).{_C.RESET}")
        sys.exit(1)

    data = np.array(all_marks)  # (N, 5)
    xy = data[:, :2].T          # (2, N)
    areas = data[:, 2]
    intensities = data[:, 3]
    circularities = data[:, 4]

    # 2D Gaussian KDE on spatial coordinates
    print(f"  {_C.CYAN}Fitting 2D spatial KDE on {len(all_marks)} marks...{_C.RESET}")
    spatial_kde = gaussian_kde(xy, bw_method="scott")
    print(f"    Bandwidth (Scott): {spatial_kde.factor:.6f}")

    # Log-Normal for area (positive, right-skewed)
    print(f"  {_C.CYAN}Fitting Log-Normal to area...{_C.RESET}")
    area_shape, area_loc, area_scale = lognorm.fit(areas, floc=0)
    print(f"    shape={area_shape:.4f}, scale={area_scale:.4f}")
    print(f"    Area: mean={np.mean(areas):.1f}, median={np.median(areas):.1f}")

    # Gaussian for intensity
    print(f"  {_C.CYAN}Fitting Gaussian to intensity...{_C.RESET}")
    int_mean, int_std = norm.fit(intensities)
    print(f"    μ={int_mean:.2f}, σ={int_std:.2f}")

    # Gaussian for circularity
    print(f"  {_C.CYAN}Fitting Gaussian to circularity...{_C.RESET}")
    circ_mean, circ_std = norm.fit(circularities)
    print(f"    μ={circ_mean:.4f}, σ={circ_std:.4f}")

    return {
        "spatial_kde": spatial_kde,
        "area_distribution": {"shape": float(area_shape), "loc": float(area_loc), "scale": float(area_scale)},
        "intensity_distribution": {"mean": float(int_mean), "std": float(int_std)},
        "circularity_distribution": {"mean": float(circ_mean), "std": float(circ_std)},
    }


# ═══════════════════════════════════════════════════════════
# PHASE 2: INTRA-PERSON VARIABILITY MODEL — P(E|Hp)
# ═══════════════════════════════════════════════════════════

def match_marks_spatial(marks_a, marks_b, max_dist=0.15):
    """
    Match marks between two images of the same person using
    spatial proximity via Hungarian optimal assignment.
    Returns list of (mark_a, mark_b) tuples.
    """
    if not marks_a or not marks_b:
        return []

    n_a, n_b = len(marks_a), len(marks_b)
    cost = np.full((n_a, n_b), 1e6)

    for i, ma in enumerate(marks_a):
        for j, mb in enumerate(marks_b):
            d = math.sqrt(
                (ma["centroid"][0] - mb["centroid"][0]) ** 2 +
                (ma["centroid"][1] - mb["centroid"][1]) ** 2
            )
            if d < max_dist:
                cost[i, j] = d

    row_ind, col_ind = linear_sum_assignment(cost)
    matched = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < max_dist:
            matched.append((marks_a[r], marks_b[c]))
    return matched


def build_intra_person_model(person_images, per_image_marks):
    """
    For all persons with 2+ images, compute mark delta vectors
    and fit a 5-D Multivariate Gaussian.
    """
    print(f"\n{_C.GOLD}═══ PHASE 2: INTRA-PERSON VARIABILITY MODEL ═══{_C.RESET}")

    # Find persons with 2+ images
    paired_persons = {k: v for k, v in person_images.items() if len(v) >= 2}
    print(f"  Persons with 2+ images: {len(paired_persons)}")

    all_deltas = []
    pairs_processed = 0
    pairs_with_marks = 0

    for person, paths in tqdm(paired_persons.items(), desc="  Phase 2 — Computing deltas"):
        # Generate all unique pairs for this person
        for i in range(len(paths)):
            for j in range(i + 1, min(len(paths), i + 4)):  # Cap at 3 pairs per person
                marks_a = per_image_marks.get(paths[i], [])
                marks_b = per_image_marks.get(paths[j], [])
                pairs_processed += 1

                matched = match_marks_spatial(marks_a, marks_b)
                if not matched:
                    continue

                pairs_with_marks += 1
                for ma, mb in matched:
                    delta = [
                        ma["centroid"][0] - mb["centroid"][0],   # Δx
                        ma["centroid"][1] - mb["centroid"][1],   # Δy
                        ma["area"] - mb["area"],                 # Δarea
                        ma["intensity"] - mb["intensity"],       # Δintensity
                        ma["circularity"] - mb["circularity"],   # Δcircularity
                    ]
                    all_deltas.append(delta)

    print(f"\n  {_C.GREEN}✓ Phase 2 complete{_C.RESET}")
    print(f"    Pairs processed:     {pairs_processed}")
    print(f"    Pairs with matches:  {pairs_with_marks}")
    print(f"    Total mark deltas:   {len(all_deltas)}")

    if len(all_deltas) < 50:
        print(f"  {_C.RED}WARNING: Low delta count ({len(all_deltas)}). "
              f"Model may be under-conditioned.{_C.RESET}")

    if len(all_deltas) < 10:
        print(f"  {_C.RED}FATAL: Insufficient deltas for covariance estimation.{_C.RESET}")
        sys.exit(1)

    deltas = np.array(all_deltas)  # (N, 5)
    mean_delta = np.mean(deltas, axis=0)
    cov_delta = np.cov(deltas, rowvar=False)

    # Regularize covariance to ensure positive-definiteness
    cov_delta += np.eye(5) * 1e-8

    print(f"\n  {_C.CYAN}Multivariate Gaussian fit (5-D):{_C.RESET}")
    dim_names = ["Δx", "Δy", "Δarea", "Δintensity", "Δcircularity"]
    for i, name in enumerate(dim_names):
        print(f"    {name}: μ={mean_delta[i]:+.6f}, σ={np.sqrt(cov_delta[i, i]):.6f}")

    return {
        "mean": mean_delta.tolist(),
        "covariance": cov_delta.tolist(),
        "n_deltas": len(all_deltas),
        "n_pairs": pairs_with_marks,
    }


# ═══════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════

def main():
    print(f"\n{_C.GOLD}══════════════════════════════════════════════════════════════════{_C.RESET}")
    print(f"  AURUMSHIELD — TIER 4 BAYESIAN CALIBRATION ENGINE")
    print(f"  Daubert-Compliant Population Model Builder")
    print(f"{_C.GOLD}══════════════════════════════════════════════════════════════════{_C.RESET}\n")

    t_start = time.time()

    # 1. Dataset
    download_and_extract_lfw()
    person_images = discover_images()
    total_images = sum(len(v) for v in person_images.values())
    total_persons = len(person_images)
    print(f"  Discovered {total_images} images across {total_persons} identities.\n")

    # 2. Phase 1 — Extract marks from entire population
    print(f"{_C.GOLD}═══ PHASE 1: POPULATION BACKGROUND MODEL ═══{_C.RESET}")
    all_marks, per_image_marks = extract_all_marks(person_images)
    population_model = fit_population_model(all_marks)

    # 3. Phase 2 — Intra-person variability from matched pairs
    intra_person_model = build_intra_person_model(person_images, per_image_marks)

    # 4. Assemble and serialize
    calibration = {
        "version": "AurumShield Tier4 Bayesian Calibration v1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": "LFW (Labeled Faces in the Wild)",
        "population_size": total_images,
        "total_marks": len(all_marks),
        "epsilon_floor": 1e-9,
        # Phase 1 models
        "spatial_kde": population_model["spatial_kde"],
        "area_distribution": population_model["area_distribution"],
        "intensity_distribution": population_model["intensity_distribution"],
        "circularity_distribution": population_model["circularity_distribution"],
        # Phase 2 model
        "intra_person_delta": intra_person_model,
    }

    # Save locally
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(calibration, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\n  {_C.GREEN}✓ Model saved: {OUTPUT_FILE} ({size_kb:.1f} KB){_C.RESET}")

    # Upload to GCS
    if GCS_AVAILABLE:
        try:
            client = gcs.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob("calibration/tier4_population_model.pkl")
            blob.upload_from_filename(OUTPUT_FILE)
            print(f"  {_C.GREEN}✓ Uploaded to gs://{GCS_BUCKET}/calibration/tier4_population_model.pkl{_C.RESET}")
        except Exception as e:
            print(f"  {_C.RED}GCS upload failed: {e}{_C.RESET}")

    elapsed = time.time() - t_start
    print(f"\n{_C.GOLD}══════════════════════════════════════════════════════════════════{_C.RESET}")
    print(f"  CALIBRATION COMPLETE")
    print(f"    Population marks:     {len(all_marks)}")
    print(f"    Intra-person deltas:  {intra_person_model['n_deltas']}")
    print(f"    Epsilon floor:        1e-9")
    print(f"    Time:                 {elapsed:.1f}s")
    print(f"{_C.GOLD}══════════════════════════════════════════════════════════════════{_C.RESET}\n")


if __name__ == "__main__":
    main()
