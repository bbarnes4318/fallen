"""
LFW Benchmark Calibration Script for Fallen
=================================================
Fetches the official LFW dataset via Hugging Face Datasets CDN and evaluates
the full biometric pipeline (ArcFace cosine, geometric L2, LBP chi-squared)
against all standardized face pairs.

Outputs:
  - calibration_data/lfw_calibration.json  (local backup)
  - gs://hoppwhistle-facial-uploads/calibration/lfw_calibration.json  (GCS)

Usage (Cloud Run Job):
  python calibrate_lfw.py

Prerequisites:
  - Network access to huggingface.co CDN
  - ArcFace model available (DeepFace)
  - MediaPipe face mesh
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import numpy as np
import json
import datetime
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import mediapipe as mp
from skimage.feature import local_binary_pattern
from deepface import DeepFace
from sklearn.metrics import roc_curve
from datasets import load_dataset
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

# ── Google Cloud Storage (optional — for backup upload) ──
try:
    from google.cloud import storage as gcs
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

# ── Constants ──
GCS_BUCKET = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lfw_calibration.json")

# ── MediaPipe Mesh ──
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# ── Preload ArcFace ──
print("Preloading ArcFace model...")
try:
    DeepFace.build_model("ArcFace")
    print("ArcFace model ready.")
except Exception as e:
    print(f"FATAL: ArcFace preload failed: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS (mirrors main.py exactly)
# ═══════════════════════════════════════════════════════════

def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def align_face_crop(image: np.ndarray, target_size: int = 256):
    """Canonical alignment — mirrors main.py align_face_crop()."""
    import math
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        resized = cv2.resize(image, (target_size, target_size))
        return resized, None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]

    left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
    right_eye_indices = [263, 362, 387, 386, 385, 373, 374, 380]

    left_eye_center = np.mean(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in left_eye_indices], axis=0
    )
    right_eye_center = np.mean(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in right_eye_indices], axis=0
    )

    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = math.degrees(math.atan2(dy, dx))

    center = ((left_eye_center[0] + right_eye_center[0]) / 2,
              (left_eye_center[1] + right_eye_center[1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    xs = [landmarks[i].x * w for i in range(len(landmarks))]
    ys = [landmarks[i].y * h for i in range(len(landmarks))]
    x_min_lm, x_max_lm = min(xs), max(xs)
    y_min_lm, y_max_lm = min(ys), max(ys)

    face_w = x_max_lm - x_min_lm
    face_h = y_max_lm - y_min_lm
    pad_w = face_w * 0.25
    pad_h = face_h * 0.35

    x_min = max(0, int(x_min_lm - pad_w))
    x_max = min(w, int(x_max_lm + pad_w))
    y_min = max(0, int(y_min_lm - pad_h))
    y_max = min(h, int(y_max_lm + pad_h))

    cropped = rotated[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        cropped = rotated

    aligned = cv2.resize(cropped, (target_size, target_size))

    rgb_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    results_aligned = face_mesh.process(rgb_aligned)

    final_landmarks = None
    if results_aligned.multi_face_landmarks:
        final_landmarks = results_aligned.multi_face_landmarks[0].landmark

    return aligned, final_landmarks


def extract_arcface_embedding(image: np.ndarray) -> np.ndarray:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = DeepFace.represent(
        img_path=rgb_image,
        model_name="ArcFace",
        enforce_detection=False,
        detector_backend="skip",
    )
    return np.array(result[0]["embedding"], dtype=np.float64)


def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def extract_geometric_ratios(landmarks) -> np.ndarray:
    coords = np.array([(l.x, l.y) for l in landmarks])
    left_eye = coords[33]
    right_eye = coords[263]
    iod = np.linalg.norm(right_eye - left_eye)
    if iod < 1e-6:
        return np.zeros(12)

    nose_tip = coords[1]
    nose_bridge = coords[6]
    chin = coords[152]
    left_mouth = coords[61]
    right_mouth = coords[291]
    forehead_top = coords[10]
    left_jaw = coords[234]
    right_jaw = coords[454]
    left_eyebrow = coords[70]
    right_eyebrow = coords[300]
    jaw_to_chin_r = np.linalg.norm(right_jaw - chin)

    ratios = np.array([
        np.linalg.norm(nose_tip - chin) / iod,
        np.linalg.norm(nose_bridge - nose_tip) / iod,
        np.linalg.norm(left_mouth - right_mouth) / iod,
        np.linalg.norm(forehead_top - chin) / iod,
        np.linalg.norm(left_jaw - right_jaw) / iod,
        np.linalg.norm(left_eyebrow - left_eye) / iod,
        np.linalg.norm(right_eyebrow - right_eye) / iod,
        np.linalg.norm(nose_tip - left_eye) / iod,
        np.linalg.norm(nose_tip - right_eye) / iod,
        np.linalg.norm(chin - left_mouth) / iod,
        np.linalg.norm(chin - right_mouth) / iod,
        np.linalg.norm(left_jaw - chin) / jaw_to_chin_r if jaw_to_chin_r > 1e-6 else 1.0,
    ])
    return ratios


def extract_lbp_histogram(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def chi_squared_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    return 0.5 * float(np.sum(((h1 - h2) ** 2) / (h1 + h2 + 1e-10)))



# ═══════════════════════════════════════════════════════════
# CALIBRATION ENGINE
# ═══════════════════════════════════════════════════════════

def run_calibration():
    """
    Main calibration procedure.
    1. Fetch LFW dataset via Hugging Face Datasets CDN
    2. Build genuine + impostor pairs from the dataset
    3. Process all pairs through the full pipeline
    4. Compute ROC, EER, FAR at specific thresholds
    5. Output calibration JSON
    """
    # 1. Fetch LFW dataset from Hugging Face CDN
    print("Fetching LFW dataset via Hugging Face Datasets CDN...")
    ds = load_dataset("logasja/lfw", split="train", trust_remote_code=True)
    print(f"Loaded {len(ds)} LFW images from Hugging Face.")

    # 2. Index images by person label for pair construction
    from collections import defaultdict
    person_images = defaultdict(list)
    for row in ds:
        label = row["label"]
        img = row["image"]  # PIL Image
        person_images[label].append(np.array(img))  # Convert PIL to numpy RGB

    # Build genuine pairs: same person, different images
    # Cap at 3,000 to match standard LFW protocol (10 folds × 300 pairs)
    MAX_PAIRS = 3000
    all_genuine_candidates = []
    for label, images in person_images.items():
        if len(images) >= 2:
            # Take up to 1 pair per person to maximize diversity
            all_genuine_candidates.append((images[0], images[1], 1))
            # If the person has more images, add more pairs (up to 3)
            for k in range(2, min(len(images), 4)):
                all_genuine_candidates.append((images[0], images[k], 1))

    import random
    random.seed(42)
    random.shuffle(all_genuine_candidates)
    genuine_pairs = all_genuine_candidates[:MAX_PAIRS]

    # Build impostor pairs: different persons (sample to match genuine count)
    person_labels = [l for l, imgs in person_images.items() if len(imgs) >= 1]
    impostor_pairs = []
    target_impostor = len(genuine_pairs)  # Match genuine count for balance
    attempts = 0
    while len(impostor_pairs) < target_impostor and attempts < target_impostor * 10:
        l1, l2 = random.sample(person_labels, 2)
        img1 = random.choice(person_images[l1])
        img2 = random.choice(person_images[l2])
        impostor_pairs.append((img1, img2, 0))
        attempts += 1

    all_pairs_data = genuine_pairs + impostor_pairs
    random.shuffle(all_pairs_data)

    n_genuine = len(genuine_pairs)
    n_impostor = len(impostor_pairs)
    print(f"Built {len(all_pairs_data)} pairs ({n_genuine} genuine, {n_impostor} impostor)")

    # 3. Process all pairs
    labels = []       # 1 = genuine, 0 = impostor
    cosine_scores = []
    l2_scores = []
    chi2_scores = []

    processed = 0
    skipped = 0

    print("\n═══ Processing All Pairs ═══")
    for img1_rgb, img2_rgb, label in tqdm(all_pairs_data, desc="Pairs"):
        # Convert RGB (from PIL) to BGR for OpenCV pipeline
        img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)

        result = process_pair_images(img1, img2)
        if result is None:
            skipped += 1
            continue

        cos, l2, chi2 = result
        cosine_scores.append(cos)
        l2_scores.append(l2)
        chi2_scores.append(chi2)
        labels.append(int(label))
        processed += 1

    print(f"\nProcessed: {processed}/{len(all_pairs_data)} pairs ({skipped} skipped)")

    if processed < 100:
        print("FATAL: Too few pairs processed. Calibration aborted.")
        sys.exit(1)

    # 3. Convert to numpy
    labels_arr = np.array(labels)
    cosine_arr = np.array(cosine_scores)
    l2_arr = np.array(l2_scores)
    chi2_arr = np.array(chi2_scores)

    # 4. Compute ROC and metrics for ArcFace cosine
    arcface_metrics = compute_metrics(labels_arr, cosine_arr, "ArcFace Cosine")

    # 5. Compute metrics for geometric L2 (invert: lower L2 = more similar)
    max_l2 = max(l2_arr.max(), 0.01)
    l2_sim = 1.0 - (l2_arr / max_l2)
    geo_metrics = compute_metrics(labels_arr, l2_sim, "Geometric L2")
    geo_metrics["raw_stats"] = {
        "genuine_mean_l2": float(np.mean(l2_arr[labels_arr == 1])),
        "genuine_std_l2": float(np.std(l2_arr[labels_arr == 1])),
        "impostor_mean_l2": float(np.mean(l2_arr[labels_arr == 0])),
        "impostor_std_l2": float(np.std(l2_arr[labels_arr == 0])),
        "normalization_max": float(max_l2),
    }

    # 6. Compute metrics for LBP chi-squared (invert: lower chi2 = more similar)
    max_chi2 = max(chi2_arr.max(), 0.01)
    chi2_sim = 1.0 - (chi2_arr / max_chi2)
    lbp_metrics = compute_metrics(labels_arr, chi2_sim, "LBP Chi-Squared")
    lbp_metrics["raw_stats"] = {
        "genuine_mean_chi2": float(np.mean(chi2_arr[labels_arr == 1])),
        "genuine_std_chi2": float(np.std(chi2_arr[labels_arr == 1])),
        "impostor_mean_chi2": float(np.mean(chi2_arr[labels_arr == 0])),
        "impostor_std_chi2": float(np.std(chi2_arr[labels_arr == 0])),
        "normalization_max": float(max_chi2),
    }

    # 7. Compute optimal fusion weights via grid search
    best_weights, best_fused_eer = find_optimal_weights(
        labels_arr, cosine_arr, l2_sim, chi2_sim
    )

    # 8. Build calibration output
    calibration = {
        "benchmark": "LFW",
        "source": "huggingface.co/datasets/logasja/lfw (official LFW mirror)",
        "pairs_evaluated": processed,
        "pairs_skipped": skipped,
        "genuine_pairs": int(np.sum(labels_arr == 1)),
        "impostor_pairs": int(np.sum(labels_arr == 0)),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "pipeline_version": "Fallen v2.0 — Forensic-Grade",
        "arcface": arcface_metrics,
        "geometric_l2": geo_metrics,
        "lbp_chi_squared": lbp_metrics,
        "fusion": {
            "optimal_weights": {
                "structural": round(best_weights[0], 4),
                "geometric": round(best_weights[1], 4),
                "micro_topology": round(best_weights[2], 4),
            },
            "fused_eer": round(best_fused_eer, 6),
            "method": "grid_search_eer_minimization",
        }
    }

    # 9. Save locally
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\n✓ Calibration saved to {OUTPUT_FILE}")

    # 10. Upload to GCS
    if GCS_AVAILABLE:
        try:
            client = gcs.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob("calibration/lfw_calibration.json")
            blob.upload_from_filename(OUTPUT_FILE)
            print(f"✓ Calibration uploaded to gs://{GCS_BUCKET}/calibration/lfw_calibration.json")
        except Exception as e:
            print(f"Warning: GCS upload failed: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Pairs evaluated:    {processed}")
    print(f"ArcFace EER:        {arcface_metrics['eer']:.4f}")
    print(f"Geometric L2 EER:   {geo_metrics['eer']:.4f}")
    print(f"LBP Chi² EER:       {lbp_metrics['eer']:.4f}")
    print(f"Optimal Weights:    {best_weights[0]:.2f} / {best_weights[1]:.2f} / {best_weights[2]:.2f}")
    print(f"Fused EER:          {best_fused_eer:.4f}")
    print("=" * 60)


def process_pair_images(img1: np.ndarray, img2: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Process a single pair of pre-loaded images through the full pipeline.
    Returns (cosine_sim, l2_distance, chi2_distance) or None on failure.
    """
    if img1 is None or img2 is None:
        return None
    if img1.size == 0 or img2.size == 0:
        return None

    # CLAHE
    img1_clahe = apply_clahe(img1)
    img2_clahe = apply_clahe(img2)

    # Align
    aligned1, lm1 = align_face_crop(img1_clahe)
    aligned2, lm2 = align_face_crop(img2_clahe)

    if lm1 is None or lm2 is None:
        return None

    # Tier 1: ArcFace cosine
    try:
        embed1 = extract_arcface_embedding(aligned1)
        embed2 = extract_arcface_embedding(aligned2)
        cosine = calculate_cosine_similarity(embed1, embed2)
    except Exception:
        return None

    # Tier 2: Geometric L2
    ratios1 = extract_geometric_ratios(lm1)
    ratios2 = extract_geometric_ratios(lm2)
    l2_dist = float(np.linalg.norm(ratios1 - ratios2))

    # Tier 3: LBP chi-squared
    lbp1 = extract_lbp_histogram(aligned1)
    lbp2 = extract_lbp_histogram(aligned2)
    chi2_dist = chi_squared_distance(lbp1, lbp2)

    return cosine, l2_dist, chi2_dist


def compute_metrics(labels: np.ndarray, scores: np.ndarray, name: str) -> dict:
    """Compute ROC, EER, and FAR at specific thresholds."""
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Equal Error Rate (where FPR == 1-TPR)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except ValueError:
        eer = 0.5  # Fallback if interpolation fails

    print(f"\n  {name} EER: {eer:.6f}")

    # FAR at specific thresholds
    threshold_points = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80]
    threshold_data = {}

    for t in threshold_points:
        # At threshold t: predictions = scores >= t
        predicted_positive = scores >= t
        actual_positive = labels == 1
        actual_negative = labels == 0

        # False Accept Rate = FP / (FP + TN)  = false positives among impostors
        false_accepts = np.sum(predicted_positive & actual_negative)
        total_negatives = np.sum(actual_negative)
        far = float(false_accepts / total_negatives) if total_negatives > 0 else 0.0

        # False Reject Rate = FN / (FN + TP) = false negatives among genuine
        false_rejects = np.sum(~predicted_positive & actual_positive)
        total_positives = np.sum(actual_positive)
        frr = float(false_rejects / total_positives) if total_positives > 0 else 0.0

        threshold_data[str(t)] = {
            "far": round(far, 8),
            "frr": round(frr, 8),
        }

        if far > 0:
            print(f"    Threshold {t:.2f}: FAR={far:.6f} (1 in {int(1/far):,}), FRR={frr:.4f}")
        else:
            print(f"    Threshold {t:.2f}: FAR=0 (no false accepts), FRR={frr:.4f}")

    return {
        "eer": round(eer, 6),
        "thresholds": threshold_data,
    }


def find_optimal_weights(
    labels: np.ndarray,
    cosine: np.ndarray,
    l2_sim: np.ndarray,
    chi2_sim: np.ndarray,
) -> Tuple[Tuple[float, float, float], float]:
    """
    Grid search over fusion weight combinations to minimize EER.
    Weights must sum to 1.0. Step size: 0.05.
    """
    best_eer = 1.0
    best_weights = (0.6, 0.25, 0.15)
    step = 0.05

    for w1 in np.arange(0.40, 0.85, step):
        for w2 in np.arange(0.05, 0.45, step):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05 or w3 > 0.40:
                continue

            fused = w1 * cosine + w2 * l2_sim + w3 * chi2_sim
            fpr, tpr, _ = roc_curve(labels, fused)

            try:
                eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
            except ValueError:
                continue

            if eer < best_eer:
                best_eer = eer
                best_weights = (round(w1, 2), round(w2, 2), round(w3, 2))

    print(f"\n  Optimal fusion weights: {best_weights} → EER={best_eer:.6f}")
    return best_weights, best_eer


if __name__ == "__main__":
    run_calibration()
