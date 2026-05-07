import os
from typing import Optional, List, Dict, Tuple, Any
import json
os.environ["TF_USE_LEGACY_KERAS"] = "1"
if os.path.exists("/app") or os.access("/", os.W_OK):
    os.environ["DEEPFACE_HOME"] = "/app"
else:
    os.environ["DEEPFACE_HOME"] = os.path.expanduser("~")

import cv2
import numpy as np
import urllib.request
import math
import hashlib
import time
from google.cloud import storage
from skimage.feature import local_binary_pattern
import mediapipe as mp
import onnxruntime as ort
from deepface import DeepFace

# Initialize ML models for the pipeline
print("Initializing ML models in pipeline_core...")
try:
    DeepFace.build_model("ArcFace")
    DeepFace.build_model("Facenet512")
except Exception as e:
    print(f"Warning: DeepFace initialization failed: {e}")

try:
    pad_session = ort.InferenceSession("models/MiniFASNetV2.onnx", providers=['CPUExecutionProvider'])
except Exception as e:
    pad_session = None
    print(f"Warning: pad_session initialization failed: {e}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def finite_or_none(value) -> float | None:
    """Safe float serializer — preserves scientific notation, rejects NaN/Inf."""
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return None

def evaluate_mark_veto_override(mark_result: dict, lr_marks: float) -> dict:
    """
    Evaluates whether independent forensic mark correspondence provides enough 
    evidence to override an ArcFace veto.
    
    SAFETY OVERRIDE (v1.153.0): Mark override is DISABLED.
    Testing revealed that mark correspondence produces false positives on 
    different-source pairs (e.g., Powell vs Djindjic scored 99.98% via override).
    The mark LR calibration is not validated for production use.
    The ArcFace veto MUST remain absolute until mark calibration is independently
    validated against a known impostor population.
    
    Original rules (preserved for future re-enablement):
    - at least 3 individual positive mark LRs > 1.0
    - aggregate lr_marks >= 100.0
    - malformed / non-numeric mark LRs are ignored
    - one extreme mark LR alone must not override
    """
    mark_lrs_raw = mark_result.get("mark_lrs", [])
    positive_mark_lrs = []
    for lr in mark_lrs_raw:
        lr_val = finite_or_none(lr)
        if lr_val is not None and lr_val > 1.0:
            positive_mark_lrs.append(lr_val)
            
    count = len(positive_mark_lrs)
    lr_marks_val = finite_or_none(lr_marks)
    
    # SAFETY: Always return ineligible until mark calibration is validated
    return {
        "eligible": False,
        "positive_mark_count": count,
        "positive_mark_lrs": positive_mark_lrs,
        "reason": f"Mark override DISABLED (v1.153.0 safety). count={count}, lr_marks={lr_marks_val}"
    }

def score_to_lr_ensemble(ensemble_score: float, temporal_delta: float = 0.0) -> float:
    """
    Convert Fused 60/40 Ensemble Score to a Likelihood Ratio using
    empirically calibrated FAR/FRR from the LFW benchmark.

    LR = P(score | Hp) / P(score | Hd) = (1 - FRR) / FAR

    Where:
      - P(score | Hp) = True Positive Rate = 1 - FRR (same person produces this score)
      - P(score | Hd) = False Acceptance Rate = FAR (different person produces this score)
      
    Temporal Invariance:
      - Exponential decay curve applied to TPR probability based on temporal delta.
      - As the time gap increases, the expected FRR naturally increases.
      - By boosting the expected TPR for degraded scores, we prevent the "Aging Problem"
        without manipulating the raw structural score or Bayesian math.
    """
    if CALIBRATION is None:
        return 1.0  # Neutral LR — no calibration data

    thresholds = CALIBRATION.get("ensemble", {}).get("thresholds", {})
    if not thresholds:
        # Fallback to arcface if ensemble calibration is not yet loaded
        thresholds = CALIBRATION.get("arcface", {}).get("thresholds", {})
        
    sorted_thresh = sorted(thresholds.keys(), key=float)

    far_value = None
    frr_value = None

    for t in sorted_thresh:
        if ensemble_score >= float(t):
            far_value = thresholds[t]["far"]
            frr_value = thresholds[t]["frr"]

    if far_value is None or far_value <= 0:
        # Score below all thresholds — strong evidence against match
        return 1e-6  # Floor: extremely low LR

    raw_tpr = 1.0 - (frr_value if frr_value is not None else 0.0)
    
    # Age-Conditioned Likelihood Ratio (Temporal Invariance)
    # The expected TPR for degraded scores is exponentially boosted ~1% per year of temporal gap
    tpr = min(1.0, raw_tpr * math.exp(0.01 * temporal_delta))

    # Epsilon floor to prevent division by zero
    far_value = max(far_value, 1e-9)
    lr = tpr / far_value
    return lr

def compute_image_hash(image: np.ndarray) -> str:
    """SHA-256 hash of raw pixel bytes of an image array (BGR, uint8).
    Used for chain-of-custody hashing at specific preprocessing stages."""
    return hashlib.sha256(image.tobytes()).hexdigest()

def fetch_image_from_url(uri: str) -> tuple:
    """
    Fetches an image from a GCS URI ONLY.
    Returns a tuple of (decoded_image_array, sha256_hash_of_raw_bytes).
    The hash is computed on the raw byte stream BEFORE OpenCV decode,
    establishing an immutable chain-of-custody fingerprint.
    """
    try:
        if uri.startswith("gs://"):
            storage_client = storage.Client()
            parts = uri.replace("gs://", "").split("/", 1)
            bucket_name_req = parts[0]
            configured_bucket = os.getenv("BUCKET_NAME") or "hoppwhistle-facial-uploads"
            from security_helpers import is_safe_image_url
            if not is_safe_image_url(uri, configured_bucket):
                raise ValueError("Unauthorized GCS bucket.")
            bucket = storage_client.bucket(configured_bucket)
            blob = bucket.blob(parts[1])
            blob.reload()
            if blob.size and blob.size > 15 * 1024 * 1024:
                raise ValueError("File exceeds 15MB size limit.")
            img_bytes = blob.download_as_bytes()
        else:
            raise ValueError("Only gs:// URIs are allowed in verification paths.")

        if not (img_bytes.startswith(b'\xff\xd8') or img_bytes.startswith(b'\x89PNG') or img_bytes.startswith(b'RIFF')):
            raise ValueError("Invalid image signature. Only JPEG/PNG/WEBP allowed.")

        # Chain of Custody: hash the raw binary BEFORE decode
        raw_hash = hashlib.sha256(img_bytes).hexdigest()

        arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
        if img is None:
            raise ValueError("Could not decode image.")
            
        h, w = img.shape[:2]
        if h > 4096 or w > 4096:
            raise ValueError("Image dimensions exceed 4096x4096.")
            
        return img, raw_hash
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to fetch image: {str(e)}")

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    MATH: g(x,y) = T(f(x,y)). 
    We transform the intensity values in localized tiles (8x8) to a uniform distribution,
    clipping the histogram at 2.0 to prevent noise over-amplification.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def align_face_crop(image: np.ndarray, target_size: int = 256):
    """
    Detects a face, aligns it by rotating to make the eye-line horizontal,
    crops tightly around the face with padding, and resizes to a canonical
    target_size × target_size image. Re-detects landmarks on the final crop
    for accurate downstream embedding extraction.
    Returns (aligned_crop, landmarks) or (resized_original, None) if no face.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        resized = cv2.resize(image, (target_size, target_size))
        return resized, None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]

    # Eye center positions for alignment
    left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
    right_eye_indices = [263, 362, 387, 386, 385, 373, 374, 380]

    left_eye_center = np.mean(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in left_eye_indices], axis=0
    )
    right_eye_center = np.mean(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in right_eye_indices], axis=0
    )

    # Rotation angle to make eyes horizontal
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eye_midpoint = (
        (left_eye_center[0] + right_eye_center[0]) / 2,
        (left_eye_center[1] + right_eye_center[1]) / 2,
    )
    M = cv2.getRotationMatrix2D(eye_midpoint, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    # Re-detect landmarks on the rotated image for an accurate bounding box
    rgb_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    results_rot = face_mesh.process(rgb_rot)
    lm = results_rot.multi_face_landmarks[0].landmark if results_rot.multi_face_landmarks else landmarks

    # Face bounding box from all landmarks
    xs = [l.x * w for l in lm]
    ys = [l.y * h for l in lm]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add 25% padding
    face_w = x_max - x_min
    face_h = y_max - y_min
    pad_x = face_w * 0.25
    pad_y = face_h * 0.25
    x_min = max(0, int(x_min - pad_x))
    x_max = min(w, int(x_max + pad_x))
    y_min = max(0, int(y_min - pad_y))
    y_max = min(h, int(y_max + pad_y))

    # Make it square (use the larger dimension)
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    if crop_w > crop_h:
        diff = crop_w - crop_h
        y_min = max(0, y_min - diff // 2)
        y_max = min(h, y_max + (diff - diff // 2))
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        x_min = max(0, x_min - diff // 2)
        x_max = min(w, x_max + (diff - diff // 2))

    cropped = rotated[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        cropped = rotated

    aligned = cv2.resize(cropped, (target_size, target_size))

    # Final landmark detection on the aligned canonical crop
    rgb_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    results_aligned = face_mesh.process(rgb_aligned)

    final_landmarks = None
    if results_aligned.multi_face_landmarks:
        final_landmarks = results_aligned.multi_face_landmarks[0].landmark

    return aligned, final_landmarks

def estimate_age(image: np.ndarray) -> float:
    """
    Estimates the apparent age of the subject using DeepFace.
    """
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(
            img_path=rgb_image,
            actions=["age"],
            enforce_detection=False,
            detector_backend="skip" # Image is already cropped/aligned
        )
        if isinstance(result, list):
            return float(result[0]["age"])
        return float(result["age"])
    except Exception as e:
        print(f"Age estimation failed: {e}")
        return 0.0

def cross_spectral_normalize(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Analyzes HSV saturation to detect if one image is grayscale/sepia and the other is color.
    If a spectral mismatch is detected, converts the color image to grayscale to match domains,
    preventing artificial texture noise in Tier 3 (LBP) and Tier 4 (Marks).
    Returns (norm_img1, norm_img2, correction_applied).
    """
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    sat1_std = np.std(hsv1[:, :, 1])
    sat2_std = np.std(hsv2[:, :, 1])
    
    threshold = 15.0 # Low saturation std indicates grayscale/monochrome
    
    is_gray1 = sat1_std < threshold
    is_gray2 = sat2_std < threshold
    
    correction_applied = False
    norm1 = img1.copy()
    norm2 = img2.copy()
    
    if is_gray1 and not is_gray2:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        norm2 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
        correction_applied = True
    elif is_gray2 and not is_gray1:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        norm1 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
        correction_applied = True
        
    return norm1, norm2, correction_applied

def extract_ensemble_embeddings(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts both ArcFace and Facenet512 embeddings.
    Returns: (arcface_embedding, facenet_embedding)
    """
    arcface_embed = extract_arcface_embedding(image)
    facenet_embed = extract_facenet_embedding(image)
    return arcface_embed, facenet_embed

def compute_ensemble_similarity(embed_pair_1: tuple[np.ndarray, np.ndarray], embed_pair_2: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, float]:
    """
    Computes cosine similarity for both models and fuses them using a weighted ensemble (60% ArcFace, 40% Facenet512).
    Returns: (fused_score, arcface_score, secondary_score)
    """
    arc1, face1 = embed_pair_1
    arc2, face2 = embed_pair_2
    
    # Needs calculate_cosine_similarity which is defined below, but Python handles forward references
    # wait, this is executed later anyway so it's fine.
    arc_score = calculate_cosine_similarity(arc1, arc2)
    face_score = calculate_cosine_similarity(face1, face2)
    
    fused_score = (arc_score * 0.60) + (face_score * 0.40)
    return fused_score, arc_score, face_score

def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    MATH: Cosine Similarity = (A • B) / (||A|| * ||B||)
    Measures orientation of the embedding vectors, highly robust to magnitude shifts.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def _run_mark_evidence_pipeline(
    probe_img: np.ndarray,
    gallery_img: Optional[np.ndarray] = None,
    probe_file_hash: Optional[str] = None,
    gallery_file_hash: Optional[str] = None,
    mode: str = "diagnostic",
    target_size: int = 1024,
) -> dict:
    """
    Shared helper for deterministic forensic facial mark detection and matching.
    Operates on raw input images, aligns, preprocesses deterministically, detects marks,
    and runs the v2 matcher if paired. 
    """
    from image_preprocessor import preprocess_for_mark_detection, IMAGE_PREPROCESSOR_VERSION
    from mark_detector import MARK_DETECTOR_VERSION

    # ── V2 MARK PIPELINE TIMING (temporary) ──
    import time as _t
    _mp_t0 = _t.time()
    def _mp_log(stage):
        elapsed = _t.time() - _mp_t0
        print(f"[V2-MARK-TIMING] {elapsed:7.2f}s | {stage}", flush=True)
    _mp_log("mark_pipeline_entered")

    has_gallery = gallery_img is not None
    
    # ── 1. Alignment on RAW image ──
    probe_aligned_raw, probe_landmarks = align_face_crop(probe_img)
    _mp_log("probe_align_done")
    gallery_aligned_raw = None
    gallery_landmarks = None
    if has_gallery:
        gallery_aligned_raw, gallery_landmarks = align_face_crop(gallery_img)
        _mp_log("gallery_align_done")

    detector_thresholds = get_detector_thresholds()
    
    probe_face_ok = probe_landmarks is not None
    gallery_face_ok = gallery_landmarks is not None if has_gallery else None

    # ── Face Detection Failure Early Exit ──
    if not probe_face_ok or (has_gallery and not gallery_face_ok):
        failed_side = []
        if not probe_face_ok: failed_side.append("probe")
        if has_gallery and not gallery_face_ok: failed_side.append("gallery")
        
        return {
            "mode": "probe_only" if not has_gallery else "paired",
            "aligned_probe_b64": None,
            "aligned_gallery_b64": None,
            "raw_probe_marks": [],
            "raw_gallery_marks": [],
            "rejected_probe_marks": [],
            "rejected_gallery_marks": [],
            "accepted_correspondences": [],
            "rejected_correspondences": [],
            "mark_match_status": "FACE_NOT_DETECTED",
            "matcher_status": "NOT_RUN_FACE_DETECTION_FAILED",
            "mark_diagnostics": {
                "raw_probe_marks_count": 0,
                "raw_gallery_marks_count": 0,
                "accepted_correspondences_count": 0,
                "rejected_candidates_count": 0,
                "detector_status": "FACE_NOT_DETECTED",
                "probe_detector_status": "FACE_NOT_DETECTED" if not probe_face_ok else "NOT_RUN",
                "gallery_detector_status": "FACE_NOT_DETECTED" if (has_gallery and not gallery_face_ok) else ("NOT_PROVIDED" if not has_gallery else "NOT_RUN"),
                "detector_status_probe": "FACE_NOT_DETECTED" if not probe_face_ok else "NOT_RUN",
                "detector_status_gallery": "FACE_NOT_DETECTED" if (has_gallery and not gallery_face_ok) else ("NOT_PROVIDED" if not has_gallery else "NOT_RUN"),
                "matcher_status": "NOT_RUN_FACE_DETECTION_FAILED",
                "calibration_status": "NOT_RUN_FACE_DETECTION_FAILED",
                "lr_marks": None,
                "mark_match_status": "FACE_NOT_DETECTED",
                "rejection_summary": f"Face detection failed for side(s): {', '.join(failed_side)}. Cannot perform mark analysis.",
                "mark_detector_trace": {"probe": None, "gallery": None},
                "matcher_thresholds": get_matcher_thresholds(),
                "technical_debt": "mark_matcher.py v2 integrated. Face detection failed."
            },
            "lr_marks": None,
            "individual_mark_lrs": [],
            "lr_calculation_trace": {
                "individual_lrs": [], "product": None, "calibration_status": "NOT_RUN_FACE_DETECTION_FAILED",
                "thresholds_used": detector_thresholds, "matcher_thresholds": get_matcher_thresholds()
            },
            "detector_thresholds": detector_thresholds,
            "matcher_thresholds": get_matcher_thresholds(),
            "mark_detector_version": MARK_DETECTOR_VERSION,
            "mark_matcher_version": MARK_MATCHER_V2_VERSION,
            "preprocessor_version": IMAGE_PREPROCESSOR_VERSION,
            "probe_preprocessing": None,
            "gallery_preprocessing": None
        }

    # ── 2. Preprocessing & Detection (Probe) ──
    probe_pp = preprocess_for_mark_detection(probe_aligned_raw, landmarks=probe_landmarks, target_size=target_size)
    _mp_log("probe_preprocess_done")
    probe_detector_input = probe_pp["images"]["mark_detector_input_bgr"]
    det_h, det_w = probe_detector_input.shape[:2]
    
    probe_structural_source = probe_pp["images"]["aligned_bgr"]
    
    marks_probe, rejected_probe_raw, occ_probe, trace_probe, overlays_probe = detect_facial_marks(
        probe_detector_input, probe_landmarks, input_is_preprocessed=True,
        structural_source_bgr=probe_structural_source
    )
    _mp_log("probe_detect_done")
    
    valid_probe_marks = []
    for m in marks_probe:
        cx, cy = int(m["centroid"][0] * det_w), int(m["centroid"][1] * det_h)
        if 0 <= cy < det_h and 0 <= cx < det_w and occ_probe[cy, cx] == 0:
            clean_m = {k: v for k, v in m.items() if k != "contour"}
            clean_m["source_side"] = "probe"
            valid_probe_marks.append(clean_m)
            
    rejected_probe_serialized = [serialize_mark_descriptor(r) for r in rejected_probe_raw]

    # ── 3. Preprocessing & Detection (Gallery) ──
    valid_gallery_marks = []
    rejected_gallery_serialized = []
    trace_gallery = None
    overlays_gallery = {}
    marks_gallery = []
    rejected_gallery_raw = []
    gallery_pp = None

    if has_gallery and gallery_face_ok:
        gallery_pp = preprocess_for_mark_detection(gallery_aligned_raw, landmarks=gallery_landmarks, target_size=target_size)
        _mp_log("gallery_preprocess_done")
        gallery_detector_input = gallery_pp["images"]["mark_detector_input_bgr"]
        gal_h, gal_w = gallery_detector_input.shape[:2]
        
        gallery_structural_source = gallery_pp["images"]["aligned_bgr"]
        
        marks_gallery, rejected_gallery_raw, occ_gallery, trace_gallery, overlays_gallery = detect_facial_marks(
            gallery_detector_input, gallery_landmarks, input_is_preprocessed=True,
            structural_source_bgr=gallery_structural_source
        )
        _mp_log("gallery_detect_done")
        for m in marks_gallery:
            cx, cy = int(m["centroid"][0] * gal_w), int(m["centroid"][1] * gal_h)
            if 0 <= cy < gal_h and 0 <= cx < gal_w and occ_gallery[cy, cx] == 0:
                clean_m = {k: v for k, v in m.items() if k != "contour"}
                clean_m["source_side"] = "gallery"
                valid_gallery_marks.append(clean_m)
        rejected_gallery_serialized = [serialize_mark_descriptor(r) for r in rejected_gallery_raw]

    # ── 4. Matching & LR ──
    rejected_probe = rejected_probe_serialized
    rejected_gallery = rejected_gallery_serialized
    correspondences = []
    rejected_cands = []
    matcher_result = None
    lr_marks = None
    individual_mark_lrs = []
    calibration_status = "NOT_APPLICABLE"

    if not has_gallery:
        mark_match_status = "NOT_RUN_SINGLE_IMAGE"
        matcher_status = "NOT_RUN_SINGLE_IMAGE"
        lr_marks = None
    else:
        exact_image_match = (probe_file_hash == gallery_file_hash) if (probe_file_hash and gallery_file_hash) else False

        if exact_image_match:
            mark_match_status = "EXACT_SELF_MATCH"
            matcher_status = "EXACT_SELF_MATCH"
            n_self = min(len(valid_probe_marks), len(valid_gallery_marks))
            lr_marks = 1.0
            individual_mark_lrs = [1.0] * n_self
            calibration_status = "NOT_APPLICABLE_SELF_MATCH"
            for si in range(n_self):
                correspondences.append({
                    "gallery_idx": si, "probe_idx": si,
                    "gallery_centroid": list(valid_gallery_marks[si]["centroid"]),
                    "probe_centroid": list(valid_probe_marks[si]["centroid"]),
                    "position_distance": 0.0, "area_ratio": 1.0,
                    "type_match": True, "region_match": True,
                    "match_quality": 1.0,
                    "mark_type": valid_gallery_marks[si].get("mark_type", "unknown"),
                    "face_region": valid_gallery_marks[si].get("face_region", "unknown"),
                    "match_cost": 0.0,
                    "lr": 1.0,
                })
            matcher_result = {
                "matched": n_self, "score": 100.0 if n_self > 0 else None,
                "lr_marks": 1.0, "mark_lrs": individual_mark_lrs,
                "matches": correspondences, "rejected_candidates": [],
                "matcher_status": "EXACT_SELF_MATCH",
                "calibration_status": calibration_status,
                "matcher_version": MARK_MATCHER_V2_VERSION,
            }
        else:
            matcher_result = match_marks_v2(
                valid_gallery_marks, valid_probe_marks,
                calibration=TIER4_CALIBRATION
            )
            matcher_status = matcher_result["matcher_status"]
            calibration_status = matcher_result.get("calibration_status", "UNKNOWN")
            lr_marks = matcher_result["lr_marks"]
            individual_mark_lrs = matcher_result["mark_lrs"]
            correspondences = matcher_result["matches"]
            rejected_cands = matcher_result["rejected_candidates"]

            if len(valid_probe_marks) < 2 or len(valid_gallery_marks) < 2:
                mark_match_status = "INSUFFICIENT_MARKS"
            elif matcher_result["matched"] > 0:
                mark_match_status = "MATCHED"
            else:
                mark_match_status = "NO_MATCHES"

    _mp_log("matcher_done")
    # ── 5. Detector Status ──
    probe_detector_status = trace_probe.get("detector_status", "UNKNOWN") if trace_probe else "UNKNOWN"
    gallery_detector_status = trace_gallery.get("detector_status", "UNKNOWN") if trace_gallery else ("NOT_PROVIDED" if not has_gallery else "UNKNOWN")

    if len(marks_probe) > 0 or (has_gallery and len(marks_gallery) > 0):
        overall_detector_status = "OK"
    elif len(marks_probe) == 0 and (not has_gallery or len(marks_gallery) == 0):
        overall_detector_status = "NO_CANDIDATES"
    else:
        overall_detector_status = "PARTIAL"

    matched_count = matcher_result["matched"] if matcher_result else 0
    mark_diagnostics_payload = {
        "raw_probe_marks_count": len(valid_probe_marks),
        "raw_gallery_marks_count": len(valid_gallery_marks),
        "accepted_correspondences_count": matched_count,
        "rejected_candidates_count": len(rejected_cands) if rejected_cands else 0,
        "detector_status": overall_detector_status,
        "probe_detector_status": probe_detector_status,
        "gallery_detector_status": gallery_detector_status,
        "detector_status_probe": probe_detector_status,
        "detector_status_gallery": gallery_detector_status,
        "matcher_status": matcher_status,
        "calibration_status": calibration_status,
        "lr_marks": finite_or_none(lr_marks),
        "mark_match_status": mark_match_status,
        "rejection_summary": _build_rejection_summary(
            valid_probe_marks, valid_gallery_marks,
            matcher_result or {}, rejected_cands, mark_match_status,
            (probe_file_hash == gallery_file_hash) if (has_gallery and probe_file_hash and gallery_file_hash) else False,
            TIER4_CALIBRATION,
            trace_probe=trace_probe, trace_gallery=trace_gallery,
        ) if has_gallery else (
            f"Probe-only mode: {len(valid_probe_marks)} mark(s) detected, {len(rejected_probe_raw)} rejected. No gallery provided for matching."
        ),
        "mark_detector_trace": {"probe": trace_probe, "gallery": trace_gallery},
        "matcher_thresholds": get_matcher_thresholds(),
        "technical_debt": "mark_matcher.py v2 integrated via shared helper.",
    }

    lr_calculation_trace = {
        "individual_lrs": [finite_or_none(lr) for lr in individual_mark_lrs],
        "product": finite_or_none(lr_marks) if has_gallery else None,
        "calibration_status": calibration_status,
        "thresholds_used": detector_thresholds,
        "matcher_thresholds": get_matcher_thresholds(),
    }

    def _pp_summary(pp_dict):
        if not pp_dict: return None
        return {
            "decoded_hash": pp_dict["decoded_hash"],
            "aligned_pre_clahe_hash": pp_dict["aligned_pre_clahe_hash"],
            "aligned_post_clahe_hash": pp_dict["aligned_post_clahe_hash"],
            "original_dimensions": pp_dict["original_dimensions"],
            "decoded_dimensions": pp_dict["decoded_dimensions"],
            "aligned_dimensions": pp_dict["aligned_dimensions"],
            "quality": pp_dict["quality"],
            "preprocessing_steps": pp_dict["preprocessing_steps"],
            "preprocessor_version": pp_dict.get("preprocessor_version", IMAGE_PREPROCESSOR_VERSION),
            "aligned_pre_clahe_b64": pp_dict.get("debug_b64", {}).get("aligned_b64"),
            "aligned_post_clahe_b64": pp_dict.get("debug_b64", {}).get("lab_clahe_b64"),
            "illumination_normalized_b64": pp_dict.get("debug_b64", {}).get("illumination_normalized_b64"),
            "mark_detector_input_b64": pp_dict.get("debug_b64", {}).get("mark_detector_input_b64"),
            "skin_mask_b64": pp_dict.get("debug_b64", {}).get("skin_mask_b64"),
        }

    _mp_log("response_assembly_start")
    return {
        "mode": "probe_only" if not has_gallery else "paired",
        "aligned_probe_b64": probe_pp["debug_b64"]["aligned_b64"] if probe_pp else None,
        "aligned_gallery_b64": gallery_pp["debug_b64"]["aligned_b64"] if gallery_pp else None,
        "raw_probe_marks": [serialize_mark_descriptor(m) for m in valid_probe_marks],
        "raw_gallery_marks": [serialize_mark_descriptor(m) for m in valid_gallery_marks],
        "rejected_probe_marks": rejected_probe_serialized,
        "rejected_gallery_marks": rejected_gallery_serialized,
        "accepted_correspondences": correspondences,
        "rejected_correspondences": rejected_cands,
        "mark_match_status": mark_match_status,
        "matcher_status": matcher_status,
        "mark_diagnostics": mark_diagnostics_payload,
        "lr_marks": finite_or_none(lr_marks) if has_gallery else None,
        "individual_mark_lrs": [finite_or_none(lr) for lr in individual_mark_lrs],
        "lr_calculation_trace": lr_calculation_trace,
        "detector_thresholds": detector_thresholds,
        "matcher_thresholds": get_matcher_thresholds(),
        "mark_detector_version": MARK_DETECTOR_VERSION,
        "mark_matcher_version": MARK_MATCHER_V2_VERSION,
        "preprocessor_version": IMAGE_PREPROCESSOR_VERSION,
        "probe_preprocessing": _pp_summary(probe_pp),
        "gallery_preprocessing": _pp_summary(gallery_pp),
    }


import pickle
from mark_detector import detect_facial_marks, serialize_mark_descriptor, get_thresholds as get_detector_thresholds, MARK_DETECTOR_VERSION
from mark_matcher import match_facial_marks as match_marks_v2, get_matcher_thresholds, MARK_MATCHER_VERSION as MARK_MATCHER_V2_VERSION

def _load_calibration():
    """Attempt to load calibration JSON from GCS, fallback to local file."""
    import json as _json

    bucket_name = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
    gcs_path = "calibration/lfw_calibration.json"

    # Try GCS first
    try:
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        if blob.exists():
            content = blob.download_as_text()
            cal = _json.loads(content)
            print(f"Calibration loaded from GCS: {cal['benchmark']} ({cal['pairs_evaluated']} pairs)")
            return cal
        else:
            print(f"No calibration blob at gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        print(f"GCS calibration load failed: {e}")

    # Fallback to local file (for dev environments)
    _local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data", "lfw_calibration.json")
    try:
        with open(_local_path, "r") as _f:
            cal = _json.load(_f)
        print(f"Calibration loaded from local file: {cal['benchmark']} ({cal['pairs_evaluated']} pairs)")
        return cal
    except FileNotFoundError:
        print("WARNING: No calibration data found (GCS or local). FAR will be reported as UNCALIBRATED.")
    except Exception as _e:
        print(f"WARNING: Failed to load calibration data: {_e}. FAR will be reported as UNCALIBRATED.")

    return None

CALIBRATION = _load_calibration()

CALIBRATION = _load_calibration()

def _load_tier4_calibration():
    """Load the Tier 4 population model from local file or GCS."""
    import json as _json
    import sys
    import numpy as np
    
    # Monkey-patch to allow unpickling numpy 2.x models in numpy 1.x environments
    if "numpy.core.numeric" in sys.modules and "numpy._core.numeric" not in sys.modules:
        sys.modules["numpy._core"] = sys.modules["numpy.core"]
        sys.modules["numpy._core.numeric"] = sys.modules["numpy.core.numeric"]
        sys.modules["numpy._core.multiarray"] = sys.modules["numpy.core.multiarray"]

    bucket_name = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
    gcs_path = "calibration/tier4_population_model.pkl"

    # Try local file first (faster)
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data", "tier4_population_model.pkl")
    try:
        with open(local_path, "rb") as f:
            cal = pickle.load(f)
        print(f"Tier 4 Bayesian model loaded from local: {cal.get('total_marks', '?')} population marks")
        return cal
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"WARNING: Failed to load local Tier 4 model: {e}")

    # Fallback to GCS
    try:
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        if blob.exists():
            pkl_bytes = blob.download_as_bytes()
            cal = pickle.loads(pkl_bytes)
            print(f"Tier 4 Bayesian model loaded from GCS: {cal.get('total_marks', '?')} population marks")
            return cal
    except Exception as e:
        print(f"GCS Tier 4 model load failed: {e}")

    print("WARNING: No Tier 4 Bayesian calibration data found. Mark LR will be unavailable.")
    return None

TIER4_CALIBRATION = _load_tier4_calibration()

TIER4_CALIBRATION = _load_tier4_calibration()

def extract_arcface_embedding(image: np.ndarray) -> np.ndarray:
    """
    Extracts a 512-D ArcFace biometric embedding from an aligned face crop.
    This is the TRUE identity discriminator — replaces MediaPipe geometric
    cosine for Tier 1 structural identity matching.

    CRITICAL: We use detector_backend='retinaface' — the ONLY backend that
    reliably handles BOTH failure modes:

    - 'skip' bypasses alignment entirely. Our align_face_crop() uses MediaPipe
      landmarks which don't match ArcFace's training alignment → poor
      discriminative power between different identities.

    - 'opencv' (Haar cascade) fails on tightly-cropped face images. With
      enforce_detection=False, failed detections produce degenerate embeddings
      that are nearly identical across all subjects → mass false matches.

    - 'retinaface' is a deep-learning face detector that successfully detects
      faces even in pre-cropped images AND provides the 5 facial landmarks
      needed for ArcFace-compatible affine alignment. This is the standard
      recommended backend for ArcFace in the DeepFace library.

    Returns a 512-D numpy array (ArcFace latent space).
    """
    # DeepFace expects RGB; our pipeline uses BGR (OpenCV)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = DeepFace.represent(
        img_path=rgb_image,
        model_name="ArcFace",
        enforce_detection=False,
        detector_backend="retinaface",
    )
    embedding = np.array(result[0]["embedding"], dtype=np.float64)
    return embedding  # 512-D vector


def extract_facenet_embedding(image: np.ndarray) -> np.ndarray:
    """
    Extracts a 512-D Facenet512 biometric embedding from an aligned face crop.
    This serves as the secondary model in the Tier 1 Neural Ensemble.
    Uses 'retinaface' detector backend for consistency with ArcFace extraction.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = DeepFace.represent(
        img_path=rgb_image,
        model_name="Facenet512",
        enforce_detection=False,
        detector_backend="retinaface",
    )
    return np.array(result[0]["embedding"], dtype=np.float64)


def _build_rejection_summary(
    valid_probe_marks: list,
    valid_gallery_marks: list,
    mark_result: dict,
    rejected_cands: list,
    mark_match_status: str,
    exact_image_match: bool,
    tier4_calibration,
    trace_probe: dict = None,
    trace_gallery: dict = None,
) -> str:
    """Return a human-readable explanation for why LR_marks is neutral or absent."""
    matched = mark_result.get("matched", 0)
    n_probe = len(valid_probe_marks)
    n_gallery = len(valid_gallery_marks)

    def _get_stage_reason(side_name, trace):
        if not trace:
            return f"No raw marks detected on {side_name}"
        if trace.get("initial_candidates", 0) == 0:
            return f"No initial mark candidates found after thresholding on {side_name}"
        if trace.get("after_skin_mask", 0) == 0:
            return f"Candidates found, but removed by occlusion/skin mask on {side_name}"
        if trace.get("after_area_filter", 0) == 0:
            return f"Initial candidates found, but all rejected by area threshold on {side_name}"
        if trace.get("after_shape_filter", 0) == 0:
            return f"Initial candidates found, but all rejected by shape/geometry filter on {side_name}"
        if trace.get("after_region_exclusion", 0) == 0:
            return f"Candidates found, but removed by facial-region exclusion mask on {side_name}"
        if trace.get("after_contrast_filter", 0) == 0:
            return f"Candidates found, but all rejected by low contrast threshold on {side_name}"
        return f"No raw marks detected on {side_name}"

    if exact_image_match:
        return "Exact self-match: mark evidence self-corresponding by identity (LR neutralized to 1.0)"
    if n_probe == 0 and n_gallery == 0:
        reason_p = _get_stage_reason("probe", trace_probe)
        reason_g = _get_stage_reason("gallery", trace_gallery)
        base_p = reason_p.replace(" on probe", "")
        base_g = reason_g.replace(" on gallery", "")
        if base_p == base_g:
            return f"{base_p} on both images"
        return f"{reason_p}. {reason_g}."
    if n_probe == 0:
        return _get_stage_reason("probe", trace_probe)
    if n_gallery == 0:
        return _get_stage_reason("gallery", trace_gallery)
    if tier4_calibration is None and matched > 0:
        return "Mark calibration data unavailable — LR defaulted to 1.0"
    if mark_match_status == "DETECTOR_UNAVAILABLE":
        return "Mark detector unavailable"
    if matched == 0 and len(rejected_cands) > 0:
        return f"All {len(rejected_cands)} candidate marks rejected by cost/distance thresholds"
    if matched == 0:
        return "Raw marks detected, but no accepted correspondences passed matching thresholds"
    if matched > 0:
        lr_marks_val = mark_result.get("lr_marks", 1.0)
        if lr_marks_val == 1.0 and tier4_calibration is None:
            return "Mark calibration data unavailable — LR defaulted to 1.0"
        if lr_marks_val is None or lr_marks_val == 1.0:
            return f"{matched} mark(s) matched but combined LR is neutral (1.0) — mark evidence neither supports nor refutes common source"
        return None  # Marks contributing normally — LR != 1.0
    return "Unknown mark pipeline state"


