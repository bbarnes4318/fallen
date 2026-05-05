"""Deterministic Forensic Image Preprocessor v1.0.0

Pure module: no FastAPI, no DB, no JWT dependencies.
No generative enhancement, no GFPGAN, no CodeFormer, no face restoration.

Provides deterministic, hashable preprocessing for mark/scar/mole/blemish
detection pipelines. Every transformation is reproducible and auditable.
"""
import hashlib
import base64
import cv2
import numpy as np

IMAGE_PREPROCESSOR_VERSION = "1.0.0"

# MediaPipe landmark index groups for skin mask feature exclusion
# (duplicated from mark_detector.py to keep this module dependency-free)
_LEFT_EYE_IDX = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
_RIGHT_EYE_IDX = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
_LEFT_BROW_IDX = [70,63,105,66,107,55,65,52,53,46]
_RIGHT_BROW_IDX = [300,293,334,296,336,285,295,282,283,276]
_NOSE_IDX = [1,2,98,327,168,6,197,195,5,4,45,220,115,48,64,102,49,131,134,236,196,3,51,281,275,440,344,278,294,331,279,360,363,456,420,399,412,351]
_LIPS_IDX = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,80,191,78]
_FACE_OVAL_IDX = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
_BORDER_MARGIN = 10
_EXCLUDE_GROUPS = [_LEFT_EYE_IDX, _RIGHT_EYE_IDX, _LEFT_BROW_IDX, _RIGHT_BROW_IDX, _NOSE_IDX, _LIPS_IDX]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sha256_image(image: np.ndarray) -> str:
    """Deterministically hash image pixel data.

    Includes shape and dtype in the hash input so two arrays with the same
    raw bytes but different interpretation (e.g. different shape) cannot
    collide silently.

    Args:
        image: numpy array (any dtype, any shape).

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    h = hashlib.sha256()
    # Encode metadata to prevent shape/dtype collisions
    meta = f"{image.shape}|{image.dtype}".encode("utf-8")
    h.update(meta)
    h.update(image.tobytes())
    return h.hexdigest()


def encode_debug_png_b64(image: np.ndarray) -> str:
    """Encode an image as a data-URI PNG base64 string.

    Args:
        image: BGR uint8 numpy array, or single-channel uint8 for masks.

    Returns:
        String in format ``data:image/png;base64,...``
    """
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("cv2.imencode failed to encode image as PNG")
    return f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"


def compute_quality_metrics(image_bgr: np.ndarray) -> dict:
    """Compute lightweight quality metrics for a BGR image.

    These metrics characterise the input quality for forensic transparency
    and can be used to gate mark detection (e.g. skip on blurry inputs).

    Args:
        image_bgr: BGR uint8 numpy array.

    Returns:
        Dict with keys: blur_laplacian, exposure_mean, exposure_std,
        underexposed_fraction, overexposed_fraction, width, height.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Laplacian variance — higher = sharper
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_laplacian = float(lap.var())

    # Exposure statistics
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    total_px = float(gray.size)
    underexposed = float(np.count_nonzero(gray < 30)) / total_px
    overexposed = float(np.count_nonzero(gray > 225)) / total_px

    return {
        "blur_laplacian": round(blur_laplacian, 2),
        "exposure_mean": round(mean_val, 2),
        "exposure_std": round(std_val, 2),
        "underexposed_fraction": round(underexposed, 4),
        "overexposed_fraction": round(overexposed, 4),
        "width": w,
        "height": h,
    }


def apply_lab_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB colour space.

    This is the single canonical CLAHE pass. It must only be called once
    per image in the pipeline — never stacked.

    Args:
        image_bgr: BGR uint8 numpy array.
        clip_limit: CLAHE clip limit (default 2.0).
        tile_grid_size: CLAHE tile grid (default (8, 8)).

    Returns:
        BGR uint8 numpy array with CLAHE applied to luminance only.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge([L_clahe, A, B])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def apply_retinex_lighting_normalization(image_bgr: np.ndarray) -> np.ndarray:
    """Conservative deterministic illumination normalisation (single-scale Retinex).

    Uses a single Gaussian-blurred illumination estimate to reduce
    large-scale lighting gradients while preserving local texture (marks,
    scars, moles). No generative reconstruction — only division-based
    normalisation that cannot hallucinate detail.

    Args:
        image_bgr: BGR uint8 numpy array.

    Returns:
        BGR uint8 numpy array with normalised illumination.
    """
    img_f = image_bgr.astype(np.float64) + 1.0  # avoid log(0)

    # Large kernel → captures only broad illumination gradients
    sigma = max(image_bgr.shape[0], image_bgr.shape[1]) // 4
    sigma = max(sigma, 15)
    # Ensure odd kernel size
    ksize = sigma if sigma % 2 == 1 else sigma + 1

    blur = cv2.GaussianBlur(img_f, (ksize, ksize), 0)
    blur = np.maximum(blur, 1.0)  # safety floor

    # Single-scale Retinex: log(image) - log(illumination)
    retinex = np.log10(img_f) - np.log10(blur)

    # Normalise per-channel to 0-255
    for c in range(retinex.shape[2]):
        ch = retinex[:, :, c]
        mn, mx = ch.min(), ch.max()
        if mx - mn > 1e-6:
            retinex[:, :, c] = (ch - mn) / (mx - mn) * 255.0
        else:
            retinex[:, :, c] = 128.0

    return np.clip(retinex, 0, 255).astype(np.uint8)


def build_skin_mask_from_landmarks(shape: tuple, landmarks) -> "np.ndarray | None":
    """Build a facial skin mask from MediaPipe-style landmarks.

    Excludes eyes, eyebrows, lips, nose-heavy regions, and image borders.

    Args:
        shape: (height, width) of the target image.
        landmarks: MediaPipe face mesh landmark object with .landmark list,
                   or None.

    Returns:
        uint8 mask (255 = skin, 0 = excluded) or None if landmarks are
        missing or unusable.
    """
    if landmarks is None:
        return None

    h, w = shape[:2]

    # Extract landmark list — handle both raw landmark objects and lists
    lm_list = None
    if hasattr(landmarks, "landmark"):
        lm_list = landmarks.landmark
    elif isinstance(landmarks, (list, tuple)) and len(landmarks) > 0:
        lm_list = landmarks
    else:
        return None

    if len(lm_list) < 468:
        return None

    def _lm_pt(idx):
        lm = lm_list[idx]
        if hasattr(lm, "x"):
            return (int(lm.x * w), int(lm.y * h))
        elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
            return (int(lm[0] * w), int(lm[1] * h))
        return (0, 0)

    # Build face oval polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    oval_pts = np.array([_lm_pt(i) for i in _FACE_OVAL_IDX], dtype=np.int32)
    cv2.fillPoly(mask, [oval_pts], 255)

    # Exclude feature regions
    for group in _EXCLUDE_GROUPS:
        pts = np.array([_lm_pt(i) for i in group], dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 0)

    # Exclude border margin
    if _BORDER_MARGIN > 0:
        mask[:_BORDER_MARGIN, :] = 0
        mask[-_BORDER_MARGIN:, :] = 0
        mask[:, :_BORDER_MARGIN] = 0
        mask[:, -_BORDER_MARGIN:] = 0

    return mask


def preprocess_for_mark_detection(
    image_bgr: np.ndarray,
    landmarks=None,
    target_size: int = 1024,
) -> dict:
    """Deterministic preprocessing pipeline for mark detection.

    Assumes input is an already-aligned face crop. Resizes to target_size,
    applies single CLAHE pass, illumination normalisation, and builds
    provenance hashes.

    The future canonical-alignment-transform work will feed into this
    function; for now it receives pre-aligned crops.

    Args:
        image_bgr: BGR uint8 numpy array (aligned face crop).
        landmarks: Optional MediaPipe landmarks for skin mask.
        target_size: Output resolution (default 1024).

    Returns:
        Dict with keys: decoded_hash, aligned_pre_clahe_hash,
        aligned_post_clahe_hash, original_dimensions, decoded_dimensions,
        aligned_dimensions, quality, images, debug_b64,
        preprocessing_steps.
    """
    steps = []

    # 0. Record original dimensions and hash
    orig_h, orig_w = image_bgr.shape[:2]
    decoded_hash = sha256_image(image_bgr)
    steps.append(f"decoded: {orig_w}x{orig_h}")

    # 1. Resize to target using high-quality interpolation
    if orig_h != target_size or orig_w != target_size:
        interp = cv2.INTER_AREA if orig_h > target_size else cv2.INTER_LANCZOS4
        aligned_bgr = cv2.resize(image_bgr, (target_size, target_size), interpolation=interp)
        steps.append(f"resized: {orig_w}x{orig_h} -> {target_size}x{target_size} ({('INTER_AREA' if interp == cv2.INTER_AREA else 'INTER_LANCZOS4')})")
    else:
        aligned_bgr = image_bgr.copy()
        steps.append("no resize needed")

    # 2. Pre-CLAHE hash (on the resized-but-not-enhanced image)
    pre_clahe_hash = sha256_image(aligned_bgr)

    # 3. Single canonical CLAHE pass
    lab_clahe_bgr = apply_lab_clahe(aligned_bgr)
    post_clahe_hash = sha256_image(lab_clahe_bgr)
    steps.append("lab_clahe: clipLimit=2.0, tileGridSize=(8,8)")

    # 4. Illumination normalisation (conservative Retinex)
    illum_norm_bgr = apply_retinex_lighting_normalization(aligned_bgr)
    steps.append("retinex_lighting_normalization: single-scale, no generative")

    # 5. Mark detector input = CLAHE output (the canonical input)
    mark_input_bgr = lab_clahe_bgr.copy()
    steps.append("mark_detector_input: lab_clahe_bgr (canonical)")

    # 6. Quality metrics
    quality = compute_quality_metrics(aligned_bgr)

    # 7. Skin mask
    skin_mask = build_skin_mask_from_landmarks(
        (target_size, target_size), landmarks
    )
    if skin_mask is not None:
        steps.append("skin_mask: built from landmarks")
    else:
        steps.append("skin_mask: None (no landmarks provided)")

    # 8. Debug base64 images
    debug_b64 = {
        "aligned_b64": encode_debug_png_b64(aligned_bgr),
        "lab_clahe_b64": encode_debug_png_b64(lab_clahe_bgr),
        "illumination_normalized_b64": encode_debug_png_b64(illum_norm_bgr),
        "mark_detector_input_b64": encode_debug_png_b64(mark_input_bgr),
        "skin_mask_b64": encode_debug_png_b64(skin_mask) if skin_mask is not None else None,
    }

    return {
        "decoded_hash": decoded_hash,
        "aligned_pre_clahe_hash": pre_clahe_hash,
        "aligned_post_clahe_hash": post_clahe_hash,
        "original_dimensions": f"{orig_w}x{orig_h}",
        "decoded_dimensions": f"{orig_w}x{orig_h}",
        "aligned_dimensions": f"{target_size}x{target_size}",
        "quality": quality,
        "images": {
            "aligned_bgr": aligned_bgr,
            "lab_clahe_bgr": lab_clahe_bgr,
            "illumination_normalized_bgr": illum_norm_bgr,
            "mark_detector_input_bgr": mark_input_bgr,
            "skin_mask": skin_mask,
        },
        "debug_b64": debug_b64,
        "preprocessing_steps": steps,
        "preprocessor_version": IMAGE_PREPROCESSOR_VERSION,
    }
