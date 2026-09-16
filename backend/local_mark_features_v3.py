"""High-resolution local facial-mark feature extraction for Fallen V3.

This module deliberately sits *beside* the existing detector. The detector
remains responsible for generating candidate marks; this module enriches those
candidates using source-resolution local appearance and neighborhood features.

The initial local-feature backend is OpenCV SIFT because it is already
available through Fallen's OpenCV dependency and gives us a deterministic,
identity-disjoint baseline before introducing a larger learned encoder.
The interface is intentionally backend-neutral so a learned descriptor can be
benchmarked later without changing the mark/constellation data model.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


LOCAL_MARK_FEATURE_VERSION = "3.0.0-sift-local"
DEFAULT_PATCH_SIZES = (32, 64, 128)


@dataclass(frozen=True)
class LocalMarkFeature:
    backend: str
    version: str
    canonical_position: Tuple[float, float]
    patch_sizes: Tuple[int, ...]
    local_embedding: Tuple[float, ...]
    contextual_embedding: Tuple[float, ...]
    keypoint_count: int
    visibility_score: float
    source_size: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12 or not np.all(np.isfinite(v)):
        return np.zeros_like(v)
    return v / norm


def _safe_crop(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    half = max(1, int(size // 2))
    x = int(round(cx * w))
    y = int(round(cy * h))
    x0, x1 = max(0, x - half), min(w, x + half)
    y0, y1 = max(0, y - half), min(h, y + half)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def _color_texture_descriptor(patch: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)

    # LBP-like 8-neighborhood histogram, normalized.
    codes = np.zeros(gray.shape, dtype=np.uint8)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
              (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(shifts):
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        codes |= ((shifted >= gray).astype(np.uint8) << bit)
    hist, _ = np.histogram(codes[1:-1, 1:-1], bins=16, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)

    # Coarse LAB histograms preserve pigmentation information without depending
    # on exact RGB values.
    color_parts: List[np.ndarray] = []
    for channel in cv2.split(lab):
        h, _ = np.histogram(channel, bins=8, range=(0, 256))
        h = h.astype(np.float32)
        h /= max(float(h.sum()), 1.0)
        color_parts.append(h)

    stats = np.array([
        float(np.mean(gray)) / 255.0,
        float(np.std(gray)) / 255.0,
        float(cv2.Laplacian(gray, cv2.CV_32F).var()) / (255.0 ** 2),
        float(np.mean(lab[:, :, 1])) / 255.0,
        float(np.mean(lab[:, :, 2])) / 255.0,
    ], dtype=np.float32)
    return _unit(np.concatenate([hist, *color_parts, stats]))


def _sift_descriptor(patch: np.ndarray) -> Tuple[np.ndarray, int]:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=96, contrastThreshold=0.01, edgeThreshold=10)
    keypoints, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(desc) == 0:
        return np.zeros(128, dtype=np.float32), 0
    # Mean-pool local SIFT descriptors into a fixed-length vector and normalize.
    pooled = desc.astype(np.float32).mean(axis=0)
    return _unit(pooled), len(keypoints)


def _position(mark: Dict[str, Any]) -> Tuple[float, float]:
    p = mark.get("canonical_position") or mark.get("centroid")
    if p is None or len(p) < 2:
        raise ValueError("mark lacks canonical_position/centroid")
    x, y = float(p[0]), float(p[1])
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def extract_local_mark_feature(
    image: np.ndarray,
    mark: Dict[str, Any],
    *,
    patch_sizes: Sequence[int] = DEFAULT_PATCH_SIZES,
) -> LocalMarkFeature:
    """Extract fixed-length local/contextual features from source-resolution imagery."""
    if image is None or image.ndim != 3:
        raise ValueError("image must be an HxWx3 BGR array")
    h, w = image.shape[:2]
    x, y = _position(mark)

    local_size = int(patch_sizes[0])
    context_sizes = tuple(int(s) for s in patch_sizes[1:])
    local_patch = _safe_crop(image, x, y, local_size)

    sift_vec, keypoints = _sift_descriptor(local_patch)
    texture_vec = _color_texture_descriptor(local_patch)
    local_embedding = _unit(np.concatenate([sift_vec, texture_vec]))

    context_parts: List[np.ndarray] = []
    for size in context_sizes:
        context_patch = _safe_crop(image, x, y, size)
        sift_ctx, _ = _sift_descriptor(context_patch)
        texture_ctx = _color_texture_descriptor(context_patch)
        context_parts.append(np.concatenate([sift_ctx, texture_ctx]))
    contextual_embedding = _unit(np.concatenate(context_parts)) if context_parts else local_embedding

    visibility = float(mark.get("visibility_confidence", mark.get("confidence", 1.0)))
    visibility = max(0.0, min(1.0, visibility))

    return LocalMarkFeature(
        backend="sift+color_texture",
        version=LOCAL_MARK_FEATURE_VERSION,
        canonical_position=(x, y),
        patch_sizes=tuple(int(s) for s in patch_sizes),
        local_embedding=tuple(float(v) for v in local_embedding),
        contextual_embedding=tuple(float(v) for v in contextual_embedding),
        keypoint_count=int(keypoints),
        visibility_score=visibility,
        source_size=(int(w), int(h)),
    )


def augment_detected_marks(
    image: np.ndarray,
    marks: Sequence[Dict[str, Any]],
    *,
    patch_sizes: Sequence[int] = DEFAULT_PATCH_SIZES,
) -> List[Dict[str, Any]]:
    """Attach high-resolution local/contextual representations to detector output."""
    enriched: List[Dict[str, Any]] = []
    for index, mark in enumerate(marks):
        item = dict(mark)
        try:
            feature = extract_local_mark_feature(image, item, patch_sizes=patch_sizes)
            item["mark_id"] = str(item.get("mark_id") or f"mark-{index}")
            item["local_embedding"] = list(feature.local_embedding)
            item["contextual_embedding"] = list(feature.contextual_embedding)
            item["local_feature_backend"] = feature.backend
            item["local_feature_version"] = feature.version
            item["local_keypoint_count"] = feature.keypoint_count
            item["local_visibility_score"] = feature.visibility_score
            item["local_source_width"] = feature.source_size[0]
            item["local_source_height"] = feature.source_size[1]
        except Exception as exc:
            item["local_feature_error"] = str(exc)
        enriched.append(item)
    return enriched


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    aa = _unit(np.asarray(a, dtype=np.float32))
    bb = _unit(np.asarray(b, dtype=np.float32))
    if aa.shape != bb.shape or not np.any(aa) or not np.any(bb):
        return None
    return float(np.clip(np.dot(aa, bb), -1.0, 1.0))


def local_feature_similarity(mark_a: Dict[str, Any], mark_b: Dict[str, Any]) -> Optional[float]:
    """Compare local and contextual feature vectors without turning the result into LR/probability."""
    local = cosine_similarity(mark_a.get("local_embedding", []), mark_b.get("local_embedding", []))
    context = cosine_similarity(mark_a.get("contextual_embedding", []), mark_b.get("contextual_embedding", []))
    if local is None and context is None:
        return None
    if local is None:
        return context
    if context is None:
        return local
    return float(0.65 * local + 0.35 * context)


def enrich_mark_observation(mark: Dict[str, Any]) -> Dict[str, Any]:
    """Map detector output into the V3 MarkObservation-compatible field names."""
    position = mark.get("canonical_position") or mark.get("centroid") or [0.0, 0.0]
    return {
        "mark_id": str(mark.get("mark_id") or "unknown"),
        "anatomical_region": str(mark.get("face_region") or mark.get("anatomical_region") or "unknown"),
        "canonical_position": [float(position[0]), float(position[1])],
        "size": float(mark.get("area", mark.get("contour_area", 0.0))),
        "shape": mark.get("shape"),
        "orientation": mark.get("orientation"),
        "local_embedding": mark.get("local_embedding"),
        "contextual_embedding": mark.get("contextual_embedding"),
        "detector_confidence": float(mark.get("confidence", 0.0)),
        "visibility": mark.get("visibility", "PRESENT"),
        "persistence_score": mark.get("persistence_score"),
        "distinctiveness_score": mark.get("distinctiveness_score"),
        "mark_type": str(mark.get("mark_type") or "unknown_distinctive_feature"),
        "local_feature_backend": mark.get("local_feature_backend"),
        "local_feature_version": mark.get("local_feature_version"),
    }
