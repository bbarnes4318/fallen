"""Image-quality primitives for Fallen V3.

Quality is evidence about observability, not evidence about identity. A poor
image must be able to produce INDETERMINATE rather than DIFFERENT PERSON.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import cv2
import numpy as np

from identity_resolution_v3 import QualityEvidence


def analyze_quality(image: np.ndarray, *, face_bbox: Optional[tuple[int, int, int, int]] = None,
                    detector_confidence: Optional[float] = None,
                    landmark_confidence: Optional[float] = None,
                    alignment_confidence: Optional[float] = None,
                    yaw: Optional[float] = None,
                    pitch: Optional[float] = None,
                    roll: Optional[float] = None,
                    occlusion: Optional[float] = None,
                    truncation: Optional[float] = None) -> QualityEvidence:
    if image is None or image.size == 0:
        return QualityEvidence(overall=0.0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    roi = gray
    face_pixels = None
    if face_bbox is not None:
        x, y, bw, bh = face_bbox
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 > x0 and y1 > y0:
            roi = gray[y0:y1, x0:x1]
            face_pixels = float(min(bw, bh))

    blur = _blur_quality(roi)
    illumination = _illumination_quality(roi)
    contrast = _contrast_quality(roi)
    compression = _compression_quality(roi)
    geometry = _optional_mean([detector_confidence, landmark_confidence, alignment_confidence])
    pose = _pose_quality(yaw, pitch, roll)
    visibility = _optional_mean([1.0 - float(occlusion) if occlusion is not None else None,
                                 1.0 - float(truncation) if truncation is not None else None])
    overall = _geometric_mean([blur, illumination, contrast, compression, geometry, pose, visibility])
    return QualityEvidence(
        overall=overall,
        face_pixels=face_pixels,
        inter_eye_distance=None,
        blur=blur,
        illumination=illumination,
        contrast=contrast,
        compression=compression,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        occlusion=occlusion,
        truncation=truncation,
        detector_confidence=detector_confidence,
        landmark_confidence=landmark_confidence,
        alignment_confidence=alignment_confidence,
    )


def quality_dict(q: QualityEvidence) -> Dict[str, Any]:
    return asdict(q)


def _blur_quality(gray: np.ndarray) -> float:
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Saturating mapping; this is an observability score, not a probability.
    return float(np.clip(variance / (variance + 150.0), 0.0, 1.0))


def _illumination_quality(gray: np.ndarray) -> float:
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    mean_term = 1.0 - min(1.0, abs(mean - 128.0) / 128.0)
    spread_term = min(1.0, std / 64.0)
    return float(np.clip(0.65 * mean_term + 0.35 * spread_term, 0.0, 1.0))


def _contrast_quality(gray: np.ndarray) -> float:
    p5, p95 = np.percentile(gray, [5, 95])
    span = float(max(0.0, p95 - p5))
    return float(np.clip(span / 160.0, 0.0, 1.0))


def _compression_quality(gray: np.ndarray) -> float:
    # Blocking proxy based on 8-pixel boundary discontinuities. Lower boundary
    # energy relative to interior gradients is treated as less suspicious.
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return 0.0
    vert = np.abs(gray[:, 8::8].astype(np.float32) - gray[:, 7::8].astype(np.float32)).mean()
    horiz = np.abs(gray[8::8, :].astype(np.float32) - gray[7::8, :].astype(np.float32)).mean()
    interior = np.abs(np.diff(gray.astype(np.float32), axis=1)).mean() + np.abs(np.diff(gray.astype(np.float32), axis=0)).mean()
    ratio = (float(vert + horiz) + 1e-6) / (float(interior) + 1e-6)
    return float(np.clip(1.0 - max(0.0, ratio - 1.0) / 3.0, 0.0, 1.0))


def _pose_quality(yaw: Optional[float], pitch: Optional[float], roll: Optional[float]) -> float:
    if yaw is None and pitch is None and roll is None:
        return 0.5
    vals = []
    for angle, limit in ((yaw, 45.0), (pitch, 35.0), (roll, 30.0)):
        if angle is not None:
            vals.append(max(0.0, 1.0 - min(abs(float(angle)) / limit, 1.0)))
    return _optional_mean(vals) if vals else 0.5


def _optional_mean(values) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.5


def _geometric_mean(values) -> float:
    vals = [max(1e-4, min(1.0, float(v))) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))
