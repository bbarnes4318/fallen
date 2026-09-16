"""Research-only V3 mark pipeline.

Runs Fallen's existing mark detector on a high-resolution aligned face and then
adds source-resolution local/contextual features. This module is intentionally
not imported by production verification yet.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from mark_detector import detect_facial_marks, MARK_DETECTOR_VERSION
from local_mark_features_v3 import augment_detected_marks


RESEARCH_MARK_PIPELINE_VERSION = "3.0.0"


def run_high_resolution_mark_pipeline(
    image: np.ndarray,
    landmarks: Any,
    *,
    patch_sizes: Sequence[int] = (32, 64, 128),
) -> Dict[str, Any]:
    """Detect marks on source-resolution imagery and enrich them with local features.

    The detector remains the authority for candidate generation. No existing
    detector thresholds are changed here.
    """
    if image is None or image.ndim != 3:
        raise ValueError("image must be an HxWx3 BGR array")

    marks, rejected, occlusion, trace, overlays = detect_facial_marks(
        image,
        landmarks,
        input_is_preprocessed=True,
        structural_source_bgr=image,
    )
    enriched = augment_detected_marks(image, marks, patch_sizes=patch_sizes)

    return {
        "pipeline_version": RESEARCH_MARK_PIPELINE_VERSION,
        "mark_detector_version": MARK_DETECTOR_VERSION,
        "local_feature_backend": "sift+color_texture",
        "local_feature_version": enriched[0].get("local_feature_version") if enriched else None,
        "source_resolution": [int(image.shape[1]), int(image.shape[0])],
        "marks": enriched,
        "rejected": rejected,
        "occlusion": occlusion,
        "detector_trace": trace,
        "overlays": overlays,
    }


def build_pair_mark_features(
    probe: np.ndarray,
    probe_landmarks: Any,
    gallery: np.ndarray,
    gallery_landmarks: Any,
) -> Dict[str, Any]:
    """Run both sides of a pair through the research mark pipeline."""
    p = run_high_resolution_mark_pipeline(probe, probe_landmarks)
    g = run_high_resolution_mark_pipeline(gallery, gallery_landmarks)
    return {"probe": p, "gallery": g}
