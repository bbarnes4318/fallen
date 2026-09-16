#!/usr/bin/env python3
"""Extract reproducible V3 image/pair features for offline ablation runs.

This is research tooling only. It uses the existing global face pipeline as the
baseline, runs the existing detector at 1024px, enriches marks with the V3 local
feature engine, and emits pair-level JSONL ready for backend.evaluation.ablation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from typing import Any, Dict, List

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from identity_resolution_v3 import MarkConstellation, MarkObservation, Visibility
from mark_matcher import match_facial_marks
from local_mark_features_v3 import local_feature_similarity
from pipeline_core import (
    align_face_crop,
    compute_ensemble_similarity,
    extract_ensemble_embeddings,
    extract_geometric_ratios_3d,
)
from quality_engine_v3 import analyze_quality
from research_mark_pipeline_v3 import run_high_resolution_mark_pipeline


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _quality_value(q) -> float:
    return float(q.overall) if q.overall is not None else 0.0


def _as_observation(mark: dict) -> MarkObservation:
    vis_raw = str(mark.get("visibility", "PRESENT"))
    try:
        vis = Visibility(vis_raw)
    except ValueError:
        vis = Visibility.PRESENT
    pos = mark.get("canonical_position") or mark.get("centroid") or [0.0, 0.0]
    return MarkObservation(
        mark_id=str(mark.get("mark_id", "unknown")),
        anatomical_region=str(mark.get("face_region") or mark.get("anatomical_region") or "unknown"),
        canonical_position=(float(pos[0]), float(pos[1])),
        size=float(mark.get("area", mark.get("contour_area", 0.0))),
        shape=mark.get("shape"),
        orientation=mark.get("orientation"),
        local_embedding=tuple(mark.get("local_embedding") or []),
        contextual_embedding=tuple(mark.get("contextual_embedding") or []),
        detector_confidence=float(mark.get("confidence", 0.0)),
        visibility=vis,
        persistence_score=mark.get("persistence_score"),
        distinctiveness_score=mark.get("distinctiveness_score"),
        mark_type=str(mark.get("mark_type") or "unknown_distinctive_feature"),
    )


@lru_cache(maxsize=2048)
def extract_image_feature(path: str) -> dict:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {path}")

    aligned_global, global_landmarks = align_face_crop(image, target_size=256)
    aligned_highres, highres_landmarks = align_face_crop(image, target_size=1024)
    if global_landmarks is None or highres_landmarks is None:
        raise ValueError(f"face not detected: {path}")

    embeddings = extract_ensemble_embeddings(aligned_global)
    geom, _angles, visibility = extract_geometric_ratios_3d(global_landmarks)
    quality = analyze_quality(aligned_highres, yaw=None, pitch=None, roll=None)
    mark_payload = run_high_resolution_mark_pipeline(aligned_highres, highres_landmarks)

    return {
        "path": path,
        "source_width": int(image.shape[1]),
        "source_height": int(image.shape[0]),
        "global_embedding_arcface": embeddings[0].astype(float).tolist(),
        "global_embedding_secondary": embeddings[1].astype(float).tolist(),
        "geometry_ratios": geom.astype(float).tolist(),
        "geometry_status": visibility.get("geometry_status", "UNKNOWN"),
        "quality_overall": _quality_value(quality),
        "mark_payload": mark_payload,
    }


def pair_features(a: dict, b: dict) -> dict:
    arc_a = np.asarray(a["global_embedding_arcface"], dtype=np.float32)
    arc_b = np.asarray(b["global_embedding_arcface"], dtype=np.float32)
    sec_a = np.asarray(a["global_embedding_secondary"], dtype=np.float32)
    sec_b = np.asarray(b["global_embedding_secondary"], dtype=np.float32)
    global_similarity = 0.60 * _cosine(arc_a, arc_b) + 0.40 * _cosine(sec_a, sec_b)

    geom_a = np.asarray(a["geometry_ratios"], dtype=np.float32)
    geom_b = np.asarray(b["geometry_ratios"], dtype=np.float32)
    n = min(len(geom_a), len(geom_b))
    if n:
        distance = float(np.linalg.norm((geom_a[:n] - geom_b[:n])))
        geometry_similarity = max(0.0, min(1.0, 1.0 - distance / 0.50))
    else:
        geometry_similarity = 0.0

    marks_a = a["mark_payload"]["marks"]
    marks_b = b["mark_payload"]["marks"]
    matched = match_facial_marks(marks_a, marks_b)
    matches = matched.get("matches", [])

    local_scores = []
    for m in matches:
        ga = int(m["gallery_idx"])
        pr = int(m["probe_idx"])
        score = local_feature_similarity(marks_a[ga], marks_b[pr])
        if score is not None:
            local_scores.append(float(score))
    local_similarity = float(np.mean(local_scores)) if local_scores else 0.0

    mark_similarity = float(matched.get("score") or 0.0) / 100.0
    obs_a = [_as_observation(m) for m in marks_a]
    obs_b = [_as_observation(m) for m in marks_b]
    constellation_a = MarkConstellation(nodes=tuple(obs_a))
    constellation_b = MarkConstellation(nodes=tuple(obs_b))
    constellation_score = float(constellation_a.compare(constellation_b)["score"])

    return {
        "global_similarity": float(global_similarity),
        "geometry_similarity": float(geometry_similarity),
        "legacy_mark_score": float(mark_similarity),
        "local_similarity": float(local_similarity),
        "mark_similarity": float(mark_similarity * (0.5 + 0.5 * local_similarity)),
        "constellation_score": constellation_score,
        "quality_overall": float(min(a["quality_overall"], b["quality_overall"])),
        "pose_delta": 0.0,
        "temporal_gap": 0.0,
        "occlusion_delta": 0.0,
        "matched_mark_count": int(len(matches)),
        "mark_count_a": len(marks_a),
        "mark_count_b": len(marks_b),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_manifest_jsonl")
    parser.add_argument("output_features_jsonl")
    args = parser.parse_args()

    pairs = _read_jsonl(args.pair_manifest_jsonl)
    cache: Dict[str, dict] = {}
    out = []
    for index, pair in enumerate(pairs, 1):
        path_a = pair.get("metadata", {}).get("path_a")
        path_b = pair.get("metadata", {}).get("path_b")
        if not path_a or not path_b:
            raise ValueError(f"pair {index} missing metadata.path_a/path_b")
        if path_a not in cache:
            cache[path_a] = extract_image_feature(path_a)
        if path_b not in cache:
            cache[path_b] = extract_image_feature(path_b)
        features = pair_features(cache[path_a], cache[path_b])
        row = dict(pair)
        row.update(features)
        out.append(row)
        if index % 25 == 0:
            print(f"processed {index}/{len(pairs)} pairs; cached_images={len(cache)}", flush=True)

    with open(args.output_features_jsonl, "w", encoding="utf-8") as handle:
        for row in out:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(json.dumps({"pairs": len(out), "unique_images": len(cache)}, indent=2))


if __name__ == "__main__":
    main()
