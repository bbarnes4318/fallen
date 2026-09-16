"""Fallen V3 identity-resolution primitives.

This module intentionally does not replace the production verification path yet.
It provides the typed, auditable evidence model and conservative profile/fusion
primitives required to migrate from score/veto logic to calibrated evidence
fusion without contaminating existing identity profiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from math import exp, isfinite, log
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class Visibility(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNOBSERVABLE = "UNOBSERVABLE"


class Decision(str, Enum):
    SAME_PERSON = "SAME_PERSON"
    DIFFERENT_PERSON = "DIFFERENT_PERSON"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class QualityEvidence:
    overall: Optional[float] = None
    face_pixels: Optional[float] = None
    inter_eye_distance: Optional[float] = None
    blur: Optional[float] = None
    illumination: Optional[float] = None
    contrast: Optional[float] = None
    compression: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    occlusion: Optional[float] = None
    truncation: Optional[float] = None
    detector_confidence: Optional[float] = None
    landmark_confidence: Optional[float] = None
    alignment_confidence: Optional[float] = None


@dataclass(frozen=True)
class MarkObservation:
    mark_id: str
    anatomical_region: str
    canonical_position: Tuple[float, float]
    size: float
    shape: Optional[str] = None
    orientation: Optional[float] = None
    local_embedding: Optional[Tuple[float, ...]] = None
    contextual_embedding: Optional[Tuple[float, ...]] = None
    detector_confidence: float = 0.0
    visibility: Visibility = Visibility.PRESENT
    persistence_score: Optional[float] = None
    distinctiveness_score: Optional[float] = None
    mark_type: str = "unknown_distinctive_feature"


@dataclass(frozen=True)
class MarkCorrespondence:
    probe_mark_id: str
    gallery_mark_id: str
    anatomical_consistency: float
    visual_similarity: float
    contextual_similarity: float
    shape_similarity: float
    size_similarity: float
    neighborhood_consistency: float
    visibility_confidence: float
    correspondence_score: float
    same_feature_probability: Optional[float] = None


@dataclass(frozen=True)
class ConstellationEdge:
    source_mark_id: str
    target_mark_id: str
    normalized_distance: float
    relative_angle: float
    region_relation: str


@dataclass(frozen=True)
class MarkConstellation:
    nodes: Tuple[MarkObservation, ...] = ()
    edges: Tuple[ConstellationEdge, ...] = ()

    def compare(self, other: "MarkConstellation") -> Dict[str, float]:
        """Interpretable relational consistency baseline.

        This is deliberately not a calibrated identity probability. It is a
        feature extractor for downstream fusion and ablation experiments.
        """
        if not self.nodes or not other.nodes:
            return {"node_overlap": 0.0, "edge_consistency": 0.0, "score": 0.0}

        def key(m: MarkObservation) -> Tuple[str, str]:
            return (m.anatomical_region, m.mark_type)

        left = {key(m): m for m in self.nodes if m.visibility is Visibility.PRESENT}
        right = {key(m): m for m in other.nodes if m.visibility is Visibility.PRESENT}
        common = set(left).intersection(right)
        node_overlap = len(common) / max(1, min(len(left), len(right)))

        # Compare normalized pairwise geometry for corresponding semantic nodes.
        pair_errors: List[float] = []
        for k in common:
            for j in common:
                if k >= j:
                    continue
                a1, a2 = left[k], left[j]
                b1, b2 = right[k], right[j]
                da = float(np.linalg.norm(np.asarray(a1.canonical_position) - np.asarray(a2.canonical_position)))
                db = float(np.linalg.norm(np.asarray(b1.canonical_position) - np.asarray(b2.canonical_position)))
                denom = max(da, db, 1e-6)
                pair_errors.append(abs(da - db) / denom)
        edge_consistency = 1.0 - min(1.0, float(np.mean(pair_errors))) if pair_errors else 0.0
        score = 0.60 * node_overlap + 0.40 * edge_consistency
        return {"node_overlap": node_overlap, "edge_consistency": edge_consistency, "score": score}


@dataclass
class IdentityProfile:
    identity_id: str
    observation_count: int = 0
    global_embedding_centroid: Optional[np.ndarray] = None
    global_embedding_variance: Optional[np.ndarray] = None
    persistent_marks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    quality_history: List[float] = field(default_factory=list)
    temporal_observations: List[float] = field(default_factory=list)
    variants: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if isinstance(result.get("global_embedding_centroid"), np.ndarray):
            result["global_embedding_centroid"] = result["global_embedding_centroid"].tolist()
        if isinstance(result.get("global_embedding_variance"), np.ndarray):
            result["global_embedding_variance"] = result["global_embedding_variance"].tolist()
        return result

    def update(self, *, embedding: Optional[np.ndarray], quality: Optional[float],
               marks: Sequence[MarkObservation], temporal_value: Optional[float],
               admission_confidence: float, min_admission_confidence: float = 0.995) -> bool:
        """Conservatively update the profile.

        A profile is not mutated unless the caller supplies an independently
        calibrated admission confidence above the configured threshold.
        """
        if admission_confidence < min_admission_confidence:
            return False

        self.observation_count += 1
        if quality is not None and isfinite(float(quality)):
            self.quality_history.append(float(quality))
        if temporal_value is not None and isfinite(float(temporal_value)):
            self.temporal_observations.append(float(temporal_value))

        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32)
            if emb.ndim != 1 or not np.all(np.isfinite(emb)):
                raise ValueError("embedding must be a finite 1-D vector")
            if self.global_embedding_centroid is None:
                self.global_embedding_centroid = emb.copy()
                self.global_embedding_variance = np.zeros_like(emb)
            else:
                n = max(1, self.observation_count - 1)
                old = self.global_embedding_centroid.copy()
                delta = emb - old
                self.global_embedding_centroid = old + delta / self.observation_count
                if self.global_embedding_variance is None:
                    self.global_embedding_variance = np.zeros_like(emb)
                self.global_embedding_variance = ((n - 1) / n) * self.global_embedding_variance + (delta * (emb - self.global_embedding_centroid)) / n

        for mark in marks:
            if mark.visibility is not Visibility.PRESENT:
                continue
            bucket = self.persistent_marks.setdefault(mark.mark_id, {
                "anatomical_region": mark.anatomical_region,
                "mark_type": mark.mark_type,
                "observations": 0,
                "visible_observations": 0,
                "positions": [],
                "sizes": [],
                "persistence_scores": [],
                "distinctiveness_scores": [],
            })
            bucket["observations"] += 1
            bucket["visible_observations"] += 1
            bucket["positions"].append(list(mark.canonical_position))
            bucket["sizes"].append(float(mark.size))
            if mark.persistence_score is not None:
                bucket["persistence_scores"].append(float(mark.persistence_score))
            if mark.distinctiveness_score is not None:
                bucket["distinctiveness_scores"].append(float(mark.distinctiveness_score))
        return True


def _safe_unit(v: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(a))
    if a.ndim != 1 or norm <= 0 or not np.all(np.isfinite(a)):
        return None
    return a / norm


def cosine_similarity(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    ua, ub = _safe_unit(a), _safe_unit(b)
    if ua is None or ub is None or ua.shape != ub.shape:
        return None
    return float(np.dot(ua, ub))


def mark_distinctiveness_information(distinctiveness_probability: Optional[float]) -> Optional[float]:
    """Return self-information -log(p) without inventing prevalence rates."""
    if distinctiveness_probability is None:
        return None
    p = float(distinctiveness_probability)
    if not (0.0 < p <= 1.0):
        return None
    return -log(p)


def build_fusion_features(*, global_similarity: Optional[float], local_similarity: Optional[float],
                          geometry_similarity: Optional[float], asymmetry_similarity: Optional[float],
                          quality: QualityEvidence, mark_correspondences: Sequence[MarkCorrespondence],
                          constellation_score: Optional[float], temporal_gap: Optional[float]) -> Dict[str, float]:
    """Build a bounded feature vector for calibration; never converts to a probability."""
    corrs = list(mark_correspondences)
    return {
        "global_similarity": _bounded(global_similarity),
        "local_similarity": _bounded(local_similarity),
        "geometry_similarity": _bounded(geometry_similarity),
        "asymmetry_similarity": _bounded(asymmetry_similarity),
        "quality_overall": _bounded(quality.overall),
        "quality_blur": _bounded(quality.blur),
        "quality_occlusion": _bounded(quality.occlusion),
        "mark_count": float(len(corrs)),
        "distinctive_mark_count": float(sum(1 for c in corrs if (c.same_feature_probability or 1.0) < 0.10)),
        "mark_correspondence_mean": float(np.mean([c.correspondence_score for c in corrs])) if corrs else 0.0,
        "mark_correspondence_max": float(max((c.correspondence_score for c in corrs), default=0.0)),
        "constellation_score": _bounded(constellation_score),
        "temporal_gap": max(0.0, float(temporal_gap or 0.0)),
    }


def _bounded(value: Optional[float]) -> float:
    if value is None or not isfinite(float(value)):
        return 0.0
    return max(-1.0, min(1.0, float(value)))


@dataclass(frozen=True)
class CalibrationSpec:
    version: str
    feature_names: Tuple[str, ...]
    positive_count: int
    negative_count: int
    hard_negative_count: int
    twin_count: int
    dataset_id: str
    created_at: str
    model_version: str


class CalibratedFusionModel:
    """Thin wrapper around a persisted sklearn classifier.

    Fitting is intentionally separated from inference. Production callers must
    load a model produced by a partitioned evaluation run; this class does not
    invent probabilities from raw similarity thresholds.
    """

    def __init__(self, estimator: Any = None, spec: Optional[CalibrationSpec] = None):
        self.estimator = estimator
        self.spec = spec

    def predict_probability(self, features: Dict[str, float]) -> Optional[float]:
        if self.estimator is None or not self.spec:
            return None
        x = np.asarray([[features.get(name, 0.0) for name in self.spec.feature_names]], dtype=np.float64)
        if hasattr(self.estimator, "predict_proba"):
            p = float(self.estimator.predict_proba(x)[0, 1])
        else:
            score = float(self.estimator.decision_function(x)[0])
            p = 1.0 / (1.0 + exp(-score))
        return max(0.0, min(1.0, p))

    def decision(self, probability: Optional[float], *, same_threshold: float = 0.995,
                 different_threshold: float = 0.005) -> Decision:
        if probability is None:
            return Decision.INDETERMINATE
        if probability >= same_threshold:
            return Decision.SAME_PERSON
        if probability <= different_threshold:
            return Decision.DIFFERENT_PERSON
        return Decision.INDETERMINATE
