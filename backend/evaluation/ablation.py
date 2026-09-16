"""Reproducible V3 ablation/evaluation runner.

The runner consumes pair-level feature records. Feature extraction is kept
separate so expensive image/model work can be cached and repeated experiments
do not silently change the source evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import brier_score, eer, tar_at_far


VARIANTS: Dict[str, Tuple[str, ...]] = {
    "global_only": ("global_similarity",),
    "global_geometry": ("global_similarity", "geometry_similarity"),
    "global_legacy_marks": ("global_similarity", "legacy_mark_score"),
    "global_local": ("global_similarity", "local_similarity"),
    "global_local_marks": ("global_similarity", "local_similarity", "mark_similarity"),
    "global_local_constellation": (
        "global_similarity", "local_similarity", "constellation_score"
    ),
    "full_v3": (
        "global_similarity",
        "local_similarity",
        "geometry_similarity",
        "mark_similarity",
        "constellation_score",
        "quality_overall",
        "pose_delta",
        "temporal_gap",
        "occlusion_delta",
    ),
}


@dataclass(frozen=True)
class AblationMetrics:
    variant: str
    feature_names: Tuple[str, ...]
    train_pairs: int
    eval_pairs: int
    eval_genuine: int
    eval_impostor: int
    eer: float
    tar_at_far_1e3: float
    tar_at_far_1e4: float
    brier: float

    def to_dict(self):
        return asdict(self)


def _identity_set(rows: Sequence[Mapping]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        ids.add(str(row["subject_a"]))
        ids.add(str(row["subject_b"]))
    return ids


def assert_identity_disjoint(train_rows: Sequence[Mapping], eval_rows: Sequence[Mapping]) -> None:
    overlap = _identity_set(train_rows) & _identity_set(eval_rows)
    if overlap:
        raise ValueError(f"identity leakage between train/eval: {sorted(overlap)[:20]}")


def _matrix(rows: Sequence[Mapping], features: Sequence[str]) -> np.ndarray:
    return np.asarray([
        [float(row.get(name, 0.0) or 0.0) for name in features]
        for row in rows
    ], dtype=np.float64)


def run_variant(
    train_rows: Sequence[Mapping],
    eval_rows: Sequence[Mapping],
    *,
    variant: str,
) -> AblationMetrics:
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant: {variant}")
    features = VARIANTS[variant]
    if not train_rows or not eval_rows:
        raise ValueError("train and eval sets must both be non-empty")

    assert_identity_disjoint(train_rows, eval_rows)

    y_train = np.asarray([int(bool(r["same_identity"])) for r in train_rows], dtype=np.int32)
    y_eval = np.asarray([int(bool(r["same_identity"])) for r in eval_rows], dtype=np.int32)
    if len(np.unique(y_train)) < 2:
        raise ValueError("training data must contain genuine and impostor pairs")

    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    model.fit(_matrix(train_rows, features), y_train)
    probabilities = model.predict_proba(_matrix(eval_rows, features))[:, 1]

    return AblationMetrics(
        variant=variant,
        feature_names=tuple(features),
        train_pairs=len(train_rows),
        eval_pairs=len(eval_rows),
        eval_genuine=int(np.sum(y_eval == 1)),
        eval_impostor=int(np.sum(y_eval == 0)),
        eer=float(eer(y_eval, probabilities)),
        tar_at_far_1e3=float(tar_at_far(y_eval, probabilities, 1e-3)["tar"]),
        tar_at_far_1e4=float(tar_at_far(y_eval, probabilities, 1e-4)["tar"]),
        brier=float(brier_score(y_eval, probabilities)),
    )


def run_ablation(records: Sequence[Mapping], *, train_split: str = "calibration", eval_split: str = "eval") -> List[AblationMetrics]:
    train = [r for r in records if r.get("split") == train_split]
    evaluation = [r for r in records if r.get("split") == eval_split]
    assert_identity_disjoint(train, evaluation)
    return [run_variant(train, evaluation, variant=name) for name in VARIANTS]


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def save_results(path: str, results: Sequence[AblationMetrics]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([r.to_dict() for r in results], handle, indent=2, sort_keys=True)
