"""Calibration utilities for V3 evidence fusion.

The caller must provide a properly partitioned dataset. This module never falls
back to a hand-authored probability mapping.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from identity_resolution_v3 import CalibrationSpec, CalibratedFusionModel


def fit_logistic_calibrator(
    feature_rows: Sequence[Dict[str, float]],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    dataset_id: str,
    created_at: str,
    model_version: str,
    n_splits: int = 5,
) -> Tuple[CalibratedFusionModel, Dict[str, float]]:
    if not feature_rows:
        raise ValueError("feature_rows cannot be empty")
    names = tuple(sorted(feature_rows[0].keys()))
    X = np.asarray([[row.get(name, 0.0) for name in names] for row in feature_rows], dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    g = np.asarray(groups)
    if len(X) != len(y) or len(y) != len(g):
        raise ValueError("feature_rows, labels, and groups must have equal length")
    if len(np.unique(y)) < 2:
        raise ValueError("both genuine and impostor examples are required")
    splits = min(n_splits, len(np.unique(g)))
    if splits < 2:
        raise ValueError("at least two identity groups are required for leakage-aware calibration")

    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=GroupKFold(n_splits=splits))
    calibrated.fit(X, y, groups=g)
    p = calibrated.predict_proba(X)[:, 1]
    diagnostics = {
        "train_auc": float(roc_auc_score(y, p)),
        "train_brier": float(brier_score_loss(y, p)),
        "sample_count": float(len(y)),
        "positive_count": float(np.sum(y == 1)),
        "negative_count": float(np.sum(y == 0)),
    }
    spec = CalibrationSpec(
        version="v3-logistic-sigmoid-1",
        feature_names=names,
        positive_count=int(np.sum(y == 1)),
        negative_count=int(np.sum(y == 0)),
        hard_negative_count=0,
        twin_count=0,
        dataset_id=dataset_id,
        created_at=created_at,
        model_version=model_version,
    )
    return CalibratedFusionModel(calibrated, spec), diagnostics


def serialize_calibration_metadata(model: CalibratedFusionModel) -> Dict[str, object]:
    if model.spec is None:
        raise ValueError("model has no calibration spec")
    return asdict(model.spec)
