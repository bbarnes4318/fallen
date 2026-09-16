"""Dataset manifest schema for leakage-aware Fallen evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PairRecord:
    subject_a: str
    subject_b: str
    same_identity: bool
    source_a: str
    source_b: str
    quality_a: Optional[float] = None
    quality_b: Optional[float] = None
    pose_a: Optional[Dict[str, float]] = None
    pose_b: Optional[Dict[str, float]] = None
    temporal_gap: Optional[float] = None
    camera_domain_a: Optional[str] = None
    camera_domain_b: Optional[str] = None
    occlusion_a: Optional[float] = None
    occlusion_b: Optional[float] = None
    mark_count_a: Optional[int] = None
    mark_count_b: Optional[int] = None
    mark_correspondence_ground_truth: Optional[bool] = None
    twin_pair: bool = False
    sibling_pair: bool = False
    difficult_pair: bool = False
    subset: Optional[str] = None
    split: str = "eval"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


REQUIRED_SUBSETS = (
    "normal_same",
    "normal_different",
    "lookalike",
    "siblings",
    "twins",
    "cross_age",
    "poor_quality_same",
    "poor_quality_different",
    "mark_rich_same",
    "mark_rich_different",
    "occluded",
    "partial_face",
    "archival",
    "cross_camera",
)


def validate_split(records: List[PairRecord]) -> List[str]:
    """Return leakage warnings/errors detectable from manifest metadata."""
    errors: List[str] = []
    identities_by_split: Dict[str, set[str]] = {}
    for r in records:
        bucket = identities_by_split.setdefault(r.split, set())
        bucket.add(r.subject_a)
        bucket.add(r.subject_b)

    splits = list(identities_by_split)
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            overlap = identities_by_split[left] & identities_by_split[right]
            if overlap:
                errors.append(f"identity leakage between {left} and {right}: {sorted(overlap)}")

    for r in records:
        if r.same_identity and r.subject_a != r.subject_b:
            errors.append(f"same_identity record has different subject IDs: {r.subject_a}/{r.subject_b}")
        if not r.same_identity and r.subject_a == r.subject_b:
            errors.append(f"impostor record reuses the same subject ID: {r.subject_a}")
    return errors
