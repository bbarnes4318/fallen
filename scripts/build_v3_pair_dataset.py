#!/usr/bin/env python3
"""Build an identity-disjoint Fallen V3 pair manifest from an image manifest.

Input JSONL fields (minimum):
  identity_id, image_id, path, split
Optional: quality, camera_domain, age, mark_count, subset, twin_group,
          sibling_group, occlusion, pose.

The builder never moves identities between splits and uses a deterministic seed
for impostor sampling. It produces PairRecord-compatible JSONL with image paths
retained under metadata for feature extraction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from typing import Dict, List


def load_images(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            for key in ("identity_id", "image_id", "path", "split"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no}: missing {key}")
            rows.append(row)
    return rows


def pair_subset(a: dict, b: dict, same: bool) -> str:
    if same and a.get("quality", 1.0) is not None and b.get("quality", 1.0) is not None:
        if float(a.get("quality", 1.0)) < 0.35 and float(b.get("quality", 1.0)) < 0.35:
            return "poor_quality_same"
    if not same and float(a.get("quality", 1.0)) < 0.35 and float(b.get("quality", 1.0)) < 0.35:
        return "poor_quality_different"
    if not same and a.get("twin_group") and a.get("twin_group") == b.get("twin_group"):
        return "twins"
    if not same and a.get("sibling_group") and a.get("sibling_group") == b.get("sibling_group"):
        return "siblings"
    if a.get("camera_domain") and b.get("camera_domain") and a["camera_domain"] != b["camera_domain"]:
        return "cross_camera"
    if same and a.get("age") is not None and b.get("age") is not None and abs(float(a["age"]) - float(b["age"])) >= 10:
        return "cross_age"
    if (a.get("mark_count") or 0) >= 3 and (b.get("mark_count") or 0) >= 3:
        return "mark_rich_same" if same else "mark_rich_different"
    if a.get("occlusion", 0.0) and b.get("occlusion", 0.0):
        return "occluded"
    return "normal_same" if same else "normal_different"


def make_record(a: dict, b: dict, same: bool) -> dict:
    subset = pair_subset(a, b, same)
    return {
        "subject_a": str(a["identity_id"]),
        "subject_b": str(b["identity_id"]),
        "same_identity": bool(same),
        "source_a": str(a["image_id"]),
        "source_b": str(b["image_id"]),
        "quality_a": a.get("quality"),
        "quality_b": b.get("quality"),
        "temporal_gap": abs(float(a["age"]) - float(b["age"])) if a.get("age") is not None and b.get("age") is not None else None,
        "camera_domain_a": a.get("camera_domain"),
        "camera_domain_b": b.get("camera_domain"),
        "occlusion_a": a.get("occlusion"),
        "occlusion_b": b.get("occlusion"),
        "mark_count_a": a.get("mark_count"),
        "mark_count_b": b.get("mark_count"),
        "twin_pair": subset == "twins",
        "sibling_pair": subset == "siblings",
        "difficult_pair": subset in {"twins", "siblings", "cross_age", "cross_camera", "poor_quality_different", "mark_rich_different"},
        "subset": subset,
        "split": a["split"],
        "metadata": {"path_a": a["path"], "path_b": b["path"], "image_id_a": a["image_id"], "image_id_b": b["image_id"]},
    }


def build(rows: List[dict], *, max_genuine_per_identity: int, impostor_multiplier: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    by_split = defaultdict(list)
    by_identity = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
        by_identity[(row["split"], row["identity_id"])].append(row)

    output: List[dict] = []
    for split_rows in by_split.values():
        identities = sorted({r["identity_id"] for r in split_rows})
        split_id_groups = {identity: by_identity[(split_rows[0]["split"], identity)] for identity in identities}
        for identity, images in split_id_groups.items():
            genuine_pairs = list(itertools.combinations(images, 2))
            rng.shuffle(genuine_pairs)
            for a, b in genuine_pairs[:max_genuine_per_identity]:
                output.append(make_record(a, b, True))

        all_impostors = []
        for i, left_id in enumerate(identities):
            for right_id in identities[i + 1:]:
                left = split_id_groups[left_id]
                right = split_id_groups[right_id]
                all_impostors.extend((a, b) for a in left for b in right)
        rng.shuffle(all_impostors)
        target = max(1, len([r for r in output if r["split"] == split_rows[0]["split"]])) * impostor_multiplier
        for a, b in all_impostors[:target]:
            output.append(make_record(a, b, False))

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("output_jsonl")
    parser.add_argument("--max-genuine-per-identity", type=int, default=100)
    parser.add_argument("--impostor-multiplier", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    rows = load_images(args.input_jsonl)
    pairs = build(rows, max_genuine_per_identity=args.max_genuine_per_identity,
                  impostor_multiplier=args.impostor_multiplier, seed=args.seed)
    with open(args.output_jsonl, "w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"images": len(rows), "pairs": len(pairs), "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
