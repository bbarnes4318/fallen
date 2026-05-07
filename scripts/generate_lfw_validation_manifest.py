#!/usr/bin/env python3
"""
LFW Validation Manifest Generator
==================================
Generates a validation_pairs CSV from the official LFW pairs text file
by mapping person names and image numbers to existing GCS URIs.

This script does NOT:
  - Download or upload images
  - Process image pixels
  - Run the validation pipeline
  - Touch production scoring or /vault/search

Usage (GitHub Actions or Cloud Shell):
  python scripts/generate_lfw_validation_manifest.py \
    --pairs-url https://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt \
    --gcs-prefix gs://hoppwhistle-facial-uploads/gallery/ \
    --output validation/validation_pairs_lfw_baseline.csv \
    --max-matched 50 \
    --max-mismatched 50

  python scripts/generate_lfw_validation_manifest.py \
    --pairs-file /path/to/pairsDevTrain.txt \
    --output validation/validation_pairs_lfw_baseline.csv \
    --check-gcs-exists

  python scripts/generate_lfw_validation_manifest.py \
    --pairs-url https://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt \
    --dry-run
"""

import argparse
import csv
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Optional


# ── LFW Pairs File Format ──
# Line 1: <num_folds>\t<pairs_per_fold>   (e.g. "10\t300")
# Matched pairs:    Name\tImageNum1\tImageNum2
# Mismatched pairs: Name1\tImageNum1\tName2\tImageNum2
#
# Image number N maps to filename: Name_NNNN.jpg (zero-padded to 4 digits)
# GCS path: gs://bucket/gallery/Name_NNNN.jpg


def lfw_image_name(person_name: str, image_num: int) -> str:
    """Convert LFW person name + image number to the filename used in GCS.

    LFW convention: Name_NNNN.jpg where NNNN is zero-padded to 4 digits.
    The mass_ingest.py script uploaded images using Path(filepath).stem as
    the blob name, preserving the original LFW naming convention.
    """
    return f"{person_name}_{image_num:04d}.jpg"


def parse_lfw_pairs(lines: List[str]) -> Tuple[List[dict], List[dict]]:
    """Parse the official LFW pairs text file.

    Returns:
        (matched_pairs, mismatched_pairs) where each entry is a dict with:
        - name1, num1, name2, num2, label
    """
    matched = []
    mismatched = []

    # Skip header line (e.g. "10\t300")
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split('\t')
        # Header line has exactly 2 numeric fields
        if len(parts) == 2:
            try:
                int(parts[0])
                int(parts[1])
                start_idx = i + 1
                break
            except ValueError:
                pass
        # If first non-empty line isn't a header, start from line 0
        start_idx = 0
        break

    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split('\t')

        if len(parts) == 3:
            # Matched pair: Name\tNum1\tNum2
            try:
                name = parts[0]
                num1 = int(parts[1])
                num2 = int(parts[2])
                matched.append({
                    "name1": name, "num1": num1,
                    "name2": name, "num2": num2,
                    "label": True,
                })
            except (ValueError, IndexError):
                print(f"  WARNING: Could not parse matched line: {stripped}")

        elif len(parts) == 4:
            # Mismatched pair: Name1\tNum1\tName2\tNum2
            try:
                name1 = parts[0]
                num1 = int(parts[1])
                name2 = parts[2]
                num2 = int(parts[3])
                mismatched.append({
                    "name1": name1, "num1": num1,
                    "name2": name2, "num2": num2,
                    "label": False,
                })
            except (ValueError, IndexError):
                print(f"  WARNING: Could not parse mismatched line: {stripped}")

    return matched, mismatched


def fetch_pairs_from_url(url: str) -> List[str]:
    """Download the LFW pairs text file (~30KB) from a URL."""
    import urllib.request
    print(f"  Fetching pairs file from: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Fallen-Validation/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        lines = content.splitlines()
        print(f"  Fetched {len(lines)} lines.")
        return lines
    except Exception as e:
        print(f"  ERROR: Failed to fetch pairs file: {e}")
        sys.exit(1)


def read_pairs_from_file(filepath: str) -> List[str]:
    """Read the LFW pairs text file from a local path."""
    print(f"  Reading pairs file from: {filepath}")
    with open(filepath, "r") as f:
        lines = f.readlines()
    print(f"  Read {len(lines)} lines.")
    return lines


def check_gcs_blob_exists(gcs_uri: str) -> bool:
    """Check whether a GCS blob exists. Requires google-cloud-storage."""
    try:
        from google.cloud import storage
    except ImportError:
        print("  WARNING: google-cloud-storage not installed. Cannot verify GCS existence.")
        return True  # Assume exists if we can't check

    # Parse gs://bucket/path
    if not gcs_uri.startswith("gs://"):
        return False
    parts = gcs_uri[5:].split("/", 1)
    if len(parts) != 2:
        return False
    bucket_name, blob_path = parts

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.exists()
    except Exception as e:
        print(f"  WARNING: GCS check failed for {gcs_uri}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate a validation manifest CSV from LFW pairs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pairs-file", type=str,
        help="Local path to an LFW pairs text file (e.g. pairsDevTrain.txt)"
    )
    source.add_argument(
        "--pairs-url", type=str,
        help="URL to download the LFW pairs text file from"
    )
    parser.add_argument(
        "--gcs-prefix", type=str,
        default="gs://hoppwhistle-facial-uploads/gallery/",
        help="GCS prefix where LFW gallery images are stored"
    )
    parser.add_argument(
        "--output", type=str,
        default="validation/validation_pairs_lfw_baseline.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--max-matched", type=int, default=50,
        help="Maximum number of matched (same-person) pairs"
    )
    parser.add_argument(
        "--max-mismatched", type=int, default=50,
        help="Maximum number of mismatched (different-person) pairs"
    )
    parser.add_argument(
        "--check-gcs-exists", action="store_true",
        help="Verify each GCS blob exists before including the pair"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and validate the pairs file without writing output"
    )

    args = parser.parse_args()

    # ── Ensure GCS prefix ends with /
    gcs_prefix = args.gcs_prefix
    if not gcs_prefix.endswith("/"):
        gcs_prefix += "/"

    # ── 1. Load pairs file ──
    if args.pairs_file:
        lines = read_pairs_from_file(args.pairs_file)
    else:
        lines = fetch_pairs_from_url(args.pairs_url)

    # ── 2. Parse pairs ──
    matched, mismatched = parse_lfw_pairs(lines)
    print(f"\n  Parsed {len(matched)} matched pairs, {len(mismatched)} mismatched pairs.")

    if not matched and not mismatched:
        print("  ERROR: No pairs found. Check the pairs file format.")
        sys.exit(1)

    # ── 3. Apply limits ──
    selected_matched = matched[:args.max_matched]
    selected_mismatched = mismatched[:args.max_mismatched]
    print(f"  Selected {len(selected_matched)} matched, {len(selected_mismatched)} mismatched.")

    # ── 4. Build manifest rows ──
    rows = []
    gcs_check_pass = 0
    gcs_check_fail = 0

    for i, pair in enumerate(selected_matched, 1):
        img1_name = lfw_image_name(pair["name1"], pair["num1"])
        img2_name = lfw_image_name(pair["name2"], pair["num2"])
        img1_uri = f"{gcs_prefix}{img1_name}"
        img2_uri = f"{gcs_prefix}{img2_name}"

        # Optional GCS existence check
        if args.check_gcs_exists:
            exists1 = check_gcs_blob_exists(img1_uri)
            exists2 = check_gcs_blob_exists(img2_uri)
            if not exists1 or not exists2:
                gcs_check_fail += 1
                missing = []
                if not exists1:
                    missing.append(img1_name)
                if not exists2:
                    missing.append(img2_name)
                print(f"  SKIP matched pair {i}: missing {', '.join(missing)}")
                continue
            gcs_check_pass += 1

        rows.append({
            "pair_id": f"lfw_matched_{i:04d}",
            "image1_url_or_gcs_path": img1_uri,
            "image2_url_or_gcs_path": img2_uri,
            "label_same_person": "true",
            "category": "same_person_normal",
            "notes": f"LFW matched pair: {pair['name1']} images {pair['num1']} and {pair['num2']}. "
                     f"Same person confirmed by LFW ground truth label.",
            "expected_challenge_type": "baseline",
        })

    for i, pair in enumerate(selected_mismatched, 1):
        img1_name = lfw_image_name(pair["name1"], pair["num1"])
        img2_name = lfw_image_name(pair["name2"], pair["num2"])
        img1_uri = f"{gcs_prefix}{img1_name}"
        img2_uri = f"{gcs_prefix}{img2_name}"

        # Optional GCS existence check
        if args.check_gcs_exists:
            exists1 = check_gcs_blob_exists(img1_uri)
            exists2 = check_gcs_blob_exists(img2_uri)
            if not exists1 or not exists2:
                gcs_check_fail += 1
                missing = []
                if not exists1:
                    missing.append(img1_name)
                if not exists2:
                    missing.append(img2_name)
                print(f"  SKIP mismatched pair {i}: missing {', '.join(missing)}")
                continue
            gcs_check_pass += 1

        rows.append({
            "pair_id": f"lfw_mismatched_{i:04d}",
            "image1_url_or_gcs_path": img1_uri,
            "image2_url_or_gcs_path": img2_uri,
            "label_same_person": "false",
            "category": "different_person_random",
            "notes": f"LFW mismatched pair: {pair['name1']} image {pair['num1']} vs "
                     f"{pair['name2']} image {pair['num2']}. "
                     f"Different person confirmed by LFW ground truth label.",
            "expected_challenge_type": "baseline",
        })

    # ── 5. Summary ──
    total = len(rows)
    same_count = sum(1 for r in rows if r["label_same_person"] == "true")
    diff_count = sum(1 for r in rows if r["label_same_person"] == "false")

    print(f"\n{'='*60}")
    print(f"  MANIFEST GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total pairs:           {total}")
    print(f"  Same person:           {same_count}")
    print(f"  Different person:      {diff_count}")
    if args.check_gcs_exists:
        print(f"  GCS exists (pass):     {gcs_check_pass}")
        print(f"  GCS exists (fail):     {gcs_check_fail}")
    print(f"  Categories present:    same_person_normal, different_person_random")
    print(f"  Categories missing:    same_person_age_gap, same_person_lighting_pose,")
    print(f"                         different_person_lookalike, mark_heavy_same_person,")
    print(f"                         mark_heavy_different_person, twins_or_high_similarity_imposters")
    print(f"{'='*60}")

    if args.dry_run:
        print(f"\n  DRY RUN — No output file written.")
        print(f"  Would write {total} rows to: {args.output}")

        # Write a dry-run summary JSON instead
        summary = {
            "dry_run": True,
            "total_pairs": total,
            "same_person": same_count,
            "different_person": diff_count,
            "output_path": args.output,
            "gcs_prefix": gcs_prefix,
            "pairs_source": args.pairs_file or args.pairs_url,
            "max_matched": args.max_matched,
            "max_mismatched": args.max_mismatched,
            "check_gcs_exists": args.check_gcs_exists,
            "first_5_rows": rows[:5] if rows else [],
        }
        # Write summary to output dir
        output_dir = os.path.dirname(args.output) or "."
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "lfw_manifest_dry_run_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Dry-run summary saved to: {summary_path}")
        return

    # ── 6. Write CSV ──
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "pair_id",
        "image1_url_or_gcs_path",
        "image2_url_or_gcs_path",
        "label_same_person",
        "category",
        "notes",
        "expected_challenge_type",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  ✓ Manifest written to: {args.output}")
    print(f"    {total} pairs ({same_count} same, {diff_count} different)")


if __name__ == "__main__":
    main()
