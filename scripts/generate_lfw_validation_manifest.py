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
  # Full LFW pairs, all matched/mismatched, with GCS verification:
  python scripts/generate_lfw_validation_manifest.py \
    --pairs-url https://vis-www.cs.umass.edu/lfw/pairs.txt \
    --gcs-prefix gs://hoppwhistle-facial-uploads/gallery/ \
    --output validation/validation_pairs_lfw_baseline.csv \
    --check-gcs-exists \
    --fail-on-missing

  # Dev-train split, limited to 50/50:
  python scripts/generate_lfw_validation_manifest.py \
    --pairs-url https://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt \
    --output validation/validation_pairs_lfw_baseline.csv \
    --max-matched 50 \
    --max-mismatched 50

  # Dry run (no CSV written):
  python scripts/generate_lfw_validation_manifest.py \
    --pairs-url https://vis-www.cs.umass.edu/lfw/pairs.txt \
    --dry-run
"""

import argparse
import csv
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Optional


# ── Well-known LFW pairs file URLs ──
LFW_FULL_PAIRS_URL = "https://vis-www.cs.umass.edu/lfw/pairs.txt"
LFW_DEV_TRAIN_PAIRS_URL = "https://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt"
LFW_DEV_TEST_PAIRS_URL = "https://vis-www.cs.umass.edu/lfw/pairsDevTest.txt"


# ── LFW Pairs File Format ──
# Line 1: <num_folds>\t<pairs_per_fold>   (e.g. "10\t300")
# Matched pairs:    Name  ImageNum1  ImageNum2
# Mismatched pairs: Name1 ImageNum1  Name2  ImageNum2
#
# Fields may be tab-delimited OR whitespace-delimited depending on source.
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

    Uses generic whitespace splitting to handle both tab-delimited and
    space-delimited variants of the pairs file.

    Returns:
        (matched_pairs, mismatched_pairs) where each entry is a dict with:
        - name1, num1, name2, num2, label
    """
    matched = []
    mismatched = []

    # Skip header line (e.g. "10\t300" or "10  300")
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
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
        parts = stripped.split()

        if len(parts) == 3:
            # Matched pair: Name Num1 Num2
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
            # Mismatched pair: Name1 Num1 Name2 Num2
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


def create_gcs_checker(gcs_prefix: str):
    """Create a reusable GCS existence checker.

    Returns a callable (gcs_uri) -> bool, or None if GCS is unavailable.
    Reuses a single storage client and bucket reference for efficiency.
    """
    try:
        from google.cloud import storage
    except ImportError:
        return None

    # Parse bucket from prefix: gs://bucket-name/path/ -> bucket-name
    if not gcs_prefix.startswith("gs://"):
        return None
    bucket_name = gcs_prefix[5:].split("/", 1)[0]

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print(f"  ERROR: Failed to create GCS client: {e}")
        return None

    def check_exists(gcs_uri: str) -> bool:
        if not gcs_uri.startswith("gs://"):
            return False
        blob_path = gcs_uri[5:].split("/", 1)
        if len(blob_path) != 2:
            return False
        try:
            blob = bucket.blob(blob_path[1])
            return blob.exists()
        except Exception as e:
            print(f"  WARNING: GCS check failed for {gcs_uri}: {e}")
            return False

    return check_exists


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
        "--max-matched", type=int, default=0,
        help="Maximum number of matched (same-person) pairs. 0 = unlimited."
    )
    parser.add_argument(
        "--max-mismatched", type=int, default=0,
        help="Maximum number of mismatched (different-person) pairs. 0 = unlimited."
    )
    parser.add_argument(
        "--check-gcs-exists", action="store_true",
        help="Verify each GCS blob exists before including the pair"
    )
    parser.add_argument(
        "--fail-on-missing", action="store_true",
        help="Exit non-zero if any selected GCS image is missing (requires --check-gcs-exists)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and validate the pairs file without writing CSV output"
    )

    args = parser.parse_args()

    # ── Validate flag combinations ──
    if args.fail_on_missing and not args.check_gcs_exists:
        print("  WARNING: --fail-on-missing has no effect without --check-gcs-exists. Enabling --check-gcs-exists.")
        args.check_gcs_exists = True

    # ── Ensure GCS prefix ends with /
    gcs_prefix = args.gcs_prefix
    if not gcs_prefix.endswith("/"):
        gcs_prefix += "/"

    # ── 1. Load pairs file ──
    pairs_source = args.pairs_file or args.pairs_url
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

    # ── 3. Apply limits (0 = unlimited) ──
    selected_matched = matched if args.max_matched <= 0 else matched[:args.max_matched]
    selected_mismatched = mismatched if args.max_mismatched <= 0 else mismatched[:args.max_mismatched]
    print(f"  Selected {len(selected_matched)} matched, {len(selected_mismatched)} mismatched.")

    # ── 4. Set up GCS checker if needed ──
    gcs_checker = None
    if args.check_gcs_exists:
        gcs_checker = create_gcs_checker(gcs_prefix)
        if gcs_checker is None:
            print("  ERROR: --check-gcs-exists is set but google-cloud-storage is not available or GCS client creation failed.")
            print("         Install google-cloud-storage or remove --check-gcs-exists.")
            sys.exit(1)
        print("  GCS existence checking enabled (single client reused).")

    # ── 5. Build manifest rows ──
    rows = []
    gcs_check_pass = 0
    gcs_check_fail = 0
    missing_pairs = []

    for i, pair in enumerate(selected_matched, 1):
        img1_name = lfw_image_name(pair["name1"], pair["num1"])
        img2_name = lfw_image_name(pair["name2"], pair["num2"])
        img1_uri = f"{gcs_prefix}{img1_name}"
        img2_uri = f"{gcs_prefix}{img2_name}"

        if gcs_checker:
            exists1 = gcs_checker(img1_uri)
            exists2 = gcs_checker(img2_uri)
            if not exists1 or not exists2:
                gcs_check_fail += 1
                missing = []
                if not exists1:
                    missing.append(img1_uri)
                if not exists2:
                    missing.append(img2_uri)
                missing_pairs.append({
                    "pair_id": f"lfw_matched_{i:04d}",
                    "type": "matched",
                    "missing_uris": missing,
                })
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

        if gcs_checker:
            exists1 = gcs_checker(img1_uri)
            exists2 = gcs_checker(img2_uri)
            if not exists1 or not exists2:
                gcs_check_fail += 1
                missing = []
                if not exists1:
                    missing.append(img1_uri)
                if not exists2:
                    missing.append(img2_uri)
                missing_pairs.append({
                    "pair_id": f"lfw_mismatched_{i:04d}",
                    "type": "mismatched",
                    "missing_uris": missing,
                })
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

    # ── 6. Summary ──
    total = len(rows)
    same_count = sum(1 for r in rows if r["label_same_person"] == "true")
    diff_count = sum(1 for r in rows if r["label_same_person"] == "false")

    print(f"\n{'='*60}")
    print(f"  MANIFEST GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Pairs source:          {pairs_source}")
    print(f"  GCS prefix:            {gcs_prefix}")
    print(f"  Selected matched:      {len(selected_matched)}")
    print(f"  Selected mismatched:   {len(selected_mismatched)}")
    print(f"  Selected total:        {len(selected_matched) + len(selected_mismatched)}")
    print(f"  Written total:         {total}")
    print(f"  Written same-person:   {same_count}")
    print(f"  Written diff-person:   {diff_count}")
    if args.check_gcs_exists:
        print(f"  GCS exists (pass):     {gcs_check_pass}")
        print(f"  GCS exists (fail):     {gcs_check_fail}")
        print(f"  Missing pairs:         {len(missing_pairs)}")
    print(f"  Categories present:    same_person_normal, different_person_random")
    print(f"  Categories missing:    same_person_age_gap, same_person_lighting_pose,")
    print(f"                         different_person_lookalike, mark_heavy_same_person,")
    print(f"                         mark_heavy_different_person, twins_or_high_similarity_imposters")
    print(f"{'='*60}")

    # ── 7. Build summary JSON ──
    summary = {
        "pairs_source": pairs_source,
        "gcs_prefix": gcs_prefix,
        "selected_matched": len(selected_matched),
        "selected_mismatched": len(selected_mismatched),
        "selected_total": len(selected_matched) + len(selected_mismatched),
        "written_total": total,
        "written_same_person": same_count,
        "written_different_person": diff_count,
        "check_gcs_exists": args.check_gcs_exists,
        "fail_on_missing": args.fail_on_missing,
        "gcs_check_pass": gcs_check_pass,
        "gcs_check_fail": gcs_check_fail,
        "missing_pairs": missing_pairs,
        "output_path": args.output,
        "dry_run": args.dry_run,
        "first_5_rows": rows[:5] if rows else [],
    }

    # Always write summary JSON next to output
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "lfw_manifest_generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")

    # ── 8. Check for zero rows ──
    if total == 0:
        print("  ERROR: Zero rows would be written. Cannot produce an empty manifest.")
        if missing_pairs:
            print(f"         {len(missing_pairs)} pairs were skipped due to missing GCS blobs.")
            print(f"         Check that mass_ingest.py has been run against the LFW dataset.")
        sys.exit(1)

    # ── 9. Check fail-on-missing ──
    if args.fail_on_missing and missing_pairs:
        print(f"\n  FAIL: {len(missing_pairs)} pairs have missing GCS images and --fail-on-missing is set.")
        print(f"        See {summary_path} for the full list of missing pairs.")
        # Still write the CSV with the valid rows so results are inspectable
        if not args.dry_run:
            _write_csv(args.output, rows)
            print(f"  Partial manifest written to: {args.output} ({total} valid rows)")
        sys.exit(1)

    # ── 10. Dry run or write ──
    if args.dry_run:
        print(f"\n  DRY RUN — No CSV written.")
        print(f"  Would write {total} rows to: {args.output}")
        return

    _write_csv(args.output, rows)
    print(f"\n  ✓ Manifest written to: {args.output}")
    print(f"    {total} pairs ({same_count} same, {diff_count} different)")


def _write_csv(output_path: str, rows: List[dict]):
    """Write manifest rows to CSV."""
    output_dir = os.path.dirname(output_path) or "."
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

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
