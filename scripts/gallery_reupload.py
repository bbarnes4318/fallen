#!/usr/bin/env python3
"""
=============================================================================
  Fallen — GALLERY THUMBNAIL RE-UPLOAD JOB
  Backfills face thumbnails for all LFW profiles missing thumbnail_url
=============================================================================

This job:
  1. Downloads the LFW dataset from GCS (same as mass_ingest.py)
  2. For each image, generates the same user_id hash as mass_ingest
  3. Checks if that profile exists with NULL thumbnail_url
  4. Aligns the face crop using MediaPipe (lightweight — no ArcFace/KMS)
  5. Uploads aligned JPEG to GCS gallery/
  6. Updates thumbnail_url in PostgreSQL

Does NOT re-run embedding extraction or KMS encryption.
Estimated runtime: ~15 minutes for 9,400 images.
"""

import os
import sys
import time
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# Path Bootstrap
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(str(PROJECT_ROOT), "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

# Config
if os.getenv("K_SERVICE") or os.getenv("CLOUD_RUN_JOB"):
    DATASETS_DIR = "/tmp/datasets"
else:
    DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

GCS_BUCKET = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
GCS_BLOB_PATH = "datasets/lfw_color.zip"
LFW_DIR = os.path.join(DATASETS_DIR, "lfwcrop_color", "faces")
BATCH_SIZE = 200


class _C:
    GOLD = "\033[38;2;212;175;55m"
    GREEN = "\033[38;2;0;255;128m"
    RED = "\033[38;2;255;60;60m"
    CYAN = "\033[38;2;80;200;255m"
    DIM = "\033[2m"
    RESET = "\033[0m"

def _ts(): return datetime.now().strftime("%H:%M:%S")


def download_and_extract_lfw():
    """Download LFW color dataset from our GCS bucket."""
    from google.cloud import storage as gcs_storage

    os.makedirs(DATASETS_DIR, exist_ok=True)
    archive_path = os.path.join(DATASETS_DIR, "lfw_color.zip")

    if not os.path.isdir(LFW_DIR):
        print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.GOLD}DOWNLOADING LFW DATASET FROM GCS...{_C.RESET}")
        client = gcs_storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_BLOB_PATH)
        blob.download_to_filename(archive_path)
        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.GREEN}DOWNLOAD COMPLETE ({size_mb:.0f} MB).{_C.RESET}")

        print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.GOLD}EXTRACTING ARCHIVE...{_C.RESET}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=DATASETS_DIR)
        os.remove(archive_path)
        print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.GREEN}EXTRACTION COMPLETE.{_C.RESET}")
    else:
        print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.CYAN}LFW DATASET FOUND. SKIPPING DOWNLOAD.{_C.RESET}")


def generate_user_id(filepath: str) -> str:
    """Same hash function as mass_ingest.py — must match exactly."""
    stem = Path(filepath).stem.lower().replace(" ", "_")
    file_hash = hashlib.sha256(Path(filepath).name.encode()).hexdigest()[:8]
    return f"{stem}_{file_hash}"


def upload_face_to_gcs(aligned_img, filename: str) -> str:
    """Uploads an aligned face crop to GCS and returns the gs:// URI."""
    from google.cloud import storage as gcs_storage
    _, buf = cv2.imencode('.jpg', aligned_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    blob_path = f"gallery/{filename}.jpg"
    client = gcs_storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(buf.tobytes(), content_type="image/jpeg")
    return f"gs://{GCS_BUCKET}/{blob_path}"


def main():
    print(f"\n{_C.GOLD}══════════════════════════════════════════════════════════════════{_C.RESET}")
    print(f"  Fallen GALLERY THUMBNAIL RE-UPLOAD ENGINE")
    print(f"  Backfilling face images for LFW profiles")
    print(f"{_C.GOLD}══════════════════════════════════════════════════════════════════{_C.RESET}\n")

    # 1. Download LFW dataset (lightweight — no TF/ArcFace needed yet)
    download_and_extract_lfw()

    # 2. Load backend alignment function (MediaPipe only — skip ArcFace)
    print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.GOLD}LOADING FACE ALIGNMENT MODULE...{_C.RESET}")
    from main import align_face_crop
    from models import SessionLocal, IdentityProfile
    print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.GREEN}ALIGNMENT MODULE LOADED.{_C.RESET}")

    # 3. Discover LFW images
    image_paths = []
    for root, _, files in os.walk(LFW_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".ppm", ".png")):
                image_paths.append(os.path.join(root, file))

    total_images = len(image_paths)
    print(f"  {_C.DIM}[{_ts()}]{_C.RESET} Discovered {total_images} LFW images.")

    # 4. Build set of user_ids that need thumbnail backfill
    session = SessionLocal()
    null_thumb_ids = {
        row[0] for row in
        session.query(IdentityProfile.user_id).filter(
            IdentityProfile.thumbnail_url.is_(None)
        ).all()
    }
    print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {len(null_thumb_ids)} profiles need thumbnail backfill.")

    if not null_thumb_ids:
        print(f"  {_C.GREEN}All profiles already have thumbnails. Nothing to do.{_C.RESET}")
        session.close()
        return

    # 5. Process images: align → upload → batch update DB
    stats = {"uploaded": 0, "skipped": 0, "no_face": 0, "error": 0}
    pending_updates = []  # list of (user_id, gs_uri)
    t_start = time.time()

    print(f"\n{_C.GOLD}▶ STARTING GALLERY UPLOAD PIPELINE...{_C.RESET}\n")

    for i, path in enumerate(image_paths):
        uid = generate_user_id(path)

        # Skip if this profile already has a thumbnail
        if uid not in null_thumb_ids:
            stats["skipped"] += 1
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  {_C.CYAN}Progress:{_C.RESET} {i+1}/{total_images} | Uploaded: {stats['uploaded']} | {rate:.1f} img/s")
            continue

        try:
            image = cv2.imread(path)
            if image is None:
                stats["error"] += 1
                continue

            # Align face crop (MediaPipe — lightweight)
            aligned, landmarks = align_face_crop(image)
            if landmarks is None:
                stats["no_face"] += 1
                continue

            # Upload to GCS
            stem = Path(path).stem  # e.g., George_W_Bush_0001
            gs_uri = upload_face_to_gcs(aligned, stem)

            pending_updates.append((uid, gs_uri))
            stats["uploaded"] += 1

            # Batch commit to DB
            if len(pending_updates) >= BATCH_SIZE:
                for update_uid, update_uri in pending_updates:
                    session.query(IdentityProfile).filter(
                        IdentityProfile.user_id == update_uid
                    ).update({"thumbnail_url": update_uri})
                session.commit()
                pending_updates.clear()
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  {_C.GREEN}✓ BATCH COMMITTED{_C.RESET} | {i+1}/{total_images} | Uploaded: {stats['uploaded']} | {rate:.1f} img/s")

        except Exception as e:
            stats["error"] += 1
            if stats["error"] <= 5:
                print(f"  {_C.RED}ERROR:{_C.RESET} {uid}: {e}")

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  {_C.CYAN}Progress:{_C.RESET} {i+1}/{total_images} | Uploaded: {stats['uploaded']} | Skipped: {stats['skipped']} | {rate:.1f} img/s")

    # Commit final batch
    if pending_updates:
        for update_uid, update_uri in pending_updates:
            session.query(IdentityProfile).filter(
                IdentityProfile.user_id == update_uid
            ).update({"thumbnail_url": update_uri})
        session.commit()
        print(f"  {_C.GREEN}✓ FINAL BATCH COMMITTED{_C.RESET}")

    session.close()

    elapsed = time.time() - t_start
    print(f"\n{_C.GOLD}══════════════════════════════════════════════════════════════════{_C.RESET}")
    print(f"  GALLERY RE-UPLOAD COMPLETE")
    print(f"  {_C.GREEN}✓ Uploaded:{_C.RESET}    {stats['uploaded']}")
    print(f"  {_C.CYAN}◉ Skipped:{_C.RESET}     {stats['skipped']} (already had thumbnail)")
    print(f"  {_C.DIM}⚠ No Face:{_C.RESET}     {stats['no_face']}")
    print(f"  {_C.RED}✗ Errors:{_C.RESET}      {stats['error']}")
    print(f"  Time:          {elapsed:.1f}s")
    print(f"{_C.GOLD}══════════════════════════════════════════════════════════════════{_C.RESET}\n")


if __name__ == "__main__":
    main()
