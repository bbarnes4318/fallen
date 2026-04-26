#!/usr/bin/env python3
"""
=============================================================================
  AURUMSHIELD — MULTI-THREADED MASS INGESTION ENGINE
  Phase 2: High-Throughput 1:N Vault Population (LFW Bootstrap)
=============================================================================
"""

import os
import sys
import time
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime
import concurrent.futures

import cv2
import numpy as np

# Path Bootstrap
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import paths differ: in the Docker container, backend code lives flat at /app/
# Locally from the project root, it's under backend/
try:
    from main import align_face_crop, extract_arcface_embedding, encrypt_embedding
    from models import engine, SessionLocal, IdentityProfile, Base
except ModuleNotFoundError:
    from backend.main import align_face_crop, extract_arcface_embedding, encrypt_embedding
    from backend.models import engine, SessionLocal, IdentityProfile, Base

# Config — Cloud Run can only write to /tmp (in-memory tmpfs)
if os.getenv("K_SERVICE") or os.getenv("CLOUD_RUN_JOB"):
    DATASETS_DIR = "/tmp/datasets"
else:
    DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

# Dataset is staged on our own GCS bucket — zero egress cost, always available
GCS_BUCKET = "hoppwhistle-facial-uploads"
GCS_BLOB_PATH = "datasets/lfw_color.zip"
LFW_DIR = os.path.join(DATASETS_DIR, "lfwcrop_color", "faces")
BATCH_SIZE = 100  # Number of profiles to commit to DB at once
MAX_WORKERS = 4   # Number of concurrent ArcFace/KMS threads

class _C:
    GOLD = "\033[38;2;212;175;55m"
    GREEN = "\033[38;2;0;255;128m"
    RED = "\033[38;2;255;60;60m"
    CYAN = "\033[38;2;80;200;255m"
    DIM = "\033[2m"
    RESET = "\033[0m"

def _ts(): return datetime.now().strftime("%H:%M:%S")

def download_and_extract_lfw():
    """Downloads LFW color dataset from our GCS bucket and extracts it."""
    from google.cloud import storage as gcs_storage
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

    archive_path = os.path.join(DATASETS_DIR, "lfw_color.zip")

    if not os.path.exists(LFW_DIR):
        if not os.path.exists(archive_path):
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
        # Free tmpfs RAM — on Cloud Run /tmp is memory-backed
        os.remove(archive_path)
        print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.GREEN}EXTRACTION COMPLETE. Archive purged from tmpfs.{_C.RESET}")
    else:
        print(f"  {_C.DIM}[{_ts()}]{_C.RESET} {_C.CYAN}LFW DATASET FOUND. SKIPPING DOWNLOAD.{_C.RESET}")

def generate_user_id(filepath: str) -> str:
    stem = Path(filepath).stem.lower().replace(" ", "_")
    file_hash = hashlib.sha256(Path(filepath).name.encode()).hexdigest()[:8]
    return f"{stem}_{file_hash}"

def process_single_image(filepath: str):
    """Worker function: Runs the heavy compute off the main thread."""
    try:
        user_id = generate_user_id(filepath)
        
        # Extract name from filename (e.g., George_W_Bush_0001.ppm -> George W Bush)
        stem = Path(filepath).stem  # George_W_Bush_0001
        person_name = "_".join(stem.split("_")[:-1]).replace("_", " ")

        image = cv2.imread(filepath)
        if image is None: return {"status": "error", "file": filepath, "msg": "Read failed"}

        aligned, landmarks = align_face_crop(image)
        if landmarks is None: return {"status": "skip", "file": filepath, "msg": "No face detected"}

        # Heavy Compute
        embedding = extract_arcface_embedding(aligned)
        
        # Network I/O (KMS Encryption)
        encrypted_payload = encrypt_embedding(embedding)

        return {
            "status": "ok",
            "file": filepath,
            "user_id": user_id,
            "person_name": person_name,
            "payload": encrypted_payload
        }
    except Exception as e:
        return {"status": "error", "file": filepath, "msg": str(e)}

def main():
    print(f"\n{_C.GOLD}══════════════════════════════════════════════════════════════════{_C.RESET}")
    print(f"  AURUMSHIELD MASS INGESTION ENGINE")
    print(f"  Target: Labeled Faces in the Wild (13,233 Profiles)")
    print(f"{_C.GOLD}══════════════════════════════════════════════════════════════════{_C.RESET}\n")

    # 1. Get Data
    download_and_extract_lfw()
    
    image_paths = []
    for root, _, files in os.walk(LFW_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".ppm", ".png")):
                image_paths.append(os.path.join(root, file))

    total_images = len(image_paths)
    print(f"  {_C.DIM}[{_ts()}]{_C.RESET} Discovered {total_images} target images.")
    
    # 2. Init DB
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    
    # Pre-cache existing user_ids to avoid DB roundtrips during checks
    print(f"  {_C.DIM}[{_ts()}]{_C.RESET} Caching existing vault indexes...")
    existing_ids = {row[0] for row in session.query(IdentityProfile.user_id).all()}

    pending_inserts = []
    stats = {"ok": 0, "skip": 0, "dup": 0, "error": 0}
    t_start = time.time()

    print(f"\n{_C.GOLD}▶ INITIATING MULTI-THREADED EXTRACTION PIPELINE...{_C.RESET}\n")

    try:
        # Use ThreadPoolExecutor to saturate CPU and parallelize KMS network requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            
            # Submit only images that aren't already in the vault
            future_to_path = {}
            for path in image_paths:
                uid = generate_user_id(path)
                if uid in existing_ids:
                    stats["dup"] += 1
                else:
                    future_to_path[executor.submit(process_single_image, path)] = path

            # Process as they complete
            processed_count = 0
            total_futures = len(future_to_path)

            for future in concurrent.futures.as_completed(future_to_path):
                processed_count += 1
                result = future.result()
                
                if result["status"] == "ok":
                    stats["ok"] += 1
                    
                    profile = IdentityProfile(
                        user_id=result["user_id"],
                        person_name=result["person_name"],
                        source="LFW Academic Dataset",
                        encrypted_facial_embedding=result["payload"],
                    )
                    pending_inserts.append(profile)

                    # Batch Commit to PostgreSQL
                    if len(pending_inserts) >= BATCH_SIZE:
                        session.bulk_save_objects(pending_inserts)
                        session.commit()
                        pending_inserts.clear()
                        print(f"  {_C.GREEN}✓ COMMITTED BATCH{_C.RESET} | {stats['ok'] + stats['dup']}/{total_images} processed")

                elif result["status"] == "skip":
                    stats["skip"] += 1
                else:
                    stats["error"] += 1
                    
                # Terminal Progress updates
                if processed_count % 50 == 0:
                    sys.stdout.write(f"\r  {_C.CYAN}Engine Status:{_C.RESET} {processed_count}/{total_futures} Extracted | Vaulted: {stats['ok']} | Errors: {stats['error']}")
                    sys.stdout.flush()

        # Commit final remaining batch
        if pending_inserts:
            session.bulk_save_objects(pending_inserts)
            session.commit()
            print(f"\n  {_C.GREEN}✓ COMMITTED FINAL BATCH{_C.RESET}")

    finally:
        session.close()

    elapsed = time.time() - t_start
    print(f"\n{_C.GOLD}══════════════════════════════════════════════════════════════════{_C.RESET}")
    print(f"  INGESTION COMPLETE")
    print(f"  {_C.GREEN}✓ Vaulted:{_C.RESET}     {stats['ok']}")
    print(f"  {_C.CYAN}◉ Duplicates:{_C.RESET}  {stats['dup']}")
    print(f"  {_C.DIM}⚠ Skipped:{_C.RESET}     {stats['skip']} (No face detected)")
    print(f"  {_C.RED}✗ Errors:{_C.RESET}      {stats['error']}")
    print(f"  Time:          {elapsed:.1f}s")
    print(f"{_C.GOLD}══════════════════════════════════════════════════════════════════{_C.RESET}\n")

if __name__ == "__main__":
    main()