#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  ARCFACE SCORCHED-EARTH MIGRATION                                ║
║  Cloud Run Job — Automated Vault Wipe & Re-Ingestion             ║
║                                                                  ║
║  Designed to run as a Cloud Run Job (NOT locally).               ║
║  Phase 1: TRUNCATE TABLE identity_profiles                       ║
║  Phase 2: Re-ingest target_profiles/ with 512-D ArcFace vectors ║
║                                                                  ║
║  Trigger from PowerShell:                                        ║
║    gcloud run jobs execute arcface-migration-job \               ║
║      --region=us-east4 --project=hoppwhistle                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import glob
import hashlib
from pathlib import Path
from datetime import datetime

import cv2

# ---------------------------------------------------------------------------
# DOCKER-AWARE IMPORTS
# ---------------------------------------------------------------------------
# In Docker: backend code is at /app/ (flat), scripts at /app/scripts/
# Locally:   backend code is at <root>/backend/, scripts at <root>/scripts/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Local development context (backend is a package)
    from backend.main import align_face_crop, extract_arcface_embedding, encrypt_embedding
    from backend.models import engine, SessionLocal, IdentityProfile, Base
except ImportError:
    # Docker context (files are flat in /app/)
    sys.path.insert(0, str(PROJECT_ROOT))
    from main import align_face_crop, extract_arcface_embedding, encrypt_embedding
    from models import engine, SessionLocal, IdentityProfile, Base


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
_backend_targets = os.path.join(PROJECT_ROOT, "backend", "target_profiles")
_flat_targets = os.path.join(PROJECT_ROOT, "target_profiles")
TARGET_DIR = _backend_targets if os.path.isdir(_backend_targets) else _flat_targets
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# TERMINAL STYLING
# ---------------------------------------------------------------------------
class _C:
    GOLD    = "\033[38;2;212;175;55m"
    GREEN   = "\033[38;2;0;255;128m"
    RED     = "\033[38;2;255;60;60m"
    YELLOW  = "\033[38;2;255;200;60m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"
    CYAN    = "\033[38;2;80;200;255m"


def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------------------
# PHASE 1: SCORCHED-EARTH TRUNCATION
# ---------------------------------------------------------------------------
def truncate_vault():
    print(f"""
{_C.RED}{'═' * 66}
  PHASE 1: SCORCHED-EARTH VAULT TRUNCATION
{'═' * 66}{_C.RESET}
""")
    session = SessionLocal()
    try:
        count_before = session.query(IdentityProfile).count()
        print(f"  {_C.DIM}[{_ts()}]{_C.RESET} Legacy rows found: {_C.YELLOW}{count_before}{_C.RESET}")

        session.execute(IdentityProfile.__table__.delete())
        session.commit()

        count_after = session.query(IdentityProfile).count()
        print(f"  {_C.DIM}[{_ts()}]{_C.RESET} Rows after truncation: {_C.GREEN}{count_after}{_C.RESET}")
        print(f"  {_C.GREEN}✓ LEGACY 1404-D VAULT WIPED{_C.RESET}")
    except Exception as e:
        session.rollback()
        print(f"  {_C.RED}✗ TRUNCATION FAILED: {e}{_C.RESET}")
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# PHASE 2: ARCFACE RE-INGESTION
# ---------------------------------------------------------------------------
def generate_user_id(filepath: str) -> str:
    stem = Path(filepath).stem.lower().replace(" ", "_")
    file_hash = hashlib.sha256(Path(filepath).name.encode()).hexdigest()[:8]
    return f"{stem}_{file_hash}"


def discover_images(target_dir: str) -> list[str]:
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(target_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(target_dir, f"*{ext.upper()}")))
    seen = set()
    unique = []
    for f in files:
        norm = os.path.normpath(f).lower()
        if norm not in seen:
            seen.add(norm)
            unique.append(f)
    return sorted(unique)


def ingest_vault():
    print(f"""
{_C.GOLD}{'═' * 66}
  PHASE 2: ARCFACE 512-D RE-INGESTION
{'═' * 66}{_C.RESET}
""")

    if not os.path.isdir(TARGET_DIR):
        print(f"  {_C.RED}✗ TARGET DIRECTORY NOT FOUND: {TARGET_DIR}{_C.RESET}")
        print(f"  {_C.DIM}  No images to ingest. Migration complete (empty vault).{_C.RESET}")
        return

    images = discover_images(TARGET_DIR)
    total = len(images)

    if total == 0:
        print(f"  {_C.YELLOW}⚠ NO IMAGES FOUND IN TARGET DIRECTORY{_C.RESET}")
        print(f"  {_C.DIM}  Path: {TARGET_DIR}{_C.RESET}")
        return

    print(f"  {_C.CYAN}TARGET DIR{_C.RESET}   {TARGET_DIR}")
    print(f"  {_C.CYAN}IMAGES FOUND{_C.RESET} {total}")
    print(f"  {_C.DIM}{'─' * 66}{_C.RESET}")

    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    count_ok = 0
    count_skip = 0
    count_dup = 0
    count_err = 0
    t_start = time.time()

    try:
        for idx, filepath in enumerate(images, start=1):
            filename = os.path.basename(filepath)
            prefix = f"  {_C.DIM}[{_ts()}]{_C.RESET} [{idx:>4}/{total}]"
            user_id = generate_user_id(filepath)

            try:
                # Check duplicates
                existing = session.query(IdentityProfile).filter_by(user_id=user_id).first()
                if existing:
                    count_dup += 1
                    print(f"{prefix} {_C.CYAN}◉ EXISTS  {_C.RESET}  {filename}")
                    continue

                # Read + align
                image = cv2.imread(filepath)
                if image is None:
                    count_err += 1
                    print(f"{prefix} {_C.RED}✗ ERROR   {_C.RESET}  {filename} — could not read")
                    continue

                aligned, landmarks = align_face_crop(image)
                if landmarks is None:
                    count_skip += 1
                    print(f"{prefix} {_C.YELLOW}⚠ NO FACE {_C.RESET}  {filename}")
                    continue

                # ArcFace 512-D embedding
                embedding = extract_arcface_embedding(aligned)
                encrypted_payload = encrypt_embedding(embedding)

                profile = IdentityProfile(
                    user_id=user_id,
                    encrypted_facial_embedding=encrypted_payload,
                )
                session.add(profile)
                session.commit()

                count_ok += 1
                print(f"{prefix} {_C.GREEN}✓ INGESTED{_C.RESET}  {filename}  →  {_C.DIM}{user_id}{_C.RESET}")

            except Exception as e:
                count_err += 1
                session.rollback()
                print(f"{prefix} {_C.RED}✗ FAILURE {_C.RESET}  {filename} — {str(e)[:60]}")

    finally:
        session.close()

    elapsed = time.time() - t_start
    print(f"""
{_C.GOLD}{'═' * 66}
  INGESTION COMPLETE
{'═' * 66}{_C.RESET}
  {_C.GREEN}✓ Ingested{_C.RESET}       {count_ok:>6}
  {_C.YELLOW}⚠ Skipped{_C.RESET}       {count_skip:>6}  (no face detected)
  {_C.CYAN}◉ Duplicates{_C.RESET}     {count_dup:>6}  (already in vault)
  {_C.RED}✗ Errors{_C.RESET}         {count_err:>6}
  {_C.DIM}─────────────────────{_C.RESET}
  {_C.BOLD}Total Processed{_C.RESET}  {total:>6}
  {_C.BOLD}Elapsed{_C.RESET}          {elapsed:>6.2f}s
  {_C.BOLD}Rate{_C.RESET}             {(count_ok / elapsed) if elapsed > 0 else 0:>6.1f} profiles/sec
{_C.GOLD}{'═' * 66}{_C.RESET}
""")


# ---------------------------------------------------------------------------
# MAIN — FULL MIGRATION PIPELINE
# ---------------------------------------------------------------------------
def main():
    print(f"""
{_C.GOLD}{'═' * 66}
  ███╗   ███╗██╗ ██████╗ ██████╗  █████╗ ████████╗███████╗
  ████╗ ████║██║██╔════╝ ██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
  ██╔████╔██║██║██║  ███╗██████╔╝███████║   ██║   █████╗
  ██║╚██╔╝██║██║██║   ██║██╔══██╗██╔══██║   ██║   ██╔══╝
  ██║ ╚═╝ ██║██║╚██████╔╝██║  ██║██║  ██║   ██║   ███████╗
  ╚═╝     ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
{'═' * 66}
  ARCFACE SCORCHED-EARTH MIGRATION
  Engine: DeepFace ArcFace 512-D  |  KMS Envelope Encryption
{'═' * 66}{_C.RESET}
""")

    t_total = time.time()

    # Phase 1: Wipe
    truncate_vault()

    # Phase 2: Re-ingest
    ingest_vault()

    elapsed_total = time.time() - t_total
    print(f"""
{_C.GOLD}{'═' * 66}
  MIGRATION COMPLETE
{'═' * 66}{_C.RESET}
  {_C.GREEN}✓{_C.RESET} Legacy 1404-D vault truncated
  {_C.GREEN}✓{_C.RESET} Fresh 512-D ArcFace embeddings ingested
  {_C.BOLD}Total Time{_C.RESET}  {elapsed_total:.2f}s
{_C.GOLD}{'═' * 66}{_C.RESET}
""")


if __name__ == "__main__":
    main()
