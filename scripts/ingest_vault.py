#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  BIOMETRIC VAULT — MASS INGESTION ENGINE                        ║
║  Phase 2: 1:N Database Population                               ║
║                                                                  ║
║  Reads raw target profile images, extracts 1404-D geometric     ║
║  embeddings via MediaPipe, encrypts via KMS envelope encryption, ║
║  and commits to PostgreSQL (pgvector-ready Cloud SQL).           ║
║                                                                  ║
║  Usage (from project root):                                      ║
║    python -m scripts.ingest_vault                                ║
║                                                                  ║
║  Ensure Cloud SQL Auth Proxy is running for local execution.     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import uuid
import glob
import hashlib
from pathlib import Path
from datetime import datetime

import cv2

# ---------------------------------------------------------------------------
# PATH BOOTSTRAP — ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import align_face_crop, extract_landmark_embedding, encrypt_embedding
from backend.models import engine, SessionLocal, IdentityProfile, Base

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TARGET_DIR = os.path.join(PROJECT_ROOT, "target_profiles")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# TERMINAL STYLING
# ---------------------------------------------------------------------------
class _C:
    """ANSI escape codes for institutional terminal output."""
    GOLD    = "\033[38;2;212;175;55m"
    GREEN   = "\033[38;2;0;255;128m"
    RED     = "\033[38;2;255;60;60m"
    YELLOW  = "\033[38;2;255;200;60m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"
    CYAN    = "\033[38;2;80;200;255m"


def _banner():
    print(f"""
{_C.GOLD}{'═' * 66}
  ██████╗ ██╗ ██████╗ ███╗   ███╗███████╗████████╗██████╗ ██╗ ██████╗
  ██╔══██╗██║██╔═══██╗████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝
  ██████╔╝██║██║   ██║██╔████╔██║█████╗     ██║   ██████╔╝██║██║     
  ██╔══██╗██║██║   ██║██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║██║     
  ██████╔╝██║╚██████╔╝██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║╚██████╗
  ╚═════╝ ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝
{'═' * 66}
  VAULT INGESTION ENGINE  ·  Phase 2: 1:N Population
{'═' * 66}{_C.RESET}
""")


def _divider():
    print(f"{_C.DIM}{'─' * 66}{_C.RESET}")


def _timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------------------
# USER ID GENERATION
# ---------------------------------------------------------------------------
def generate_user_id(filepath: str) -> str:
    """
    Generates a deterministic user_id from the filename.
    Uses a SHA-256 truncated hash prefixed with the sanitized stem
    to remain human-readable yet collision-resistant.

    Example: 'john_doe_a1b2c3d4'
    """
    stem = Path(filepath).stem.lower().replace(" ", "_")
    # Deterministic hash from full absolute path
    file_hash = hashlib.sha256(Path(filepath).name.encode()).hexdigest()[:8]
    return f"{stem}_{file_hash}"


# ---------------------------------------------------------------------------
# CORE INGESTION
# ---------------------------------------------------------------------------
def discover_images(target_dir: str) -> list[str]:
    """Discovers all supported image files in the target directory."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(target_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(target_dir, f"*{ext.upper()}")))
    # Deduplicate (case-insensitive OSes may double-count)
    seen = set()
    unique = []
    for f in files:
        norm = os.path.normpath(f).lower()
        if norm not in seen:
            seen.add(norm)
            unique.append(f)
    return sorted(unique)


def ingest_single(filepath: str, session) -> str:
    """
    Processes a single image through the full biometric pipeline.

    Returns:
        'ok'      — successfully ingested
        'skip'    — no face detected
        'dup'     — duplicate user_id (already in vault)
        'error'   — unexpected failure
    """
    user_id = generate_user_id(filepath)

    # Check for duplicates
    existing = session.query(IdentityProfile).filter_by(user_id=user_id).first()
    if existing:
        return "dup"

    # 1. Read image
    image = cv2.imread(filepath)
    if image is None:
        return "error"

    # 2. Align & crop — returns (aligned_crop, landmarks)
    aligned, landmarks = align_face_crop(image)
    if landmarks is None:
        return "skip"

    # 3. Extract 1404-D geometric embedding
    embedding = extract_landmark_embedding(landmarks)

    # 4. KMS envelope encryption
    encrypted_payload = encrypt_embedding(embedding)

    # 5. Commit to vault
    profile = IdentityProfile(
        user_id=user_id,
        encrypted_facial_embedding=encrypted_payload,
    )
    session.add(profile)
    session.commit()

    return "ok"


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    _banner()

    # Validate target directory
    if not os.path.isdir(TARGET_DIR):
        print(f"{_C.RED}  ✗ TARGET DIRECTORY NOT FOUND{_C.RESET}")
        print(f"    Expected: {TARGET_DIR}")
        print(f"    Create the directory and populate it with target profile images.")
        sys.exit(1)

    # Discover images
    images = discover_images(TARGET_DIR)
    total = len(images)

    if total == 0:
        print(f"{_C.YELLOW}  ⚠ NO IMAGES FOUND IN TARGET DIRECTORY{_C.RESET}")
        print(f"    Path: {TARGET_DIR}")
        print(f"    Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    print(f"  {_C.CYAN}TARGET DIR{_C.RESET}   {TARGET_DIR}")
    print(f"  {_C.CYAN}IMAGES FOUND{_C.RESET} {total}")
    _divider()

    # Ensure DB schema exists
    print(f"  {_C.DIM}[{_timestamp()}]{_C.RESET} Initializing database schema...", end="")
    Base.metadata.create_all(bind=engine)
    print(f" {_C.GREEN}OK{_C.RESET}")
    _divider()

    # Ingestion loop
    session = SessionLocal()
    count_ok = 0
    count_skip = 0
    count_dup = 0
    count_err = 0
    t_start = time.time()

    try:
        for idx, filepath in enumerate(images, start=1):
            filename = os.path.basename(filepath)
            prefix = f"  {_C.DIM}[{_timestamp()}]{_C.RESET} [{idx:>4}/{total}]"

            try:
                result = ingest_single(filepath, session)

                if result == "ok":
                    count_ok += 1
                    uid = generate_user_id(filepath)
                    print(f"{prefix} {_C.GREEN}✓ INGESTED{_C.RESET}  {filename}  →  {_C.DIM}{uid}{_C.RESET}")

                elif result == "skip":
                    count_skip += 1
                    print(f"{prefix} {_C.YELLOW}⚠ NO FACE {_C.RESET}  {filename}  —  skipped")

                elif result == "dup":
                    count_dup += 1
                    print(f"{prefix} {_C.CYAN}◉ EXISTS  {_C.RESET}  {filename}  —  already in vault")

                elif result == "error":
                    count_err += 1
                    print(f"{prefix} {_C.RED}✗ ERROR   {_C.RESET}  {filename}  —  could not read image")

            except Exception as e:
                count_err += 1
                session.rollback()
                print(f"{prefix} {_C.RED}✗ FAILURE {_C.RESET}  {filename}  —  {str(e)[:60]}")

    finally:
        session.close()

    # Final report
    elapsed = time.time() - t_start
    _divider()
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


if __name__ == "__main__":
    main()
