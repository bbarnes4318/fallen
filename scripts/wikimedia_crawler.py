"""
===============================================================
  WIKIMEDIA AUTONOMOUS BIOMETRIC CRAWLER
  Sovereign Identity Acquisition Daemon
===============================================================
  Queries Wikidata SPARQL for high-profile entities, pulls images
  via Wikimedia Commons API, runs facial extraction + KMS encryption,
  and vaults encrypted vectors with full legal attribution metadata.

  Designed to run as a Cloud Run Job using the backend Docker image.
===============================================================
"""

import sys
import os
import time
import re
import urllib.parse
import requests
import cv2
import numpy as np
from datetime import datetime

# -- Path setup: /app in Docker (backend code), or local repo --
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# In Docker: WORKDIR=/app contains backend code directly
# Locally: backend/ is a subdirectory of the project root
sys.path.insert(0, "/app")                             # Docker container
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))  # Local dev
sys.path.insert(0, PROJECT_ROOT)

from models import SessionLocal, IdentityProfile, init_db
from main import align_face_crop, extract_landmark_embedding, encrypt_embedding, apply_clahe

# -- Constants --
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
COMMONS_API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SovereignBiometricCrawler/1.0 (https://hoppwhistle.com; mailto:admin@hoppwhistle.com)"
MIN_FACE_SIZE = 60
REQUEST_DELAY = 1.0

# -- Terminal formatting --
GOLD = "\033[38;2;212;175;55m"
RED = "\033[91m"
GREEN = "\033[92m"
DIM = "\033[90m"
RESET = "\033[0m"
BOLD = "\033[1m"


def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix_map = {
        "INFO": f"{DIM}[{timestamp}]{RESET} {GOLD}>{RESET}",
        "OK": f"{DIM}[{timestamp}]{RESET} {GREEN}+{RESET}",
        "SKIP": f"{DIM}[{timestamp}]{RESET} {DIM}-{RESET}",
        "ERR": f"{DIM}[{timestamp}]{RESET} {RED}x{RESET}",
        "HEAD": f"{DIM}[{timestamp}]{RESET} {GOLD}{BOLD}#{RESET}",
    }
    prefix = prefix_map.get(level, f"[{timestamp}]")
    print(f"{prefix} {msg}")


# -------------------------------------------------------------
# WIKIDATA SPARQL FETCHER
# -------------------------------------------------------------

SPARQL_QUERY = """
SELECT ?person ?personLabel ?occupationLabel ?image WHERE {
  ?person wdt:P31 wd:Q5;
          wdt:P18 ?image;
          wdt:P106 ?occupation.
  VALUES ?occupation { wd:Q33999 wd:Q177220 wd:Q639669 wd:Q2066131 wd:Q3665646 wd:Q937857 }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} LIMIT 1000
"""


def fetch_wikidata_targets() -> list[dict]:
    """
    Queries Wikidata SPARQL endpoint for high-profile entities
    (actors, singers, musicians, athletes) with P18 image property.
    """
    log("Querying Wikidata SPARQL for high-profile entities...", "HEAD")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/sparql-results+json",
    }
    params = {"query": SPARQL_QUERY, "format": "json"}

    for attempt in range(3):
        try:
            resp = requests.get(WIKIDATA_SPARQL_ENDPOINT, params=params, headers=headers, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < 2:
                wait = (attempt + 1) * 5
                log(f"SPARQL connection failed, retrying in {wait}s... ({e.__class__.__name__})", "ERR")
                time.sleep(wait)
            else:
                raise

    results = data.get("results", {}).get("bindings", [])
    log(f"SPARQL returned {len(results)} raw results", "INFO")

    targets = []
    seen_ids = set()

    for row in results:
        person_uri = row.get("person", {}).get("value", "")
        wikidata_id = person_uri.split("/")[-1] if person_uri else None
        if not wikidata_id or wikidata_id in seen_ids:
            continue
        seen_ids.add(wikidata_id)

        person_name = row.get("personLabel", {}).get("value", "Unknown")
        occupation = row.get("occupationLabel", {}).get("value", "Unknown")
        image_url = row.get("image", {}).get("value", "")

        image_filename = urllib.parse.unquote(image_url.split("/")[-1]) if image_url else None
        if not image_filename:
            continue

        targets.append({
            "wikidata_id": wikidata_id,
            "person_name": person_name,
            "occupation": occupation,
            "image_filename": image_filename,
        })

    log(f"Extracted {len(targets)} unique targets", "OK")
    return targets


# -------------------------------------------------------------
# WIKIMEDIA COMMONS API INTEGRATION
# -------------------------------------------------------------

def fetch_image_metadata(filename: str) -> dict | None:
    """
    Queries the Wikimedia Commons API for full image metadata
    including URL, creator, license, and attribution requirements.
    Downloads the image bytes into memory.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": "800",
        "titles": f"File:{filename}",
    }
    headers = {"User-Agent": USER_AGENT}

    resp = requests.get(COMMONS_API_ENDPOINT, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    if "imageinfo" not in page:
        return None

    info = page["imageinfo"][0]
    image_url = info.get("url", "")
    thumb_api_url = info.get("thumburl", "")
    meta = info.get("extmetadata", {})

    def extract_meta(key: str) -> str:
        return meta.get(key, {}).get("value", "") if meta else ""

    file_page_url = info.get("descriptionurl", "")
    creator = extract_meta("Artist")
    creator = re.sub(r"<[^>]+>", "", creator).strip() if creator else ""
    license_short = extract_meta("LicenseShortName")
    license_url_val = extract_meta("LicenseUrl")
    credit = extract_meta("Credit")
    credit = re.sub(r"<[^>]+>", "", credit).strip() if credit else ""
    attribution_required = extract_meta("AttributionRequired").lower() in ("true", "yes", "1")
    source = extract_meta("ImageDescription")
    source = re.sub(r"<[^>]+>", "", source).strip() if source else ""

    # Use thumbnail URL from API (already resized, avoids CDN 403s)
    download_url = thumb_api_url or image_url
    if not download_url:
        return None

    dl_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "image/jpeg,image/png,image/webp,image/*,*/*;q=0.8",
    }
    img_resp = requests.get(download_url, headers=dl_headers, timeout=60)
    img_resp.raise_for_status()
    image_bytes = img_resp.content

    if len(image_bytes) < 1000:
        return None

    thumb_url = thumb_api_url or image_url

    return {
        "image_url": image_url,
        "thumbnail_url": thumb_url,
        "file_page_url": file_page_url,
        "creator": creator[:500] if creator else None,
        "license_short_name": license_short[:255] if license_short else None,
        "license_url": license_url_val or None,
        "credit": credit[:500] if credit else None,
        "attribution_required": attribution_required,
        "source": source[:500] if source else None,
        "image_bytes": image_bytes,
    }


# -------------------------------------------------------------
# BIOMETRIC PIPELINE & VAULT INSERTION
# -------------------------------------------------------------

def process_and_vault_target(target: dict, metadata: dict, session) -> bool:
    """
    Runs the biometric pipeline on downloaded image bytes:
    1. Decode -> CLAHE -> Align -> Extract -> Encrypt
    2. Insert IdentityProfile with attribution metadata
    """
    img_array = np.frombuffer(metadata["image_bytes"], dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        log(f"  Failed to decode image for {target['person_name']}", "ERR")
        return False

    clahe_img = apply_clahe(image)
    aligned, landmarks = align_face_crop(clahe_img)

    if landmarks is None:
        log(f"  No face detected: {target['person_name']}", "SKIP")
        return False

    h, w = aligned.shape[:2]
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        log(f"  Face too small ({w}x{h}): {target['person_name']}", "SKIP")
        return False

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 50.0:
        log(f"  Image too blurry (score={blur_score:.1f}): {target['person_name']}", "SKIP")
        return False

    embedding = extract_landmark_embedding(landmarks)
    encrypted_blob = encrypt_embedding(embedding)

    user_id = f"wiki_{target['wikidata_id']}"

    existing = session.query(IdentityProfile).filter_by(user_id=user_id).first()
    if existing:
        log(f"  Already in vault: {target['person_name']} ({user_id})", "SKIP")
        return False

    profile = IdentityProfile(
        user_id=user_id,
        encrypted_facial_embedding=encrypted_blob,
        image_url=metadata["image_url"],
        thumbnail_url=metadata["thumbnail_url"],
        file_page_url=metadata["file_page_url"],
        creator=metadata["creator"],
        license_short_name=metadata["license_short_name"],
        license_url=metadata["license_url"],
        credit=metadata["credit"],
        attribution_required=metadata["attribution_required"],
        source=metadata["source"],
        person_name=target["person_name"],
        wikidata_id=target["wikidata_id"],
    )
    session.add(profile)
    session.commit()

    log(f"  {GREEN}VAULTED{RESET}: {target['person_name']} -> {user_id} ({metadata['license_short_name'] or 'N/A'})", "OK")
    return True


# -------------------------------------------------------------
# MAIN EXECUTION LOOP
# -------------------------------------------------------------

def run_crawler():
    print(f"\n{'=' * 60}")
    print(f"  SOVEREIGN IDENTITY ACQUISITION DAEMON")
    print(f"  Wikimedia Commons -> Biometric Vault Pipeline")
    print(f"{'=' * 60}\n")

    log("Initializing database schema...", "HEAD")
    init_db()
    log("Schema ready", "OK")

    targets = fetch_wikidata_targets()
    if not targets:
        log("No targets returned from Wikidata. Exiting.", "ERR")
        return

    total = len(targets)
    vaulted = 0
    skipped = 0
    errors = 0

    log(f"Beginning ingestion of {total} targets...\n", "HEAD")

    session = SessionLocal()
    try:
        for i, target in enumerate(targets):
            progress = f"[{i + 1}/{total}]"
            log(f"{progress} Processing: {BOLD}{target['person_name']}{RESET} ({target['wikidata_id']}) - {target['occupation']}", "INFO")

            try:
                user_id = f"wiki_{target['wikidata_id']}"
                existing = session.query(IdentityProfile).filter_by(user_id=user_id).first()
                if existing:
                    log(f"  Already in vault, skipping", "SKIP")
                    skipped += 1
                    continue

                metadata = fetch_image_metadata(target["image_filename"])
                if not metadata:
                    log(f"  No image metadata available", "SKIP")
                    skipped += 1
                    time.sleep(REQUEST_DELAY)
                    continue

                success = process_and_vault_target(target, metadata, session)
                if success:
                    vaulted += 1
                else:
                    skipped += 1

            except requests.exceptions.RequestException as e:
                log(f"  Network error: {str(e)[:100]}", "ERR")
                errors += 1
            except Exception as e:
                log(f"  Pipeline error: {str(e)[:100]}", "ERR")
                errors += 1
                session.rollback()

            time.sleep(REQUEST_DELAY)

            if (i + 1) % 50 == 0:
                print(f"\n{DIM}-- Progress: {i + 1}/{total} | Vaulted: {vaulted} | Skipped: {skipped} | Errors: {errors} --{RESET}\n")

    finally:
        session.close()

    print(f"\n{'=' * 60}")
    print(f"  CRAWL COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total Targets:  {total}")
    print(f"  {GREEN}Vaulted:        {vaulted}{RESET}")
    print(f"  {DIM}Skipped:        {skipped}{RESET}")
    print(f"  {RED}Errors:         {errors}{RESET}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run_crawler()
