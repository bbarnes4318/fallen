"""
═══════════════════════════════════════════════════════════════════
  WIKIMEDIA AUTONOMOUS BIOMETRIC CRAWLER
  Sovereign Identity Acquisition Daemon
═══════════════════════════════════════════════════════════════════
  Queries Wikidata SPARQL for high-profile entities, pulls images
  via Wikimedia Commons API, runs facial extraction + KMS encryption,
  and vaults encrypted vectors with full legal attribution metadata.
═══════════════════════════════════════════════════════════════════
"""

import sys
import os
import time
import re
import struct
import urllib.parse
import requests
import cv2
import numpy as np
from datetime import datetime

# ── Path setup: allow imports from backend/ ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from backend.models import SessionLocal, IdentityProfile, init_db

# ── Conditional imports for encryption ──
try:
    from google.cloud import kms
    from cryptography.fernet import Fernet
    HAS_KMS = True
except ImportError:
    HAS_KMS = False

# ── MediaPipe Face Mesh (Tasks API for 0.10.30+) ──
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        # Legacy API (< 0.10.30)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        USE_LEGACY_MP = True
    else:
        # New Tasks API (0.10.30+)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        
        # Download face landmarker model if not present
        MODEL_PATH = os.path.join(PROJECT_ROOT, "face_landmarker.task")
        if not os.path.exists(MODEL_PATH):
            print("[SETUP] Downloading MediaPipe Face Landmarker model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            resp = requests.get(model_url, timeout=60)
            with open(MODEL_PATH, "wb") as f:
                f.write(resp.content)
            print("[SETUP] Model downloaded successfully.")
        
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(options)
        USE_LEGACY_MP = False
        face_mesh = None
except Exception as e:
    print(f"[WARN] MediaPipe init failed: {e}")
    face_mesh = None
    face_landmarker = None
    USE_LEGACY_MP = False

# ── Constants ──
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
COMMONS_API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SovereignBiometricCrawler/1.0 (https://hoppwhistle.com; mailto:admin@hoppwhistle.com)"
KMS_KEY_NAME = os.getenv("KMS_KEY_NAME") or "projects/hoppwhistle/locations/us-central1/keyRings/facial-keyring/cryptoKeys/facial-dek"
MIN_FACE_SIZE = 60
REQUEST_DELAY = 1.0

# ── Terminal formatting ──
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


# ─────────────────────────────────────────────────────────────
# BIOMETRIC PIPELINE (Self-contained, no main.py dependency)
# ─────────────────────────────────────────────────────────────

def align_face_crop_local(image: np.ndarray, target_size: int = 256):
    """
    Detects face landmarks and returns an aligned crop + landmarks.
    Works with both legacy and Tasks MediaPipe APIs.
    Returns (aligned_crop, landmarks) or (image, None) on failure.
    """
    if USE_LEGACY_MP and face_mesh is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return image, None
        
        landmarks = result.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Extract eye centers for alignment
        left_eye_indices = [33, 133, 160, 144, 153, 158]
        right_eye_indices = [362, 263, 387, 373, 380, 385]
        
        left_eye = np.mean([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in left_eye_indices], axis=0)
        right_eye = np.mean([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in right_eye_indices], axis=0)
        
        # Compute rotation angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate image
        center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (w, h))
        
        # Crop face region
        all_x = [landmarks.landmark[i].x * w for i in range(468)]
        all_y = [landmarks.landmark[i].y * h for i in range(468)]
        x_min, x_max = int(min(all_x)), int(max(all_x))
        y_min, y_max = int(min(all_y)), int(max(all_y))
        
        padding = int((x_max - x_min) * 0.15)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        crop = aligned[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return image, None
        
        crop = cv2.resize(crop, (target_size, target_size))
        return crop, landmarks
    
    elif not USE_LEGACY_MP and face_landmarker is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = face_landmarker.detect(mp_image)
        
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return image, None
        
        landmarks_list = result.face_landmarks[0]
        h, w = image.shape[:2]
        
        # Extract eye centers
        left_eye_indices = [33, 133, 160, 144, 153, 158]
        right_eye_indices = [362, 263, 387, 373, 380, 385]
        
        left_eye = np.mean([(landmarks_list[i].x * w, landmarks_list[i].y * h) for i in left_eye_indices], axis=0)
        right_eye = np.mean([(landmarks_list[i].x * w, landmarks_list[i].y * h) for i in right_eye_indices], axis=0)
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (w, h))
        
        all_x = [lm.x * w for lm in landmarks_list]
        all_y = [lm.y * h for lm in landmarks_list]
        x_min, x_max = int(min(all_x)), int(max(all_x))
        y_min, y_max = int(min(all_y)), int(max(all_y))
        
        padding = int((x_max - x_min) * 0.15)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        crop = aligned[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return image, None
        
        crop = cv2.resize(crop, (target_size, target_size))
        return crop, landmarks_list
    
    return image, None


def extract_landmark_embedding_local(landmarks) -> np.ndarray:
    """
    Extracts a geometric embedding from facial landmarks.
    Compatible with both legacy and Tasks API landmark formats.
    """
    coords = []
    if USE_LEGACY_MP:
        # Legacy: landmarks.landmark[i].x/y/z
        for lm in landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
    else:
        # Tasks API: landmarks is a list of NormalizedLandmark
        for lm in landmarks:
            coords.extend([lm.x, lm.y, lm.z])
    
    return np.array(coords, dtype=np.float64)


def encrypt_embedding_local(embedding: np.ndarray) -> bytes:
    """
    KMS Envelope Encryption with fallback for local development.
    """
    if not HAS_KMS:
        return b"MOCK_ENCRYPTED_PACKET"
    
    try:
        dek = Fernet.generate_key()
        cipher = Fernet(dek)
        payload_bytes = embedding.tobytes()
        encrypted_payload = cipher.encrypt(payload_bytes)
        
        client = kms.KeyManagementServiceClient()
        encrypt_response = client.encrypt(request={'name': KMS_KEY_NAME, 'plaintext': dek})
        encrypted_dek = encrypt_response.ciphertext
        
        dek_len = struct.pack(">I", len(encrypted_dek))
        return dek_len + encrypted_dek + encrypted_payload
    except Exception as e:
        print(f"  KMS Encryption warning/fallback: {e}")
        return b"MOCK_ENCRYPTED_PACKET"


def apply_clahe_local(image: np.ndarray) -> np.ndarray:
    """CLAHE histogram equalization for contrast normalization."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────
# TASK 2: WIKIDATA SPARQL FETCHER
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# TASK 3: WIKIMEDIA COMMONS API INTEGRATION
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# TASK 4: BIOMETRIC PIPELINE & VAULT INSERTION
# ─────────────────────────────────────────────────────────────

def process_and_vault_target(target: dict, metadata: dict, session) -> bool:
    """
    Runs the biometric pipeline on downloaded image bytes:
    1. Decode → CLAHE → Align → Extract → Encrypt
    2. Insert IdentityProfile with attribution metadata
    """
    img_array = np.frombuffer(metadata["image_bytes"], dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        log(f"  Failed to decode image for {target['person_name']}", "ERR")
        return False

    clahe_img = apply_clahe_local(image)
    aligned, landmarks = align_face_crop_local(clahe_img)
    
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

    embedding = extract_landmark_embedding_local(landmarks)
    encrypted_blob = encrypt_embedding_local(embedding)

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


# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION LOOP
# ─────────────────────────────────────────────────────────────

def run_crawler():
    print(f"\n{GOLD}{'=' * 60}{RESET}")
    print(f"{GOLD}  SOVEREIGN IDENTITY ACQUISITION DAEMON{RESET}")
    print(f"{GOLD}  Wikimedia Commons -> Biometric Vault Pipeline{RESET}")
    print(f"{GOLD}{'=' * 60}{RESET}\n")

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

    print(f"\n{GOLD}{'=' * 60}{RESET}")
    print(f"{GOLD}  CRAWL COMPLETE{RESET}")
    print(f"{GOLD}{'=' * 60}{RESET}")
    print(f"  Total Targets:  {total}")
    print(f"  {GREEN}Vaulted:        {vaulted}{RESET}")
    print(f"  {DIM}Skipped:        {skipped}{RESET}")
    print(f"  {RED}Errors:         {errors}{RESET}")
    print(f"{GOLD}{'=' * 60}{RESET}\n")


if __name__ == "__main__":
    run_crawler()
