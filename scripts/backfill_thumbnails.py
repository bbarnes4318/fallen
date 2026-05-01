"""
Fallen — Backfill thumbnail_url for LFW profiles
=====================================================
Sets thumbnail_url = gs://hoppwhistle-facial-uploads/gallery/{stem}.jpg
for all profiles where thumbnail_url IS NULL and source = 'LFW Academic Dataset'.

The stem is derived by removing the trailing hash from user_id:
  user_id: george_w_bush_0001_24e0100f
  stem:    George_W_Bush_0001  (original filename capitalization lost, but GCS paths are case-sensitive)

Since the gallery blobs were uploaded with the original filename stem (preserving case),
we reconstruct the stem from the directory listing of gs://hoppwhistle-facial-uploads/gallery/.
"""

import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from models import SessionLocal, IdentityProfile
from google.cloud import storage

BUCKET_NAME = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
GALLERY_PREFIX = "gallery/"


def main():
    print("[BACKFILL] Starting thumbnail_url backfill for LFW profiles...")

    # 1. List all gallery blobs to build a lowercase-to-actual mapping
    print("[BACKFILL] Listing gallery blobs from GCS...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=GALLERY_PREFIX))
    print(f"[BACKFILL] Found {len(blobs)} gallery blobs")

    # Build map: lowercase_stem -> gs:// URI
    # e.g., "george_w_bush_0001" -> "gs://bucket/gallery/George_W_Bush_0001.jpg"
    gallery_map = {}
    for blob in blobs:
        # blob.name = "gallery/George_W_Bush_0001.jpg"
        filename = blob.name.replace(GALLERY_PREFIX, "")  # "George_W_Bush_0001.jpg"
        stem = os.path.splitext(filename)[0]  # "George_W_Bush_0001"
        lowercase_stem = stem.lower()  # "george_w_bush_0001"
        gallery_map[lowercase_stem] = f"gs://{BUCKET_NAME}/{blob.name}"

    print(f"[BACKFILL] Built gallery map with {len(gallery_map)} entries")

    # 2. Query profiles with NULL thumbnail_url
    session = SessionLocal()
    try:
        null_profiles = session.query(IdentityProfile).filter(
            IdentityProfile.thumbnail_url.is_(None)
        ).all()
        print(f"[BACKFILL] Found {len(null_profiles)} profiles with NULL thumbnail_url")

        updated = 0
        not_found = 0
        for profile in null_profiles:
            # user_id format: george_w_bush_0001_24e0100f (stem + _ + 8-char hash)
            # Remove the trailing _XXXXXXXX hash to get the original stem (lowercased)
            parts = profile.user_id.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) == 8:
                lowercase_stem = parts[0]
            else:
                lowercase_stem = profile.user_id

            gs_uri = gallery_map.get(lowercase_stem)
            if gs_uri:
                profile.thumbnail_url = gs_uri
                updated += 1
            else:
                not_found += 1

        if updated > 0:
            session.commit()
            print(f"[BACKFILL] Updated {updated} profiles with thumbnail_url")
        if not_found > 0:
            print(f"[BACKFILL] WARN: {not_found} profiles had no matching gallery blob")

        print(f"[BACKFILL] Done. Updated: {updated}, Not found: {not_found}")

    except Exception as e:
        print(f"[BACKFILL] FATAL: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
