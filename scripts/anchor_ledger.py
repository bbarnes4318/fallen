"""
AurumShield — WORM Ledger Anchoring (Legal Non-Repudiation)
============================================================
Cryptographically anchors the mutable PostgreSQL VerificationEvent
ledger to an immutable GCS WORM bucket to prove records were not
altered post-transaction.

This script:
  1. Queries all VerificationEvent records from the last 24 hours
  2. Serializes them into a deterministic, sorted JSON string
  3. Computes the SHA-256 hash (Daily Anchor Hash)
  4. Writes a tiny JSON manifest with the date, record count, and hash
  5. Uploads to gs://[BUCKET]/worm_anchors/YYYY-MM-DD.json

Designed to run nightly via Google Cloud Scheduler.

NOTE: The target GCS bucket path should be manually configured with a
Bucket Lock / Retention Policy to ensure these anchor hashes can never
be deleted or overwritten, achieving true WORM compliance.

Usage:
    python anchor_ledger.py

Environment Variables:
    BUCKET_NAME     — GCS bucket (default: hoppwhistle-facial-uploads)
    DB_USER, DB_PASS, DB_NAME, CLOUD_SQL_CONNECTION_NAME — DB credentials
"""

import os
import sys
import json
import hashlib
import datetime
from google.cloud import storage

# Add parent directory to path for models import
# Add paths for both Docker container (/app) and local dev (../backend)
sys.path.insert(0, "/app")                                                      # Docker container
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))  # Local dev
from models import SessionLocal, VerificationEvent


def anchor_daily_ledger():
    """
    Core anchor logic.
    Queries 24h window -> serializes -> hashes -> uploads anchor manifest.
    """
    bucket_name = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
    now = datetime.datetime.utcnow()
    anchor_date = now.strftime("%Y-%m-%d")
    cutoff = now - datetime.timedelta(hours=24)

    print(f"[ANCHOR] Starting daily ledger anchoring for {anchor_date}")
    print(f"[ANCHOR] Query window: {cutoff.isoformat()}Z -> {now.isoformat()}Z")

    session = SessionLocal()
    try:
        # 1. Query all verification events in the last 24 hours
        events = (
            session.query(VerificationEvent)
            .filter(VerificationEvent.timestamp >= cutoff)
            .order_by(VerificationEvent.id.asc())
            .all()
        )

        print(f"[ANCHOR] Found {len(events)} verification events in window")

        # 2. Serialize to deterministic JSON (sorted by id, sorted keys)
        records = []
        for event in events:
            record = {
                "id": event.id,
                "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                "probe_hash": event.probe_hash,
                "gallery_hash": event.gallery_hash,
                "matched_user_id": event.matched_user_id,
                "fused_score_x100": event.fused_score_x100,
                "conclusion": event.conclusion,
                "pipeline_version": event.pipeline_version,
                "calibration_benchmark": event.calibration_benchmark,
                "false_acceptance_rate": event.false_acceptance_rate,
                "veto_triggered": event.veto_triggered,
                "structural_score_x100": event.structural_score_x100,
                "geometric_score_x100": event.geometric_score_x100,
                "micro_topology_score_x100": event.micro_topology_score_x100,
                "receipt_url": getattr(event, "receipt_url", None),
            }
            records.append(record)

        # 3. Deterministic serialization (sorted keys, compact separators)
        canonical_json = json.dumps(records, sort_keys=True, separators=(",", ":"))

        # 4. Compute SHA-256 anchor hash
        anchor_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        print(f"[ANCHOR] Canonical JSON size: {len(canonical_json)} bytes")
        print(f"[ANCHOR] Daily Anchor Hash: {anchor_hash}")

        # 5. Build manifest
        manifest = {
            "anchor_date": anchor_date,
            "anchor_timestamp_utc": now.isoformat() + "Z",
            "query_window_start": cutoff.isoformat() + "Z",
            "query_window_end": now.isoformat() + "Z",
            "record_count": len(records),
            "anchor_hash_sha256": anchor_hash,
            "hash_algorithm": "SHA-256",
            "serialization": "JSON (sorted keys, compact separators)",
        }

        manifest_json = json.dumps(manifest, indent=2)

        # 6. Upload to GCS
        gcs_path = f"worm_anchors/{anchor_date}.json"
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(manifest_json, content_type="application/json")

        print(f"[ANCHOR] Anchor manifest uploaded to gs://{bucket_name}/{gcs_path}")
        print(f"[ANCHOR] Manifest:\n{manifest_json}")

    except Exception as e:
        print(f"[ANCHOR] FATAL: Ledger anchoring failed: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    anchor_daily_ledger()
