"""
AurumShield — Cold Storage Lifecycle Policy Automation
=======================================================
Applies GCS Bucket Lifecycle rules to transition raw biometric uploads
to COLDLINE storage after 24 hours.

Rules applied:
  - Objects with prefix "probe_"   -> COLDLINE after 1 day
  - Objects with prefix "gallery_" -> COLDLINE after 1 day

Objects NOT affected:
  - receipts/   — Composite forensic PNGs stay in STANDARD (hot) indefinitely
  - topology/   — Pre-computed graph JSON stays hot
  - calibration/ — Calibration data stays hot
  - worm_anchors/ — Anchor manifests stay hot

This script is idempotent — safe to run multiple times.
It replaces any existing lifecycle rules on the bucket.

NOTE: In production, ensure the bucket has tight IAM restrictions
on the cold-stored biometric assets.

Usage:
    python apply_cold_storage_rules.py

Environment Variables:
    BUCKET_NAME — GCS bucket (default: hoppwhistle-facial-uploads)
"""

import os
from google.cloud import storage


def apply_lifecycle_rules():
    """
    Applies lifecycle rules for raw biometric upload cold-storage transition.
    """
    bucket_name = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")

    print(f"[COLD STORAGE] Applying lifecycle rules to gs://{bucket_name}")

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    # Clear existing lifecycle rules to ensure idempotency
    bucket.lifecycle_rules = []

    # Rule 1: Transition probe_* objects to COLDLINE after 1 day
    bucket.add_lifecycle_set_storage_class_rule(
        storage_class="COLDLINE",
        age=1,
        matches_prefix=["probe_"],
    )
    print("[COLD STORAGE] Rule added: probe_* -> COLDLINE after 1 day")

    # Rule 2: Transition gallery_* objects to COLDLINE after 1 day
    bucket.add_lifecycle_set_storage_class_rule(
        storage_class="COLDLINE",
        age=1,
        matches_prefix=["gallery_"],
    )
    print("[COLD STORAGE] Rule added: gallery_* -> COLDLINE after 1 day")

    # Commit the lifecycle configuration
    bucket.patch()

    print(f"[COLD STORAGE] Lifecycle rules applied successfully to gs://{bucket_name}")
    print("[COLD STORAGE] Current rules:")
    for rule in bucket.lifecycle_rules:
        print(f"  - Action: {rule['action']} | Condition: {rule['condition']}")


if __name__ == "__main__":
    apply_lifecycle_rules()
