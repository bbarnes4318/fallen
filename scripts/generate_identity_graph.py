"""
AurumShield — Identity Graph Generator (Cloud Run Job)
=======================================================
Decoupled from the /vault/network API endpoint to eliminate O(N²)
KMS decryption and cosine computation at request time.

This script:
  1. Queries all IdentityProfile records from PostgreSQL
  2. Decrypts each KMS-encrypted 512-D ArcFace embedding
  3. Computes upper-triangle NxN cosine similarity matrix
  4. Filters links > 97% match threshold (capped at 5,000 strongest)
  5. Uploads the resulting JSON graph to GCS

Designed to run as an asynchronous Cloud Run Job triggered by
Cloud Scheduler (e.g., every 30 minutes or on-demand).

Usage:
    python generate_identity_graph.py

Environment Variables:
    BUCKET_NAME     — GCS bucket (default: hoppwhistle-facial-uploads)
    KMS_KEY_NAME    — GCP KMS key resource path
    DB_USER, DB_PASS, DB_NAME, CLOUD_SQL_CONNECTION_NAME — DB credentials
"""

import os
import sys
import json
import struct
import time
import datetime
import numpy as np
from google.cloud import storage, kms
import google.oauth2.service_account
from cryptography.fernet import Fernet

# Add parent directory to path for models import
# Add paths for both Docker container (/app) and local dev (../backend)
sys.path.insert(0, "/app")                                                      # Docker container
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))  # Local dev
from models import SessionLocal, IdentityProfile


# ---------------------------------------------------------
# ISOLATED CRYPTO FUNCTIONS (No TensorFlow/MediaPipe deps)
# ---------------------------------------------------------

KMS_KEY_NAME = (
    os.getenv("KMS_KEY_NAME")
    or "projects/hoppwhistle/locations/us-central1/keyRings/facial-keyring/cryptoKeys/facial-dek"
)


def decrypt_embedding(packet: bytes) -> np.ndarray:
    """
    Decrypts a KMS envelope-encrypted biometric embedding.
    Duplicated from main.py to avoid importing the full
    TensorFlow/MediaPipe/ONNX stack (~2GB cold start).
    """
    if packet == b"MOCK_ENCRYPTED_PACKET":
        return np.random.rand(512)

    try:
        dek_len = struct.unpack(">I", packet[:4])[0]
        encrypted_dek = packet[4 : 4 + dek_len]
        encrypted_payload = packet[4 + dek_len :]

        client = kms.KeyManagementServiceClient()
        decrypt_response = client.decrypt(
            request={"name": KMS_KEY_NAME, "ciphertext": encrypted_dek}
        )
        dek = decrypt_response.plaintext

        cipher = Fernet(dek)
        payload_bytes = cipher.decrypt(encrypted_payload)

        embedding = np.frombuffer(payload_bytes, dtype=np.float64)
        return embedding
    except Exception as e:
        raise ValueError(f"Decryption failed: {e}")


def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine Similarity = (A . B) / (||A|| x ||B||)"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


# ── Cached signing infrastructure (built once, reused for all URLs) ──
_signing_cache = {}

def _get_signing_infra():
    """Build and cache signing credentials + client for Cloud Run IAM signing."""
    if _signing_cache:
        return _signing_cache

    import google.auth
    from google.auth.transport import requests as auth_requests
    from google.auth import compute_engine

    credentials, project = google.auth.default()
    client = storage.Client(credentials=credentials, project=project)

    signing_creds = None
    if isinstance(credentials, compute_engine.Credentials):
        # Cloud Run's default creds return 'default' as service_account_email.
        # Fetch the actual email from the metadata server.
        import requests as _req
        sa_email = credentials.service_account_email
        if sa_email == "default" or not sa_email:
            try:
                resp = _req.get(
                    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
                    headers={"Metadata-Flavor": "Google"},
                    timeout=5,
                )
                sa_email = resp.text.strip()
            except Exception:
                sa_email = "facial-runtime-sa@hoppwhistle.iam.gserviceaccount.com"
        
        print(f"[GRAPH] Using service account for signing: {sa_email}")
        
        from google.auth import iam
        signer = iam.Signer(
            request=auth_requests.Request(),
            credentials=credentials,
            service_account_email=sa_email,
        )
        signing_creds = google.oauth2.service_account.Credentials(
            signer=signer,
            service_account_email=sa_email,
            token_uri="https://oauth2.googleapis.com/token",
        )

    _signing_cache["client"] = client
    _signing_cache["signing_creds"] = signing_creds
    return _signing_cache


def sign_gcs_url(gs_uri: str, expiration_days: int = 7) -> str:
    """
    Converts a gs://bucket/path URI to a signed URL for browser access.
    Uses IAM signBlob API for Cloud Run (compute engine creds can't sign directly).
    Passes through https:// URLs unchanged (e.g., Wikimedia thumbnails).
    Returns empty string if signing fails.
    """
    if not gs_uri:
        return ""
    if gs_uri.startswith("https://") or gs_uri.startswith("http://"):
        return gs_uri
    if not gs_uri.startswith("gs://"):
        return ""
    try:
        infra = _get_signing_infra()
        client = infra["client"]
        signing_creds = infra["signing_creds"]

        parts = gs_uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        sign_kwargs = {
            "version": "v4",
            "expiration": datetime.timedelta(days=expiration_days),
            "method": "GET",
        }
        if signing_creds:
            sign_kwargs["credentials"] = signing_creds

        return blob.generate_signed_url(**sign_kwargs)
    except Exception as e:
        print(f"[GRAPH] WARN: Failed to sign {gs_uri}: {e}")
        return ""


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def generate_graph():
    """
    Core graph generation logic.
    Queries vault -> decrypts -> computes NxN -> uploads JSON to GCS.
    """
    bucket_name = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
    gcs_path = "topology/network_graph.json"

    print(f"[GRAPH] Starting identity graph generation...")
    print(f"[GRAPH] Target: gs://{bucket_name}/{gcs_path}")

    session = SessionLocal()
    try:
        profiles = session.query(IdentityProfile).all()
        print(f"[GRAPH] Found {len(profiles)} identity profiles in vault.")

        if not profiles:
            graph_data = {"nodes": [], "links": []}
            _upload_graph(bucket_name, gcs_path, graph_data)
            print("[GRAPH] Empty vault — uploaded empty graph.")
            return

        # 1. Decrypt all embeddings
        decrypted = []
        decrypt_start = time.perf_counter()

        for profile in profiles:
            try:
                vec = decrypt_embedding(profile.encrypted_facial_embedding)
                decrypted.append({
                    "user_id": profile.user_id,
                    "person_name": profile.person_name,
                    "thumbnail_url": profile.thumbnail_url or profile.image_url,
                    "embedding": vec,
                })
            except Exception as e:
                print(f"[GRAPH] WARN: Skipping {profile.user_id} — decryption failed: {e}")
                continue

        decrypt_elapsed = (time.perf_counter() - decrypt_start) * 1000
        print(f"[GRAPH] Decrypted {len(decrypted)}/{len(profiles)} embeddings in {decrypt_elapsed:.0f}ms")

        # 2. Generate signed URLs for thumbnails (batch)
        print(f"[GRAPH] Generating signed thumbnail URLs...")
        thumbnail_map = {}
        for d in decrypted:
            if d["thumbnail_url"]:
                thumbnail_map[d["user_id"]] = sign_gcs_url(d["thumbnail_url"])
            else:
                thumbnail_map[d["user_id"]] = ""
        print(f"[GRAPH] Signed {len(thumbnail_map)} thumbnail URLs")

        # 3. Build nodes
        nodes = [
            {
                "id": d["user_id"],
                "name": d["person_name"] or d["user_id"],
                "group": 1,
                "thumbnail": thumbnail_map.get(d["user_id"], ""),
            }
            for d in decrypted
        ]

        # 3. Compute NxN cosine similarity (upper triangle only)
        links = []
        comparisons = 0
        compute_start = time.perf_counter()

        for i in range(len(decrypted)):
            for j in range(i + 1, len(decrypted)):
                vec_a = decrypted[i]["embedding"]
                vec_b = decrypted[j]["embedding"]

                # Skip dimension mismatch (legacy 1404-D vs current 512-D)
                if vec_a.shape[0] != vec_b.shape[0]:
                    continue

                comparisons += 1
                score = calculate_cosine_similarity(vec_a, vec_b) * 100

                if score > 97.0:
                    links.append({
                        "source": decrypted[i]["user_id"],
                        "target": decrypted[j]["user_id"],
                        "value": round(score, 1),
                    })

        compute_elapsed = (time.perf_counter() - compute_start) * 1000
        print(f"[GRAPH] Computed {comparisons} pairwise comparisons in {compute_elapsed:.0f}ms")
        print(f"[GRAPH] Found {len(links)} links above 97% threshold")

        # Cap links to strongest 5,000 for browser renderability
        MAX_LINKS = 5000
        if len(links) > MAX_LINKS:
            links.sort(key=lambda l: l["value"], reverse=True)
            links = links[:MAX_LINKS]
            print(f"[GRAPH] Capped to top {MAX_LINKS} strongest links")

        # 4. Mark high-connectivity nodes as anomalies (group 2)
        connection_counts = {}
        for link in links:
            connection_counts[link["source"]] = connection_counts.get(link["source"], 0) + 1
            connection_counts[link["target"]] = connection_counts.get(link["target"], 0) + 1

        avg_connections = (
            (sum(connection_counts.values()) / len(connection_counts))
            if connection_counts
            else 0
        )
        for node in nodes:
            if connection_counts.get(node["id"], 0) > avg_connections * 1.5:
                node["group"] = 2

        # Prune orphan nodes (no links) to keep the graph compact
        linked_ids = set()
        for link in links:
            linked_ids.add(link["source"])
            linked_ids.add(link["target"])
        nodes = [n for n in nodes if n["id"] in linked_ids]
        print(f"[GRAPH] Pruned to {len(nodes)} connected nodes")

        graph_data = {"nodes": nodes, "links": links}

        # 5. Upload to GCS
        _upload_graph(bucket_name, gcs_path, graph_data)
        print(f"[GRAPH] Graph uploaded to gs://{bucket_name}/{gcs_path}")
        print(f"[GRAPH] Summary: {len(nodes)} nodes, {len(links)} links")

    except Exception as e:
        print(f"[GRAPH] FATAL: Graph generation failed: {e}")
        raise
    finally:
        session.close()


def _upload_graph(bucket_name: str, gcs_path: str, graph_data: dict):
    """Serializes and uploads graph JSON to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(
        json.dumps(graph_data, separators=(',', ':')),
        content_type="application/json",
    )


if __name__ == "__main__":
    generate_graph()
