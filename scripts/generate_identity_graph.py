"""
AurumShield — Identity Graph Generator (Cloud Run Job)
=======================================================
Decoupled from the /vault/network API endpoint to eliminate O(N²)
KMS decryption and cosine computation at request time.

This script:
  1. Queries all IdentityProfile records from PostgreSQL
  2. Decrypts each KMS-encrypted 512-D ArcFace embedding
  3. Computes upper-triangle NxN cosine similarity matrix
  4. Filters links > 90% match threshold (capped at 5,000 strongest)
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
import numpy as np
from google.cloud import storage, kms
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
                    "embedding": vec,
                })
            except Exception as e:
                print(f"[GRAPH] WARN: Skipping {profile.user_id} — decryption failed: {e}")
                continue

        decrypt_elapsed = (time.perf_counter() - decrypt_start) * 1000
        print(f"[GRAPH] Decrypted {len(decrypted)}/{len(profiles)} embeddings in {decrypt_elapsed:.0f}ms")

        # 2. Build nodes
        nodes = [
            {
                "id": d["user_id"],
                "name": d["person_name"] or d["user_id"],
                "group": 1,
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

                if score > 90.0:
                    links.append({
                        "source": decrypted[i]["user_id"],
                        "target": decrypted[j]["user_id"],
                        "value": round(score, 1),
                    })

        compute_elapsed = (time.perf_counter() - compute_start) * 1000
        print(f"[GRAPH] Computed {comparisons} pairwise comparisons in {compute_elapsed:.0f}ms")
        print(f"[GRAPH] Found {len(links)} links above 90% threshold")

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
