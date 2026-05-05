"""Extract GCS URLs from the matching job for event #17."""
import os, sys, json
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://facial_app:Fv966468!Sec5677@127.0.0.1:5432/facial_db")
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # Get all jobs created after 21:00 UTC May 4 (event #17 is at 21:54:42)
    jobs = conn.execute(text("""
        SELECT job_id, created_at, result_payload
        FROM verification_jobs
        WHERE created_at > '2026-05-04 21:00:00'
        ORDER BY created_at DESC
    """)).fetchall()
    
    print(f"Jobs after 21:00 UTC on May 4: {len(jobs)}")
    
    for j in jobs:
        print(f"\n=== Job: {j[0]}, Created: {j[1]} ===")
        payload = json.loads(j[2])
        
        # Print fused_score
        fs = payload.get("fused_score")
        print(f"  fused_score: {fs}")
        
        md = payload.get("mark_diagnostics", {})
        print(f"  probe_marks: {md.get('raw_probe_marks_count')}")
        print(f"  gallery_marks: {md.get('raw_gallery_marks_count')}")
        
        # Find ALL string values containing gs://
        def find_gs(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    find_gs(v, f"{prefix}.{k}" if prefix else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    find_gs(v, f"{prefix}[{i}]")
            elif isinstance(obj, str) and ("gs://" in obj or "upload" in obj.lower()):
                print(f"  GCS URL [{prefix}]: {obj}")
        
        find_gs(payload)
        
        # Check for image URL keys
        for k in ["probe_url", "gallery_url", "probe_gs_uri", "gallery_gs_uri",
                   "gallery_image_url", "probe_image_url", "receipt_url",
                   "probe_source_url", "gallery_source_url"]:
            if k in payload and payload[k]:
                print(f"  {k}: {payload[k]}")
    
    # Also: the VerificationResponse model might store URLs differently
    # Let's check the full top-level key list of the matching job payload
    if jobs:
        latest = json.loads(jobs[0][2])
        print(f"\n=== Payload top-level keys for latest job ===")
        for k in sorted(latest.keys()):
            v = latest[k]
            if isinstance(v, str) and len(v) > 200:
                print(f"  {k}: [string, {len(v)} chars, starts with: {v[:80]}...]")
            elif isinstance(v, list):
                print(f"  {k}: [list, {len(v)} items]")
            elif isinstance(v, dict):
                print(f"  {k}: [dict, {len(v)} keys: {list(v.keys())[:10]}]")
            else:
                print(f"  {k}: {v}")
    
    # Check GCS bucket for recently uploaded images
    print("\n=== Searching GCS for uploaded images matching hashes ===")
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket("hoppwhistle-facial-uploads")
    
    # List recent blobs in uploads/ prefix
    blobs = list(bucket.list_blobs(prefix="uploads/", max_results=50))
    blobs.sort(key=lambda b: b.updated, reverse=True)
    print(f"Recent uploads: {len(blobs)}")
    for b in blobs[:20]:
        print(f"  gs://hoppwhistle-facial-uploads/{b.name} | {b.size} bytes | {b.updated}")
