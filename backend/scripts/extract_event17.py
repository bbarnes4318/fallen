"""Extract full details for verification event #17 (86.2% / 0 marks)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://facial_app:Fv966468!Sec5677@127.0.0.1:5432/facial_db")
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    print("=" * 80)
    print("FULL DETAILS FOR VERIFICATION EVENT #17 (86.2% / 0 marks)")
    print("=" * 80)
    
    # Get full event record
    event = conn.execute(text("""
        SELECT * FROM verification_events WHERE id = 17
    """)).fetchone()
    
    if event:
        cols = conn.execute(text("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'verification_events'
            ORDER BY ordinal_position
        """)).fetchall()
        col_names = [c[0] for c in cols]
        
        print("\n--- Full VerificationEvent #17 ---")
        for i, col in enumerate(col_names):
            val = event[i] if i < len(event) else "N/A"
            if val is not None and str(val) != "":
                print(f"  {col}: {val}")
    
    # Get all verification jobs and find the one matching event #17 timestamp
    print("\n--- Finding matching VerificationJob ---")
    jobs = conn.execute(text("""
        SELECT job_id, status, created_at, 
               result_payload
        FROM verification_jobs
        ORDER BY created_at DESC
        LIMIT 20
    """)).fetchall()
    
    # Event 17 timestamp: 2026-05-04 21:54:42.577876+00:00
    for j in jobs:
        job_id = j[0]
        created = j[2]
        payload_str = j[3]
        
        try:
            payload = json.loads(payload_str)
        except:
            continue
        
        # Check if this job's payload has fused_score ~86.2 
        fused = payload.get("fused_score")
        probe_marks_count = 0
        gallery_marks_count = 0
        
        md = payload.get("mark_diagnostics", {})
        probe_marks_count = md.get("raw_probe_marks_count", -1)
        gallery_marks_count = md.get("raw_gallery_marks_count", -1)
        
        # The 86.2% job
        if (fused is not None and abs(fused - 86.2) < 0.5) or \
           (probe_marks_count == 0 and gallery_marks_count == 0 and str(created) > "2026-05-04 21:"):
            print(f"\n*** MATCH: Job {job_id} ***")
            print(f"  Created: {created}")
            print(f"  Status: {j[1]}")
            print(f"  fused_score: {fused}")
            
            # Extract image URLs from payload
            for key in sorted(payload.keys()):
                val = payload[key]
                if isinstance(val, str) and ("gs://" in val or "upload" in val.lower() or "url" in key.lower() or "uri" in key.lower() or "image" in key.lower()):
                    print(f"  {key}: {val}")
            
            # Look for probe_url/gallery_url in ANY nested structure
            print(f"\n  --- All URL-like fields in payload ---")
            def find_urls(obj, prefix=""):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        find_urls(v, f"{prefix}.{k}" if prefix else k)
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        find_urls(v, f"{prefix}[{i}]")
                elif isinstance(obj, str) and ("gs://" in obj or "http" in obj):
                    print(f"    {prefix}: {obj}")
            find_urls(payload)
            
            # Print mark_diagnostics
            print(f"\n  --- mark_diagnostics ---")
            print(json.dumps(md, indent=4))
            
            # Print full payload keys
            print(f"\n  --- Payload top-level keys ---")
            for k in sorted(payload.keys()):
                v = payload[k]
                if isinstance(v, (str, int, float, bool, type(None))):
                    print(f"    {k}: {v}")
                elif isinstance(v, list):
                    print(f"    {k}: [{len(v)} items]")
                elif isinstance(v, dict):
                    print(f"    {k}: {{...{len(v)} keys}}")
    
    # Also check: the probe_hash and gallery_hash from event #17 may help trace back to GCS
    print("\n--- Image hashes from event #17 ---")
    hashes = conn.execute(text("""
        SELECT probe_hash, gallery_hash, 
               probe_source_file_hash, gallery_source_file_hash,
               probe_decoded_image_hash, gallery_decoded_image_hash,
               probe_original_dimensions, gallery_original_dimensions,
               probe_aligned_dimensions, gallery_aligned_dimensions
        FROM verification_events WHERE id = 17
    """)).fetchone()
    
    if hashes:
        print(f"  probe_hash: {hashes[0]}")
        print(f"  gallery_hash: {hashes[1]}")
        print(f"  probe_source_file_hash: {hashes[2]}")
        print(f"  gallery_source_file_hash: {hashes[3]}")
        print(f"  probe_decoded_image_hash: {hashes[4]}")
        print(f"  gallery_decoded_image_hash: {hashes[5]}")
        print(f"  probe_original_dimensions: {hashes[6]}")
        print(f"  gallery_original_dimensions: {hashes[7]}")
        print(f"  probe_aligned_dimensions: {hashes[8]}")
        print(f"  gallery_aligned_dimensions: {hashes[9]}")

print("\n" + "=" * 80)
