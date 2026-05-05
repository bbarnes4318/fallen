"""Extract the exact aligned crops from job fd92cb69-3ae6-4c86-8bed-2256e1313ad8 as PNG files.
Also extract the full mark_diagnostics and audit_log mark data."""
import os, sys, json, base64, hashlib
from pathlib import Path
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://facial_app:Fv966468!Sec5677@127.0.0.1:5432/facial_db")
engine = create_engine(DATABASE_URL)

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "audit_output"
OUT_DIR.mkdir(exist_ok=True)

with engine.connect() as conn:
    job = conn.execute(text("""
        SELECT result_payload FROM verification_jobs 
        WHERE job_id = 'fd92cb69-3ae6-4c86-8bed-2256e1313ad8'
    """)).fetchone()
    
    if not job:
        print("ERROR: Job not found!")
        sys.exit(1)
    
    payload = json.loads(job[0])
    
    # Extract aligned crops
    for side in ["probe", "gallery"]:
        b64_key = f"{side}_aligned_b64"
        b64_data = payload.get(b64_key, "")
        if b64_data:
            # Strip data URI prefix
            if "base64," in b64_data:
                b64_data = b64_data.split("base64,")[1]
            img_bytes = base64.b64decode(b64_data)
            out_path = OUT_DIR / f"production_{side}_aligned.png"
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            h = hashlib.sha256(img_bytes).hexdigest()
            print(f"  Saved {out_path} ({len(img_bytes)} bytes, sha256={h[:16]}...)")
    
    # Also try to find the original images via GCS listing
    # The original images were 276x182 (probe) and 194x259 (gallery)
    # Let's search GCS for ALL blobs, not just uploads/
    print("\n=== Searching GCS for ALL recent image blobs ===")
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket("hoppwhistle-facial-uploads")
    
    # List ALL blobs
    all_blobs = []
    for blob in bucket.list_blobs():
        all_blobs.append(blob)
    
    all_blobs.sort(key=lambda b: b.updated, reverse=True)
    print(f"Total blobs in bucket: {len(all_blobs)}")
    
    # Show the most recent 40
    for b in all_blobs[:40]:
        print(f"  gs://hoppwhistle-facial-uploads/{b.name} | {b.size:>10} bytes | {b.updated} | {b.content_type}")
    
    # Now look for blobs whose hash matches the probe/gallery hashes
    probe_hash = "d51875cfea4cd61041042bd5da15f5e091c29d40717ac4e2440481d116abf6fc"
    gallery_hash = "3c888aca6f95dc3e80bf47c0a6c2f9ead34526b827c6e92a171be23e8e2970b2"
    
    print(f"\n=== Checking SHA-256 matches for production hashes ===")
    print(f"  Looking for probe_hash: {probe_hash[:24]}...")
    print(f"  Looking for gallery_hash: {gallery_hash[:24]}...")
    
    # Download any image blobs and check their hashes
    image_blobs = [b for b in all_blobs if b.name.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    print(f"\n  Image blobs to check: {len(image_blobs)}")
    
    for b in image_blobs:
        data = b.download_as_bytes()
        h = hashlib.sha256(data).hexdigest()
        match_tag = ""
        if h == probe_hash:
            match_tag = " *** PROBE MATCH ***"
            out_path = OUT_DIR / f"production_probe_original{os.path.splitext(b.name)[1]}"
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"  SAVED probe original: {out_path}")
        if h == gallery_hash:
            match_tag = " *** GALLERY MATCH ***"
            out_path = OUT_DIR / f"production_gallery_original{os.path.splitext(b.name)[1]}"
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"  SAVED gallery original: {out_path}")
        if match_tag:
            print(f"  {b.name}: hash={h[:24]}...{match_tag}")
    
    # Save full mark diagnostics trace
    md = payload.get("mark_diagnostics", {})
    md_path = OUT_DIR / "production_mark_diagnostics.json"
    with open(md_path, "w") as f:
        json.dump(md, f, indent=2)
    print(f"\n  Saved mark_diagnostics: {md_path}")
    
    # Save audit_log mark data
    al = payload.get("audit_log", {})
    mark_fields = {k: v for k, v in al.items() if "mark" in k.lower() or "lr_mark" in k.lower()}
    al_path = OUT_DIR / "production_audit_log_marks.json"
    with open(al_path, "w") as f:
        json.dump(mark_fields, f, indent=2)
    print(f"  Saved audit_log mark fields: {al_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"PRODUCTION CASE SUMMARY")
    print(f"{'='*80}")
    print(f"  Job ID: fd92cb69-3ae6-4c86-8bed-2256e1313ad8")
    print(f"  Event ID: 17")
    print(f"  fused_identity_score: {payload.get('fused_identity_score')}")
    print(f"  bayesian_fused_score: {payload.get('bayesian_fused_score')}")
    print(f"  conclusion: {payload.get('conclusion')}")
    print(f"  Probe original dims: 276x182")
    print(f"  Gallery original dims: 194x259")
    print(f"  marks_detected_probe: {payload.get('marks_detected_probe')}")
    print(f"  marks_detected_gallery: {payload.get('marks_detected_gallery')}")
    print(f"  mark_match_status: {payload.get('mark_match_status')}")
    print(f"  lr_marks: {payload.get('lr_marks')}")
    print(f"  detector_status: {md.get('detector_status')}")
    print(f"  probe_detector_status: {md.get('probe_detector_status')}")
    print(f"  gallery_detector_status: {md.get('gallery_detector_status')}")
    
    # Full detector trace
    dt = md.get("mark_detector_trace", {})
    for side in ["probe", "gallery"]:
        print(f"\n  --- {side.upper()} detector trace ---")
        st = dt.get(side, {})
        for k, v in st.items():
            print(f"    {k}: {v}")
