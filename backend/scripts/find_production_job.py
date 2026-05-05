"""Find the exact production verification job that produced 86.2% / 0 marks."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("ENVIRONMENT", "development")

from models import SessionLocal, VerificationEvent, VerificationJob
from sqlalchemy import text

db = SessionLocal()

# 86.2% = 8620 in x100 format (could also be 8619 or 8621 due to rounding)
print("=" * 80)
print("SEARCHING FOR 86.2% / 0-marks PRODUCTION JOB")
print("=" * 80)

# Strategy 1: Search VerificationEvent for fused_score_x100 near 8620
print("\n--- Strategy 1: VerificationEvent with fused_score_x100 near 8620 ---")
try:
    events = db.execute(text("""
        SELECT id, timestamp, fused_score_x100, bayesian_fused_score_x100,
               marks_detected_probe, marks_detected_gallery, 
               lr_marks_product, mark_match_status,
               probe_hash, gallery_hash,
               probe_original_dimensions, gallery_original_dimensions,
               probe_aligned_dimensions, gallery_aligned_dimensions,
               mark_detector_version,
               mark_lrs_json, accepted_mark_correspondences_json,
               conclusion
        FROM verification_events 
        WHERE fused_score_x100 BETWEEN 8600 AND 8650
        ORDER BY timestamp DESC
        LIMIT 10
    """)).fetchall()
    
    print(f"Found {len(events)} events with score near 86.2%")
    for e in events:
        print(f"\n  Event ID: {e[0]}")
        print(f"  Timestamp: {e[1]}")
        print(f"  fused_score_x100: {e[2]} ({e[2]/100.0:.2f}%)")
        print(f"  bayesian_fused_score_x100: {e[3]} ({e[3]/100.0:.2f}% if set)" if e[3] else f"  bayesian_fused_score_x100: None")
        print(f"  marks_detected_probe: {e[4]}")
        print(f"  marks_detected_gallery: {e[5]}")
        print(f"  lr_marks_product: {e[6]}")
        print(f"  mark_match_status: {e[7]}")
        print(f"  probe_hash: {e[8]}")
        print(f"  gallery_hash: {e[9]}")
        print(f"  probe_original_dimensions: {e[10]}")
        print(f"  gallery_original_dimensions: {e[11]}")
        print(f"  probe_aligned_dimensions: {e[12]}")
        print(f"  gallery_aligned_dimensions: {e[13]}")
        print(f"  mark_detector_version: {e[14]}")
        print(f"  conclusion: {e[17]}")
except Exception as ex:
    print(f"  Error: {ex}")

# Strategy 2: Also search for events with 0 probe marks AND 0 gallery marks
print("\n--- Strategy 2: VerificationEvent with 0 marks on both sides ---")
try:
    zero_mark_events = db.execute(text("""
        SELECT id, timestamp, fused_score_x100, bayesian_fused_score_x100,
               marks_detected_probe, marks_detected_gallery,
               lr_marks_product, mark_match_status,
               probe_hash, gallery_hash,
               conclusion
        FROM verification_events
        WHERE marks_detected_probe = 0 AND marks_detected_gallery = 0
        ORDER BY timestamp DESC
        LIMIT 10
    """)).fetchall()
    
    print(f"Found {len(zero_mark_events)} events with 0 marks on both sides")
    for e in zero_mark_events:
        print(f"\n  Event ID: {e[0]}, Score: {e[2]/100.0:.2f}%, Marks P/G: {e[4]}/{e[5]}, Status: {e[7]}")
        print(f"  probe_hash: {e[8]}, gallery_hash: {e[9]}")
        print(f"  conclusion: {e[10]}")
except Exception as ex:
    print(f"  Error: {ex}")

# Strategy 3: Search VerificationJob for result_payload containing 86.2
print("\n--- Strategy 3: VerificationJob result_payload search ---")
try:
    jobs = db.execute(text("""
        SELECT job_id, status, created_at, paid_at,
               substring(result_payload, 1, 500) as payload_preview
        FROM verification_jobs
        ORDER BY created_at DESC
        LIMIT 20
    """)).fetchall()
    
    print(f"Found {len(jobs)} total recent jobs")
    for j in jobs:
        print(f"\n  Job ID: {j[0]}, Status: {j[1]}, Created: {j[2]}")
        # Check if this job's payload mentions 86.2 or has 0 marks
        preview = j[4] or ""
        if "86.2" in preview or '"probe_marks": 0' in preview.lower():
            print(f"  *** POTENTIAL MATCH ***")
        print(f"  Payload preview: {preview[:200]}...")
except Exception as ex:
    print(f"  Error: {ex}")

# Strategy 4: Get full payload for jobs near 86.2%
print("\n--- Strategy 4: Full payload extraction for matching jobs ---")
try:
    matching_jobs = db.execute(text("""
        SELECT job_id, result_payload, created_at
        FROM verification_jobs
        WHERE result_payload LIKE '%86.2%'
           OR result_payload LIKE '%8620%'
           OR result_payload LIKE '%8619%'
           OR result_payload LIKE '%8621%'
        ORDER BY created_at DESC
        LIMIT 5
    """)).fetchall()
    
    print(f"Found {len(matching_jobs)} jobs with 86.2% in payload")
    for j in matching_jobs:
        print(f"\n  Job ID: {j[0]}, Created: {j[2]}")
        try:
            payload = json.loads(j[1])
            fused = payload.get("fused_score") or payload.get("displayed_score")
            print(f"  fused_score: {fused}")
            
            # Extract mark diagnostics
            md = payload.get("mark_diagnostics", {})
            print(f"  mark_diagnostics: {json.dumps(md, indent=4)[:1000]}")
            
            # Extract image URLs
            probe_url = payload.get("probe_url") or payload.get("probe_image_url") or payload.get("probe_gs_uri")
            gallery_url = payload.get("gallery_url") or payload.get("gallery_image_url") or payload.get("gallery_gs_uri")
            print(f"  probe_url: {probe_url}")
            print(f"  gallery_url: {gallery_url}")
            
            # Check for marks data
            probe_marks = payload.get("raw_probe_marks", [])
            gallery_marks = payload.get("raw_gallery_marks", [])
            print(f"  raw_probe_marks count: {len(probe_marks) if isinstance(probe_marks, list) else probe_marks}")
            print(f"  raw_gallery_marks count: {len(gallery_marks) if isinstance(gallery_marks, list) else gallery_marks}")
            
        except json.JSONDecodeError:
            print(f"  (Could not parse payload as JSON)")
except Exception as ex:
    print(f"  Error: {ex}")

# Strategy 5: Get ALL recent verification events to find the right one
print("\n--- Strategy 5: All recent verification events ---")
try:
    all_events = db.execute(text("""
        SELECT id, timestamp, fused_score_x100,
               marks_detected_probe, marks_detected_gallery,
               lr_marks_product, probe_hash, gallery_hash,
               conclusion
        FROM verification_events
        ORDER BY timestamp DESC
        LIMIT 30
    """)).fetchall()
    
    print(f"Last {len(all_events)} events:")
    for e in all_events:
        score = e[2] / 100.0 if e[2] else 0
        print(f"  ID={e[0]:4d} | {e[1]} | {score:6.2f}% | P={e[3]}, G={e[4]} | LR_marks={e[5]} | {e[8][:60] if e[8] else 'N/A'}")
except Exception as ex:
    print(f"  Error: {ex}")

db.close()
print("\n" + "=" * 80)
print("SEARCH COMPLETE")
print("=" * 80)
