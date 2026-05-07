import requests
import json
import sys
import time
import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--single', type=str, help='Run only cases containing this string')
parser.add_argument('--limit', type=int, help='Max number of cases to run')
parser.add_argument('--skip-v1', action='store_true', help='Skip V1 (Legacy)')
parser.add_argument('--skip-v2', action='store_true', help='Skip V2 (Smoke)')
parser.add_argument('--timeout-seconds', type=int, default=180, help='Max seconds per request/poll loop')
parser.add_argument('--v2-url', type=str, default="https://v2smoke---facial-backend-vkd6b6ijxa-uk.a.run.app", help='Target V2 URL (must be the smoke revision facial-backend-00188-hmw)')
parser.add_argument('--fresh', action='store_true', help='Remove old validation_results*.json files')
args = parser.parse_args()

LEGACY_URL = "https://facial-backend-vkd6b6ijxa-uk.a.run.app"
V2_URL = args.v2_url

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE_TS = f"validation_results_{timestamp_str}.json"
OUTPUT_FILE_LATEST = "validation_results_latest.json"

if args.fresh:
    for f in ["validation_results.json", OUTPUT_FILE_LATEST, "validation_results_partial.json"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

print(f"Logging in to {LEGACY_URL}...")
login_resp = requests.post(f"{LEGACY_URL}/login", json={"password": "aurum-admin-99"}, timeout=(10, 30))
if login_resp.status_code != 200:
    print("Login failed:", login_resp.text)
    sys.exit(1)

token = login_resp.json().get("access_token")
headers = {"Authorization": f"Bearer {token}"}

URL_GENUINE = "gs://hoppwhistle-facial-uploads/test_genuine.jpg"
URL_FAKE = "gs://hoppwhistle-facial-uploads/test_fake.jpg"
URL_DIFF = "gs://hoppwhistle-facial-uploads/probe_025a3a4f7c574a7289ac1f63221ff65d.jpg"
URL_LOW_QUALITY = "gs://hoppwhistle-facial-uploads/low_quality.jpg"
URL_NO_FACE = "gs://hoppwhistle-facial-uploads/no_face.jpg"

cases = [
    {
        "name": "1. exact_self_match (Exact same image)",
        "payload": {"probe_url": URL_GENUINE, "gallery_url": URL_GENUINE, "require_liveness": False}
    },
    {
        "name": "2. same person, different image",
        "payload": {"probe_url": URL_GENUINE, "gallery_url": URL_FAKE, "require_liveness": False}
    },
    {
        "name": "3. obvious different person",
        "payload": {"probe_url": URL_GENUINE, "gallery_url": URL_DIFF, "require_liveness": False}
    },
    {
        "name": "4. low-quality/compressed image pair",
        "payload": {"probe_url": URL_LOW_QUALITY, "gallery_url": URL_LOW_QUALITY, "require_liveness": False}
    },
    {
        "name": "5. face not detected or invalid image",
        "payload": {"probe_url": URL_NO_FACE, "gallery_url": URL_GENUINE, "require_liveness": False}
    },
    {
        "name": "6. one image with visible marks/scars/moles",
        "payload": {"probe_url": URL_DIFF, "gallery_url": URL_DIFF, "require_liveness": False}
    },
    {
        "name": "7. one pair where marks should not match",
        "payload": {"probe_url": URL_DIFF, "gallery_url": URL_GENUINE, "require_liveness": False}
    }
]

if args.single:
    cases = [c for c in cases if args.single.lower() in c["name"].lower()]

if args.limit:
    cases = cases[:args.limit]

print(f"\n--- STARTING BATCH ---")
print(f"V1 URL: {LEGACY_URL}")
print(f"V2 URL: {V2_URL} (Default: facial-backend-00188-hmw)")
print(f"Skip V1: {args.skip_v1}")
print(f"Skip V2: {args.skip_v2}")
print(f"Output files: {OUTPUT_FILE_TS}, {OUTPUT_FILE_LATEST}")
print(f"Cases to run: {len(cases)}")
for c in cases:
    print(f" - {c['name']}")
print("----------------------\n")

def check_diagnostics(result_data, variant):
    if not result_data:
        return
    
    # Try top-level keys first (VerificationResponse), then fall back to audit_log
    audit = result_data.get("audit_log", {}) or {}
    
    # mark_detector_version / mark_matcher_version: top-level or in audit_log
    mdv = result_data.get("mark_detector_version") or audit.get("mark_detector_version")
    mmv = result_data.get("mark_matcher_version") or audit.get("mark_matcher_version")
    mms = result_data.get("mark_match_status") or audit.get("mark_match_status")
    print(f"  > mark_detector_version: {mdv}")
    print(f"  > mark_matcher_version: {mmv}")
    print(f"  > mark_match_status: {mms}")
    
    # lr_marks: top-level or in audit_log
    lr = result_data.get("lr_marks")
    if lr is None:
        lr = audit.get("lr_marks")
    print(f"  > lr_marks: {lr}")
    
    # mark_diagnostics: top-level or in audit_log
    mark_diagnostics = result_data.get("mark_diagnostics") or audit.get("mark_diagnostics") or {}
    probe_trace = {}
    if isinstance(mark_diagnostics, dict):
        probe_trace = mark_diagnostics.get("mark_detector_trace", {}).get("probe", {}) or {}
    
    print(f"  > probe aligned dimensions: {probe_trace.get('aligned_dimensions')}")
    print(f"  > preprocessing steps: {probe_trace.get('preprocessing_steps')}")
    # Extract counts
    raw_probe_marks = mark_diagnostics.get("raw_probe_marks", [])
    raw_gallery_marks = mark_diagnostics.get("raw_gallery_marks", [])
    accepted = mark_diagnostics.get("accepted_correspondences", [])
    rejected = mark_diagnostics.get("rejected_correspondences", [])
    
    print(f"  > raw_probe_marks count: {len(raw_probe_marks) if isinstance(raw_probe_marks, list) else 0}")
    print(f"  > raw_gallery_marks count: {len(raw_gallery_marks) if isinstance(raw_gallery_marks, list) else 0}")
    print(f"  > accepted_correspondences count: {len(accepted) if isinstance(accepted, list) else 0}")
    print(f"  > rejected_correspondences count: {len(rejected) if isinstance(rejected, list) else 0}")
    input_is_preprocessed = probe_trace.get("input_is_preprocessed")
    internal_clahe_applied = probe_trace.get("internal_clahe_applied")
    print(f"  > input_is_preprocessed: {input_is_preprocessed}")
    print(f"  > internal_clahe_applied: {internal_clahe_applied}")
    
    if variant == "V2" and not args.skip_v2:
        if input_is_preprocessed is not True or internal_clahe_applied is not False:
            print(f"  [FAIL] V2 trace not showing input_is_preprocessed=true and internal_clahe_applied=false")
        if mms == "EXACT_SELF_MATCH" and lr != 1.0:
            print(f"  [FAIL] exact_self_match must return lr_marks=1.0. Got {lr}")
        if not mark_diagnostics:
            print(f"  [FAIL] missing mark_diagnostics")

def run_job(url, payload, name, variant):
    start_time = time.time()
    print(f"[{variant}] {name}")
    print(f"  Target URL: {url}")
    print(f"  Request start: {time.strftime('%H:%M:%S')}")
    
    try:
        req_start = time.time()
        resp = requests.post(f"{url}/verify/fuse", json=payload, headers=headers, timeout=(10, args.timeout_seconds))
        print(f"  POST /verify/fuse -> {resp.status_code} (took {time.time() - req_start:.1f}s)")
        if resp.status_code != 200:
            return resp.status_code, resp.text, None
        
        resp_json = resp.json()
        job_id = resp_json.get("job_id")
        
        if not job_id:
            print(f"  [SYNC RESPONSE] No job_id returned. Processing synchronous result.")
            if resp_json.get("status") == "success" or "audit_log" in resp_json or "conclusion" in resp_json:
                print(f"  [SUCCESS] Completed synchronously in {time.time() - start_time:.1f}s")
                print(f"  > HTTP status: {resp.status_code}")
                check_diagnostics(resp_json, variant)
                return resp.status_code, None, resp_json
            else:
                err = resp_json.get("error", "Unknown synchronous error")
                print(f"  [FAILED] Job failed synchronously: {err}")
                return resp.status_code, err, resp_json
                
        print(f"  Got job_id: {job_id}")
        
        for attempt in range(1, 31):
            if time.time() - start_time > args.timeout_seconds:
                print(f"  Timeout reached ({args.timeout_seconds}s)")
                return 504, "Hard Timeout Reached", None
                
            time.sleep(5)
            poll_req_start = time.time()
            try:
                poll_resp = requests.get(f"{url}/verify/result/{job_id}?bypass_code=aurum-admin-99", headers=headers, timeout=(10, 30))
                print(f"  Poll {attempt}/30 -> {poll_resp.status_code} (took {time.time() - poll_req_start:.1f}s)")
                
                if poll_resp.status_code == 200:
                    result_data = poll_resp.json()
                    status = result_data.get("status", "unknown")
                    print(f"    Status: {status}")
                    
                    if status == "completed" or "audit_log" in result_data:
                        print(f"  [SUCCESS] Completed in {time.time() - start_time:.1f}s")
                        print(f"  > HTTP status: {poll_resp.status_code}")
                        check_diagnostics(result_data, variant)
                        return poll_resp.status_code, None, result_data
                    if status == "failed":
                        err = result_data.get("error", "Unknown error")
                        print(f"  [FAILED] Job failed: {err}")
                        print(f"  > HTTP status: {poll_resp.status_code}")
                        return poll_resp.status_code, err, result_data
                elif poll_resp.status_code >= 400:
                    print(f"  [ERROR] Poll returned {poll_resp.status_code}: {poll_resp.text}")
                    return poll_resp.status_code, poll_resp.text, None
            except requests.exceptions.RequestException as e:
                print(f"  [ERROR] Poll request exception: {e}")
                
        print(f"  [TIMEOUT] Max poll attempts reached")
        return 504, "Max poll attempts reached", None
        
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] POST request exception: {e}")
        return 500, str(e), None

results = []

for case in cases:
    print(f"\n==========================================")
    print(f"CASE: {case['name']}")
    print(f"==========================================")
    
    v1_status, v1_err, v1_data = None, None, None
    if not args.skip_v1:
        v1_status, v1_err, v1_data = run_job(LEGACY_URL, case["payload"], case["name"], "V1")
    
    v2_status, v2_err, v2_data = None, None, None
    if not args.skip_v2:
        v2_status, v2_err, v2_data = run_job(V2_URL, case["payload"], case["name"], "V2")
    
    results.append({
        "name": case["name"],
        "v1": {"status": v1_status, "err": v1_err, "data": v1_data},
        "v2": {"status": v2_status, "err": v2_err, "data": v2_data}
    })
    
    # Save partial results
    with open('validation_results_partial.json', 'w') as f:
        json.dump(results, f, indent=2)

print("\nValidation batch complete!")
with open(OUTPUT_FILE_TS, 'w') as f:
    json.dump(results, f, indent=2)
with open(OUTPUT_FILE_LATEST, 'w') as f:
    json.dump(results, f, indent=2)
