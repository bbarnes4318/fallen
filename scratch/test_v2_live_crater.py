import json
import time
import requests
import sys
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--single", type=str, help="Run a single test case (e.g. exact_self_match)")
parser.add_argument("--limit", type=int, default=3, help="Max requests to run")
parser.add_argument("--timeout-seconds", type=int, default=240, help="Read timeout")
parser.add_argument("--no-poll", action="store_true", help="Skip polling")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

V2_URL = "https://facial-backend-v2-marks-196207148120.us-east4.run.app/verify/fuse"
V2_POLL_BASE = "https://facial-backend-v2-marks-196207148120.us-east4.run.app/verify/result"

CASES = {
    "crater_test": {
        "probe_url": "gs://hoppwhistle-facial-uploads/crater_test.jpg",
        "gallery_url": "gs://hoppwhistle-facial-uploads/crater_test.jpg",
        "require_liveness": False
    }
}

print("Logging in to get token...")
try:
    login_resp = requests.post("https://facial-backend-vkd6b6ijxa-uk.a.run.app/login", json={"password": "aurum-admin-99"}, timeout=(10, 30))
    if login_resp.status_code != 200:
        print("Login failed:", login_resp.text)
        sys.exit(1)
    token = login_resp.json().get("access_token")
    HEADERS = {"Authorization": f"Bearer {token}"}
except Exception as e:
    print("Login exception:", e)
    sys.exit(1)

requests_to_run = []
if args.single:
    if args.single in CASES:
        requests_to_run.append((args.single, CASES[args.single]))
    else:
        print(f"Unknown case: {args.single}")
        sys.exit(1)
else:
    for i, (k, v) in enumerate(CASES.items()):
        if i < args.limit:
            requests_to_run.append((k, v))

print("\n--- STARTUP CONFIG ---")
print(f"Target service URL: {V2_URL}")
print(f"Exact endpoint: POST /verify/fuse")
print(f"Connect Timeout: 10s | Read Timeout: {args.timeout_seconds}s")
print(f"Number of requests planned: {len(requests_to_run)}")
print(f"Sequential or Concurrent: Sequential")
print("----------------------\n")

results = []

for idx, (case_name, payload) in enumerate(requests_to_run):
    req_num = idx + 1
    now = datetime.now().isoformat()
    print(f"[{now}] PROGRESS: Starting Request {req_num}/{len(requests_to_run)} | Case: {case_name}")
    print(f"URL: {V2_URL}")
    if args.verbose:
        print(f"Payload: {payload}")
    
    start_time = time.time()
    try:
        res = requests.post(V2_URL, json=payload, headers=HEADERS, timeout=(10, args.timeout_seconds))
        elapsed = time.time() - start_time
        print(f"HTTP Status: {res.status_code}")
        print(f"Elapsed time: {elapsed:.2f}s")
        if res.status_code != 200:
            print(f"Response (first 500 chars): {res.text[:500]}")
            results.append({"case": case_name, "error": f"HTTP {res.status_code}", "elapsed": elapsed})
            continue
            
        data = res.json()
        job_id = data.get("job_id")
        
        final_data = None
        if job_id:
            print(f"Got job_id immediately: {job_id}")
            if not args.no_poll:
                poll_attempts = 30
                poll_delay = 5
                completed = False
                for p in range(poll_attempts):
                    time.sleep(poll_delay)
                    poll_url = f"{V2_POLL_BASE}/{job_id}?bypass_code=aurum-admin-99"
                    print(f"  Poll {p+1}/{poll_attempts} at {poll_url}")
                    poll_res = requests.get(poll_url, headers=HEADERS, timeout=(10, 60))
                    print(f"  Poll HTTP Status: {poll_res.status_code}")
                    if poll_res.status_code == 200:
                        poll_data = poll_res.json()
                        status = poll_data.get("status")
                        print(f"  Poll status text: {status}")
                        if status in ["completed", "success", "failed"] or "conclusion" in poll_data or "structural_score" in poll_data:
                            completed = True
                            final_data = poll_data
                            break
                    else:
                        print(f"  Poll failed, response: {poll_res.text[:200]}")
                if not completed:
                    print("Polling timed out.")
        else:
            print("No job_id returned. Assuming synchronous completion.")
            final_data = data
            
        if final_data:
            diag = final_data.get("mark_diagnostics", {})
            has_diag = "mark_detector_trace" in diag
            preprocessed = diag.get("mark_detector_trace", {}).get("probe", {}).get("input_is_preprocessed") if has_diag else None
            clahe = diag.get("mark_detector_trace", {}).get("probe", {}).get("internal_clahe_applied") if has_diag else None
            
            result = {
                "case": case_name,
                "post_elapsed": elapsed,
                "total_elapsed": time.time() - start_time,
                "conclusion": final_data.get("conclusion"),
                "mark_match_status": final_data.get("mark_match_status"),
                "lr_marks": final_data.get("lr_marks"),
                "raw_probe_marks": len(final_data.get("raw_probe_marks", [])),
                "raw_gallery_marks": len(final_data.get("raw_gallery_marks", [])),
                "accepted_correspondences": len(final_data.get("correspondences", [])),
                "rejected_correspondences": diag.get("rejected_candidates_count", 0),
                "input_is_preprocessed": preprocessed,
                "internal_clahe_applied": clahe,
                "structural_cands": diag.get("mark_detector_trace", {}).get("probe", {}).get("structural_depression_final_candidates") if has_diag else None,
                "fallback_used": diag.get("mark_detector_trace", {}).get("probe", {}).get("fallback_used") if has_diag else None,
            }
            results.append(result)
            print("--- Request Result ---")
            for k, v in result.items():
                print(f"{k}: {v}")
            if has_diag:
                print("\n--- Probe Trace ---")
                print(json.dumps(diag.get("mark_detector_trace", {}).get("probe", {}), indent=2))
                
                # Save overlays
                overlays = diag.get("mark_detector_overlays", {}).get("probe", {})
                if overlays:
                    import base64
                    import os
                    out_dir = "scratch/live_overlays"
                    os.makedirs(out_dir, exist_ok=True)
                    for k, b64 in overlays.items():
                        if b64:
                            b64_data = b64.split(",")[1] if "," in b64 else b64
                            out_path = os.path.join(out_dir, f"live_overlay_{k}.png")
                            with open(out_path, "wb") as f:
                                f.write(base64.b64decode(b64_data))
                    print(f"Saved {len(overlays)} overlays to {out_dir}")
            print("----------------------")
        else:
            results.append({"case": case_name, "error": "No final data retrieved", "total_elapsed": time.time() - start_time})
            
    except requests.exceptions.Timeout as e:
        elapsed = time.time() - start_time
        print(f"Timeout Error after {elapsed:.2f}s: {e}")
        results.append({"case": case_name, "error": "Timeout", "elapsed": elapsed})
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Error after {elapsed:.2f}s: {e}")
        results.append({"case": case_name, "error": str(e), "elapsed": elapsed})
        
    with open("scratch/v2_isolated_partial.json", "w") as f:
        json.dump(results, f, indent=2)
        
print("\n--- FINAL RESULTS ---")
print(json.dumps(results, indent=2))
