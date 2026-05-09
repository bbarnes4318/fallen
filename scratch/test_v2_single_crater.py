"""
Exactly ONE live V2 crater request — with structural_source diagnostics.
No batching. No concurrency. No local builds.
"""
import json
import time
import requests
import sys

V2_URL = "https://facial-backend-v2-marks-196207148120.us-east4.run.app/verify/fuse"
V2_POLL_BASE = "https://facial-backend-v2-marks-196207148120.us-east4.run.app/verify/result"

# Login
print("=== LOGIN ===")
login_resp = requests.post(
    "https://facial-backend-vkd6b6ijxa-uk.a.run.app/login",
    json={"password": "aurum-admin-99"},
    timeout=(10, 30)
)
if login_resp.status_code != 200:
    print(f"Login failed: {login_resp.status_code} {login_resp.text[:300]}")
    sys.exit(1)
token = login_resp.json().get("access_token")
HEADERS = {"Authorization": f"Bearer {token}"}
print("Login OK")

# Single request
payload = {
    "probe_url": "gs://hoppwhistle-facial-uploads/crater_test.jpg",
    "gallery_url": "gs://hoppwhistle-facial-uploads/crater_test.jpg",
    "require_liveness": False
}

print(f"\n=== SENDING 1 REQUEST to {V2_URL} ===")
t0 = time.time()
res = requests.post(V2_URL, json=payload, headers=HEADERS, timeout=(10, 240))
elapsed = time.time() - t0
print(f"HTTP Status: {res.status_code}")
print(f"Elapsed: {elapsed:.2f}s")

if res.status_code != 200:
    print(f"ERROR RESPONSE: {res.text[:1000]}")
    sys.exit(1)

data = res.json()

# Poll if async
final_data = None
job_id = data.get("job_id")
if job_id:
    print(f"job_id: {job_id}")
    for p in range(30):
        time.sleep(5)
        poll_url = f"{V2_POLL_BASE}/{job_id}?bypass_code=aurum-admin-99"
        pr = requests.get(poll_url, headers=HEADERS, timeout=(10, 60))
        if pr.status_code == 200:
            pd = pr.json()
            if pd.get("status") in ["completed", "success", "failed"] or "conclusion" in pd or "structural_score" in pd:
                final_data = pd
                print(f"Poll {p+1} => completed")
                break
            else:
                print(f"Poll {p+1} => status={pd.get('status')}")
        else:
            print(f"Poll {p+1} => HTTP {pr.status_code}")
    if not final_data:
        print("Polling timed out")
        sys.exit(1)
else:
    final_data = data

# Save full response
with open("scratch/v2_single_crater_full_response.json", "w") as f:
    json.dump(final_data, f, indent=2, default=str)
print("Full response saved to scratch/v2_single_crater_full_response.json")

# Extract every requested field
diag = final_data.get("mark_diagnostics", {})
trace = diag.get("mark_detector_trace", {})
probe_trace = trace.get("probe", {})
gallery_trace = trace.get("gallery", {})

raw_probe = final_data.get("raw_probe_marks", [])
raw_gallery = final_data.get("raw_gallery_marks", [])
rej_probe = final_data.get("rejected_probe_marks", [])
rej_gallery = final_data.get("rejected_gallery_marks", [])

print("\n" + "="*60)
print("=== REQUIRED DIAGNOSTIC FIELDS ===")
print("="*60)

fields = [
    ("mark_detector_version", final_data.get("mark_detector_version")),
    ("mark_matcher_version", final_data.get("mark_matcher_version")),
    ("raw_probe_marks.length", len(raw_probe)),
    ("raw_gallery_marks.length", len(raw_gallery)),
    ("mark_diagnostics.raw_probe_marks_count", diag.get("raw_probe_marks_count")),
    ("mark_diagnostics.raw_gallery_marks_count", diag.get("raw_gallery_marks_count")),
    ("mark_diagnostics.probe_detector_status", diag.get("probe_detector_status")),
    ("mark_diagnostics.gallery_detector_status", diag.get("gallery_detector_status")),
    ("mark_diagnostics.matcher_status", diag.get("matcher_status")),
    ("probe.input_is_preprocessed", probe_trace.get("input_is_preprocessed")),
    ("probe.internal_clahe_applied", probe_trace.get("internal_clahe_applied")),
    ("probe.initial_candidates", probe_trace.get("initial_candidates")),
    ("probe.dark_lesion_initial_candidates", probe_trace.get("dark_lesion_initial_candidates")),
    ("probe.bright_scar_initial_candidates", probe_trace.get("bright_scar_initial_candidates")),
    ("probe.linear_scar_initial_candidates", probe_trace.get("linear_scar_initial_candidates")),
    ("probe.texture_anomaly_initial_candidates", probe_trace.get("texture_anomaly_initial_candidates")),
    ("probe.structural_depression_initial_candidates", probe_trace.get("structural_depression_initial_candidates")),
    ("probe.structural_depression_area_pass", probe_trace.get("structural_depression_area_pass")),
    ("probe.structural_depression_contrast_relaxed_pass", probe_trace.get("structural_depression_contrast_relaxed_pass")),
    ("probe.structural_depression_final_candidates", probe_trace.get("structural_depression_final_candidates")),
    ("probe.structural_source_used", probe_trace.get("structural_source_used")),
    ("probe.structural_source_dimensions", probe_trace.get("structural_source_dimensions")),
    ("probe.structural_depression_algorithm", probe_trace.get("structural_depression_algorithm")),
    ("probe.structural_depression_n_face_pixels", probe_trace.get("structural_depression_n_face_pixels")),
    ("probe.structural_depression_mask_nonzero_pixels", probe_trace.get("structural_depression_mask_nonzero_pixels")),
    ("probe.structural_depression_score_max", probe_trace.get("structural_depression_score_max")),
    ("probe.structural_depression_score_mean", probe_trace.get("structural_depression_score_mean")),
    ("probe.structural_depression_score_percentile_threshold", probe_trace.get("structural_depression_score_percentile_threshold")),
    ("probe.structural_depression_dog_range", probe_trace.get("structural_depression_dog_range")),
    ("probe.structural_depression_log_range", probe_trace.get("structural_depression_log_range")),
    ("probe.structural_depression_var_range", probe_trace.get("structural_depression_var_range")),
    ("probe.structural_depression_depth_max", probe_trace.get("structural_depression_depth_max")),
    ("probe.structural_depression_depth_mean", probe_trace.get("structural_depression_depth_mean")),
    ("probe.structural_depression_threshold_used", probe_trace.get("structural_depression_threshold_used")),
    ("probe.after_area_filter", probe_trace.get("after_area_filter")),
    ("probe.after_shape_filter", probe_trace.get("after_shape_filter")),
    ("probe.after_region_exclusion", probe_trace.get("after_region_exclusion")),
    ("probe.after_contrast_filter", probe_trace.get("after_contrast_filter")),
    ("probe.strict_final_valid_marks", probe_trace.get("strict_final_valid_marks")),
    ("probe.fallback_used", probe_trace.get("fallback_used")),
    ("probe.fallback_candidates", probe_trace.get("fallback_candidates")),
    ("probe.final_valid_marks", probe_trace.get("final_valid_marks")),
    ("probe.detector_status", probe_trace.get("detector_status")),
    ("gallery.initial_candidates", gallery_trace.get("initial_candidates")),
    ("gallery.structural_depression_initial_candidates", gallery_trace.get("structural_depression_initial_candidates")),
    ("gallery.structural_depression_final_candidates", gallery_trace.get("structural_depression_final_candidates")),
    ("gallery.structural_source_used", gallery_trace.get("structural_source_used")),
    ("gallery.structural_depression_algorithm", gallery_trace.get("structural_depression_algorithm")),
    ("gallery.structural_depression_mask_nonzero_pixels", gallery_trace.get("structural_depression_mask_nonzero_pixels")),
    ("gallery.structural_depression_score_max", gallery_trace.get("structural_depression_score_max")),
    ("gallery.structural_depression_score_mean", gallery_trace.get("structural_depression_score_mean")),
    ("gallery.final_valid_marks", gallery_trace.get("final_valid_marks")),
    ("gallery.detector_status", gallery_trace.get("detector_status")),
]

for name, val in fields:
    print(f"  {name}: {val}")

print(f"\n=== FIRST 10 raw_probe_marks ===")
for i, m in enumerate(raw_probe[:10]):
    print(f"  [{i}] channel={m.get('channel')} type={m.get('mark_type')} area={m.get('area')} contrast={m.get('contrast_score')} centroid={m.get('centroid_px')} confidence={m.get('confidence')}")
if not raw_probe:
    print("  (empty)")

print(f"\n=== FIRST 10 raw_gallery_marks ===")
for i, m in enumerate(raw_gallery[:10]):
    print(f"  [{i}] channel={m.get('channel')} type={m.get('mark_type')} area={m.get('area')} contrast={m.get('contrast_score')} centroid={m.get('centroid_px')} confidence={m.get('confidence')}")
if not raw_gallery:
    print("  (empty)")

print(f"\n=== FIRST 10 rejected_probe_marks ===")
for i, m in enumerate(rej_probe[:10]):
    print(f"  [{i}] channel={m.get('channel')} area={m.get('area')} contrast={m.get('contrast_score')} reason={m.get('rejection_reason')}")
if not rej_probe:
    print("  (empty)")

print(f"\n=== FIRST 10 rejected_gallery_marks ===")
for i, m in enumerate(rej_gallery[:10]):
    print(f"  [{i}] channel={m.get('channel')} area={m.get('area')} contrast={m.get('contrast_score')} reason={m.get('rejection_reason')}")
if not rej_gallery:
    print("  (empty)")

# Decision tree
print("\n" + "="*60)
print("=== DECISION TREE ===")
print("="*60)
sd_init = probe_trace.get("structural_depression_initial_candidates")
sd_mask_px = probe_trace.get("structural_depression_mask_nonzero_pixels")
sd_src = probe_trace.get("structural_source_used")
sd_algo = probe_trace.get("structural_depression_algorithm")
sd_score_max = probe_trace.get("structural_depression_score_max")
sd_score_mean = probe_trace.get("structural_depression_score_mean")
sd_pct_thresh = probe_trace.get("structural_depression_score_percentile_threshold")

if sd_algo is None or sd_algo == "none":
    print("RESULT: structural_depression_algorithm is MISSING or 'none'.")
    print("ACTION: Deployed code does not include v3 changes.")
elif sd_mask_px is not None and sd_mask_px == 0:
    print(f"RESULT: structural_depression_mask_nonzero_pixels = 0")
    print(f"  algorithm = {sd_algo}")
    print(f"  structural_source_used = {sd_src}")
    print(f"  score_max = {sd_score_max}")
    print(f"  score_mean = {sd_score_mean}")
    print(f"  percentile_threshold = {sd_pct_thresh}")
    print("ACTION: No structural pixels above percentile threshold. Review score distribution.")
elif sd_init is not None and sd_init == 0:
    print(f"RESULT: structural_depression_initial_candidates = 0 (mask_px={sd_mask_px})")
    print("ACTION: Mask has pixels but no contours survived _generate_contours.")
elif sd_init is not None and sd_init > 0 and len(raw_probe) == 0:
    print(f"RESULT: structural_depression_initial_candidates = {sd_init} but raw_probe_marks.length = 0")
    print("ACTION: Filtering is dropping all marks. Check rejection details.")
elif len(raw_probe) > 0:
    has_structural = any(m.get("channel") == "structural_depression" for m in raw_probe)
    has_crater_type = any(m.get("mark_type") in ("structural_crater", "depression_scar") for m in raw_probe)
    print(f"RESULT: raw_probe_marks.length = {len(raw_probe)}")
    print(f"  has_structural_depression_channel = {has_structural}")
    print(f"  has_crater_or_depression_type = {has_crater_type}")
    if has_structural:
        print("SUCCESS: Structural depression marks detected in live response!")
    else:
        print("PARTIAL: Marks detected but none from structural_depression channel.")

print("\nDone.")

