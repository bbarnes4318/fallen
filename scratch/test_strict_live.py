"""Quick isolated live test for strict V2 Cloud Run service."""
import requests
import time
import json
import sys

BASE_URL = "https://facial-backend-v2-strict-marks-196207148120.us-east4.run.app"
OP_PASS = "aurum-admin-99"

TESTS = [
    {
        "name": "SELF_MATCH",
        "desc": "Exact same image (Aaron Eckhart)",
        "image1": "gs://hoppwhistle-facial-uploads/gallery/Aaron_Eckhart_0001.jpg",
        "image2": "gs://hoppwhistle-facial-uploads/gallery/Aaron_Eckhart_0001.jpg",
    },
    {
        "name": "DIFFERENT_PERSON",
        "desc": "Aaron Eckhart vs George W Bush",
        "image1": "gs://hoppwhistle-facial-uploads/gallery/Aaron_Eckhart_0001.jpg",
        "image2": "gs://hoppwhistle-facial-uploads/gallery/George_W_Bush_0001.jpg",
    },
    {
        "name": "SAME_PERSON_DIFF_IMAGE",
        "desc": "George W Bush #1 vs #2 (same person, different photos)",
        "image1": "gs://hoppwhistle-facial-uploads/gallery/George_W_Bush_0001.jpg",
        "image2": "gs://hoppwhistle-facial-uploads/gallery/George_W_Bush_0002.jpg",
    },
]


def get_jwt():
    """Login to get operator JWT."""
    print("Authenticating...")
    r = requests.post(f"{BASE_URL}/login", json={"password": OP_PASS}, timeout=30)
    if r.status_code != 200:
        print(f"  Login failed: {r.status_code} {r.text[:300]}")
        sys.exit(1)
    token = r.json().get("access_token")
    print(f"  JWT obtained (length={len(token)})")
    return token


def run_test(test, jwt_token):
    print(f"\n{'='*60}")
    print(f"TEST: {test['name']} — {test['desc']}")
    print(f"{'='*60}")

    body = {
        "gallery_url": test["image1"],
        "probe_url": test["image2"],
    }

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }

    t0 = time.time()
    try:
        r = requests.post(f"{BASE_URL}/verify/fuse", json=body, headers=headers, timeout=180)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    elapsed = int((time.time() - t0) * 1000)
    print(f"  HTTP: {r.status_code}  Elapsed: {elapsed}ms")

    if r.status_code != 200:
        print(f"  Error: {r.text[:500]}")
        return

    d = r.json()

    # The API returns {job_id, locked, preview} — full data is behind payment.
    # Preview has the summary we need.
    print(f"  job_id: {d.get('job_id')}")
    print(f"  locked: {d.get('locked')}")

    preview = d.get("preview", {})
    print(f"  --- PREVIEW ---")
    for k, v in sorted(preview.items()):
        if isinstance(v, str) and len(v) > 100:
            print(f"    {k}: [str len={len(v)}]")
        else:
            print(f"    {k}: {v}")

    # Also try to unlock via the job endpoint for full telemetry
    job_id = d.get("job_id")
    if job_id:
        r2 = requests.get(
            f"{BASE_URL}/verify/result/{job_id}?password={OP_PASS}",
            headers={"Authorization": f"Bearer {jwt_token}"},
            timeout=30,
        )
        if r2.status_code == 200:
            full = r2.json()
            print(f"  --- FULL RESULT (unlocked) ---")
            print(f"    conclusion: {full.get('conclusion')}")
            print(f"    fused_identity_score: {full.get('fused_identity_score')}")
            print(f"    mark_match_status: {full.get('mark_match_status')}")
            print(f"    lr_marks: {full.get('lr_marks')}")
            print(f"    mark_matcher_version: {full.get('mark_matcher_version')}")
            print(f"    marks_detected_probe: {full.get('marks_detected_probe')}")
            print(f"    marks_detected_gallery: {full.get('marks_detected_gallery')}")
            print(f"    marks_matched: {full.get('marks_matched')}")
            print(f"    correspondences: {len(full.get('correspondences', []))}")
            print(f"    bayesian_fused_score: {full.get('bayesian_fused_score')}")
            print(f"    veto_triggered: {full.get('veto_triggered')}")

            al = full.get("audit_log", {}) or {}
            if al:
                print(f"    audit.lr_marks: {al.get('lr_marks')}")
                print(f"    audit.lr_total: {al.get('lr_total')}")
                print(f"    audit.posterior: {al.get('posterior_probability')}")
                print(f"    audit.mark_matcher_version: {al.get('mark_matcher_version')}")
        else:
            print(f"  Full result fetch failed: {r2.status_code} {r2.text[:200]}")

    return d


if __name__ == "__main__":
    print(f"Service: {BASE_URL}")
    print(f"Testing strict V2 mark matcher live service...\n")

    jwt_token = get_jwt()

    for test in TESTS:
        run_test(test, jwt_token)

    print(f"\n{'='*60}")
    print("All tests complete.")
    print(f"{'='*60}")
