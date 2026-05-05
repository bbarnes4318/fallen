"""
Production Forensic Pipeline Verification — API-Only (v1.153.0)
Runs 3 required verification cases against live Cloud Run endpoint.
"""
import requests
import json
import sys
import time

BASE_URL = "https://facial-backend-vkd6b6ijxa-uk.a.run.app"

def login():
    resp = requests.post(f"{BASE_URL}/login", json={"password": "aurum-admin-99"})
    if resp.status_code != 200:
        print(f"LOGIN FAILED: {resp.status_code} {resp.text}")
        sys.exit(1)
    return resp.json()["access_token"]

def verify(token, probe_url, gallery_url, label):
    print(f"\n{'='*60}")
    print(f"CASE: {label}")
    print(f"Probe:   {probe_url}")
    print(f"Gallery: {gallery_url}")
    print(f"{'='*60}")

    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "probe_url": probe_url,
        "gallery_url": gallery_url,
        "require_liveness": False,
        "force_cross_spectral": False,
    }

    resp = requests.post(f"{BASE_URL}/verify/fuse", json=payload, headers=headers, timeout=300)
    if resp.status_code != 200:
        print(f"VERIFY FAILED: {resp.status_code} {resp.text}")
        return None
    data = resp.json()
    print("\n--- API Response ---")
    print(json.dumps(data, indent=2))

    preview = data.get("preview", data)
    score = preview.get("fused_identity_score") or data.get("fused_score", 0)
    conclusion = preview.get("conclusion") or data.get("conclusion", "")
    veto = preview.get("veto_triggered", data.get("veto_triggered", False))
    print(f"\n--- Summary ---")
    print(f"  Score:      {score}")
    print(f"  Conclusion: {conclusion}")
    print(f"  Veto:       {veto}")
    return data

if __name__ == "__main__":
    token = login()
    print(f"Authenticated.")

    # Case A: Same Image
    verify(token, 
        "gs://hoppwhistle-facial-uploads/test_genuine.jpg", 
        "gs://hoppwhistle-facial-uploads/test_genuine.jpg",
        "A. Exact Same Image")
    time.sleep(3)

    # Case B: Same Person, Different Images
    verify(token, 
        "gs://hoppwhistle-facial-uploads/gallery/Colin_Powell_0001.jpg", 
        "gs://hoppwhistle-facial-uploads/gallery/Colin_Powell_0005.jpg",
        "B. Different Images, Same Person (Colin Powell)")
    time.sleep(3)

    # Case C: Different People
    verify(token, 
        "gs://hoppwhistle-facial-uploads/gallery/Colin_Powell_0001.jpg", 
        "gs://hoppwhistle-facial-uploads/gallery/Zoran_Djindjic_0001.jpg",
        "C. Different People (Powell vs Djindjic)")
