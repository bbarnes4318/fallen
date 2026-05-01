import requests
import json
import time

API_URL = "https://facial-backend-vkd6b6ijxa-uk.a.run.app/verify/fuse"
# We need to use JWT token. The operator password is 'aurum-admin-99'
# Wait, the auth is a bearer token. Where do we get it?
# /login endpoint
def get_token():
    try:
        resp = requests.post(
            "https://facial-backend-vkd6b6ijxa-uk.a.run.app/login", 
            json={"password": "aurum-admin-99"}
        )
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception as e:
        print("Login failed, trying without token or with generic token...")
        return "dummy_token"

def test_pipeline():
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    print("=========================================")
    print(" LIVE FIRE TEST: Phase 7 Gatekeeper      ")
    print("=========================================\n")
    
    # 1. Test Case A: The Veto (Synthetic Media)
    print("Executing Test Case A: Synthetic Media (VETO Expected)")
    payload_fake = {
        "probe_url": "gs://hoppwhistle-facial-uploads/test_fake.jpg",
        "gallery_url": "gs://hoppwhistle-facial-uploads/test_genuine.jpg"
    }
    
    res_fake = requests.post(API_URL, json=payload_fake, headers=headers)
    print(f"Status Code: {res_fake.status_code}")
    try:
        data = res_fake.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        assert data.get("conclusion") == "VETO: Synthetic Media Detected"
        print("Test Case A Passed: VETO Triggered Successfully!\n")
    except Exception as e:
        print(f"Test Case A Failed: {e}\n")

    # 2. Test Case B: The Pass (Genuine Human Face)
    print("Executing Test Case B: Genuine Media (PASS Expected)")
    payload_real = {
        "probe_url": "gs://hoppwhistle-facial-uploads/test_genuine.jpg",
        "gallery_url": "gs://hoppwhistle-facial-uploads/test_genuine.jpg"
    }
    
    res_real = requests.post(API_URL, json=payload_real, headers=headers)
    print(f"Status Code: {res_real.status_code}")
    try:
        data = res_real.json()
        # Could print partial to avoid huge b64 strings
        print(f"Conclusion: {data.get('conclusion')}")
        print(f"Fused Score: {data.get('fused_identity_score')}")
        assert "VETO: Synthetic Media Detected" not in data.get("conclusion", "")
        print("Test Case B Passed: Gatekeeper cleared genuine media!\n")
    except Exception as e:
        print(f"Test Case B Failed: {e}\n")

if __name__ == "__main__":
    test_pipeline()
