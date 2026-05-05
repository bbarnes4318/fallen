import requests
import json
import sys

BASE_URL = "https://facial-backend-vkd6b6ijxa-uk.a.run.app"

# 1. Login
login_resp = requests.post(f"{BASE_URL}/login", json={"password": "aurum-admin-99"})
if login_resp.status_code != 200:
    print("Login failed:", login_resp.text)
    sys.exit(1)

token = login_resp.json().get("access_token")
if not token:
    print("No access token in response:", login_resp.text)
    sys.exit(1)

# 2. Verify
headers = {"Authorization": f"Bearer {token}"}
payload = {
    "probe_url": "gs://hoppwhistle-facial-uploads/test_genuine.jpg",
    "gallery_url": "gs://hoppwhistle-facial-uploads/test_genuine.jpg"
}
resp = requests.post(f"{BASE_URL}/verify/fuse", json=payload, headers=headers)
if resp.status_code != 200:
    print(f"Error: {resp.status_code} {resp.text}")
    sys.exit(1)

data = resp.json()
print("Verification Success")
print(json.dumps(data, indent=2))

# Save the DB ID for lookup
with open('live_verify_result.json', 'w') as f:
    json.dump(data, f)
