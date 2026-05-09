import requests
import json
import sys
import time

BASE_URL = "https://facial-backend-v2-strict-marks-196207148120.us-east4.run.app"

# 1. Login
print("Logging in to", BASE_URL)
login_resp = requests.post(f"{BASE_URL}/login", json={"password": "aurum-admin-99"})
print("Login status:", login_resp.status_code)
if login_resp.status_code != 200:
    print("Login failed:", login_resp.text)
    sys.exit(1)

token = login_resp.json().get("access_token")

# 2. Verify
headers = {"Authorization": f"Bearer {token}"}
payload = {
    "probe_url": "gs://hoppwhistle-facial-uploads/test_genuine.jpg",
    "gallery_url": "gs://hoppwhistle-facial-uploads/test_genuine.jpg"
}
print("Starting verify fuse...")
resp = requests.post(f"{BASE_URL}/verify/fuse", json=payload, headers=headers)
print("Verify fuse status:", resp.status_code)
if resp.status_code != 200:
    print(f"Error starting job: {resp.status_code} {resp.text}")
    sys.exit(1)

data = resp.json()
job_id = data.get("job_id")
print("Job started:", job_id)

# 3. Poll for result
for _ in range(30):
    time.sleep(2)
    poll_resp = requests.get(f"{BASE_URL}/verify/result/{job_id}", headers=headers, params={'bypass_code': 'aurum-admin-99'})
    if poll_resp.status_code == 200:
        result_data = poll_resp.json()
        if result_data.get("status") == "completed" or "audit_log" in result_data:
            print("Job completed!")
            with open('live_verify_result.json', 'w') as f:
                json.dump(result_data, f, indent=2)
            sys.exit(0)
        else:
            print("Still processing...", result_data.get("status"))
    else:
        print(f"Poll failed: {poll_resp.status_code} {poll_resp.text}")
        
print("Timeout waiting for job")
sys.exit(1)
