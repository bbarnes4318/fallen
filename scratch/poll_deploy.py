import time, subprocess, sys, json

target = "afcaeaf"
print(f"Waiting for deployment of {target}...")

for i in range(40):
    try:
        out = subprocess.check_output(
            ["gcloud", "run", "services", "describe",
             "facial-backend-v2-marks", "--region", "us-east4", "--format=json"]
        ).decode()
        data = json.loads(out)
        img = data["spec"]["template"]["spec"]["containers"][0]["image"]
        ready = data["status"]["conditions"][0]["status"]
        if target in img and ready == "True":
            print(f"DEPLOYED AND READY: {img}")
            sys.exit(0)
        elif target in img:
            print(f"Image found but not ready yet...")
        else:
            short = img.split(":")[-1][:12]
            print(f"[{i+1}] Still on {short}...")
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(15)

print("Timeout")
sys.exit(1)
