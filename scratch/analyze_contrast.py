import json

with open('scratch/autopsy_output/autopsy_report.json', 'r') as f:
    r = json.load(f)

lc = [c for c in r['rejected'] if c['rejection_reason'] == 'low_contrast']
print(f"Total low_contrast rejected: {len(lc)}")
for i, c in enumerate(lc[:20]):
    print(f"[{i}] area: {c['area']}, contrast: {c['contrast']:.2f}, channel: {c['channel']}")
