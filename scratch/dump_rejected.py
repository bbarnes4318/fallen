import json

with open('scratch/autopsy_output/autopsy_report.json', 'r') as f:
    r = json.load(f)

for c in r['rejected']:
    if c['rejection_reason'] == 'low_contrast' and c['channel'] == 'dark_lesion':
        print(c)
