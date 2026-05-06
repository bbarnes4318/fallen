import json
for r in json.load(open('validation_results.json')):
    if r['v1']['err']: print(f"V1 {r['name']}:", r['v1']['err'])
    if r['v2']['err']: print(f"V2 {r['name']}:", r['v2']['err'])
