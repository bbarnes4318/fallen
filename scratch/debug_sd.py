import json
with open('scratch/autopsy_output_after_structural_fix/autopsy_report.json') as f:
    r = json.load(f)

# check marks
sd_marks = [m for m in r['marks'] if m.get('channel') == 'structural_depression']
print(f"Accepted SD marks: {len(sd_marks)}")

# check rejected
sd_rej = [m for m in r.get('rejected', []) if m.get('channel') == 'structural_depression']
print(f"Rejected SD marks: {len(sd_rej)}")

for m in sd_rej:
    print("Rejected:", m['rejection_reason'], "Area:", m['area'], "Contrast:", m['contrast_score'])
