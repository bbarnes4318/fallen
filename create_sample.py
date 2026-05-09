import csv
import os

source_path = r'C:\Users\jimbo\OneDrive\Documents\facial\validation\ci_artifacts\run_25507545407\validation-results-20260507_160934\validation_pairs_lfw_baseline.csv'
target_path = r'C:\Users\jimbo\OneDrive\Documents\facial\validation\validation_pairs_lfw_50_50_sample.csv'

matched_count = 0
mismatched_count = 0
sampled_rows = []
header = None

with open(source_path, 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
    for row in reader:
        # Assuming label_same_person is "true" or "false"
        label = row.get('label_same_person', '').strip().lower()
        if label == 'true' and matched_count < 50:
            sampled_rows.append(row)
            matched_count += 1
        elif label == 'false' and mismatched_count < 50:
            sampled_rows.append(row)
            mismatched_count += 1
        
        if matched_count == 50 and mismatched_count == 50:
            break

with open(target_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(sampled_rows)

print(f"Matched: {matched_count}, Mismatched: {mismatched_count}")
