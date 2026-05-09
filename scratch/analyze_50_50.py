import json
import csv

artifact_dir = 'dl_artifact/validation-results-20260507_200820'

# 1. False negatives
print('--- False Negatives ---')
with open(f'{artifact_dir}/false_negatives.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Pair ID: {row.get('pair_id')}")
        print(f"  Label: {row.get('label_same_person')}")
        print(f"  structural_sim: {row.get('structural_sim')}")
        print(f"  arcface_sim: {row.get('arcface_sim')}")
        print(f"  facenet_sim: {row.get('facenet_sim')}")
        print(f"  fused_score: {row.get('fused_score')}")
        print(f"  bayesian_fused_score: {row.get('bayesian_fused_score')}")
        print(f"  veto_triggered: {row.get('veto_triggered')}")
        print(f"  veto_reason: {row.get('veto_reason')}")
        print(f"  lr_ensemble: {row.get('lr_ensemble')}")
        print(f"  lr_marks: {row.get('lr_marks')}")
        print(f"  raw_probe_marks_count: {row.get('raw_probe_marks_count')}")
        print(f"  raw_gallery_marks_count: {row.get('raw_gallery_marks_count')}")
        print(f"  accepted_correspondences_count: {row.get('accepted_correspondences_count')}")
        print(f"  conclusion: {row.get('conclusion')}")
        print('')

# 2. FACE_NOT_DETECTED failures
print('--- FACE_NOT_DETECTED Failures ---')
with open(f'{artifact_dir}/validation_results.jsonl', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if row.get('error') and 'FACE_NOT_DETECTED' in row['error']:
            print(f"Pair ID: {row.get('pair_id')}")
            print(f"  label: {row.get('label_same_person')}")
            print(f"  category: {row.get('category')}")
            print(f"  error: {row.get('error')}")

# 3. Conflicting evidence cases
print('--- Conflicting Evidence Cases ---')
conflicts = {'face_weak_marks_strong': [], 'face_strong_marks_weak': []}
with open(f'{artifact_dir}/conflicting_evidence_cases.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        conflict_type = row.get('conflict_type')
        if conflict_type in conflicts:
            conflicts[conflict_type].append(row)

for c_type, rows in conflicts.items():
    print(f"\nGroup: {c_type}")
    print(f"  Count: {len(rows)}")
    same_count = sum(1 for r in rows if str(r.get('label_same_person')).lower() == 'true')
    diff_count = len(rows) - same_count
    print(f"  Same-person: {same_count}, Different-person: {diff_count}")
    print(f"  Representative pair_ids: {[r.get('pair_id') for r in rows[:3]]}")
    # Did the hard safety rule prevent false positives?
    # Hard rule: Score > 50 and NO veto. If different person (diff_count) and hard rule would have said Match, that's an FP.
    fp_prevented = 0
    for r in rows:
        if str(r.get('label_same_person')).lower() == 'false':
            score = float(r.get('fused_score', 0))
            veto = str(r.get('veto_triggered')).lower() == 'true'
            # If the base score was high but veto stopped it, it prevented an FP.
            if score > 50.0 and veto:
                fp_prevented += 1
    print(f"  False positives prevented by hard rule/veto: {fp_prevented}")

# 4. Mark LR explosion analysis
print('\n--- Mark LR Explosion Analysis ---')
all_results = []
with open(f'{artifact_dir}/validation_results.jsonl', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if not row.get('error'):
            all_results.append(row)

top_marks = sorted(all_results, key=lambda x: x.get('lr_marks', 1.0), reverse=True)[:10]
for i, r in enumerate(top_marks):
    print(f"Rank {i+1}: lr_marks = {r.get('lr_marks')}")
    print(f"  Pair ID: {r.get('pair_id')}")
    print(f"  Label: {r.get('label_same_person')}")
    print(f"  accepted_correspondences_count: {r.get('accepted_correspondences_count')}")
    print(f"  raw_probe_marks_count: {r.get('raw_probe_marks_count')}")
    print(f"  raw_gallery_marks_count: {r.get('raw_gallery_marks_count')}")
    # Getting contributing types from evidence if present
    evidence = r.get('evidence', [])
    mark_ev = next((e for e in evidence if e.get('type') == 'mark_match'), None)
    if mark_ev and 'details' in mark_ev and 'correspondences' in mark_ev['details']:
        corrs = mark_ev['details']['correspondences']
        top_corrs = sorted(corrs, key=lambda x: x.get('lr', 1.0), reverse=True)[:3]
        print(f"  Top 3 mark types/LRs: {[(c.get('type', 'unknown'), c.get('lr', 1.0)) for c in top_corrs]}")

# 5. Mark LR aggregation alternatives
print('\n--- Mark LR Aggregation Alternatives ---')
with open(f'{artifact_dir}/mark_analysis/mark_lr_comparison.json', encoding='utf-8') as f:
    comparisons = json.load(f)
    for model_name, model_data in comparisons.items():
        print(f"Model: {model_name}")
        print(f"  Max LR: {model_data.get('max_fused_lr')}")
        print(f"  Median LR: {model_data.get('average_fused_lr')} (Note: metric says average but usually median is preferred)")

# Number of pairs > 100, 1e6, 1e12
import math
with open(f'{artifact_dir}/mark_analysis/mark_lr_distribution.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    
    # We'll map columns. naive_lr, capped_100_lr, decay_0_9_lr, log_cap_lr, cluster_lr
    models = {
        'naive_multiplication': 'naive_lr',
        'region_cap': 'capped_100_lr',
        'distance_decay': 'decay_0_9_lr',
        'log_lr_cap': 'log_cap_lr',
        'composite_clustering': 'cluster_lr'
    }
    
    for model_name, col in models.items():
        if col not in rows[0]:
            print(f"Column {col} not found for model {model_name}")
            continue
            
        lrs = [float(r[col]) for r in rows if r.get(col)]
        lrs.sort()
        if lrs:
            median = lrs[len(lrs)//2]
        else:
            median = 0
            
        gt_100 = sum(1 for lr in lrs if lr > 100)
        gt_1e6 = sum(1 for lr in lrs if lr > 1e6)
        gt_1e12 = sum(1 for lr in lrs if lr > 1e12)
        
        # Any different person pairs with extreme LR? Let's check > 100
        extreme_diff = sum(1 for r in rows if float(r.get(col, 0)) > 100 and str(r.get('label_same_person')).lower() == 'false')
        
        print(f"Model: {model_name} (from CSV)")
        print(f"  Median LR: {median}")
        print(f"  Pairs > 100: {gt_100}")
        print(f"  Pairs > 1e6: {gt_1e6}")
        print(f"  Pairs > 1e12: {gt_1e12}")
        print(f"  Different-person pairs with LR > 100: {extreme_diff}")
        print('')
