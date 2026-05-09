import csv
import json

artifact_dir = 'dl_artifact/validation-results-20260507_200820'

print('\n--- Mark LR Aggregation Alternatives ---')

models = ['naive_multiplication', 'region_cap', 'distance_decay', 'log_lr_cap', 'composite_clustering']
model_data = {m: {'lrs': [], 'diff_gt_100': 0} for m in models}

with open(f'{artifact_dir}/mark_analysis/mark_lr_distribution.csv', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        m = row['model']
        if m in model_data:
            lr = float(row['model_lr']) if row['model_lr'] else 0.0
            model_data[m]['lrs'].append(lr)
            if lr > 100 and str(row['label_same_person']).lower() == 'false':
                model_data[m]['diff_gt_100'] += 1

for m in models:
    lrs = model_data[m]['lrs']
    lrs.sort()
    median = lrs[len(lrs)//2] if lrs else 0
    gt_100 = sum(1 for lr in lrs if lr > 100)
    gt_1e6 = sum(1 for lr in lrs if lr > 1e6)
    gt_1e12 = sum(1 for lr in lrs if lr > 1e12)
    diff_gt_100 = model_data[m]['diff_gt_100']
    
    max_lr = max(lrs) if lrs else 0
    
    print(f"Model: {m}")
    print(f"  Max LR: {max_lr}")
    print(f"  Median LR: {median}")
    print(f"  Pairs > 100: {gt_100}")
    print(f"  Pairs > 1e6: {gt_1e6}")
    print(f"  Pairs > 1e12: {gt_1e12}")
    print(f"  Different-person pairs with LR > 100: {diff_gt_100}")
    print('')
