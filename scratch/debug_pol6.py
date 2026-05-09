import json
import csv
artifact_path = 'dl_artifact/validation-results-20260507_200820/validation_results.jsonl'
pairs = []
with open(artifact_path, 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if not row.get('error'):
            pairs.append(row)

for p in pairs:
    corrs = p.get('accepted_correspondences_count', 0)
    struct = p.get('structural_sim', 0.0)
    fused = p.get('fused_score', 0.0)
    veto = p.get('veto_triggered', False)
    lr = p.get('lr_marks', 1.0)
    
    if corrs < 3 or struct < 0.40:
        lr = 1.0
        
    bayesian = fused * lr
    dec = bayesian > 50.0 and not veto
    
    if p['label_same_person'] and not dec:
        print(f"FN: {p['pair_id']}, corrs: {corrs}, struct: {struct}, fused: {fused}, veto: {veto}, lr: {lr}, bayesian: {bayesian}")
