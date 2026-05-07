import json
import csv
import os

def avg(lst): return sum(lst)/len(lst) if lst else 0.0

def safe_float(v, default=0.0):
    try:
        return float(v)
    except:
        return default

def simulate():
    artifact_path = 'dl_artifact/validation_results.jsonl'
    
    # Try alternate path
    if not os.path.exists(artifact_path):
        import glob
        matches = glob.glob('dl_artifact/validation-results-*/validation_results.jsonl')
        if matches:
            # Sort to get the most recent one
            matches.sort(key=os.path.getmtime, reverse=True)
            artifact_path = matches[0]

    if not os.path.exists(artifact_path):
        print(f"Waiting for artifact at {artifact_path}...")
        return
        
    pairs = []
    with open(artifact_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if not row.get('error'):
                pairs.append(row)

    same_person_corrs = []
    diff_person_corrs = []
    same_person_dist = []
    diff_person_dist = []
    same_person_regions = []
    diff_person_regions = []
    generic_only_diff_count = 0

    generic_types = {'freckle', 'pore', 'texture_anomaly'}
    distinctive_types = {'mole', 'scar', 'linear_scar', 'structural_crater', 'depression_scar'}

    for p in pairs:
        label = p['label_same_person']
        agg = p.get('evidence_aggregate', {})
        corrs = agg.get('accepted_correspondences_count', 0)
        dist = agg.get('average_position_distance', 0.0)
        regions = agg.get('distinct_face_regions_count', 0)
        
        if label:
            same_person_corrs.append(corrs)
            same_person_dist.append(dist)
            same_person_regions.append(regions)
        else:
            diff_person_corrs.append(corrs)
            diff_person_dist.append(dist)
            diff_person_regions.append(regions)
            
            top_types = agg.get('top_5_mark_types', [])
            has_distinctive = any(t not in generic_types for t in top_types)
            if not has_distinctive:
                generic_only_diff_count += 1

    print(f"Total Evaluated: {len(pairs)}")
    print(f"FACE_NOT_DETECTED: {100 - len(pairs)}") # assuming 100 manifest pairs
    print(f"Avg Corrs (Same): {avg(same_person_corrs):.2f}")
    print(f"Avg Corrs (Diff): {avg(diff_person_corrs):.2f}")
    print(f"Avg Dist (Same): {avg(same_person_dist):.2f}")
    print(f"Avg Dist (Diff): {avg(diff_person_dist):.2f}")
    print(f"Avg Regions (Same): {avg(same_person_regions):.2f}")
    print(f"Avg Regions (Diff): {avg(diff_person_regions):.2f}")
    print(f"Diff-Person Pairs with Only Generic Marks: {generic_only_diff_count}")
    
    # 1. Baseline
    def baseline(p):
        return True, p.get('lr_marks', 1.0)
        
    # 2. Gate A
    def gate_a(p):
        agg = p.get('evidence_aggregate', {})
        struct = p.get('structural_sim', 0.0)
        corrs = agg.get('accepted_correspondences_count', 0)
        avg_dist = agg.get('average_position_distance', 999.0)
        regions = agg.get('distinct_face_regions_count', 0)
        largest_region = agg.get('largest_single_region_correspondence_count', 0)
        
        if struct >= 0.40 and corrs >= 3 and avg_dist <= 20.0 and regions >= 2 and largest_region <= (corrs * 0.5):
            return True, p.get('lr_marks', 1.0)
        return False, 1.0
        
    # 3. Gate B
    def gate_b(p):
        pass_a, lr = gate_a(p)
        if not pass_a: return False, 1.0
        
        agg = p.get('evidence_aggregate', {})
        top_types = agg.get('top_5_mark_types', [])
        
        has_distinctive = False
        for t in top_types:
            if t not in generic_types:
                has_distinctive = True
                break
                
        if not has_distinctive:
            return False, 1.0
        return True, lr
        
    # 4. Gate C
    def gate_c(p):
        agg = p.get('evidence_aggregate', {})
        top_types = agg.get('top_5_mark_types', [])
        has_distinctive = False
        for t in top_types:
            if t in distinctive_types:
                has_distinctive = True
                break
                
        if not has_distinctive:
            return False, 1.0
        return True, p.get('lr_marks', 1.0)
        
    # 5. Gate D
    def gate_d(p):
        struct = p.get('structural_sim', 0.0)
        # return 'review' if struct < 0.60 else 'auto' -> simulated by dropping LR to 1 for auto decision
        if struct < 0.60:
            return False, 1.0
        return True, p.get('lr_marks', 1.0)

    results = []
    
    for name, func in [('baseline', baseline), ('gate_a', gate_a), ('gate_b', gate_b), ('gate_c', gate_c), ('gate_d', gate_d)]:
        tp, fp, tn, fn = 0, 0, 0, 0
        diff_gt_100 = 0
        for p in pairs:
            label = p['label_same_person']
            struct = p.get('structural_sim', 0.0)
            fused = p.get('fused_score', 0.0)
            veto = p.get('veto_triggered', False)
            
            pass_gate, lr = func(p)
                
            bayesian = fused * lr
            dec = bayesian > 50.0 and not veto
            
            if lr > 100 and not label: diff_gt_100 += 1
            if label and dec: tp += 1
            elif label and not dec: fn += 1
            elif not label and dec: fp += 1
            elif not label and not dec: tn += 1
            
        print(f"--- {name} ---")
        print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"Diff-Person LR > 100: {diff_gt_100}")
        print()

if __name__ == '__main__':
    simulate()
