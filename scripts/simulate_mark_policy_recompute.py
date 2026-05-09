import json
import os

def avg(lst): return sum(lst)/len(lst) if lst else 0.0

def simulate():
    artifact_path = 'dl_artifact/validation_results.jsonl'
    
    if not os.path.exists(artifact_path):
        import glob
        matches = glob.glob('dl_artifact/validation-results-*/validation_results.jsonl')
        if matches:
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
    generic_only_diff_count = 0

    # For channels, generic vs distinctive:
    # generic: dark, light, dark_lesion
    # distinctive: bright_scar, linear_scar_v2, structural_crater, structural_depression, bright_scar_v2
    generic_channels = {'dark', 'light', 'dark_lesion'}

    for p in pairs:
        label = p['label_same_person']
        
        # Recompute accepted mark types/channels
        accepted_details = p.get('accepted_correspondences_detail', [])
        raw_probe = p.get('raw_probe_marks_summary', [])
        
        accepted_channels = []
        for c in accepted_details:
            pidx = c.get('probe_idx')
            if pidx is not None and pidx < len(raw_probe):
                ch = raw_probe[pidx].get('channel')
                if ch:
                    accepted_channels.append(ch)
                    
        # Update our metrics based on recomputed data
        corrs = len(accepted_details)
        avg_dist = avg([c.get('position_distance', 0) for c in accepted_details])
        
        # Save recomputed data back into the pair dict so gates can use it
        p['_recomputed'] = {
            'corrs': corrs,
            'avg_dist': avg_dist,
            'accepted_channels': accepted_channels
        }
        
        if label:
            same_person_corrs.append(corrs)
            same_person_dist.append(avg_dist)
        else:
            diff_person_corrs.append(corrs)
            diff_person_dist.append(avg_dist)
            
            has_distinctive = any(ch not in generic_channels for ch in accepted_channels)
            if not has_distinctive:
                generic_only_diff_count += 1

    print(f"Total Evaluated: {len(pairs)}")
    print(f"FACE_NOT_DETECTED: {100 - len(pairs)}") 
    print(f"Avg Corrs (Same): {avg(same_person_corrs):.2f}")
    print(f"Avg Corrs (Diff): {avg(diff_person_corrs):.2f}")
    print(f"Avg Dist (Same): {avg(same_person_dist):.2f}")
    print(f"Avg Dist (Diff): {avg(diff_person_dist):.2f}")
    print(f"Diff-Person Pairs with Only Generic Marks: {generic_only_diff_count}")
    
    # 1. Baseline
    def baseline(p):
        return True, p.get('lr_marks', 1.0)
        
    # 2. Gate A: Requires at least 3 marks, tight distance
    def gate_a(p):
        struct = p.get('structural_sim', 0.0)
        r = p['_recomputed']
        
        if struct >= 0.40 and r['corrs'] >= 3 and r['avg_dist'] <= 20.0:
            return True, p.get('lr_marks', 1.0)
        return False, 1.0
        
    # 3. Gate B: Gate A + requires at least one distinctive mark
    def gate_b(p):
        pass_a, lr = gate_a(p)
        if not pass_a: return False, 1.0
        
        r = p['_recomputed']
        has_distinctive = any(ch not in generic_channels for ch in r['accepted_channels'])
                
        if not has_distinctive:
            return False, 1.0
        return True, lr
        
    # 4. Gate C: Just requires at least one distinctive mark
    def gate_c(p):
        r = p['_recomputed']
        has_distinctive = any(ch not in generic_channels for ch in r['accepted_channels'])
                
        if not has_distinctive:
            return False, 1.0
        return True, p.get('lr_marks', 1.0)
        
    # 5. Gate D: Only allow mark LR if struct >= 0.60
    def gate_d(p):
        struct = p.get('structural_sim', 0.0)
        if struct < 0.60:
            return False, 1.0
        return True, p.get('lr_marks', 1.0)

    for name, func in [('baseline', baseline), ('gate_a', gate_a), ('gate_b', gate_b), ('gate_c', gate_c), ('gate_d', gate_d)]:
        tp, fp, tn, fn = 0, 0, 0, 0
        diff_gt_100 = 0
        for p in pairs:
            label = p['label_same_person']
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
