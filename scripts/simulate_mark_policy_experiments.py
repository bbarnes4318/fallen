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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="dl_artifact", help="Directory containing validation_results.jsonl")
    args = parser.parse_args()
    
    artifact_path = os.path.join(args.artifacts_dir, 'validation_results.jsonl')
    
    if not os.path.exists(artifact_path):
        import glob
        matches = glob.glob(f'{args.artifacts_dir}/validation-results-*/validation_results.jsonl')
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            artifact_path = matches[0]

    if not os.path.exists(artifact_path):
        print(f"Waiting for artifact at {artifact_path}...")
        return
        
    pairs = []
    error_count = 0
    with open(artifact_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if not row.get('error'):
                pairs.append(row)
            else:
                error_count += 1

    same_person_corrs = []
    diff_person_corrs = []
    same_person_dist_marks = []
    diff_person_dist_marks = []
    same_person_gen_marks = []
    diff_person_gen_marks = []
    generic_only_diff_count = 0
    gate_c_diff_count = 0

    for p in pairs:
        label = p['label_same_person']
        agg = p.get('evidence_aggregate', {})
        corrs = agg.get('accepted_correspondences_count', 0)
        dist_marks = agg.get('distinctive_mark_count', 0)
        gen_marks = agg.get('generic_mark_count', 0)
        gen_only = agg.get('generic_only_match', False)
        
        if label:
            same_person_corrs.append(corrs)
            same_person_dist_marks.append(dist_marks)
            same_person_gen_marks.append(gen_marks)
        else:
            diff_person_corrs.append(corrs)
            diff_person_dist_marks.append(dist_marks)
            diff_person_gen_marks.append(gen_marks)
            if gen_only:
                generic_only_diff_count += 1
            if dist_marks > 0:
                gate_c_diff_count += 1

    print(f"Total Evaluated: {len(pairs)}")
    print(f"FACE_NOT_DETECTED: {error_count}")
    print(f"Avg Corrs (Same): {avg(same_person_corrs):.2f}")
    print(f"Avg Corrs (Diff): {avg(diff_person_corrs):.2f}")
    print(f"Avg Distinctive Marks (Same): {avg(same_person_dist_marks):.2f}")
    print(f"Avg Distinctive Marks (Diff): {avg(diff_person_dist_marks):.2f}")
    print(f"Avg Generic Marks (Same): {avg(same_person_gen_marks):.2f}")
    print(f"Avg Generic Marks (Diff): {avg(diff_person_gen_marks):.2f}")
    print(f"Diff-Person Pairs with Only Generic Marks: {generic_only_diff_count}")
    print(f"Diff-Person Pairs passing Gate C: {gate_c_diff_count}")
    print()
    
    def log_cap(lr, cap=100.0):
        if lr <= 1.0: return lr
        return min(lr, cap)

    def p1_baseline(p):
        return True, p.get('lr_marks', 1.0)

    def p2_log_cap(p):
        return True, log_cap(p.get('lr_marks', 1.0), 100.0)

    def p3_log_cap_face_floor(p):
        if p.get('structural_sim', 0.0) < 0.40: return False, 1.0
        return True, log_cap(p.get('lr_marks', 1.0), 100.0)

    def p4_region_cap(p):
        agg = p.get('evidence_aggregate', {})
        regions = agg.get('distinct_face_regions_count', 0)
        lr = p.get('lr_marks', 1.0)
        if regions < 2: return True, min(lr, 10.0)
        return True, log_cap(lr, 100.0)

    def p5_review_only(p):
        return False, 1.0

    def p6_strict_gate(p):
        agg = p.get('evidence_aggregate', {})
        corrs = agg.get('accepted_correspondences_count', 0)
        avg_dist = agg.get('average_position_distance', 999.0)
        regions = agg.get('distinct_face_regions_count', 0)
        if corrs >= 3 and avg_dist <= 20.0 and regions >= 2:
            return True, log_cap(p.get('lr_marks', 1.0), 100.0)
        return False, 1.0

    def p7_gate_c(p):
        agg = p.get('evidence_aggregate', {})
        dist_count = agg.get('distinctive_mark_count', 0)
        if dist_count == 0:
            return False, 1.0
        return True, p.get('lr_marks', 1.0)

    def p8_gate_c_face_region(p):
        agg = p.get('evidence_aggregate', {})
        dist_count = agg.get('distinctive_mark_count', 0)
        regions = agg.get('distinct_face_regions_count', 0)
        struct = p.get('structural_sim', 0.0)
        if dist_count == 0 or regions < 2 or struct < 0.40:
            return False, 1.0
        return True, p.get('lr_marks', 1.0)

    def p9_gate_c_face_region_dist(p):
        agg = p.get('evidence_aggregate', {})
        dist_count = agg.get('distinctive_mark_count', 0)
        regions = agg.get('distinct_face_regions_count', 0)
        avg_dist = agg.get('average_position_distance', 999.0)
        struct = p.get('structural_sim', 0.0)
        if dist_count == 0 or regions < 2 or struct < 0.40 or avg_dist > 20.0:
            return False, 1.0
        return True, p.get('lr_marks', 1.0)

    policies = [
        ('Current baseline', p1_baseline),
        ('Log LR cap only', p2_log_cap),
        ('Log LR cap + face floor', p3_log_cap_face_floor),
        ('Region cap + log cap', p4_region_cap),
        ('Mark evidence as review only', p5_review_only),
        ('Strict acceptance gate', p6_strict_gate),
        ('Gate C telemetry only', p7_gate_c),
        ('Gate C + face floor + region diversity', p8_gate_c_face_region),
        ('Gate C + face floor + region diversity + distance cap', p9_gate_c_face_region_dist)
    ]
    
    for name, func in policies:
        tp, fp, tn, fn = 0, 0, 0, 0
        diff_gt_100 = 0
        diff_dist_count = 0
        diff_gen_only = 0
        review_count = 0
        fps = []
        fns = []
        
        for p in pairs:
            label = p['label_same_person']
            struct = p.get('structural_sim', 0.0)
            
            # Reconstruct original decision (before mark LR was applied, if doing Bayesian)
            # Actually, the original fused score included the old mark LR.
            # We want to re-fuse it.
            lr_ensemble = p.get('lr_ensemble', 1.0)
            veto = p.get('veto_triggered', False)
            
            pass_gate, lr = func(p)
            
            lr_total = lr_ensemble * lr
            PRIOR = 0.5
            posterior = (PRIOR * lr_total) / ((PRIOR * lr_total) + (1.0 - PRIOR))
            fused = posterior * 100.0
            
            # The override logic in production:
            veto_override_applied = False
            if veto:
                # If mark match status matched, override (simplified)
                # Production veto override depends on mark_result which we don't fully simulate.
                # Let's assume veto override applies if lr > 1.0 and pass_gate is True.
                if pass_gate and lr > 1.0:
                    veto_override_applied = True
            
            dec = fused > 50.0
            if veto and not veto_override_applied:
                dec = False
                
            agg = p.get('evidence_aggregate', {})
            dist_marks = agg.get('distinctive_mark_count', 0)
            gen_only = agg.get('generic_only_match', False)
            
            if not label:
                if lr > 100.0: diff_gt_100 += 1
                if dist_marks > 0: diff_dist_count += 1
                if gen_only: diff_gen_only += 1
                
            # If the score is marginal, it might be sent for manual review
            if 50.0 < fused <= 75.0:
                review_count += 1
                
            if label and dec: tp += 1
            elif label and not dec: 
                fn += 1
                fns.append(p['pair_id'])
            elif not label and dec: 
                fp += 1
                fps.append(p['pair_id'])
            elif not label and not dec: tn += 1
            
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"--- {name} ---")
        print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
        print(f"FAR: {far:.4f} | FRR: {frr:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"Human Review Count: {review_count}")
        print(f"Diff-Person LR > 100: {diff_gt_100}")
        print(f"Diff-Person with Distinctive Marks: {diff_dist_count}")
        print(f"Diff-Person with Generic-Only Matches: {diff_gen_only}")
        print(f"False Positives: {fps}")
        print(f"False Negatives: {fns}")
        print()

if __name__ == '__main__':
    simulate()
