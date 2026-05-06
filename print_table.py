import json

results = json.load(open('validation_results.json'))

print('| Test Case | V1 Status | V2 Status | V1 Conclusion | V2 Conclusion | V1 Match Status | V2 Match Status | V1 LR | V2 LR | V1 Marks (P/G) | V2 Marks (P/G) | V2 Accepted | V2 Rejected | V2 Det Status | V2 Match Status | V2 No Double-CLAHE | Errors Logs |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')

def safe_get(d, keys, default='N/A'):
    if d is None: return default
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d

for r in results:
    v1 = r['v1']
    v2 = r['v2']
    
    v1_stat = v1['status']
    v2_stat = v2['status']
    
    v1_conc = safe_get(v1['data'], ['verification_conclusion'])
    v2_conc = safe_get(v2['data'], ['verification_conclusion'])
    
    v1_ms = safe_get(v1['data'], ['mark_match_status'])
    v2_ms = safe_get(v2['data'], ['mark_match_status'])
    
    v1_lr = safe_get(v1['data'], ['lr_marks'])
    v2_lr = safe_get(v2['data'], ['lr_marks'])
    
    v1_pm = safe_get(v1['data'], ['marks_detected_probe'])
    v1_gm = safe_get(v1['data'], ['marks_detected_gallery'])
    if v1_pm == 'N/A':
        v1_pm = len(safe_get(v1['data'], ['raw_probe_marks'], []))
    if v1_gm == 'N/A':
        v1_gm = len(safe_get(v1['data'], ['raw_gallery_marks'], []))
        
    v2_diag = safe_get(v2['data'], ['mark_diagnostics'], {})
    v2_pm = v2_diag.get('raw_probe_marks_count', 'N/A')
    v2_gm = v2_diag.get('raw_gallery_marks_count', 'N/A')
    if v2_pm == 'N/A':
        v2_pm = len(safe_get(v2['data'], ['raw_probe_marks'], []))
    if v2_gm == 'N/A':
        v2_gm = len(safe_get(v2['data'], ['raw_gallery_marks'], []))
        
    v2_acc = v2_diag.get('accepted_correspondences_count', len(safe_get(v2['data'], ['accepted_correspondences'], [])))
    v2_rej = v2_diag.get('rejected_candidates_count', len(safe_get(v2_diag, ['rejected_correspondences'], [])))
    
    v2_det_stat = v2_diag.get('detector_status', 'N/A')
    v2_m_stat = v2_diag.get('matcher_status', 'N/A')
    
    trace = v2_diag.get('mark_detector_trace', {}).get('probe', {})
    no_clahe = trace.get('input_is_preprocessed', False) and not trace.get('internal_clahe_applied', True)
    
    errors = []
    if v1['err']: errors.append(f'V1:{v1["err"]}')
    if v2['err']: errors.append(f'V2:{v2["err"]}')
    err_str = '; '.join(errors) if errors else 'None'
    
    print(f'| {r["name"]} | {v1_stat} | {v2_stat} | {v1_conc} | {v2_conc} | {v1_ms} | {v2_ms} | {v1_lr} | {v2_lr} | {v1_pm}/{v1_gm} | {v2_pm}/{v2_gm} | {v2_acc} | {v2_rej} | {v2_det_stat} | {v2_m_stat} | {no_clahe} | {err_str} |')
