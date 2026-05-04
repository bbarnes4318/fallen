import sys

def patch_file():
    with open('backend/main.py', 'r', encoding='utf-8') as f:
        content = f.read()

    target = """        rejection_summary = _build_rejection_summary(
            valid_probe_marks, valid_gallery_marks,
            mark_result, rejected_cands, mark_match_status,
            exact_image_match, TIER4_CALIBRATION,
        )"""
    
    replacement = """        rejection_summary = _build_rejection_summary(
            valid_probe_marks, valid_gallery_marks,
            mark_result, rejected_cands, mark_match_status,
            exact_image_match, TIER4_CALIBRATION,
            trace_probe=trace_probe, trace_gallery=trace_gallery,
        )"""
    
    content = content.replace(target, replacement)
    
    with open('backend/main.py', 'w', encoding='utf-8') as f:
        f.write(content)

patch_file()
