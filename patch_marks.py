import sys
import re

def patch_file():
    with open('backend/main.py', 'r', encoding='utf-8') as f:
        content = f.read()

    sig_pattern = r'def detect_facial_marks\(aligned_crop: np\.ndarray, landmarks\) -> tuple\[list, list, np\.ndarray\]:'
    new_sig = 'def detect_facial_marks(aligned_crop: np.ndarray, landmarks) -> tuple[list, list, np.ndarray, dict, dict]:'
    
    if not re.search(sig_pattern, content):
        print("Signature not found!")
        if 'tuple[list, list, np.ndarray, dict, dict]' in content:
            print("Already patched signature.")
        else:
            return

    content = re.sub(sig_pattern, new_sig, content)

    # 1. Initialize trace counters
    init_counters = """    valid_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(occ_mask))

    trace = {
        "initial_candidates": 0,
        "after_skin_mask": 0,
        "after_area_filter": 0,
        "after_shape_filter": 0,
        "after_region_exclusion": 0,
        "after_contrast_filter": 0,
        "final_valid_marks": 0
    }
    overlays = {}"""
    content = content.replace('    valid_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(occ_mask))', init_counters)

    # 2. Add overlays if DEBUG_FORENSIC
    debug_overlays = """    # ── Pass 4: Texture clusters (bilateral filter difference) ──
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
    texture_diff = cv2.absdiff(gray, smoothed)
    _, texture_thresh = cv2.threshold(texture_diff, 12, 255, cv2.THRESH_BINARY)
    texture_masked = cv2.bitwise_and(texture_thresh, valid_mask)
    texture_cleaned = cv2.morphologyEx(texture_masked, cv2.MORPH_OPEN, kernel)
    texture_contours, _ = cv2.findContours(texture_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate initial candidates before mask (approximate by finding contours on raw thresholds)
    dark_initial_cnts, _ = cv2.findContours(cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    light_initial_cnts, _ = cv2.findContours(cv2.morphologyEx(light_thresh, cv2.MORPH_OPEN, kernel), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges_initial = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    linear_initial_cnts, _ = cv2.findContours(edges_initial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    texture_initial_cnts, _ = cv2.findContours(cv2.morphologyEx(texture_thresh, cv2.MORPH_OPEN, kernel), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    trace["initial_candidates"] = len(dark_initial_cnts) + len(light_initial_cnts) + len(linear_initial_cnts) + len(texture_initial_cnts)

    all_contours = [
        (dark_contours, "dark"),
        (light_contours, "light"),
        (linear_contours, "linear_scar"),
        (texture_contours, "texture_cluster"),
    ]
    
    trace["after_skin_mask"] = sum(len(c[0]) for c in all_contours)

    import os
    if os.getenv("DEBUG_FORENSIC") == "true":
        import base64
        _, sm_buf = cv2.imencode('.png', skin_mask)
        overlays["skin_mask_b64"] = f"data:image/png;base64,{base64.b64encode(sm_buf).decode('utf-8')}"
        
        cand_mask = np.zeros((h, w), dtype=np.uint8)
        for cnt_list, _ in all_contours:
            cv2.drawContours(cand_mask, cnt_list, -1, 255, 1)
        _, cm_buf = cv2.imencode('.png', cand_mask)
        overlays["candidate_mask_b64"] = f"data:image/png;base64,{base64.b64encode(cm_buf).decode('utf-8')}"
"""
    pattern_pass4 = r'    # ── Pass 4: Texture clusters \(bilateral filter difference\) ──.*?    all_contours = \[\n        \(dark_contours, "dark"\),\n        \(light_contours, "light"\),\n        \(linear_contours, "linear_scar"\),\n        \(texture_contours, "texture_cluster"\),\n    \]'
    content = re.sub(pattern_pass4, debug_overlays.replace('\\', '\\\\'), content, flags=re.DOTALL)

    # 3. Add trace counters inside the loop
    # area filter
    area_filter = """            # ── Rejection: area bounds ──
            if area < 8:
                continue  # Too small — noise
            if area > 500:
                continue  # Too large — shadow/region artifact
            
            trace["after_area_filter"] += 1"""
    content = re.sub(r'            # ── Rejection: area bounds ──\n            if area < 8:\n                continue  # Too small — noise\n            if area > 500:\n                continue  # Too large — shadow/region artifact', area_filter, content)

    # shape filter
    shape_filter = """            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue  # Invalid geometry — degenerate contour
            
            trace["after_shape_filter"] += 1"""
    content = re.sub(r'            M = cv2\.moments\(cnt\)\n            if M\["m00"\] == 0:\n                continue  # Invalid geometry — degenerate contour', shape_filter, content)

    # region exclusion
    region_filter = """            # Check 3: Contour-mask overlap ratio must be >= 70%
            if rejection_reason is None:
                contour_pixels = np.count_nonzero(mark_mask)
                if contour_pixels > 0:
                    overlap = cv2.bitwise_and(mark_mask, skin_mask)
                    overlap_pixels = np.count_nonzero(overlap)
                    overlap_ratio = overlap_pixels / contour_pixels
                    if overlap_ratio < _MIN_OVERLAP_RATIO:
                        rejection_reason = f"insufficient_face_overlap ({overlap_ratio:.2f})"
                        
            if rejection_reason is None:
                trace["after_region_exclusion"] += 1"""
    content = re.sub(r'            # Check 3: Contour-mask overlap ratio must be >= 70%.*?                        rejection_reason = f"insufficient_face_overlap \(\{overlap_ratio:\.2f\}\)"', region_filter, content, flags=re.DOTALL)

    # contrast filter
    contrast_filter = """            if rejection_reason is not None:
                mark_descriptor["rejection_reason"] = rejection_reason
                # Remove contour before adding to rejected list (not serializable)
                rejected_desc = {k: v for k, v in mark_descriptor.items() if k != "contour"}
                rejected_marks.append(rejected_desc)
                continue
                
            trace["after_contrast_filter"] += 1"""
    content = re.sub(r'            if rejection_reason is not None:\n                mark_descriptor\["rejection_reason"\] = rejection_reason\n                # Remove contour before adding to rejected list \(not serializable\)\n                rejected_desc = {k: v for k, v in mark_descriptor\.items\(\) if k != "contour"}\n                rejected_marks\.append\(rejected_desc\)\n                continue', contrast_filter, content)

    # final valid marks
    final_valid = """            # Mark it used to prevent overlaps
            cv2.drawContours(used_mask, [cnt], -1, 255, -1)
            mark_index += 1
            marks.append(mark_descriptor)
            trace["final_valid_marks"] += 1"""
    content = re.sub(r'            # Mark it used to prevent overlaps\n            cv2\.drawContours\(used_mask, \[cnt\], -1, 255, -1\)\n            mark_index \+= 1\n            marks\.append\(mark_descriptor\)', final_valid, content)

    # return statement
    return_stmt = """    if os.getenv("DEBUG_FORENSIC") == "true":
        rej_overlay = aligned_crop.copy()
        for rm in rejected_marks:
            cv2.rectangle(rej_overlay, (rm["bbox"][0], rm["bbox"][1]), (rm["bbox"][0]+rm["bbox"][2], rm["bbox"][1]+rm["bbox"][3]), (0, 0, 255), 1)
        _, ro_buf = cv2.imencode('.png', rej_overlay)
        overlays["rejected_overlay_b64"] = f"data:image/png;base64,{base64.b64encode(ro_buf).decode('utf-8')}"
        
        fin_overlay = aligned_crop.copy()
        for m in marks:
            cv2.rectangle(fin_overlay, (m["bbox"][0], m["bbox"][1]), (m["bbox"][0]+m["bbox"][2], m["bbox"][1]+m["bbox"][3]), (0, 255, 0), 1)
        _, fo_buf = cv2.imencode('.png', fin_overlay)
        overlays["final_marks_overlay_b64"] = f"data:image/png;base64,{base64.b64encode(fo_buf).decode('utf-8')}"

    return marks, rejected_marks, occ_mask, trace, overlays"""
    content = content.replace('    return marks, rejected_marks, occ_mask', return_stmt)

    with open('backend/main.py', 'w', encoding='utf-8') as f:
        f.write(content)

patch_file()
