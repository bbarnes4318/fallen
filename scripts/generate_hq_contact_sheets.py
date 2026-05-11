import json
import os
import math
import subprocess
import sys
import csv
import argparse
from PIL import Image, ImageDraw, ImageFont

def get_basename(path):
    if not path: return ""
    return path.split("/")[-1]

def download_gcs_file(gcs_uri, local_path):
    if os.path.exists(local_path): 
        return True, ""
    gsutil_cmd = "gsutil.cmd" if sys.platform == "win32" else "gsutil"
    try:
        res = subprocess.run([gsutil_cmd, "cp", gcs_uri, local_path], capture_output=True, text=True, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def get_non_white_pixels(img):
    gray = img.convert("L")
    data = list(gray.getdata())
    return sum(1 for p in data if p < 255)

def create_contact_sheets(input_jsonl, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    crop_tmp_dir = os.path.join(output_dir, "tmp_crops")
    os.makedirs(crop_tmp_dir, exist_ok=True)
    
    crop_targets = {
        "retained_light_scar": [],
        "suppressed_light_scar": [],
        "same_person_light_scar_corresps": [],
        "impostor_light_scar_corresps": [],
        "retained_dark_spot_strict": [],
        "suppressed_dark_spot_strict": []
    }

    manifest_map = {}
    import glob
    for manifest_path in glob.glob("validation/validation_pairs*.csv"):
        try:
            with open(manifest_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "pair_id" in row and "image2_url_or_gcs_path" in row:
                        manifest_map[row["pair_id"]] = {
                            "probe": row.get("image1_url_or_gcs_path", ""),
                            "gallery": row.get("image2_url_or_gcs_path", ""),
                            "is_same": str(row.get("label_same_person", "")).lower() in ["true", "1", "yes"]
                        }
        except Exception as e:
            print(f"Skipping {manifest_path}: {e}")

    pairs = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))

    for p in pairs:
        pair_id = p.get("pair_id", "unknown")
        is_same = p.get("label_same_person", False)
        image_path = manifest_map.get(pair_id, {}).get("gallery", "")
        
        raw_marks = p.get("raw_gallery_marks_summary", [])
        
        def distance(m1, m2):
            c1 = m1.get("centroid", [0, 0])
            c2 = m2.get("centroid", [0, 0])
            return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

        # 1. Dark Spot Strict
        for m in raw_marks:
            if m.get("mark_type") != "dark_spot": continue
            area = m.get("area") or 0
            salience = m.get("salience") or 0
            isolated = m.get("isolated", False)
            keep = True
            reason = ""
            if area < 15: keep, reason = False, "dark_spot_strict_area"
            elif salience > 0 and salience < 3.0: keep, reason = False, "dark_spot_strict_salience"
            elif not isolated: keep, reason = False, "dark_spot_strict_cluster"
            
            if keep:
                crop_targets["retained_dark_spot_strict"].append({"mark": m, "img": image_path, "pair_id": pair_id, "is_same": is_same, "variant": "dark_spot_strict", "suppression_reason": ""})
            else:
                m_sup = dict(m)
                m_sup["suppression_reason"] = reason
                crop_targets["suppressed_dark_spot_strict"].append({"mark": m_sup, "img": image_path, "pair_id": pair_id, "is_same": is_same, "variant": "dark_spot_strict"})

        # 2. Light Scar Cluster Aware
        ls_retained_phase1 = []
        for m in raw_marks:
            if m.get("mark_type") != "light_scar": continue
            area = m.get("area") or 0
            salience = m.get("salience") or 0
            confidence = m.get("confidence") or 0
            keep = True
            reason = ""
            if area < 10: keep, reason = False, "area_too_small"
            elif salience > 0 and salience < 0.05: keep, reason = False, "salience_too_low"
            elif confidence > 0 and confidence < 0.5: keep, reason = False, "confidence_too_low"
            
            if keep:
                ls_retained_phase1.append(m)
            else:
                m_sup = dict(m)
                m_sup["suppression_reason"] = reason
                crop_targets["suppressed_light_scar"].append({"mark": m_sup, "img": image_path, "pair_id": pair_id, "is_same": is_same, "variant": "light_scar_cluster_aware"})
                
        # Cluster logic for Light Scar
        clusters = []
        for m in ls_retained_phase1:
            added = False
            for c in clusters:
                if any(distance(m, cm) < 0.02 for cm in c):
                    c.append(m)
                    added = True
                    break
            if not added:
                clusters.append([m])
                
        for c in clusters:
            c.sort(key=lambda x: (x.get("salience") or 0) * (x.get("confidence") or 0), reverse=True)
            crop_targets["retained_light_scar"].append({"mark": c[0], "img": image_path, "pair_id": pair_id, "is_same": is_same, "variant": "light_scar_cluster_aware", "suppression_reason": ""})
            for suppressed_m in c[1:]:
                m_sup = dict(suppressed_m)
                m_sup["suppression_reason"] = "cluster_duplicate"
                crop_targets["suppressed_light_scar"].append({"mark": m_sup, "img": image_path, "pair_id": pair_id, "is_same": is_same, "variant": "light_scar_cluster_aware"})

        # Correspondences (gallery side only)
        corresps = p.get("accepted_correspondences_detail", [])
        for c in corresps:
            if c.get("mark_type") == "light_scar":
                item = {
                    "corresp": c, 
                    "img": image_path, 
                    "centroid": c.get("gallery_centroid"), 
                    "pair_id": pair_id, 
                    "is_same": is_same, 
                    "variant": "baseline_current_detector"
                }
                if is_same: crop_targets["same_person_light_scar_corresps"].append(item)
                else: crop_targets["impostor_light_scar_corresps"].append(item)

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 10)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()

    reviewer_csv_rows = []
    debug_summary_rows = []
    failed_rows = []
    
    def generate_sheet(target_key, output_filename, is_corresp=False, is_mixed=False):
        if is_mixed and target_key == "dark_spot_retained_suppressed":
            items = crop_targets["retained_dark_spot_strict"] + crop_targets["suppressed_dark_spot_strict"]
        else:
            items = crop_targets[target_key]
            
        rows_loaded = len(items)
        if not items:
            debug_summary_rows.append({
                "sheet_name": output_filename,
                "rows_loaded": 0,
                "crop_targets_selected": 0,
                "successful_downloads": 0,
                "successful_crops": 0,
                "pasted_tiles": 0,
                "failed_rows": 0,
                "non_white_pixels": 0,
                "is_blank": True
            })
            return
        
        if is_corresp: items.sort(key=lambda x: x["corresp"].get("patch_combined_similarity") or 0, reverse=True)
        else: items.sort(key=lambda x: x["mark"].get("contrast_score", x["mark"].get("salience", 0)) or 0, reverse=True)
            
        top_items = items[:50]
        crop_targets_selected = len(top_items)
        
        cols = 5
        rows = math.ceil(len(top_items) / cols)
        
        tile_w = 150
        tile_h = 250
        
        sheet_img = Image.new('RGB', (cols * tile_w, rows * tile_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(sheet_img)
        
        successful_downloads = 0
        successful_crops = 0
        pasted_tiles = 0
        failed_count = 0
        labels_drawn = 0
        
        for i, item in enumerate(top_items):
            gcs_path = item.get("img")
            pair_id = item["pair_id"]
            is_same = "same_person" if item["is_same"] else "different_person"
            variant = item["variant"]
            
            if is_corresp:
                m = item["corresp"]
                mark_type = m.get("mark_type", "unknown")
                region = m.get("regional_canonical_region_gallery", "unknown")
                confidence = "N/A"
                contrast = "N/A"
                area = "N/A"
                match_quality = str(round(m.get("patch_combined_similarity") or 0, 3))
                sup_reason = "N/A"
                centroid = item["centroid"]
            else:
                m = item["mark"]
                mark_type = m.get("mark_type", "unknown")
                region = m.get("face_region", "unknown")
                confidence = str(round(m.get("confidence") or 0, 3))
                contrast = str(round(m.get("contrast_score", m.get("salience", 0)) or 0, 3))
                area = str(round(m.get("area") or 0, 1))
                match_quality = "N/A"
                sup_reason = m.get("suppression_reason", "None")
                centroid = m.get("centroid")
                
            if not gcs_path:
                failed_rows.append({"sheet": output_filename, "pair_id": pair_id, "error": "Missing image path"})
                failed_count += 1
                continue
                
            if not centroid:
                print(f"DEBUG {pair_id} MISSING CENTROID: centroid={centroid}, type={type(centroid)}")
                failed_rows.append({"sheet": output_filename, "pair_id": pair_id, "error": "Missing centroid"})
                failed_count += 1
                continue

            local_img_path = os.path.join(crop_tmp_dir, get_basename(gcs_path))
            cropped_img = None
            
            success, err_msg = download_gcs_file(gcs_path, local_img_path)
            if not success:
                failed_rows.append({"sheet": output_filename, "pair_id": pair_id, "error": f"Download failed: {err_msg}"})
                failed_count += 1
                continue
            
            successful_downloads += 1
            
            try:
                img = Image.open(local_img_path)
                w, h = img.size
                cx, cy = int(centroid[0] * w), int(centroid[1] * h)
                box_size = 150
                left = max(0, cx - box_size // 2)
                top = max(0, cy - box_size // 2)
                right = min(w, cx + box_size // 2)
                bottom = min(h, cy + box_size // 2)
                cropped = img.crop((left, top, right, bottom))
                cropped_img = Image.new("RGB", (box_size, box_size), (0,0,0))
                cropped_img.paste(cropped, (0,0))
                successful_crops += 1
            except Exception as e:
                failed_rows.append({"sheet": output_filename, "pair_id": pair_id, "error": f"Crop failed: {e}"})
                failed_count += 1
                continue
            
            col_idx = i % cols
            row_idx = i // cols
            x = col_idx * tile_w
            y = row_idx * tile_h
            
            if cropped_img:
                sheet_img.paste(cropped_img, (x, y))
                pasted_tiles += 1
            
            text_y = y + 155
            lines = [
                f"ID: {pair_id} ({is_same})",
                f"Var: {variant}",
                f"Type: {mark_type} | {region}",
                f"Conf: {confidence} | Cont: {contrast}",
                f"Area: {area} | Match: {match_quality}",
                f"Supp: {sup_reason}"
            ]
            for line in lines:
                draw.text((x + 5, text_y), line, font=font, fill=(0,0,0))
                text_y += 15
            labels_drawn += 1

            crop_filename = f"{target_key}_{i:02d}.jpg"
            reviewer_csv_rows.append({
                "review_id": crop_filename,
                "crop_filename": crop_filename,
                "pair_id": pair_id,
                "label_same_person": item["is_same"],
                "variant": variant,
                "mark_type": mark_type,
                "region": region,
                "confidence": confidence,
                "contrast_score": contrast,
                "salience_score": contrast,
                "area": area,
                "match_quality": match_quality,
                "suppression_reason": sup_reason,
                "human_label_real_mark_yes_no": "",
                "human_label_texture_noise_yes_no": "",
                "human_label_lighting_or_makeup_yes_no": "",
                "human_label_good_correspondence_yes_no": "",
                "review_notes": ""
            })

        sheet_img.save(os.path.join(output_dir, output_filename))
        print(f"Generated {output_filename}")
        
        non_white_pixels = get_non_white_pixels(sheet_img)
        is_blank = non_white_pixels == 0
        
        debug_summary_rows.append({
            "sheet_name": output_filename,
            "rows_loaded": rows_loaded,
            "crop_targets_selected": crop_targets_selected,
            "successful_downloads": successful_downloads,
            "successful_crops": successful_crops,
            "pasted_tiles": pasted_tiles,
            "failed_rows": failed_count,
            "non_white_pixels": non_white_pixels,
            "is_blank": is_blank
        })
        
        if crop_targets_selected > 0 and pasted_tiles == 0:
            print(f"FATAL ERROR: Contact sheet {output_filename} has 0 pasted tiles despite {crop_targets_selected} targets.")
            return True
            
        if is_blank:
            print(f"FATAL ERROR: Contact sheet {output_filename} is a blank canvas.")
            return True
            
        if crop_targets_selected > 0 and successful_downloads == 0:
            print(f"FATAL ERROR: Contact sheet {output_filename} had 0 successful downloads for {crop_targets_selected} targets.")
            return True
            
        if crop_targets_selected > 0 and failed_count == crop_targets_selected:
            print(f"FATAL ERROR: Contact sheet {output_filename} failed on all {crop_targets_selected} targets (missing centroid, path, etc.).")
            return True
            
        return False

    print("Generating contact sheets...")
    fatal = False
    fatal = generate_sheet("retained_light_scar", "light_scar_retained_contact_sheet.jpg") or fatal
    fatal = generate_sheet("suppressed_light_scar", "light_scar_suppressed_contact_sheet.jpg") or fatal
    fatal = generate_sheet("same_person_light_scar_corresps", "light_scar_same_person_correspondence_contact_sheet.jpg", is_corresp=True) or fatal
    fatal = generate_sheet("impostor_light_scar_corresps", "light_scar_impostor_correspondence_contact_sheet.jpg", is_corresp=True) or fatal
    fatal = generate_sheet("dark_spot_retained_suppressed", "dark_spot_retained_suppressed_contact_sheet.jpg", is_mixed=True) or fatal
    
    if reviewer_csv_rows:
        keys = reviewer_csv_rows[0].keys()
        with open(os.path.join(output_dir, "hq_visual_audit_review_sheet.csv"), "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(reviewer_csv_rows)
            
    if debug_summary_rows:
        keys = debug_summary_rows[0].keys()
        with open(os.path.join(output_dir, "hq_contact_sheet_debug_summary.csv"), "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(debug_summary_rows)
            
    if failed_rows:
        keys = failed_rows[0].keys()
        with open(os.path.join(output_dir, "hq_contact_sheet_failed_rows.csv"), "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(failed_rows)
            
    if fatal:
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    create_contact_sheets(args.input, args.output_dir)
