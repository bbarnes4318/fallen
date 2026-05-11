import json
import os
import csv
import math
import subprocess
import sys
import argparse

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageDraw, ImageFont

def distance(m1, m2):
    c1 = m1.get("centroid", [0, 0])
    c2 = m2.get("centroid", [0, 0])
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def download_gcs_file(gcs_uri, local_path):
    if os.path.exists(local_path): return True
    gsutil_cmd = "gsutil.cmd" if sys.platform == "win32" else "gsutil"
    try:
        subprocess.run([gsutil_cmd, "cp", gcs_uri, local_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {gcs_uri}: {e}")
        return False

def get_basename(path):
    return path.split("/")[-1].split("\\")[-1]

def filter_marks(marks, variant):
    retained = []
    suppressed = []
    
    for m in marks:
        m_type = m.get("mark_type")
        area = m.get("area") or 0
        contrast = m.get("contrast_score", m.get("salience", 0)) or 0
        confidence = m.get("confidence") or 0
        region = m.get("face_region", "unknown")
        
        isolated = True
        for other in marks:
            if other == m: continue
            if distance(m, other) < 0.005:
                isolated = False
                break
                
        def apply_dark_spot_strict(mark):
            if mark.get("mark_type") == "dark_spot":
                if area < 15: return False, "dark_spot_strict_area"
                if contrast > 0 and contrast < 3.0: return False, "dark_spot_strict_salience"
                if not isolated: return False, "dark_spot_strict_cluster"
            return True, None

        def get_light_scar_contrast_relaxed(mark):
            if mark.get("mark_type") == "light_scar":
                if area < 10: return False, "area_too_small"
                if contrast > 0 and contrast < 0.05: return False, "salience_too_low"
                if confidence > 0 and confidence < 0.5: return False, "confidence_too_low"
            return True, None

        keep = True
        reason = None
        
        if variant == "dark_spot_strict":
            keep, reason = apply_dark_spot_strict(m)
        elif variant == "light_scar_cluster_aware":
            keep, reason = get_light_scar_contrast_relaxed(m)
            
        if keep:
            retained.append(m)
        else:
            m_sup = dict(m)
            m_sup["suppression_reason"] = reason
            suppressed.append(m_sup)

    # 2nd pass for cluster logic
    if variant == "light_scar_cluster_aware":
        final_retained = []
        light_scars = [m for m in retained if m.get("mark_type") == "light_scar"]
        other_marks = [m for m in retained if m.get("mark_type") != "light_scar"]
        
        clusters = []
        for m in light_scars:
            added = False
            for c in clusters:
                if any(distance(m, cm) < 0.02 for cm in c):
                    c.append(m)
                    added = True
                    break
            if not added:
                clusters.append([m])
                
        for c in clusters:
            c.sort(key=lambda x: (x.get("contrast_score", x.get("salience", 0)) or 0) * (x.get("confidence") or 0), reverse=True)
            final_retained.append(c[0])
            for suppressed_m in c[1:]:
                m_sup = dict(suppressed_m)
                m_sup["suppression_reason"] = "cluster_duplicate"
                suppressed.append(m_sup)
                
        retained = other_marks + final_retained

    return retained, suppressed


def create_contact_sheets(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    crop_tmp_dir = os.path.join(output_dir, "crop_source_tmp")
    os.makedirs(crop_tmp_dir, exist_ok=True)
    
    # Load manifest
    manifest_map = {}
    manifest_path = "validation/validation_pairs_hq_headshots.csv"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest_map[row["pair_id"]] = {
                    "probe": row["image1_url_or_gcs_path"],
                    "gallery": row["image2_url_or_gcs_path"],
                    "is_same": str(row.get("label_same_person", "")).lower() in ["true", "1", "yes"]
                }
    
    pairs = []
    with open(input_file, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                pairs.append(json.loads(line.strip()))
            except:
                pass
                
    crop_targets = {
        "retained_light_scar": [],
        "suppressed_light_scar": [],
        "same_person_light_scar_corresps": [],
        "impostor_light_scar_corresps": [],
        "retained_dark_spot_strict": [],
        "suppressed_dark_spot_strict": []
    }

    # Extract marks
    for p in pairs:
        pair_id = p.get("pair_id")
        is_same = manifest_map.get(pair_id, {}).get("is_same", False)
        
        retained_ls, suppressed_ls = filter_marks(p.get("raw_gallery_marks_summary", []), "light_scar_cluster_aware")
        retained_ds, suppressed_ds = filter_marks(p.get("raw_gallery_marks_summary", []), "dark_spot_strict")
        
        image_path = manifest_map.get(pair_id, {}).get("gallery", "")
        
        for m in retained_ls:
            if m.get("mark_type") == "light_scar":
                crop_targets["retained_light_scar"].append({"mark": m, "img": image_path, "pair_id": pair_id, "is_same": is_same, "variant": "light_scar_cluster_aware"})
        for m in suppressed_ls:
            if m.get("mark_type") == "light_scar":
                crop_targets["suppressed_light_scar"].append({"mark": m, "img": image_path, "pair_id": pair_id, "is_same": is_same, "variant": "light_scar_cluster_aware"})
        for m in retained_ds:
            if m.get("mark_type") == "dark_spot":
                crop_targets["retained_dark_spot_strict"].append({"mark": m, "img": image_path, "pair_id": pair_id, "is_same": is_same, "variant": "dark_spot_strict", "suppression_reason": ""})
        for m in suppressed_ds:
            if m.get("mark_type") == "dark_spot":
                crop_targets["suppressed_dark_spot_strict"].append({"mark": m, "img": image_path, "pair_id": pair_id, "is_same": is_same, "variant": "dark_spot_strict"})

        # Correspondences (only gallery side for visualization)
        corresps = p.get("accepted_correspondences_detail", [])
        for c in corresps:
            if c.get("mark_type") == "light_scar":
                item = {
                    "corresp": c, 
                    "img": image_path, 
                    "centroid": c.get("regional_canonical_centroid_gallery"), 
                    "pair_id": pair_id, 
                    "is_same": is_same, 
                    "variant": "baseline_current_detector"
                }
                if is_same: crop_targets["same_person_light_scar_corresps"].append(item)
                else: crop_targets["impostor_light_scar_corresps"].append(item)

    # Prepare fonts
    try:
        # standard fallback on windows
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 10)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()

    reviewer_csv_rows = []
    
    def generate_sheet(target_key, output_filename, is_corresp=False, is_mixed=False):
        if is_mixed and target_key == "dark_spot_retained_suppressed":
            items = crop_targets["retained_dark_spot_strict"] + crop_targets["suppressed_dark_spot_strict"]
        else:
            items = crop_targets[target_key]
        if not items: return
        
        if is_corresp: items.sort(key=lambda x: x["corresp"].get("patch_combined_similarity") or 0, reverse=True)
        else: items.sort(key=lambda x: x["mark"].get("contrast_score", x["mark"].get("salience", 0)) or 0, reverse=True)
            
        top_items = items[:50]
        
        cols = 5
        rows = math.ceil(len(top_items) / cols)
        
        # Tile size: 150 crop + 100 for text
        tile_w = 150
        tile_h = 250
        
        sheet_img = Image.new('RGB', (cols * tile_w, rows * tile_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(sheet_img)
        
        for i, item in enumerate(top_items):
            gcs_path = item.get("img")
            if not gcs_path: continue
            
            # Metadata
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

            # Crop
            local_img_path = os.path.join(crop_tmp_dir, get_basename(gcs_path))
            cropped_img = None
            if download_gcs_file(gcs_path, local_img_path) and centroid:
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
                    # pad if smaller
                    cropped_img = Image.new("RGB", (box_size, box_size), (0,0,0))
                    cropped_img.paste(cropped, (0,0))
                except Exception as e:
                    print(f"Crop failed: {e}")
            
            # Place on sheet
            col_idx = i % cols
            row_idx = i // cols
            x = col_idx * tile_w
            y = row_idx * tile_h
            
            if cropped_img:
                sheet_img.paste(cropped_img, (x, y))
            
            # Draw text
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

            # Append to Reviewer CSV
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
                "salience_score": contrast, # Fallback mapping
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

    print("Generating contact sheets...")
    generate_sheet("retained_light_scar", "light_scar_retained_contact_sheet.jpg")
    generate_sheet("suppressed_light_scar", "light_scar_suppressed_contact_sheet.jpg")
    generate_sheet("same_person_light_scar_corresps", "light_scar_same_person_correspondence_contact_sheet.jpg", is_corresp=True)
    generate_sheet("impostor_light_scar_corresps", "light_scar_impostor_correspondence_contact_sheet.jpg", is_corresp=True)
    generate_sheet("dark_spot_retained_suppressed", "dark_spot_retained_suppressed_contact_sheet.jpg", is_mixed=True)
    
    # Save reviewer CSV
    if reviewer_csv_rows:
        keys = reviewer_csv_rows[0].keys()
        with open(os.path.join(output_dir, "hq_visual_audit_review_sheet.csv"), "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(reviewer_csv_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    create_contact_sheets(args.input, args.output_dir)
