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
        subprocess.run([gsutil_cmd, "cp", gcs_uri, local_path], capture_output=True, text=True, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def get_non_white_pixels(img):
    gray = img.convert("L")
    data = list(gray.getdata())
    return sum(1 for p in data if p < 255)

def draw_mark(draw, centroid, bbox, img_w, img_h, color=(0, 255, 0)):
    cx, cy = int(centroid[0] * img_w), int(centroid[1] * img_h)
    r = 5
    draw.ellipse([(cx-r, cy-r), (cx+r, cy+r)], outline=color, width=2)
    if bbox:
        x_min = int(bbox[0] * img_w)
        y_min = int(bbox[1] * img_h)
        x_max = int(bbox[2] * img_w)
        y_max = int(bbox[3] * img_h)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

def create_phase2d_audit(input_jsonl, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    crop_tmp_dir = os.path.join(output_dir, "tmp_crops")
    os.makedirs(crop_tmp_dir, exist_ok=True)
    
    crop_sizes = [256, 512, 768]
    for size in crop_sizes:
        os.makedirs(os.path.join(output_dir, f"crops/{size}"), exist_ok=True)
        
    for cat in ["light_scar_same_person", "light_scar_impostor", "dark_spot_same_person", "dark_spot_impostor"]:
        os.makedirs(os.path.join(output_dir, f"{cat}_overlay_examples"), exist_ok=True)

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
            pass

    pairs = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))

    targets = {
        "light_scar_same_person": [],
        "light_scar_impostor": [],
        "dark_spot_same_person": [],
        "dark_spot_impostor": []
    }

    for p in pairs:
        pair_id = p.get("pair_id", "unknown")
        is_same = p.get("label_same_person", False)
        
        gallery_path = manifest_map.get(pair_id, {}).get("gallery", "")
        probe_path = manifest_map.get(pair_id, {}).get("probe", "")
        
        raw_probe = p.get("raw_probe_marks_summary", [])
        raw_gallery = p.get("raw_gallery_marks_summary", [])
        
        corresps = p.get("accepted_correspondences_detail", [])
        for c in corresps:
            m_type = c.get("mark_type")
            if m_type not in ["light_scar", "dark_spot"]:
                continue
                
            probe_idx = c.get("probe_idx")
            gallery_idx = c.get("gallery_idx")
            
            p_mark = raw_probe[probe_idx] if probe_idx is not None and probe_idx < len(raw_probe) else {}
            g_mark = raw_gallery[gallery_idx] if gallery_idx is not None and gallery_idx < len(raw_gallery) else {}
            
            item = {
                "corresp": c,
                "probe_img": probe_path,
                "gallery_img": gallery_path,
                "pair_id": pair_id,
                "is_same": is_same,
                "mark_type": m_type,
                "p_mark": p_mark,
                "g_mark": g_mark,
                "all_probe_marks": raw_probe,
                "all_gallery_marks": raw_gallery,
                "all_corresps": corresps
            }
            
            cat_key = f"{m_type}_{'same_person' if is_same else 'impostor'}"
            targets[cat_key].append(item)

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        except:
            font = ImageFont.load_default()

    reviewer_csv_rows = []
    debug_summary_rows = []
    failed_rows = []
    fatal = False
    
    def generate_paired_sheet(target_key, output_filename):
        nonlocal fatal
        items = targets[target_key]
        rows_loaded = len(items)
        if not items:
            debug_summary_rows.append({
                "sheet_name": output_filename,
                "rows_loaded": 0,
                "targets_selected": 0,
                "probe_downloads": 0,
                "gallery_downloads": 0,
                "probe_crops": 0,
                "gallery_crops": 0,
                "pasted_tiles": 0,
                "failed_rows": 0,
                "non_white_pixels": 0,
                "is_blank": True,
                "manually_reviewable": False
            })
            return
            
        items.sort(key=lambda x: x["corresp"].get("patch_combined_similarity") or 0, reverse=True)
        top_items = items[:50]
        targets_selected = len(top_items)
        
        tile_w = 1536
        tile_h = 662
        
        cols = 1
        rows = len(top_items)
        
        sheet_img = Image.new('RGB', (tile_w, rows * tile_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(sheet_img)
        
        successful_probe_downloads = 0
        successful_gallery_downloads = 0
        successful_probe_crops = 0
        successful_gallery_crops = 0
        pasted_tiles = 0
        failed_count = 0
        
        for i, item in enumerate(top_items):
            c = item["corresp"]
            p_mark = item["p_mark"]
            g_mark = item["g_mark"]
            
            p_img_path = item["probe_img"]
            g_img_path = item["gallery_img"]
            pair_id = item["pair_id"]
            
            local_p_img = os.path.join(crop_tmp_dir, get_basename(p_img_path))
            local_g_img = os.path.join(crop_tmp_dir, get_basename(g_img_path))
            
            p_dl, p_err = download_gcs_file(p_img_path, local_p_img)
            g_dl, g_err = download_gcs_file(g_img_path, local_g_img)
            
            if not p_dl or not g_dl:
                failed_rows.append({"sheet": output_filename, "pair_id": pair_id, "error": f"DL failed. P: {p_err}, G: {g_err}"})
                failed_count += 1
                continue
                
            successful_probe_downloads += 1
            successful_gallery_downloads += 1
            
            p_centroid = c.get("probe_centroid")
            g_centroid = c.get("gallery_centroid")
            
            if not p_centroid or not g_centroid:
                failed_rows.append({"sheet": output_filename, "pair_id": pair_id, "error": "Missing centroid in corresp"})
                failed_count += 1
                continue
                
            try:
                p_im = Image.open(local_p_img)
                g_im = Image.open(local_g_img)
            except Exception as e:
                failed_rows.append({"sheet": output_filename, "pair_id": pair_id, "error": f"Image open failed: {e}"})
                failed_count += 1
                continue
            
            def extract_crop(im, centroid, size):
                w, h = im.size
                cx, cy = int(centroid[0] * w), int(centroid[1] * h)
                left = max(0, cx - size // 2)
                top = max(0, cy - size // 2)
                right = min(w, cx + size // 2)
                bottom = min(h, cy + size // 2)
                cropped = im.crop((left, top, right, bottom))
                res = Image.new("RGB", (size, size), (0,0,0))
                paste_x = max(0, (size // 2) - cx)
                paste_y = max(0, (size // 2) - cy)
                res.paste(cropped, (paste_x, paste_y))
                return res

            p_crops = {}
            g_crops = {}
            for size in crop_sizes:
                try:
                    p_c = extract_crop(p_im, p_centroid, size)
                    g_c = extract_crop(g_im, g_centroid, size)
                    
                    p_draw = ImageDraw.Draw(p_c)
                    g_draw = ImageDraw.Draw(g_c)
                    
                    ccx, ccy = size // 2, size // 2
                    r = 5
                    p_draw.ellipse([(ccx-r, ccy-r), (ccx+r, ccy+r)], outline=(0,255,0), width=2)
                    g_draw.ellipse([(ccx-r, ccy-r), (ccx+r, ccy+r)], outline=(0,255,0), width=2)
                    
                    p_bbox = p_mark.get("bbox")
                    if p_bbox:
                        pw, ph = p_im.size
                        bx_min, by_min, bx_max, by_max = p_bbox[0]*pw, p_bbox[1]*ph, p_bbox[2]*pw, p_bbox[3]*ph
                        cx_orig, cy_orig = int(p_centroid[0]*pw), int(p_centroid[1]*ph)
                        p_draw.rectangle([bx_min - cx_orig + ccx, by_min - cy_orig + ccy, bx_max - cx_orig + ccx, by_max - cy_orig + ccy], outline=(0,255,0), width=2)
                        
                    g_bbox = g_mark.get("bbox")
                    if g_bbox:
                        gw, gh = g_im.size
                        bx_min, by_min, bx_max, by_max = g_bbox[0]*gw, g_bbox[1]*gh, g_bbox[2]*gw, g_bbox[3]*gh
                        cx_orig, cy_orig = int(g_centroid[0]*gw), int(g_centroid[1]*gh)
                        g_draw.rectangle([bx_min - cx_orig + ccx, by_min - cy_orig + ccy, bx_max - cx_orig + ccx, by_max - cy_orig + ccy], outline=(0,255,0), width=2)
                        
                    p_crops[size] = p_c
                    g_crops[size] = g_c
                    
                    p_c_name = f"{target_key}_probe_{pair_id}_{i}_{size}.jpg"
                    g_c_name = f"{target_key}_gallery_{pair_id}_{i}_{size}.jpg"
                    p_c.save(os.path.join(output_dir, f"crops/{size}", p_c_name))
                    g_c.save(os.path.join(output_dir, f"crops/{size}", g_c_name))
                except Exception as e:
                    pass
            
            if 512 not in p_crops or 512 not in g_crops:
                failed_rows.append({"sheet": output_filename, "pair_id": pair_id, "error": "512 Crop failed"})
                failed_count += 1
                continue
                
            successful_probe_crops += 1
            successful_gallery_crops += 1
            
            p_loc = p_im.copy()
            p_loc.thumbnail((256, 256))
            p_draw_loc = ImageDraw.Draw(p_loc)
            draw_mark(p_draw_loc, p_centroid, p_mark.get("bbox"), p_loc.size[0], p_loc.size[1], color=(255,0,0))
            
            g_loc = g_im.copy()
            g_loc.thumbnail((256, 256))
            g_draw_loc = ImageDraw.Draw(g_loc)
            draw_mark(g_draw_loc, g_centroid, g_mark.get("bbox"), g_loc.size[0], g_loc.size[1], color=(255,0,0))
            
            overlay_filename = ""
            if i < 5:
                p_ov = p_im.copy()
                g_ov = g_im.copy()
                po_draw = ImageDraw.Draw(p_ov)
                go_draw = ImageDraw.Draw(g_ov)
                
                for om in item["all_probe_marks"]:
                    if om.get("mark_type") == item["mark_type"]:
                        draw_mark(po_draw, om.get("centroid", [0,0]), om.get("bbox"), p_ov.size[0], p_ov.size[1], color=(255,255,0))
                for om in item["all_gallery_marks"]:
                    if om.get("mark_type") == item["mark_type"]:
                        draw_mark(go_draw, om.get("centroid", [0,0]), om.get("bbox"), g_ov.size[0], g_ov.size[1], color=(255,255,0))
                
                ov_w = p_ov.size[0] + g_ov.size[0]
                ov_h = max(p_ov.size[1], g_ov.size[1])
                combined_ov = Image.new('RGB', (ov_w, ov_h), (0,0,0))
                combined_ov.paste(p_ov, (0,0))
                combined_ov.paste(g_ov, (p_ov.size[0], 0))
                comb_draw = ImageDraw.Draw(combined_ov)
                
                for oc in item["all_corresps"]:
                    if oc.get("mark_type") == item["mark_type"]:
                        pc = oc.get("probe_centroid", [0,0])
                        gc = oc.get("gallery_centroid", [0,0])
                        p_cx, p_cy = int(pc[0]*p_ov.size[0]), int(pc[1]*p_ov.size[1])
                        g_cx, g_cy = int(gc[0]*g_ov.size[0]) + p_ov.size[0], int(gc[1]*g_ov.size[1])
                        comb_draw.line([(p_cx, p_cy), (g_cx, g_cy)], fill=(0,255,0), width=3)
                
                overlay_filename = f"{target_key}_overlay_{pair_id}.jpg"
                combined_ov.save(os.path.join(output_dir, f"{target_key}_overlay_examples", overlay_filename))
            
            y_offset = i * tile_h
            
            p_loc_y = y_offset + (512 - p_loc.size[1]) // 2
            g_loc_y = y_offset + (512 - g_loc.size[1]) // 2
            
            sheet_img.paste(p_loc, (0, p_loc_y))
            sheet_img.paste(p_crops[512], (256, y_offset))
            sheet_img.paste(g_crops[512], (256 + 512, y_offset))
            sheet_img.paste(g_loc, (256 + 512 + 512, g_loc_y))
            
            pasted_tiles += 1
            
            text_y = y_offset + 512 + 10
            
            labels = [
                f"pair_id: {pair_id} | label_same_person: {item['is_same']}",
                f"variant: {item['mark_type']} | mark_type: {item['mark_type']}",
                f"canonical_region: {c.get('regional_canonical_region_probe')} | face_region: {p_mark.get('face_region')}",
                f"P: conf: {p_mark.get('confidence')} cont: {p_mark.get('contrast_score')} sal: {p_mark.get('salience')} area: {p_mark.get('area')}",
                f"G: conf: {g_mark.get('confidence')} cont: {g_mark.get('contrast_score')} sal: {g_mark.get('salience')} area: {g_mark.get('area')}",
                f"match_quality: {round(c.get('patch_combined_similarity') or 0, 3)} | regional_uv_distance: {c.get('regional_uv_distance')}"
            ]
            for idx, label in enumerate(labels):
                draw.text((10, text_y + (idx*20)), label, font=font, fill=(0,0,0))

            bbox_missing = p_mark.get('bbox') is None or g_mark.get('bbox') is None

            reviewer_csv_rows.append({
                "review_id": f"{target_key}_{i}",
                "pair_id": pair_id,
                "label_same_person": item["is_same"],
                "probe_image_path": p_img_path,
                "gallery_image_path": g_img_path,
                "contact_sheet_filename": output_filename,
                "overlay_filename": overlay_filename,
                "probe_crop_filename": f"{target_key}_probe_{pair_id}_{i}_512.jpg",
                "gallery_crop_filename": f"{target_key}_gallery_{pair_id}_{i}_512.jpg",
                "mark_type": item["mark_type"],
                "variant": item["mark_type"],
                "canonical_region_probe": c.get('regional_canonical_region_probe'),
                "canonical_region_gallery": c.get('regional_canonical_region_gallery'),
                "face_region_probe": p_mark.get("face_region"),
                "face_region_gallery": g_mark.get("face_region"),
                "probe_centroid": c.get("probe_centroid"),
                "gallery_centroid": c.get("gallery_centroid"),
                "probe_bbox": p_mark.get("bbox"),
                "gallery_bbox": g_mark.get("bbox"),
                "bbox_missing": bbox_missing,
                "confidence_probe": p_mark.get("confidence"),
                "confidence_gallery": g_mark.get("confidence"),
                "contrast_probe": p_mark.get("contrast_score"),
                "contrast_gallery": g_mark.get("contrast_score"),
                "salience_probe": p_mark.get("salience"),
                "salience_gallery": g_mark.get("salience"),
                "area_probe": p_mark.get("area"),
                "area_gallery": g_mark.get("area"),
                "match_quality": c.get("patch_combined_similarity"),
                "regional_uv_distance": c.get("regional_uv_distance"),
                "human_same_physical_mark_yes_no": "",
                "human_real_mark_probe_yes_no": "",
                "human_real_mark_gallery_yes_no": "",
                "human_texture_noise_probe_yes_no": "",
                "human_texture_noise_gallery_yes_no": "",
                "human_lighting_makeup_hair_yes_no": "",
                "human_correspondence_confidence_1_to_5": "",
                "review_notes": ""
            })

        sheet_img.save(os.path.join(output_dir, output_filename))
        print(f"Generated {output_filename}")
        
        non_white_pixels = get_non_white_pixels(sheet_img)
        is_blank = non_white_pixels == 0
        
        debug_summary_rows.append({
            "sheet_name": output_filename,
            "rows_loaded": rows_loaded,
            "targets_selected": targets_selected,
            "probe_downloads": successful_probe_downloads,
            "gallery_downloads": successful_gallery_downloads,
            "probe_crops": successful_probe_crops,
            "gallery_crops": successful_gallery_crops,
            "pasted_tiles": pasted_tiles,
            "failed_rows": failed_count,
            "non_white_pixels": non_white_pixels,
            "is_blank": is_blank,
            "manually_reviewable": not is_blank and pasted_tiles > 0
        })
        
        if targets_selected > 0 and pasted_tiles == 0:
            print(f"FATAL ERROR: Contact sheet {output_filename} has 0 pasted tiles despite {targets_selected} targets.")
            fatal = True
            
        if is_blank:
            print(f"FATAL ERROR: Contact sheet {output_filename} is a blank canvas.")
            fatal = True
            
        if targets_selected > 0 and successful_probe_downloads == 0:
            print(f"FATAL ERROR: Contact sheet {output_filename} had 0 successful probe downloads.")
            fatal = True
            
        if targets_selected > 0 and successful_gallery_downloads == 0:
            print(f"FATAL ERROR: Contact sheet {output_filename} had 0 successful gallery downloads.")
            fatal = True
            
        if targets_selected > 0 and successful_probe_crops == 0:
            print(f"FATAL ERROR: Contact sheet {output_filename} had 0 successful probe crops.")
            fatal = True

        if targets_selected > 0 and successful_gallery_crops == 0:
            print(f"FATAL ERROR: Contact sheet {output_filename} had 0 successful gallery crops.")
            fatal = True

    generate_paired_sheet("light_scar_same_person", "light_scar_same_person_paired_contact_sheet.jpg")
    generate_paired_sheet("light_scar_impostor", "light_scar_impostor_paired_contact_sheet.jpg")
    generate_paired_sheet("dark_spot_same_person", "dark_spot_same_person_paired_contact_sheet.jpg")
    generate_paired_sheet("dark_spot_impostor", "dark_spot_impostor_paired_contact_sheet.jpg")

    if reviewer_csv_rows:
        keys = reviewer_csv_rows[0].keys()
        with open(os.path.join(output_dir, "hq_phase2d_visual_review_sheet.csv"), "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(reviewer_csv_rows)
            
    if debug_summary_rows:
        keys = debug_summary_rows[0].keys()
        with open(os.path.join(output_dir, "hq_phase2d_visual_audit_debug_summary.csv"), "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(debug_summary_rows)
            
    if failed_rows:
        keys = failed_rows[0].keys()
        with open(os.path.join(output_dir, "hq_phase2d_failed_rows.csv"), "w", newline='', encoding='utf-8') as f:
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
    create_phase2d_audit(args.input, args.output_dir)
