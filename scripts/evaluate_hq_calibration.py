import json
import os
import csv
import math
import argparse
import subprocess
import sys

# Ensure PIL is available for image cropping
try:
    from PIL import Image
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

def distance(m1, m2):
    c1 = m1.get("centroid", [0, 0])
    c2 = m2.get("centroid", [0, 0])
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def is_high_value(m):
    return m.get("mark_type") in ["dark_spot", "dark_mole", "light_scar"]

def filter_marks(marks, variant):
    retained = []
    suppressed = []
    
    if variant == "baseline_current_detector":
        return marks, []
        
    for m in marks:
        m_type = m.get("mark_type")
        area = m.get("area", 0)
        salience = m.get("salience", 0)
        confidence = m.get("confidence", 0)
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
                if salience < 3.0: return False, "dark_spot_strict_salience"
                if not isolated: return False, "dark_spot_strict_cluster"
            return True, None

        def get_light_scar_contrast_relaxed(mark):
            if mark.get("mark_type") == "light_scar":
                if area < 10: return False, "area_too_small"
                if salience < 0.05: return False, "salience_too_low"
                if confidence < 0.5: return False, "confidence_too_low"
            return True, None

        def get_light_scar_shape_linear(mark):
            if mark.get("mark_type") == "light_scar":
                if "bbox" not in mark and "width" not in mark:
                    return False, "shape_variant_skipped_due_to_missing_bbox=true"
                width = mark.get("width", mark.get("bbox", [0,0,0,0])[2] if "bbox" in mark else 0)
                height = mark.get("height", mark.get("bbox", [0,0,0,0])[3] if "bbox" in mark else 0)
                if width > 0 and height > 0:
                    ratio = max(width, height) / min(width, height)
                    if ratio < 2.0: return False, "not_linear"
                if area < 10: return False, "area_too_small"
            return True, None

        def get_light_scar_region_safe(mark):
            if mark.get("mark_type") == "light_scar":
                noisy = ["philtrum", "chin", "jawline", "periocular", "lower_cheek", "mid_cheek"]
                if region in noisy:
                    if area < 20: return False, "region_safe_area_strict"
                    if salience < 0.1: return False, "region_safe_salience_strict"
                    if confidence < 0.7: return False, "region_safe_confidence_strict"
                else:
                    if area < 10 or salience < 0.05: return False, "region_safe_normal_filtered"
            return True, None
            
        keep = True
        reason = None
        
        if variant == "dark_spot_strict":
            keep, reason = apply_dark_spot_strict(m)
        elif variant == "light_scar_contrast_relaxed":
            keep, reason = get_light_scar_contrast_relaxed(m)
        elif variant == "light_scar_shape_linear":
            keep, reason = get_light_scar_shape_linear(m)
        elif variant == "light_scar_cluster_aware":
            keep, reason = get_light_scar_contrast_relaxed(m) # Apply base relaxed filter first
        elif variant == "light_scar_region_safe":
            keep, reason = get_light_scar_region_safe(m)
        elif variant == "dark_spot_strict_plus_light_scar_v2":
            keep1, reason1 = apply_dark_spot_strict(m)
            keep2, reason2 = get_light_scar_contrast_relaxed(m)
            if not keep1: keep, reason = False, reason1
            elif not keep2: keep, reason = False, reason2
        elif variant == "dark_spot_strict_plus_cluster_aware_light_scar":
            keep1, reason1 = apply_dark_spot_strict(m)
            keep2, reason2 = get_light_scar_contrast_relaxed(m) # Base filter applied before clustering
            if not keep1: keep, reason = False, reason1
            elif not keep2: keep, reason = False, reason2

        if keep:
            retained.append(m)
        else:
            m_sup = dict(m)
            m_sup["suppression_reason"] = reason
            suppressed.append(m_sup)

    # 2nd pass for cluster logic
    if "cluster_aware" in variant or variant == "dark_spot_strict_plus_light_scar_v2":
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
            c.sort(key=lambda x: x.get("salience", 0) * x.get("confidence", 0), reverse=True)
            final_retained.append(c[0])
            for suppressed_m in c[1:]:
                m_sup = dict(suppressed_m)
                m_sup["suppression_reason"] = "cluster_duplicate"
                suppressed.append(m_sup)
                
        retained = other_marks + final_retained

    return retained, suppressed

def download_gcs_file(gcs_uri, local_path):
    if os.path.exists(local_path): return True
    try:
        subprocess.run(["gsutil", "cp", gcs_uri, local_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {gcs_uri}: {e}")
        return False

def generate_crop(image_path, centroid, output_path):
    try:
        img = Image.open(image_path)
        w, h = img.size
        cx, cy = int(centroid[0] * w), int(centroid[1] * h)
        box_size = 150
        left = max(0, cx - box_size // 2)
        top = max(0, cy - box_size // 2)
        right = min(w, cx + box_size // 2)
        bottom = min(h, cy + box_size // 2)
        crop = img.crop((left, top, right, bottom))
        crop.save(output_path)
        return True
    except Exception as e:
        print(f"Crop failed for {image_path}: {e}")
        return False

def get_basename(path):
    return path.split("/")[-1].split("\\")[-1]

def evaluate_calibration(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    crop_tmp_dir = os.path.join(output_dir, "crop_source_tmp")
    os.makedirs(crop_tmp_dir, exist_ok=True)
    
    variants = [
        "baseline_current_detector",
        "dark_spot_strict",
        "light_scar_contrast_relaxed",
        "light_scar_shape_linear",
        "light_scar_cluster_aware",
        "light_scar_region_safe",
        "dark_spot_strict_plus_light_scar_v2",
        "dark_spot_strict_plus_cluster_aware_light_scar"
    ]
    
    # Load manifest to resolve image URLs
    manifest_map = {}
    manifest_path = "validation/validation_pairs_hq_headshots.csv"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest_map[row["pair_id"]] = {
                    "probe": row["image1_url_or_gcs_path"],
                    "gallery": row["image2_url_or_gcs_path"]
                }
    
    pairs = []
    with open(input_file, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                pairs.append(json.loads(line.strip()))
            except:
                pass
                
    results = {v: {
        "images_total_marks": [], "images_hv_marks": [],
        "counts": {t: 0 for t in ["light_scar", "dark_spot", "dark_mole", "depression_scar", "linear_scar", "structural_crater"]},
        "images_gt_20_hv": 0, "images_gt_30_total": 0,
        "accepted_marks_total": 0, "generated_marks_total": 0,
        "false_support": 0, "same_retention": 0,
        "total_same_corresps_baseline": 0, "total_diff_corresps_baseline": 0,
        "accepted_per_pair": [],
        "false_support_regions": [],
        "false_support_types": []
    } for v in variants}
    
    crop_targets = {
        "retained_light_scar": [],
        "suppressed_light_scar": [],
        "same_person_light_scar_corresps": [],
        "impostor_light_scar_corresps": [],
        "retained_dark_spot_strict": [],
        "suppressed_dark_spot_strict": []
    }

    retained_csv = []
    suppressed_csv = []
    same_csv = []
    impostor_csv = []
    false_support_regions_csv = []
    
    for v in variants:
        retained_map = {}
        for p in pairs:
            pair_id = p.get("pair_id")
            for role in ["probe", "gallery"]:
                summary_key = f"raw_{role}_marks_summary"
                image_path = manifest_map.get(pair_id, {}).get(role, "")
                
                marks = p.get(summary_key, [])
                ret, sup = filter_marks(marks, v)
                
                retained_map[f"{pair_id}_{role}"] = ret
                
                for m in ret:
                    mtype = m.get("mark_type")
                    if mtype in results[v]["counts"]: results[v]["counts"][mtype] += 1
                    results[v]["generated_marks_total"] += 1
                    
                    if v == "light_scar_contrast_relaxed" and mtype == "light_scar":
                        crop_targets["retained_light_scar"].append({"mark": m, "img": image_path})
                    if v == "dark_spot_strict" and mtype == "dark_spot":
                        crop_targets["retained_dark_spot_strict"].append({"mark": m, "img": image_path})
                        
                    retained_csv.append({
                        "pair_id": pair_id, "variant_name": v, "mark_type": mtype,
                        "canonical_region": m.get("face_region"), "salience": m.get("salience")
                    })
                    
                for m in sup:
                    mtype = m.get("mark_type")
                    if v == "light_scar_contrast_relaxed" and mtype == "light_scar":
                        crop_targets["suppressed_light_scar"].append({"mark": m, "img": image_path})
                    if v == "dark_spot_strict" and mtype == "dark_spot":
                        crop_targets["suppressed_dark_spot_strict"].append({"mark": m, "img": image_path})
                        
                    suppressed_csv.append({
                        "pair_id": pair_id, "variant_name": v, "mark_type": mtype,
                        "canonical_region": m.get("face_region"),
                        "suppression_reason": m.get("suppression_reason"), "salience": m.get("salience")
                    })
                
                hv_count = sum(1 for m in ret if is_high_value(m))
                tot_count = len(ret)
                results[v]["images_total_marks"].append(tot_count)
                results[v]["images_hv_marks"].append(hv_count)

        for p in pairs:
            is_same = p.get("label_same_person", False)
            corresps = p.get("accepted_correspondences_detail", [])
            ret1 = {m.get("index") for m in retained_map.get(f"{p.get('pair_id')}_probe", [])}
            ret2 = {m.get("index") for m in retained_map.get(f"{p.get('pair_id')}_gallery", [])}
            
            retained_corresps = []
            for c in corresps:
                if v == "baseline_current_detector":
                    if is_same: results[v]["total_same_corresps_baseline"] += 1
                    else: results[v]["total_diff_corresps_baseline"] += 1
                    
                i1, i2 = c.get("probe_idx"), c.get("gallery_idx")
                if i1 is None or i2 is None: continue
                
                if i1 in ret1 and i2 in ret2:
                    retained_corresps.append(c)
                    results[v]["accepted_marks_total"] += 2
                    
                    if c.get("mark_type") == "light_scar" and v == "light_scar_contrast_relaxed":
                        item = {"corresp": c, "img": manifest_map.get(p.get("pair_id"), {}).get("gallery", ""), "centroid": c.get("regional_canonical_centroid_gallery")}
                        if is_same: crop_targets["same_person_light_scar_corresps"].append(item)
                        else: crop_targets["impostor_light_scar_corresps"].append(item)
            
            results[v]["accepted_per_pair"].append(len(retained_corresps))
            
            if is_same: 
                results[v]["same_retention"] += len(retained_corresps)
            else: 
                results[v]["false_support"] += len(retained_corresps)
            
            for c in retained_corresps:
                row = {
                    "pair_id": p.get("pair_id"), "variant_name": v,
                    "mark_type": c.get("mark_type"),
                    "canonical_region": c.get("regional_canonical_region_gallery")
                }
                if is_same: same_csv.append(row)
                else: 
                    impostor_csv.append(row)
                    false_support_regions_csv.append(row)
                    results[v]["false_support_regions"].append(c.get("regional_canonical_region_gallery"))
                    results[v]["false_support_types"].append(c.get("mark_type"))

    # Generate crops
    def crop_top(target_key, output_subfolder, is_corresp=False):
        items = crop_targets[target_key]
        if not items: return
        if is_corresp: items.sort(key=lambda x: x["corresp"].get("patch_combined_similarity", 0), reverse=True)
        else: items.sort(key=lambda x: x["mark"].get("salience", 0), reverse=True)
            
        top_items = items[:50]
        out_dir = os.path.join(output_dir, "crops", output_subfolder)
        os.makedirs(out_dir, exist_ok=True)
        
        for i, item in enumerate(top_items):
            gcs_path = item.get("img")
            if not gcs_path: continue
            local_img_path = os.path.join(crop_tmp_dir, get_basename(gcs_path))
            if download_gcs_file(gcs_path, local_img_path):
                centroid = item["centroid"] if is_corresp else item["mark"].get("centroid")
                if centroid:
                    generate_crop(local_img_path, centroid, os.path.join(out_dir, f"{i:02d}_{get_basename(gcs_path)}"))
                    
    print("Generating visual audit crops...")
    crop_top("retained_light_scar", "retained_light_scar")
    crop_top("suppressed_light_scar", "suppressed_light_scar")
    crop_top("same_person_light_scar_corresps", "same_person_light_scar_corresps", is_corresp=True)
    crop_top("impostor_light_scar_corresps", "impostor_light_scar_corresps", is_corresp=True)
    crop_top("retained_dark_spot_strict", "retained_dark_spot_strict")
    crop_top("suppressed_dark_spot_strict", "suppressed_dark_spot_strict")

    # Generate summary CSV
    summary_csv = []
    from collections import Counter
    for v in variants:
        res = results[v]
        total_same_base = max(1, results["baseline_current_detector"]["same_retention"])
        total_diff_base = max(1, results["baseline_current_detector"]["false_support"])
        
        same_ret_rate = res["same_retention"] / total_same_base if total_same_base > 0 else 0
        diff_sup_rate = 1.0 - (res["false_support"] / total_diff_base) if total_diff_base > 0 else 0
        gen = res["generated_marks_total"]
        acc = res["accepted_marks_total"]
        precision = acc / gen if gen > 0 else 0
        avg_hv = sum(res["images_hv_marks"]) / len(res["images_hv_marks"]) if res["images_hv_marks"] else 0
        
        top_false_regions = "; ".join([f"{k}({v})" for k,v in Counter(res["false_support_regions"]).most_common(3)])
        top_false_types = "; ".join([f"{k}({v})" for k,v in Counter(res["false_support_types"]).most_common(3)])
        
        summary_csv.append({
            "variant_name": v,
            "avg_hv_marks": round(avg_hv, 2),
            "light_scar_retained_count": res["counts"]["light_scar"],
            "light_scar_suppressed_count": sum(1 for row in suppressed_csv if row["variant_name"] == v and row["mark_type"] == "light_scar"),
            "same_person_retention_rate": round(same_ret_rate, 4),
            "different_person_suppression_rate": round(diff_sup_rate, 4),
            "precision_proxy": round(precision, 4),
            "false_support_count": res["false_support"],
            "top_false_support_regions": top_false_regions,
            "top_false_support_types": top_false_types
        })

    def write_csv(path, dicts):
        if not dicts: return
        with open(path, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=dicts[0].keys())
            writer.writeheader()
            writer.writerows(dicts)

    write_csv(os.path.join(output_dir, "hq_light_scar_v2_variant_summary.csv"), summary_csv)
    write_csv(os.path.join(output_dir, "hq_light_scar_retained_marks.csv"), [d for d in retained_csv if d["mark_type"] == "light_scar"])
    write_csv(os.path.join(output_dir, "hq_light_scar_suppressed_marks.csv"), [d for d in suppressed_csv if d["mark_type"] == "light_scar"])
    write_csv(os.path.join(output_dir, "hq_light_scar_same_person_correspondences.csv"), [d for d in same_csv if d["mark_type"] == "light_scar"])
    write_csv(os.path.join(output_dir, "hq_light_scar_impostor_correspondences.csv"), [d for d in impostor_csv if d["mark_type"] == "light_scar"])
    write_csv(os.path.join(output_dir, "hq_dark_spot_strict_retained_suppressed.csv"), [d for d in retained_csv+suppressed_csv if d["mark_type"] == "dark_spot"])
    write_csv(os.path.join(output_dir, "hq_false_support_regions.csv"), false_support_regions_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    evaluate_calibration(args.input, args.output_dir)
