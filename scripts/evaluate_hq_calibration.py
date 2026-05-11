import argparse
import json
import os
import csv
import math

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
        
    if variant == "dark_spot_strict":
        for m in marks:
            if m.get("mark_type") == "dark_spot":
                area = m.get("area", 0)
                contrast = m.get("contrast", 0)
                
                # Check isolation
                isolated = True
                for other in marks:
                    if other == m: continue
                    if distance(m, other) < 0.005:
                        isolated = False
                        break
                        
                if area < 15:
                    m_sup = dict(m); m_sup["suppression_reason"] = "dark_spot_strict_area"; suppressed.append(m_sup)
                elif contrast < 3.0:
                    m_sup = dict(m); m_sup["suppression_reason"] = "dark_spot_strict_contrast"; suppressed.append(m_sup)
                elif not isolated:
                    m_sup = dict(m); m_sup["suppression_reason"] = "dark_spot_strict_cluster"; suppressed.append(m_sup)
                else:
                    retained.append(m)
            else:
                retained.append(m)
        return retained, suppressed

    if variant == "light_scar_strict":
        for m in marks:
            if m.get("mark_type") == "light_scar":
                area = m.get("area", 0)
                contrast = m.get("contrast", 0)
                confidence = m.get("confidence", 0)
                
                # Check isolation
                isolated = True
                for other in marks:
                    if other == m: continue
                    if distance(m, other) < 0.005:
                        isolated = False
                        break
                        
                if area < 50:
                    m_sup = dict(m); m_sup["suppression_reason"] = "light_scar_strict_area"; suppressed.append(m_sup)
                elif contrast < 0.15:
                    m_sup = dict(m); m_sup["suppression_reason"] = "light_scar_strict_contrast"; suppressed.append(m_sup)
                elif confidence < 0.8:
                    m_sup = dict(m); m_sup["suppression_reason"] = "light_scar_strict_confidence"; suppressed.append(m_sup)
                elif not isolated:
                    m_sup = dict(m); m_sup["suppression_reason"] = "light_scar_strict_cluster"; suppressed.append(m_sup)
                else:
                    retained.append(m)
            else:
                retained.append(m)
        return retained, suppressed

    if variant == "dark_spot_strict_plus_light_scar_strict":
        for m in marks:
            if m.get("mark_type") == "dark_spot":
                area = m.get("area", 0)
                contrast = m.get("contrast", 0)
                isolated = True
                for other in marks:
                    if other == m: continue
                    if distance(m, other) < 0.005:
                        isolated = False
                        break
                if area < 15:
                    m_sup = dict(m); m_sup["suppression_reason"] = "dark_spot_strict_area"; suppressed.append(m_sup)
                elif contrast < 3.0:
                    m_sup = dict(m); m_sup["suppression_reason"] = "dark_spot_strict_contrast"; suppressed.append(m_sup)
                elif not isolated:
                    m_sup = dict(m); m_sup["suppression_reason"] = "dark_spot_strict_cluster"; suppressed.append(m_sup)
                else:
                    retained.append(m)
            elif m.get("mark_type") == "light_scar":
                area = m.get("area", 0)
                contrast = m.get("contrast", 0)
                confidence = m.get("confidence", 0)
                isolated = True
                for other in marks:
                    if other == m: continue
                    if distance(m, other) < 0.005:
                        isolated = False
                        break
                if area < 50:
                    m_sup = dict(m); m_sup["suppression_reason"] = "light_scar_strict_area"; suppressed.append(m_sup)
                elif contrast < 0.15:
                    m_sup = dict(m); m_sup["suppression_reason"] = "light_scar_strict_contrast"; suppressed.append(m_sup)
                elif confidence < 0.8:
                    m_sup = dict(m); m_sup["suppression_reason"] = "light_scar_strict_confidence"; suppressed.append(m_sup)
                elif not isolated:
                    m_sup = dict(m); m_sup["suppression_reason"] = "light_scar_strict_cluster"; suppressed.append(m_sup)
                else:
                    retained.append(m)
            else:
                retained.append(m)
        return retained, suppressed

    if variant == "permanent_mark_only_experiment":
        allowed = ["dark_mole", "depression_scar", "linear_scar", "structural_crater"]
        for m in marks:
            if m.get("mark_type") in allowed:
                retained.append(m)
            else:
                m_sup = dict(m); m_sup["suppression_reason"] = "not_permanent"; suppressed.append(m_sup)
        return retained, suppressed

    return marks, []


def evaluate_calibration(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    variants = [
        "baseline_current_detector",
        "dark_spot_strict",
        "light_scar_strict",
        "dark_spot_strict_plus_light_scar_strict",
        "permanent_mark_only_experiment"
    ]
    
    pairs = []
    with open(input_path, "r") as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
            
    results = {v: {
        "images_total_marks": [],
        "images_hv_marks": [],
        "images_gt_20_hv": 0,
        "images_gt_30_total": 0,
        "counts": {
            "light_scar": 0,
            "dark_spot": 0,
            "dark_mole": 0,
            "depression_scar": 0,
            "linear_scar": 0,
            "structural_crater": 0
        },
        "accepted_per_pair": [],
        "false_support": 0,
        "same_retention": 0,
        "diff_suppression": 0,
        "total_same_corresps_baseline": 0,
        "total_diff_corresps_baseline": 0,
        "generated_marks_total": 0,
        "accepted_marks_total": 0
    } for v in variants}
    
    retained_csv = []
    suppressed_csv = []
    same_csv = []
    impostor_csv = []
    by_image_csv = []
    by_mark_type_csv = []
    by_region_csv = []
    
    for v in variants:
        # Precompute retained marks by image path
        retained_map = {}
        for p in pairs:
            for side, marks_key in [("probe", "raw_probe_marks_summary"), ("gallery", "raw_gallery_marks_summary")]:
                key = f"{p.get('pair_id')}_{side}"
                if key not in retained_map:
                    marks = p.get(marks_key, [])
                    # Inject index based on array position since it's missing in json
                    for i, m in enumerate(marks):
                        if "index" not in m: m["index"] = i
                    ret, sup = filter_marks(marks, v)
                    retained_map[key] = ret
                    
                    results[v]["generated_marks_total"] += len(ret)
                    
                    for m in ret:
                        mtype = m.get("mark_type")
                        if mtype in results[v]["counts"]:
                            results[v]["counts"][mtype] += 1
                            
                        # CSV export
                        if len(retained_csv) < 1000:
                            retained_csv.append({
                                "pair_id": p.get("pair_id"),
                                "image_path": key,
                                "variant_name": v,
                                "mark_type": mtype,
                                "canonical_region": m.get("face_region"),
                                "centroid": m.get("centroid"),
                                "area": m.get("area"),
                                "contrast": m.get("contrast"),
                                "confidence": m.get("confidence"),
                                "suppression_reason": "",
                                "retained": "Y"
                            })
                    for m in sup:
                        if len(suppressed_csv) < 1000:
                            suppressed_csv.append({
                                "pair_id": p.get("pair_id"),
                                "image_path": key,
                                "variant_name": v,
                                "mark_type": m.get("mark_type"),
                                "canonical_region": m.get("face_region"),
                                "centroid": m.get("centroid"),
                                "area": m.get("area"),
                                "contrast": m.get("contrast"),
                                "confidence": m.get("confidence"),
                                "suppression_reason": m.get("suppression_reason"),
                                "retained": "N"
                            })
                            
                    hv_count = sum(1 for m in ret if is_high_value(m))
                    tot_count = len(ret)
                    
                    results[v]["images_total_marks"].append(tot_count)
                    results[v]["images_hv_marks"].append(hv_count)
                    if hv_count > 20: results[v]["images_gt_20_hv"] += 1
                    if tot_count > 30: results[v]["images_gt_30_total"] += 1
                    
                    # Store for before/after by image
                    if v == "baseline_current_detector":
                        by_image_csv.append({
                            "pair_id": p.get("pair_id"),
                            "image_path": key,
                            "baseline_total": tot_count,
                            "baseline_hv": hv_count
                        })
                    
                    # Group by mark type
                    from collections import Counter
                    type_counts = Counter(m.get("mark_type") for m in ret)
                    for mtype, count in type_counts.items():
                        by_mark_type_csv.append({
                            "image_path": key,
                            "variant_name": v,
                            "mark_type": mtype,
                            "count": count
                        })
                    
                    # Group by region
                    region_counts = Counter(m.get("face_region", "unknown") for m in ret)
                    for reg, count in region_counts.items():
                        by_region_csv.append({
                            "image_path": key,
                            "variant_name": v,
                            "canonical_region": reg,
                            "count": count
                        })
                        
        # Now evaluate correspondences
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
                    
                # Strict Matcher logic: if mark was suppressed, correspondence is invalid
                i1 = c.get("probe_idx")
                i2 = c.get("gallery_idx")
                
                # If either is None, it means the structure is malformed, skip
                if i1 is None or i2 is None: continue
                
                if i1 in ret1 and i2 in ret2:
                    retained_corresps.append(c)
                    results[v]["accepted_marks_total"] += 2
            
            results[v]["accepted_per_pair"].append(len(retained_corresps))
            
            if is_same:
                results[v]["same_retention"] += len(retained_corresps)
            else:
                results[v]["false_support"] += len(retained_corresps)
                
            # Store correspondences for review
            for c in retained_corresps:
                row = {
                    "pair_id": p.get("pair_id"),
                    "variant_name": v,
                    "mark_type": c.get("mark_type"),
                    "canonical_region": c.get("regional_canonical_region_gallery"),
                    "patch_sim": c.get("patch_combined_similarity"),
                    "uv_dist": c.get("regional_uv_distance")
                }
                if is_same and len(same_csv) < 1000:
                    same_csv.append(row)
                elif not is_same and len(impostor_csv) < 1000:
                    impostor_csv.append(row)
                    
    # Generate summary CSV
    summary_csv = []
    for v in variants:
        res = results[v]
        total_same_base = max(1, results["A_baseline_current_detector"]["same_retention"])
        total_diff_base = max(1, results["A_baseline_current_detector"]["false_support"])
        
        same_ret_rate = res["same_retention"] / total_same_base
        diff_sup_rate = 1.0 - (res["false_support"] / total_diff_base)
        
        gen = res["generated_marks_total"]
        acc = res["accepted_marks_total"]
        precision = acc / gen if gen > 0 else 0
        
        avg_tot = sum(res["images_total_marks"]) / len(res["images_total_marks"]) if res["images_total_marks"] else 0
        avg_hv = sum(res["images_hv_marks"]) / len(res["images_hv_marks"]) if res["images_hv_marks"] else 0
        avg_acc = sum(res["accepted_per_pair"]) / len(res["accepted_per_pair"]) if res["accepted_per_pair"] else 0
        
        summary_csv.append({
            "variant_name": v,
            "avg_total_marks": round(avg_tot, 2),
            "avg_hv_marks": round(avg_hv, 2),
            "images_gt_20_hv": res["images_gt_20_hv"],
            "images_gt_30_total": res["images_gt_30_total"],
            "light_scar": res["counts"]["light_scar"],
            "dark_spot": res["counts"]["dark_spot"],
            "dark_mole": res["counts"]["dark_mole"],
            "depression_scar": res["counts"]["depression_scar"],
            "linear_scar": res["counts"]["linear_scar"],
            "structural_crater": res["counts"]["structural_crater"],
            "avg_accepted_per_pair": round(avg_acc, 2),
            "false_support": res["false_support"],
            "same_retention_rate": round(same_ret_rate, 4),
            "diff_suppression_rate": round(diff_sup_rate, 4),
            "precision_proxy": round(precision, 4)
        })

    def write_csv(path, dicts):
        if not dicts: return
        with open(path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=dicts[0].keys())
            writer.writeheader()
            writer.writerows(dicts)

    write_csv(os.path.join(output_dir, "hq_calibration_variant_summary.csv"), summary_csv)
    write_csv(os.path.join(output_dir, "hq_calibration_retained_marks.csv"), retained_csv)
    write_csv(os.path.join(output_dir, "hq_calibration_suppressed_marks.csv"), suppressed_csv)
    write_csv(os.path.join(output_dir, "hq_calibration_same_person_correspondences.csv"), same_csv)
    write_csv(os.path.join(output_dir, "hq_calibration_impostor_correspondences.csv"), impostor_csv)
    write_csv(os.path.join(output_dir, "hq_calibration_before_after_by_image.csv"), by_image_csv)
    write_csv(os.path.join(output_dir, "hq_calibration_before_after_by_mark_type.csv"), by_mark_type_csv)
    write_csv(os.path.join(output_dir, "hq_calibration_before_after_by_region.csv"), by_region_csv)
    
    # Dump metrics json
    with open(os.path.join(output_dir, "hq_calibration_metrics.json"), "w") as f:
        json.dump(summary_csv, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    evaluate_calibration(args.input, args.output_dir)
