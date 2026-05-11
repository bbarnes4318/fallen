import json
import os
import csv
import math
import argparse

def distance(m1, m2):
    c1 = m1.get("centroid", [0, 0])
    c2 = m2.get("centroid", [0, 0])
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def is_high_value(m):
    return m.get("mark_type") in ["dark_spot", "dark_mole", "light_scar"]

def run_sweep(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
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
    else:
        print("Manifest not found")
        return
        
    pairs = []
    with open(input_file, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                pairs.append(json.loads(line.strip()))
            except:
                pass
                
    # Define sweep grid
    contrast_thresholds = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    area_thresholds = [8, 10, 15, 20, 30, 50]
    confidence_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    cluster_radii = [0.01, 0.02, 0.03, 0.05]
    max_marks_per_cluster = [1, 2, 3]
    
    # Pre-parse baseline to avoid repeating
    total_same_corresps_baseline = 0
    total_diff_corresps_baseline = 0
    
    baseline_retained_map = {}
    for p in pairs:
        pair_id = p.get("pair_id")
        for role in ["probe", "gallery"]:
            summary_key = f"raw_{role}_marks_summary"
            marks = p.get(summary_key, [])
            for i, m in enumerate(marks):
                m["index"] = i
            baseline_retained_map[f"{pair_id}_{role}"] = marks

    for p in pairs:
        is_same = manifest_map.get(p.get("pair_id"), {}).get("is_same", False)
        corresps = p.get("accepted_correspondences_detail", [])
        ret1 = {m.get("index") for m in baseline_retained_map.get(f"{p.get('pair_id')}_probe", [])}
        ret2 = {m.get("index") for m in baseline_retained_map.get(f"{p.get('pair_id')}_gallery", [])}
        for c in corresps:
            i1, i2 = c.get("probe_idx"), c.get("gallery_idx")
            if i1 is None or i2 is None: continue
            if i1 in ret1 and i2 in ret2:
                if is_same: total_same_corresps_baseline += 1
                else: total_diff_corresps_baseline += 1

    total_same_base = max(1, total_same_corresps_baseline)
    total_diff_base = max(1, total_diff_corresps_baseline)
    
    print(f"Baseline Same-Person Corresps: {total_same_corresps_baseline}")
    print(f"Baseline Diff-Person Corresps: {total_diff_corresps_baseline}")

    results = []
    
    total_perms = len(contrast_thresholds) * len(area_thresholds) * len(confidence_thresholds) * len(cluster_radii) * len(max_marks_per_cluster)
    print(f"Running {total_perms} permutations...")

    count = 0
    for contrast_t in contrast_thresholds:
        for area_t in area_thresholds:
            for conf_t in confidence_thresholds:
                for rad_t in cluster_radii:
                    for max_marks in max_marks_per_cluster:
                        count += 1
                        if count % 100 == 0: print(f"Processing {count}/{total_perms}")
                        
                        v_name = f"c{contrast_t}_a{area_t}_conf{conf_t}_r{rad_t}_m{max_marks}"
                        
                        same_retention = 0
                        false_support = 0
                        generated_marks_total = 0
                        accepted_marks_total = 0
                        images_hv_marks = []
                        light_scar_retained_count = 0
                        light_scar_suppressed_count = 0
                        
                        retained_map = {}
                        
                        for p in pairs:
                            pair_id = p.get("pair_id")
                            for role in ["probe", "gallery"]:
                                marks = baseline_retained_map.get(f"{pair_id}_{role}", [])
                                retained = []
                                suppressed_count_img = 0
                                
                                # Step 1: Base Filter
                                for m in marks:
                                    mtype = m.get("mark_type")
                                    if mtype != "light_scar":
                                        retained.append(m)
                                        continue
                                        
                                    area = m.get("area") or 0
                                    # Note: using salience as contrast if contrast_score is not present
                                    contrast = m.get("contrast_score", m.get("salience", 0)) or 0
                                    confidence = m.get("confidence") or 0
                                    
                                    if area < area_t or contrast < contrast_t or confidence < conf_t:
                                        suppressed_count_img += 1
                                        continue
                                        
                                    retained.append(m)
                                
                                # Step 2: Clustering for light scars
                                final_retained = []
                                light_scars = [m for m in retained if m.get("mark_type") == "light_scar"]
                                other_marks = [m for m in retained if m.get("mark_type") != "light_scar"]
                                
                                clusters = []
                                for m in light_scars:
                                    added = False
                                    for c in clusters:
                                        if any(distance(m, cm) < rad_t for cm in c):
                                            c.append(m)
                                            added = True
                                            break
                                    if not added:
                                        clusters.append([m])
                                        
                                for c in clusters:
                                    # Sort by salience * confidence
                                    c.sort(key=lambda x: (x.get("contrast_score", x.get("salience", 0))) * (x.get("confidence") or 0), reverse=True)
                                    final_retained.extend(c[:max_marks])
                                    suppressed_count_img += len(c) - len(c[:max_marks])
                                        
                                ret = other_marks + final_retained
                                retained_map[f"{pair_id}_{role}"] = ret
                                
                                generated_marks_total += len(ret)
                                hv_count = sum(1 for m in ret if is_high_value(m))
                                images_hv_marks.append(hv_count)
                                light_scar_retained_count += len(final_retained)
                                light_scar_suppressed_count += suppressed_count_img

                        for p in pairs:
                            is_same = manifest_map.get(p.get("pair_id"), {}).get("is_same", False)
                            corresps = p.get("accepted_correspondences_detail", [])
                            ret1 = {m.get("index") for m in retained_map.get(f"{p.get('pair_id')}_probe", [])}
                            ret2 = {m.get("index") for m in retained_map.get(f"{p.get('pair_id')}_gallery", [])}
                            
                            num_retained = 0
                            for c in corresps:
                                i1, i2 = c.get("probe_idx"), c.get("gallery_idx")
                                if i1 is None or i2 is None: continue
                                if i1 in ret1 and i2 in ret2:
                                    num_retained += 1
                            
                            accepted_marks_total += num_retained * 2
                            if is_same: same_retention += num_retained
                            else: false_support += num_retained
                                
                        same_ret_rate = same_retention / total_same_base
                        diff_sup_rate = 1.0 - (false_support / total_diff_base)
                        precision = accepted_marks_total / generated_marks_total if generated_marks_total > 0 else 0
                        avg_hv = sum(images_hv_marks) / len(images_hv_marks) if images_hv_marks else 0
                        
                        results.append({
                            "variant_name": v_name,
                            "contrast_threshold": contrast_t,
                            "area_threshold": area_t,
                            "confidence_threshold": conf_t,
                            "cluster_radius": rad_t,
                            "max_marks_per_cluster": max_marks,
                            "avg_hv_marks": round(avg_hv, 2),
                            "light_scar_retained_count": light_scar_retained_count,
                            "light_scar_suppressed_count": light_scar_suppressed_count,
                            "same_person_retention_rate": round(same_ret_rate, 4),
                            "different_person_suppression_rate": round(diff_sup_rate, 4),
                            "precision_proxy": round(precision, 4),
                            "false_support_count": false_support
                        })

    # Output full results
    def write_csv(path, dicts):
        if not dicts: return
        keys = []
        for d in dicts:
            for k in d.keys():
                if k not in keys: keys.append(k)
        with open(path, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(dicts)

    write_csv(os.path.join(output_dir, "sweep_hq_calibration_all_results.csv"), results)
    
    # Pareto calculation
    pareto_frontier = []
    valid_results = [r for r in results if r["precision_proxy"] > 0.7407 and r["same_person_retention_rate"] >= 0.80 and r["different_person_suppression_rate"] >= 0.10]
    
    def is_strictly_better(r1, r2):
        return (r1["precision_proxy"] >= r2["precision_proxy"] and 
                r1["same_person_retention_rate"] >= r2["same_person_retention_rate"] and 
                r1["different_person_suppression_rate"] >= r2["different_person_suppression_rate"] and 
                (r1["precision_proxy"] > r2["precision_proxy"] or 
                 r1["same_person_retention_rate"] > r2["same_person_retention_rate"] or 
                 r1["different_person_suppression_rate"] > r2["different_person_suppression_rate"]))

    for r in results:
        dominated = False
        for other in results:
            if is_strictly_better(other, r):
                dominated = True
                break
        if not dominated:
            pareto_frontier.append(r)
            
    pareto_frontier.sort(key=lambda x: x["precision_proxy"], reverse=True)
    write_csv(os.path.join(output_dir, "sweep_hq_calibration_pareto_frontier.csv"), pareto_frontier)
    
    valid_results.sort(key=lambda x: x["precision_proxy"], reverse=True)
    write_csv(os.path.join(output_dir, "sweep_hq_calibration_summary.csv"), valid_results)
    
    print("\nMinimum Wiring Bar Check:")
    if valid_results:
        print(f"Found {len(valid_results)} configurations that beat the minimum bar!")
    else:
        print("No detector calibration variant is ready. Manual audit or new detector features are required.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_sweep(args.input, args.output_dir)
