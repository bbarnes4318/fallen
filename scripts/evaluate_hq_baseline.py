import json
import argparse
import os
import csv
from collections import defaultdict

def write_csv(filename, headers, rows):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def generate_hq_report(input_jsonl, strict_mark_report_json, output_md, output_json, output_dir="."):
    pairs = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
        
    if not pairs:
        print("No pairs found in report.")
        return

    # Load strict mark report JSON for Phases 2B, 3A, 3B
    strict_report = {}
    if os.path.exists(strict_mark_report_json):
        with open(strict_mark_report_json, 'r') as f:
            strict_report = json.load(f)

    # 1. Quality Gate
    total_pairs = len(pairs)
    same_person_pairs = [p for p in pairs if p.get("label_same_person", False) in [True, "true", "True"]]
    diff_person_pairs = [p for p in pairs if p.get("label_same_person", False) in [False, "false", "False"]]
    
    face_not_detected = sum(1 for p in pairs if p.get("error") == "FACE_NOT_DETECTED")
    image_too_large = sum(1 for p in pairs if p.get("error") and "Image dimensions exceed" in p.get("error"))
    evaluated_pairs = [p for p in pairs if not p.get("error")]

    # 2. FACE_NOT_DETECTED Failures
    failure_rows = []
    for p in pairs:
        if p.get("error") == "FACE_NOT_DETECTED":
            pair_id = p.get("pair_id")
            # Usually only one image fails. We don't have the exact image from the error string, but we can log the pair.
            failure_rows.append([
                pair_id,
                p.get("image1_path", ""),
                p.get("image2_path", ""),
                p.get("error"),
                "Likely profile angle, occlusion, or extreme crop"
            ])
    write_csv(os.path.join(output_dir, "hq_failure_diagnostics.csv"), 
              ["pair_id", "image1_path", "image2_path", "error", "likely_reason"], failure_rows)

    cand_counts = []
    hv_counts = []
    surviving_hv_corresp = []
    accepted_corresp = []
    
    # Store images info for ranking
    images_info = []
    
    # Mark type reliability
    mark_type_stats = defaultdict(lambda: {"raw": 0, "accepted": 0, "same": 0, "diff": 0, 
                                           "same_uv": [], "diff_uv": [],
                                           "same_patch": [], "diff_patch": [],
                                           "same_qual": [], "diff_qual": [],
                                           "false_support": 0})
    
    # Region reliability
    region_stats = defaultdict(lambda: {"raw": 0, "accepted": 0, "same": 0, "diff": 0, 
                                        "same_uv": [], "diff_uv": [],
                                        "false_support": 0})

    # Correspondences for ranking
    all_correspondences = []
    
    over_detected_images = []

    for p in evaluated_pairs:
        pair_id = p.get("pair_id")
        is_same = p.get("label_same_person", False) in [True, "true", "True"]
        
        # pair metrics
        surviving = p.get("strict_mark_data", {}).get("distinctive_preserved", 0)
        surviving_hv_corresp.append(surviving)
        accepted = p.get("accepted_correspondences_count", 0)
        accepted_corresp.append(accepted)
        
        # image level metrics
        cand1 = p.get("raw_probe_marks_count", 0)
        cand2 = p.get("raw_gallery_marks_count", 0)
        cand_counts.extend([cand1, cand2])
        
        m_probe = p.get("raw_probe_marks_summary", [])
        m_gallery = p.get("raw_gallery_marks_summary", [])
        
        hv1 = sum(1 for m in m_probe if m.get("mark_type") in ["dark_spot", "dark_mole", "light_scar"])
        hv2 = sum(1 for m in m_gallery if m.get("mark_type") in ["dark_spot", "dark_mole", "light_scar"])
        hv_counts.extend([hv1, hv2])
        
        noisy1 = sum(1 for m in m_probe if m.get("mark_type") in ["light_scar", "texture_noise", "unknown"])
        noisy2 = sum(1 for m in m_gallery if m.get("mark_type") in ["light_scar", "texture_noise", "unknown"])
        
        images_info.append({"pair_id": pair_id, "image_path": p.get("image1_path"), "high_value_count": hv1, "noisy_count": noisy1, "total_marks": cand1})
        images_info.append({"pair_id": pair_id, "image_path": p.get("image2_path"), "high_value_count": hv2, "noisy_count": noisy2, "total_marks": cand2})
        
        # Over-detection checks
        for img_info in images_info[-2:]:
            hv = img_info["high_value_count"]
            tot = img_info["total_marks"]
            if hv > 20 or tot > 30 or (tot > 0 and hv/tot > 0.8):
                over_detected_images.append(img_info)

        # Correspondences analysis
        corresps = p.get("accepted_correspondences_detail", [])
        for c in corresps:
            mtype = c.get("mark_type", "unknown")
            cregion = c.get("regional_canonical_region_gallery", "unknown")
            uv_dist = c.get("regional_uv_distance", 0)
            patch_sim = c.get("patch_combined_similarity", 0)
            qual = c.get("match_quality", 0)
            
            mark_type_stats[mtype]["accepted"] += 1
            region_stats[cregion]["accepted"] += 1
            
            if is_same:
                mark_type_stats[mtype]["same"] += 1
                mark_type_stats[mtype]["same_uv"].append(uv_dist)
                mark_type_stats[mtype]["same_patch"].append(patch_sim)
                mark_type_stats[mtype]["same_qual"].append(qual)
                region_stats[cregion]["same"] += 1
                region_stats[cregion]["same_uv"].append(uv_dist)
            else:
                mark_type_stats[mtype]["diff"] += 1
                mark_type_stats[mtype]["diff_uv"].append(uv_dist)
                mark_type_stats[mtype]["diff_patch"].append(patch_sim)
                mark_type_stats[mtype]["diff_qual"].append(qual)
                mark_type_stats[mtype]["false_support"] += 1
                region_stats[cregion]["diff"] += 1
                region_stats[cregion]["diff_uv"].append(uv_dist)
                region_stats[cregion]["false_support"] += 1

            all_correspondences.append({
                "pair_id": pair_id,
                "label_same_person": is_same,
                "image1_gcs_path": p.get("image1_path"),
                "image2_gcs_path": p.get("image2_path"),
                "mark_type": mtype,
                "mark_class": c.get("mark_class_gallery"),
                "canonical_region": cregion,
                "face_region": c.get("face_region"),
                "centroid_gallery": c.get("gallery_centroid"),
                "centroid_probe": c.get("probe_centroid"),
                "match_quality": qual,
                "regional_uv_distance": uv_dist,
                "patch_similarity": patch_sim,
                "reason_selected_for_review": "Top quality" if qual > 0.8 else ("False support" if not is_same else "General review")
            })

    # Sort and slice top images
    images_info_sorted_hv = sorted(images_info, key=lambda x: x["high_value_count"], reverse=True)
    images_info_sorted_noisy = sorted(images_info, key=lambda x: x["noisy_count"], reverse=True)
    
    write_csv(os.path.join(output_dir, "hq_top_high_value_images.csv"), 
              ["pair_id", "image_path", "high_value_count", "total_marks"], 
              [[r["pair_id"], r["image_path"], r["high_value_count"], r["total_marks"]] for r in images_info_sorted_hv[:20]])
              
    write_csv(os.path.join(output_dir, "hq_top_noisy_images.csv"), 
              ["pair_id", "image_path", "noisy_count", "total_marks"], 
              [[r["pair_id"], r["image_path"], r["noisy_count"], r["total_marks"]] for r in images_info_sorted_noisy[:20]])

    # Top correspondences
    c_fields = ["pair_id", "label_same_person", "image1_gcs_path", "image2_gcs_path", "mark_type", "mark_class", "canonical_region", "face_region", "centroid_gallery", "centroid_probe", "match_quality", "regional_uv_distance", "patch_similarity", "reason_selected_for_review"]
    c_sorted_qual = sorted(all_correspondences, key=lambda x: x["match_quality"], reverse=True)
    c_impostors = [c for c in all_correspondences if not c["label_same_person"]]
    c_sorted_impostor = sorted(c_impostors, key=lambda x: x["match_quality"], reverse=True)
    
    write_csv(os.path.join(output_dir, "hq_top_match_quality_correspondences.csv"), c_fields, [[c[f] for f in c_fields] for c in c_sorted_qual[:20]])
    write_csv(os.path.join(output_dir, "hq_top_impostor_correspondences.csv"), c_fields, [[c[f] for f in c_fields] for c in c_sorted_impostor[:20]])
    
    # General sanity review (sample of 100)
    write_csv(os.path.join(output_dir, "hq_detector_sanity_review.csv"), c_fields, [[c[f] for f in c_fields] for c in c_sorted_qual[:100]])

    perc_3hv_imgs = (sum(1 for c in hv_counts if c >= 3) / max(len(hv_counts), 1)) * 100 if hv_counts else 0
    perc_5hv_imgs = (sum(1 for c in hv_counts if c >= 5) / max(len(hv_counts), 1)) * 100 if hv_counts else 0
    perc_3hv_pairs = (sum(1 for c in surviving_hv_corresp if c >= 3) / max(len(surviving_hv_corresp), 1)) * 100 if surviving_hv_corresp else 0
    perc_5hv_pairs = (sum(1 for c in surviving_hv_corresp if c >= 5) / max(len(surviving_hv_corresp), 1)) * 100 if surviving_hv_corresp else 0

    report_md = f"""# HQ Phase 0B Validation Report

## 1. Quality Gate
- **Total pairs**: {total_pairs}
- **Same-person pairs**: {len(same_person_pairs)}
- **Different-person pairs**: {len(diff_person_pairs)}
- **Evaluated pairs**: {len(evaluated_pairs)}
- **FACE_NOT_DETECTED errors**: {face_not_detected}
- **Image dimension errors**: {image_too_large}

### Mark Density & Over-Detection
- **Avg candidate count per image**: {sum(cand_counts)/max(len(cand_counts), 1):.1f}
- **Avg high-value mark count per image**: {sum(hv_counts)/max(len(hv_counts), 1):.1f}
- **Images exceeding over-detection thresholds (>20 HV or >30 total)**: {len(over_detected_images)}

## 2. Reliability Breakdowns
*See generated CSVs for full datasets.*

### Mark-Type Reliability
"""
    for mtype, s in mark_type_stats.items():
        avg_s_uv = sum(s["same_uv"])/max(len(s["same_uv"]),1)
        avg_d_uv = sum(s["diff_uv"])/max(len(s["diff_uv"]),1)
        avg_s_patch = sum(s["same_patch"])/max(len(s["same_patch"]),1)
        avg_d_patch = sum(s["diff_patch"])/max(len(s["diff_patch"]),1)
        priority = "High" if s["false_support"] > 5 else "Normal"
        report_md += f"- **{mtype}**: Accepted={s['accepted']}, Same={s['same']}, Diff={s['diff']}, False Support={s['false_support']}, Same Avg UV={avg_s_uv:.4f}, Diff Avg UV={avg_d_uv:.4f}, Same Patch={avg_s_patch:.4f}, Diff Patch={avg_d_patch:.4f}. Review Priority: {priority}\n"

    report_md += "\n### Region Reliability\n"
    for cregion, s in region_stats.items():
        avg_s_uv = sum(s["same_uv"])/max(len(s["same_uv"]),1)
        avg_d_uv = sum(s["diff_uv"])/max(len(s["diff_uv"]),1)
        priority = "High" if s["false_support"] > 5 else "Normal"
        report_md += f"- **{cregion}**: Accepted={s['accepted']}, Same={s['same']}, Diff={s['diff']}, False Support={s['false_support']}, Same Avg UV={avg_s_uv:.4f}, Diff Avg UV={avg_d_uv:.4f}. Review Priority: {priority}\n"

    report_md += "\n## 3. Review Recommendations\n"
    report_md += "See the following CSVs generated for manual sanity check:\n"
    report_md += "- `hq_detector_sanity_review.csv`\n"
    report_md += "- `hq_top_high_value_images.csv`\n"
    report_md += "- `hq_top_noisy_images.csv`\n"
    report_md += "- `hq_top_match_quality_correspondences.csv`\n"
    report_md += "- `hq_top_impostor_correspondences.csv`\n"
    report_md += "- `hq_failure_diagnostics.csv`\n"

    with open(output_md, "w") as f:
        f.write(report_md)
        
    out_json = {
        "quality_gate": {
            "total_pairs": total_pairs,
            "evaluated": len(evaluated_pairs),
            "face_not_detected": face_not_detected,
            "perc_3hv_imgs": perc_3hv_imgs,
            "perc_5hv_imgs": perc_5hv_imgs
        }
    }
    with open(output_json, "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"Report and CSV artifacts generated at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="hq_validation_results/validation_results.jsonl")
    parser.add_argument("--strict_json", default="hq_validation_results/strict_mark_analysis/strict_mark_report.json")
    parser.add_argument("--output_md", default="hq_baseline_report.md")
    parser.add_argument("--output_json", default="hq_baseline_report.json")
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output_md)
    if not output_dir:
        output_dir = "."
        
    if os.path.exists(args.input):
        generate_hq_report(args.input, args.strict_json, args.output_md, args.output_json, output_dir)
    else:
        print(f"Input file {args.input} not found. Run the validation pipeline first.")
