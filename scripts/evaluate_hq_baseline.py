import json
import argparse
import os

def generate_hq_report(input_jsonl, strict_mark_report_json, output_md, output_json):
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

    cand_counts = []
    hv_counts = []
    surviving_hv_corresp = []
    accepted_corresp = []
    
    for p in evaluated_pairs:
        # pair metrics
        surviving = p.get("strict_mark_data", {}).get("distinctive_preserved", 0)
        surviving_hv_corresp.append(surviving)
        accepted = p.get("accepted_correspondences_count", 0)
        accepted_corresp.append(accepted)
        
        # image level metrics
        cand1 = p.get("raw_probe_marks_count", 0)
        cand2 = p.get("raw_gallery_marks_count", 0)
        cand_counts.extend([cand1, cand2])
        
        hv1 = sum(1 for m in p.get("raw_probe_marks_summary", []) if m.get("mark_type") in ["dark_spot", "dark_mole", "light_scar"])
        hv2 = sum(1 for m in p.get("raw_gallery_marks_summary", []) if m.get("mark_type") in ["dark_spot", "dark_mole", "light_scar"])
        hv_counts.extend([hv1, hv2])

    perc_3hv_imgs = (sum(1 for c in hv_counts if c >= 3) / max(len(hv_counts), 1)) * 100 if hv_counts else 0
    perc_5hv_imgs = (sum(1 for c in hv_counts if c >= 5) / max(len(hv_counts), 1)) * 100 if hv_counts else 0
    perc_3hv_pairs = (sum(1 for c in surviving_hv_corresp if c >= 3) / max(len(surviving_hv_corresp), 1)) * 100 if surviving_hv_corresp else 0
    perc_5hv_pairs = (sum(1 for c in surviving_hv_corresp if c >= 5) / max(len(surviving_hv_corresp), 1)) * 100 if surviving_hv_corresp else 0

    # Phase 2B Rerun Data
    bary_avail = strict_report.get("barycentric_available_rate", 0) * 100
    bary_same = strict_report.get("same_person_avg_barycentric_distance", 0)
    bary_diff = strict_report.get("different_person_avg_barycentric_distance", 0)
    
    reg_avail = strict_report.get("regional_available_rate", 0) * 100
    reg_same = strict_report.get("same_person_avg_regional_uv_distance", 0)
    reg_diff = strict_report.get("different_person_avg_regional_uv_distance", 0)
    
    patch_avail = strict_report.get("patch_available_rate", 0) * 100
    patch_same = strict_report.get("same_person_avg_patch_combined_similarity", 0)
    patch_diff = strict_report.get("different_person_avg_patch_combined_similarity", 0)

    # Phase 3A
    sp = strict_report.get("signal_purification_filters", {})
    sp_incl = sp.get("include_all_current", {}).get("uv_separation_delta", 0)
    sp_hv = sp.get("high_value_types_only", {}).get("uv_separation_delta", 0)
    sp_dist = sp.get("distinctive_only", {}).get("uv_separation_delta", 0)

    # Phase 3B
    same_avg_marks = strict_report.get("same_person_avg_scoring_eligible_correspondences", 0)
    diff_avg_marks = strict_report.get("different_person_avg_scoring_eligible_correspondences", 0)
    fn_count = strict_report.get("false_negative_count", 0)
    fp_count = strict_report.get("false_positive_count", 0)

    report_md = f"""# HQ Phase 0 / Baseline Validation Report

## 1. Quality Gate
- **Total pairs**: {total_pairs}
- **Same-person pairs**: {len(same_person_pairs)}
- **Different-person pairs**: {len(diff_person_pairs)}
- **Evaluated pairs**: {len(evaluated_pairs)}
- **FACE_NOT_DETECTED errors**: {face_not_detected}
- **Image dimension errors**: {image_too_large}

### Mark Density
- **Avg candidate count per image**: {sum(cand_counts)/max(len(cand_counts), 1):.1f}
- **Avg high-value mark count per image**: {sum(hv_counts)/max(len(hv_counts), 1):.1f}
- **Avg accepted correspondence count per pair**: {sum(accepted_corresp)/max(len(accepted_corresp), 1):.1f}
- **Images with 3+ high-value marks**: {perc_3hv_imgs:.1f}%
- **Images with 5+ high-value marks**: {perc_5hv_imgs:.1f}%
- **Pairs with 3+ surviving high-value correspondences**: {perc_3hv_pairs:.1f}%
- **Pairs with 5+ surviving high-value correspondences**: {perc_5hv_pairs:.1f}%

## 2. Phase 2B Rerun: Telemetry Fallbacks (Telemetry-Only)
- **Barycentric Cost** (Avail: {bary_avail:.1f}%): Same={bary_same:.4f}, Diff={bary_diff:.4f}
- **Regional UV Distance** (Avail: {reg_avail:.1f}%): Same={reg_same:.4f}, Diff={reg_diff:.4f}
- **Patch Combined Similarity** (Avail: {patch_avail:.1f}%): Same={patch_same:.4f}, Diff={patch_diff:.4f}
*(All metrics are derived from telemetry metadata without altering operational scoring)*

## 3. Phase 3A Rerun: Signal Purification 
- **Include All (Current)**: UV separation delta = {sp_incl:+.4f}
- **High-Value Types Only**: UV separation delta = {sp_hv:+.4f}
- **Distinctive Only**: UV separation delta = {sp_dist:+.4f}

## 4. Phase 3B Rerun: Shadow Scoring
- **Same-person average scoring correspondences**: {same_avg_marks:.2f}
- **Different-person average scoring correspondences**: {diff_avg_marks:.2f}
- **False Negatives (Strict Matcher)**: {fn_count}
- **False Positives (Strict Matcher)**: {fp_count}
*(Marks capped to LR=1.0 per Phase 3B production policy)*
"""

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

    print(f"Report generated at {output_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="hq_validation_results/validation_results.jsonl")
    parser.add_argument("--strict_json", default="hq_validation_results/strict_mark_analysis/strict_mark_report.json")
    parser.add_argument("--output_md", default="hq_baseline_report.md")
    parser.add_argument("--output_json", default="hq_baseline_report.json")
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        generate_hq_report(args.input, args.strict_json, args.output_md, args.output_json)
    else:
        print(f"Input file {args.input} not found. Run the validation pipeline first.")
