import json
import argparse
import os

def generate_hq_report(input_json, output_md, output_json):
    with open(input_json, 'r') as f:
        data = json.load(f)
        
    pairs = data.get("pairs", [])
    if not pairs:
        print("No pairs found in report.")
        return

    # 1. Quality Gate
    total_pairs = len(pairs)
    same_person_pairs = [p for p in pairs if p.get("label_same_person", False) in [True, "true", "True"]]
    diff_person_pairs = [p for p in pairs if p.get("label_same_person", False) in [False, "false", "False"]]
    
    face_not_detected = sum(1 for p in pairs if p.get("status") == "FACE_NOT_DETECTED" or p.get("error") == "FACE_NOT_DETECTED")
    evaluated_pairs = [p for p in pairs if p.get("status") == "SUCCESS"]

    resolutions = []
    crop_sizes = []
    cand_counts = []
    hv_counts = []
    ur_counts = []
    surviving_hv_corresp = []
    accepted_corresp = []
    
    for p in evaluated_pairs:
        # pair metrics
        surviving = p.get("mark_matcher_result", {}).get("telemetry", {}).get("correspondences_surviving_high_value", 0)
        surviving_hv_corresp.append(surviving)
        accepted = p.get("mark_matcher_result", {}).get("accepted_correspondences_count", 0)
        accepted_corresp.append(accepted)
        
        # image level metrics (approximated from marks if raw image metadata missing)
        # Assuming the pipeline outputs total candidate count as 'candidate_count_image1', etc.
        cand1 = len(p.get("evidence", {}).get("image1_marks", []))
        cand2 = len(p.get("evidence", {}).get("image2_marks", []))
        if cand1 > 0: cand_counts.append(cand1)
        if cand2 > 0: cand_counts.append(cand2)
        
        hv1 = sum(1 for m in p.get("evidence", {}).get("image1_marks", []) if m.get("mark_type") in ["dark_spot", "dark_mole", "light_scar"])
        hv2 = sum(1 for m in p.get("evidence", {}).get("image2_marks", []) if m.get("mark_type") in ["dark_spot", "dark_mole", "light_scar"])
        hv_counts.extend([hv1, hv2])

    perc_3hv_imgs = (sum(1 for c in hv_counts if c >= 3) / max(len(hv_counts), 1)) * 100
    perc_5hv_imgs = (sum(1 for c in hv_counts if c >= 5) / max(len(hv_counts), 1)) * 100
    perc_3hv_pairs = (sum(1 for c in surviving_hv_corresp if c >= 3) / max(len(surviving_hv_corresp), 1)) * 100
    perc_5hv_pairs = (sum(1 for c in surviving_hv_corresp if c >= 5) / max(len(surviving_hv_corresp), 1)) * 100

    report_md = f"""# HQ Phase 0 / Baseline Validation Report

## 1. Quality Gate
- **Total pairs**: {total_pairs}
- **Same-person pairs**: {len(same_person_pairs)}
- **Different-person pairs**: {len(diff_person_pairs)}
- **Evaluated pairs**: {len(evaluated_pairs)}
- **FACE_NOT_DETECTED count**: {face_not_detected}

### Mark Density
- **Avg candidate count per image**: {sum(cand_counts)/max(len(cand_counts), 1):.1f}
- **Avg high-value mark count per image**: {sum(hv_counts)/max(len(hv_counts), 1):.1f}
- **Avg accepted correspondence count per pair**: {sum(accepted_corresp)/max(len(accepted_corresp), 1):.1f}
- **Images with 3+ high-value marks**: {perc_3hv_imgs:.1f}%
- **Images with 5+ high-value marks**: {perc_5hv_imgs:.1f}%
- **Pairs with 3+ surviving high-value correspondences**: {perc_3hv_pairs:.1f}%
- **Pairs with 5+ surviving high-value correspondences**: {perc_5hv_pairs:.1f}%

*(Image resolution, face crop size, and blur metrics will require adding explicitly to the extraction pipeline if missing here)*

## 2. Detector Sanity Review
*(Requires pipeline integration with mark distribution telemetry from evaluate_strict_mark_matcher.py)*

## 3. HQ Phase 2B Rerun
*(Refer to Phase 2B telemetry in strict_mark_report.json)*

## 4. HQ Phase 3A Rerun
*(Refer to Signal Purification metrics in strict_mark_report.json)*

## 5. HQ Phase 3B Shadow Scoring
*(Refer to Precision/Recall simulation in strict_mark_report.json)*
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
    parser.add_argument("--input", default="validation_results/strict_mark_report.json")
    parser.add_argument("--output_md", default="hq_baseline_report.md")
    parser.add_argument("--output_json", default="hq_baseline_report.json")
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        generate_hq_report(args.input, args.output_md, args.output_json)
    else:
        print(f"Input file {args.input} not found. Run the validation pipeline first.")
