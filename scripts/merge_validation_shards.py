import os
import sys
import json
import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Merge validation shards.")
    parser.add_argument("--input-dir", required=True, help="Directory containing shard subdirectories")
    parser.add_argument("--output-dir", required=True, help="Directory to write merged results")
    return parser.parse_args()

def _write_csv(path, rows):
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("(no results)\n")
        return
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    jsonl_files = glob.glob(os.path.join(args.input_dir, "**", "validation_results.jsonl"), recursive=True)
    if not jsonl_files:
        print(f"FATAL: No validation_results.jsonl files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} shard files.")
    
    all_results = []
    for jf in jsonl_files:
        with open(jf, "r") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    print(f"Merged {len(all_results)} total rows.")

    merged_jsonl = os.path.join(args.output_dir, "validation_results.jsonl")
    with open(merged_jsonl, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    tp, fp, tn, fn = 0, 0, 0, 0
    false_positives = []
    false_negatives = []
    conflicting_cases = []
    face_not_detected_count = 0

    for r in all_results:
        if r.get("error"):
            if "FACE_NOT_DETECTED" in r["error"]:
                face_not_detected_count += 1
            continue
            
        label = r["label_same_person"]
        score = r.get("fused_score", 0)
        veto = r.get("veto_triggered", False)
        predicted_match = score > 50.0 and not (veto and not r.get("veto_override_applied", False))

        if predicted_match and label:
            tp += 1
        elif predicted_match and not label:
            fp += 1
            false_positives.append(r)
        elif not predicted_match and not label:
            tn += 1
        elif not predicted_match and label:
            fn += 1
            false_negatives.append(r)

        face_strong = r.get("structural_sim", 0) >= 0.60
        face_weak = r.get("structural_sim", 0) < 0.40
        marks_strong = r.get("lr_marks", 1.0) > 10.0
        marks_weak = r.get("lr_marks", 1.0) <= 1.0

        if (face_weak and marks_strong) or (face_strong and marks_weak):
            conflict_type = "face_weak_marks_strong" if face_weak else "face_strong_marks_weak"
            conflicting_cases.append({**r, "conflict_type": conflict_type})

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    errors = sum(1 for r in all_results if r.get("error"))

    summary = {
        "total_pairs": len(all_results),
        "errors": errors,
        "face_not_detected_count": face_not_detected_count,
        "evaluated": total,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "FAR": round(far, 4),
        "FRR": round(frr, 4),
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "conflicting_evidence_count": len(conflicting_cases),
    }

    summary_path = os.path.join(args.output_dir, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _write_csv(os.path.join(args.output_dir, "false_positives.csv"), false_positives)
    _write_csv(os.path.join(args.output_dir, "false_negatives.csv"), false_negatives)
    _write_csv(os.path.join(args.output_dir, "conflicting_evidence_cases.csv"), conflicting_cases)

    print(f"\n{'='*60}")
    print(f"Merge complete.")
    print(f"  Total: {len(all_results)}, Evaluated: {total}, Errors: {errors} (FACE_NOT_DETECTED: {face_not_detected_count})")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Precision={precision:.4f} Recall={recall:.4f} FAR={far:.4f} FRR={frr:.4f}")
    print(f"  Results: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
