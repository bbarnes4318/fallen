"""
Validate Embeddings — Offline embedding-only diagnostic.

*** NON-PRODUCTION-EQUIVALENT ***
This script extracts real ArcFace and Facenet512 embeddings from validation images
but bypasses the mark detection, LBP, geometry, and Bayesian fusion stages.
It is designed for rapid threshold sweeps and model comparison only.

It tests:
  - ArcFace alone
  - Facenet512 alone
  - Current 60/40 ensemble
  - Alternative ensemble weights (50/50, 70/30, 80/20)
  - Full threshold sweep from 0.00 to 1.00

Outputs:
  - threshold_sweep.csv
  - embedding_metrics.json

Usage:
  python scripts/validate_embeddings.py --manifest validation/validation_pairs.csv --output-dir validation/results/embeddings
"""

import argparse
import csv
import json
import math
import os
import sys
import time

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embedding-only diagnostic (Non-Production-Equivalent)."
    )
    parser.add_argument("--manifest", required=True, help="Path to validation_pairs.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to write results")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest only")
    return parser.parse_args()


def extract_pair_embeddings(pair, modules):
    """Fetch images and extract ArcFace + Facenet512 embeddings for one pair.

    Uses the same image loading, CLAHE, alignment, and embedding extraction
    as the production pipeline to ensure embedding equivalence.
    """
    fetch_image_from_url = modules["fetch_image_from_url"]
    apply_clahe = modules["apply_clahe"]
    align_face_crop = modules["align_face_crop"]
    extract_arcface_embedding = modules["extract_arcface_embedding"]
    extract_facenet_embedding = modules["extract_facenet_embedding"]
    calculate_cosine_similarity = modules["calculate_cosine_similarity"]
    cross_spectral_normalize = modules["cross_spectral_normalize"]

    # Load images
    img1, _ = fetch_image_from_url(pair["image1_url_or_gcs_path"])
    img2, _ = fetch_image_from_url(pair["image2_url_or_gcs_path"])

    # CLAHE
    img1_clahe = apply_clahe(img1)
    img2_clahe = apply_clahe(img2)

    # Align
    aligned1, lm1 = align_face_crop(img1_clahe)
    aligned2, lm2 = align_face_crop(img2_clahe)

    if lm1 is None or lm2 is None:
        return None  # Face not detected

    # Cross-spectral normalize
    aligned1, aligned2, _ = cross_spectral_normalize(aligned1, aligned2)

    # Extract embeddings
    arc1 = extract_arcface_embedding(aligned1)
    arc2 = extract_arcface_embedding(aligned2)
    fn1 = extract_facenet_embedding(aligned1)
    fn2 = extract_facenet_embedding(aligned2)

    arc_sim = calculate_cosine_similarity(arc1, arc2)
    fn_sim = calculate_cosine_similarity(fn1, fn2)

    return {
        "arcface_sim": arc_sim,
        "facenet_sim": fn_sim,
    }


def sweep_thresholds(scores, score_key):
    """Sweep thresholds from 0.0 to 1.0 and compute FAR/FRR at each point."""
    import numpy as np
    thresholds = np.arange(0.0, 1.005, 0.01)
    results = []
    for t in thresholds:
        tp, fp, tn, fn_count = 0, 0, 0, 0
        for s in scores:
            predicted = s[score_key] >= t
            actual = s["label_same_person"]
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and not actual:
                tn += 1
            else:
                fn_count += 1

        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn_count / (fn_count + tp) if (fn_count + tp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0

        results.append({
            "threshold": round(float(t), 3),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn_count,
            "FAR": round(far, 6), "FRR": round(frr, 6),
            "precision": round(precision, 6), "recall": round(recall, 6),
        })
    return results


def compute_eer(sweep_results):
    """Find the approximate EER (where FAR ~ FRR)."""
    best = min(sweep_results, key=lambda x: abs(x["FAR"] - x["FRR"]))
    return {
        "eer": round((best["FAR"] + best["FRR"]) / 2, 6),
        "eer_threshold": best["threshold"],
        "eer_far": best["FAR"],
        "eer_frr": best["FRR"],
    }


def compute_auc(sweep_results):
    """Compute AUC using the trapezoidal rule on the ROC curve (FAR vs TPR)."""
    # Sort by FAR ascending
    points = sorted(sweep_results, key=lambda x: x["FAR"])
    auc = 0.0
    for i in range(1, len(points)):
        dx = points[i]["FAR"] - points[i - 1]["FAR"]
        tpr_cur = 1.0 - points[i]["FRR"]
        tpr_prev = 1.0 - points[i - 1]["FRR"]
        auc += dx * (tpr_cur + tpr_prev) / 2.0
    return round(auc, 6)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate manifest
    pairs = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)
    print(f"Manifest loaded: {len(pairs)} pairs")

    if args.dry_run:
        print("DRY-RUN complete. Manifest valid.")
        return

    # Import backend modules
    try:
        from main import (
            fetch_image_from_url,
            apply_clahe,
            align_face_crop,
            extract_arcface_embedding,
            extract_facenet_embedding,
            calculate_cosine_similarity,
            cross_spectral_normalize,
        )
    except ImportError as e:
        print(f"FATAL: Cannot import backend modules. Run inside the backend container.\n{e}")
        sys.exit(1)

    modules = {
        "fetch_image_from_url": fetch_image_from_url,
        "apply_clahe": apply_clahe,
        "align_face_crop": align_face_crop,
        "extract_arcface_embedding": extract_arcface_embedding,
        "extract_facenet_embedding": extract_facenet_embedding,
        "calculate_cosine_similarity": calculate_cosine_similarity,
        "cross_spectral_normalize": cross_spectral_normalize,
    }

    # Extract embeddings for all pairs
    scores = []
    for idx, pair in enumerate(pairs):
        pair_id = pair["pair_id"]
        label = pair["label_same_person"].strip().lower() == "true"
        print(f"[{idx+1}/{len(pairs)}] Extracting embeddings for {pair_id}...", end=" ", flush=True)

        try:
            result = extract_pair_embeddings(pair, modules)
            if result is None:
                print("SKIP (face not detected)")
                continue

            scores.append({
                "pair_id": pair_id,
                "label_same_person": label,
                "category": pair.get("category", "unknown"),
                "arcface_sim": result["arcface_sim"],
                "facenet_sim": result["facenet_sim"],
            })
            print(f"arc={result['arcface_sim']:.4f} fn={result['facenet_sim']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    if not scores:
        print("ERROR: No valid embedding pairs extracted. Exiting.")
        sys.exit(1)

    # ── Compute ensemble scores for all weight configurations ──
    ensemble_configs = {
        "arcface_alone": (1.0, 0.0),
        "facenet_alone": (0.0, 1.0),
        "ensemble_60_40": (0.6, 0.4),
        "ensemble_50_50": (0.5, 0.5),
        "ensemble_70_30": (0.7, 0.3),
        "ensemble_80_20": (0.8, 0.2),
    }

    all_sweep_rows = []
    metrics = {}

    for config_name, (w_arc, w_fn) in ensemble_configs.items():
        print(f"\nSweeping thresholds for: {config_name} (ArcFace={w_arc}, Facenet={w_fn})")

        # Compute the weighted score for each pair
        for s in scores:
            s[f"score_{config_name}"] = (s["arcface_sim"] * w_arc) + (s["facenet_sim"] * w_fn)

        sweep = sweep_thresholds(scores, f"score_{config_name}")
        eer_info = compute_eer(sweep)
        auc = compute_auc(sweep)

        metrics[config_name] = {
            "weights": {"arcface": w_arc, "facenet": w_fn},
            "AUC": auc,
            **eer_info,
            "total_pairs": len(scores),
        }

        for row in sweep:
            row["configuration"] = config_name
            all_sweep_rows.append(row)

    # ── Write outputs ──
    sweep_path = os.path.join(args.output_dir, "threshold_sweep.csv")
    with open(sweep_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["configuration", "threshold", "TP", "FP", "TN", "FN", "FAR", "FRR", "precision", "recall"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_sweep_rows)

    metrics_path = os.path.join(args.output_dir, "embedding_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("Embedding validation complete.")
    for name, m in metrics.items():
        print(f"  {name}: AUC={m['AUC']}, EER={m['eer']} @ threshold={m['eer_threshold']}")
    print(f"  Results: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
