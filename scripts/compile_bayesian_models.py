#!/usr/bin/env python3
"""
Fallen - MAP-REDUCE COMPILER
Compiles Bayesian models using online/streaming algorithms.
- Welford's Algorithm for 5D Covariance and Univariate Stats.
- Reservoir Sampling for 2D Spatial KDE.
- Computes Ensemble FAR/FRR thresholds.
"""

import os
import sys
import math
import pickle
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import gaussian_kde, lognorm, norm
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    from google.cloud import storage as gcs
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Streaming Math Helpers ---

class WelfordUnivariate:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self):
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

class WelfordMultivariate:
    def __init__(self, dim):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim)
        self.cov_m2 = np.zeros((dim, dim))

    def update(self, vec):
        self.count += 1
        delta = vec - self.mean
        self.mean += delta / self.count
        delta2 = vec - self.mean
        self.cov_m2 += np.outer(delta, delta2)

    @property
    def covariance(self):
        if self.count < 2:
            return np.zeros((self.dim, self.dim))
        return self.cov_m2 / (self.count - 1)

class ReservoirSampler:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.reservoir = []
        self.seen = 0

    def update(self, item):
        self.seen += 1
        if len(self.reservoir) < self.capacity:
            self.reservoir.append(item)
        else:
            j = np.random.randint(0, self.seen)
            if j < self.capacity:
                self.reservoir[j] = item

    def get_array(self):
        return np.array(self.reservoir)

# --- Compute Similarity for Thresholds ---
def compute_fused_ensemble_score(emb_a, emb_b):
    """
    Computes Structural Sim = 60% ArcFace + 40% Facenet512
    Cosine Similarity calculation.
    """
    if "arcface" not in emb_a or "arcface" not in emb_b:
        return 0.0
    
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    
    arc_sim = cosine_sim(emb_a["arcface"], emb_b["arcface"])
    facenet_sim = cosine_sim(emb_a.get("facenet", emb_a["arcface"]), emb_b.get("facenet", emb_b["arcface"]))
    
    # 60/40 Split
    fused_sim = (arc_sim * 0.60) + (facenet_sim * 0.40)
    return max(0.0, min(1.0, float(fused_sim)))

def match_marks_spatial(marks_a, marks_b, max_dist=0.15):
    if not marks_a or not marks_b:
        return []
    n_a, n_b = len(marks_a), len(marks_b)
    cost = np.full((n_a, n_b), 1e6)
    for i, ma in enumerate(marks_a):
        for j, mb in enumerate(marks_b):
            d = math.sqrt((ma["centroid"][0] - mb["centroid"][0])**2 + (ma["centroid"][1] - mb["centroid"][1])**2)
            if d < max_dist:
                cost[i, j] = d
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < max_dist:
            matched.append((marks_a[r], marks_b[c]))
    return matched

# --- Main Compiler ---
def compile_models(chunks_dir, output_file):
    print(f"Scanning for chunks in {chunks_dir}...")
    chunk_files = sorted(list(Path(chunks_dir).glob("chunk_*.pkl")))
    if not chunk_files:
        print("FATAL: No chunk files found.")
        sys.exit(1)

    # State objects
    spatial_sampler = ReservoirSampler(capacity=50000)
    area_stats = WelfordUnivariate()
    intensity_stats = WelfordUnivariate()
    circularity_stats = WelfordUnivariate()
    
    delta_stats = WelfordMultivariate(dim=5)
    
    # Accumulate similarity scores for FAR/FRR calculation
    # We will also use reservoir sampling for threshold calculations if the dataset is massive
    same_person_scores = []
    diff_person_scores = []
    
    # Track images per person to construct intra-person deltas
    person_records = {} # memory bound, but stores just refs/metadata. Alternatively, if memory is very tight, we process this in another pass.
    # For now, we store metadata in memory to construct pairs
    
    total_marks = 0
    total_images = 0
    
    print("Reading chunks and streaming data...")
    for chunk_path in tqdm(chunk_files, desc="Processing chunks"):
        with open(chunk_path, "rb") as f:
            chunk_data = pickle.load(f)
            
        for record in chunk_data:
            total_images += 1
            person = record["person"]
            marks = record["marks"]
            embeds = record["ensemble_embeds"]
            
            if person not in person_records:
                person_records[person] = []
            
            # Store lightweight representation for pairing
            person_records[person].append({
                "marks": marks,
                "embeds": embeds
            })
            
            for m in marks:
                total_marks += 1
                spatial_sampler.update([m["centroid"][0], m["centroid"][1]])
                area_stats.update(m["area"])
                intensity_stats.update(m["intensity"])
                circularity_stats.update(m["circularity"])

    # Phase 2: Pairing for deltas and FAR/FRR
    print("Computing Intra-Person Deltas and FAR/FRR scores...")
    persons = list(person_records.keys())
    
    for i, person in enumerate(tqdm(persons, desc="Evaluating pairs")):
        records = person_records[person]
        
        # True Positives (Same person pairs)
        for j in range(len(records)):
            for k in range(j + 1, min(len(records), j + 4)): # limit max pairs per person
                rec_a = records[j]
                rec_b = records[k]
                
                # FAR/FRR Score
                fused_sim = compute_fused_ensemble_score(rec_a["embeds"], rec_b["embeds"])
                same_person_scores.append(fused_sim)
                
                # Mark deltas
                matched = match_marks_spatial(rec_a["marks"], rec_b["marks"])
                for ma, mb in matched:
                    delta = np.array([
                        ma["centroid"][0] - mb["centroid"][0],
                        ma["centroid"][1] - mb["centroid"][1],
                        ma["area"] - mb["area"],
                        ma["intensity"] - mb["intensity"],
                        ma["circularity"] - mb["circularity"]
                    ])
                    delta_stats.update(delta)
        
        # True Negatives (Different person pairs) - pick a random record from next person
        if i < len(persons) - 1:
            diff_person = persons[i+1]
            if len(person_records[diff_person]) > 0 and len(records) > 0:
                rec_a = records[0]
                rec_b = person_records[diff_person][0]
                fused_sim = compute_fused_ensemble_score(rec_a["embeds"], rec_b["embeds"])
                diff_person_scores.append(fused_sim)

    # Compile Final Models
    print("Compiling final distributions...")
    
    # KDE
    xy = spatial_sampler.get_array().T
    if xy.shape[1] > 0:
        spatial_kde = gaussian_kde(xy, bw_method="scott")
    else:
        spatial_kde = None
        
    # FAR/FRR Calibration
    # Evaluate at various thresholds [0.10, 0.15, ... 0.95]
    thresholds = {}
    tp_array = np.array(same_person_scores)
    tn_array = np.array(diff_person_scores)
    
    for t in np.arange(0.10, 1.0, 0.05):
        t_val = round(t, 2)
        if len(tp_array) > 0:
            fn = np.sum(tp_array < t_val)
            frr = float(fn / len(tp_array))
        else:
            frr = 1.0
            
        if len(tn_array) > 0:
            fp = np.sum(tn_array >= t_val)
            far = float(fp / len(tn_array))
        else:
            far = 1.0
            
        thresholds[f"{t_val:.2f}"] = {
            "far": far,
            "frr": frr
        }
        
    covariance = delta_stats.covariance
    # Regularize covariance to ensure positive-definiteness
    covariance += np.eye(5) * 1e-8
        
    calibration = {
        "version": "Fallen Tier4 Bayesian Calibration v2.0 (Hyperscale)",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": "Distributed Extracted Chunks",
        "population_size": total_images,
        "total_marks": total_marks,
        "epsilon_floor": 1e-9,
        "ensemble": {
            "thresholds": thresholds
        },
        "spatial_kde": spatial_kde,
        "area_distribution": {
            # Approximating lognorm or just storing mean/variance for fallback
            "mean": float(area_stats.mean),
            "variance": float(area_stats.variance),
            # Ideally lognorm fitting from Welford isn't exact, fallback to gaussian or method of moments.
            # Passing raw mean/std to keep it mathematically stable without loading all points.
        },
        "intensity_distribution": {"mean": float(intensity_stats.mean), "std": float(np.sqrt(intensity_stats.variance))},
        "circularity_distribution": {"mean": float(circularity_stats.mean), "std": float(np.sqrt(circularity_stats.variance))},
        "intra_person_delta": {
            "mean": delta_stats.mean.tolist(),
            "covariance": covariance.tolist(),
            "n_deltas": delta_stats.count
        }
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(calibration, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    size_kb = os.path.getsize(output_file) / 1024
    print(f"\nModel saved: {output_file} ({size_kb:.1f} KB)")
    print("Compilation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Bayesian models.")
    parser.add_argument("--chunks", type=str, default=str(PROJECT_ROOT / "datasets" / "extracted_chunks"), help="Path to input chunks directory")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "backend" / "calibration_data" / "tier4_population_model.pkl"), help="Path to output model file")
    args = parser.parse_args()
    
    compile_models(args.chunks, args.output)
