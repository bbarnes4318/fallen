#!/usr/bin/env python3
"""
Fallen - MAP-REDUCE WORKER: EXTRACTION
Extracts marks and Neural Ensemble embeddings from datasets.
Outputs chunk files for compile_bayesian_models.py.
"""

import os
import sys
import json
import math
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

# Ensure backend modules are available
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Mocked imports for main/models if necessary, or we can just import from main.
# We need align_face_crop, detect_facial_marks, extract_ensemble_embeddings
try:
    from main import apply_clahe, align_face_crop, detect_facial_marks, extract_ensemble_embeddings
except ImportError:
    print("FATAL: Could not import from backend.main. Ensure PYTHONPATH is correct.")
    sys.exit(1)

def discover_images(dataset_dir):
    """Groups images by person identity. Returns {name: [paths]}."""
    person_images = {}
    for root, _, files in os.walk(dataset_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".ppm", ".png", ".jpeg")):
                fp = os.path.join(root, f)
                stem = Path(fp).stem
                # Assuming LFW format: Name_Name_0001.jpg
                parts = stem.split("_")
                person = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
                if person not in person_images:
                    person_images[person] = []
                person_images[person].append(fp)
    return person_images

def process_shard(dataset_dir, output_dir, chunk_size=500):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = Path(output_dir) / "worker_checkpoint.json"
    
    processed_files = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            state = json.load(f)
            processed_files = set(state.get("processed_files", []))
            print(f"Resuming from checkpoint: {len(processed_files)} files already processed.")

    person_images = discover_images(dataset_dir)
    all_paths = [(person, fp) for person, paths in person_images.items() for fp in paths]
    
    # Filter out already processed
    paths_to_process = [(person, fp) for person, fp in all_paths if fp not in processed_files]
    
    print(f"Total images found: {len(all_paths)}")
    print(f"Images to process in this run: {len(paths_to_process)}")
    
    chunk_data = []
    chunk_index = len(processed_files) // chunk_size
    
    for i, (person, filepath) in enumerate(tqdm(paths_to_process, desc="Extracting features")):
        try:
            image = cv2.imread(filepath)
            if image is None:
                continue
                
            clahe_img = apply_clahe(image)
            aligned, landmarks = align_face_crop(clahe_img)
            
            if landmarks is None:
                continue
                
            # Extract Tier 4 Marks
            marks = detect_facial_marks(aligned, landmarks)
            
            # Extract Tier 1 Neural Ensemble Embeddings
            ensemble_embeds = extract_ensemble_embeddings(aligned)
            
            record = {
                "filepath": filepath,
                "person": person,
                "marks": marks,
                "ensemble_embeds": ensemble_embeds
            }
            chunk_data.append(record)
            
        except Exception as e:
            print(f"\nError processing {filepath}: {e}")
            
        processed_files.add(filepath)
        
        # Save chunk and checkpoint
        if len(chunk_data) >= chunk_size or i == len(paths_to_process) - 1:
            chunk_file = Path(output_dir) / f"chunk_{chunk_index:04d}.pkl"
            with open(chunk_file, "wb") as f:
                pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            # Update checkpoint
            with open(checkpoint_file, "w") as f:
                json.dump({"processed_files": list(processed_files)}, f)
                
            chunk_data = []
            chunk_index += 1

    print(f"Worker complete. Output chunks saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract marks and embeddings.")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT / "datasets" / "lfwcrop_color" / "faces"), help="Path to dataset directory")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "datasets" / "extracted_chunks"), help="Path to output chunks")
    parser.add_argument("--chunk-size", type=int, default=500, help="Number of images per chunk")
    args = parser.parse_args()
    
    process_shard(args.dataset, args.output, args.chunk_size)
