#!/usr/bin/env python3
"""
Fallen — Hyperscale Dataset Ingestion
==========================================
Streams demographic-balanced datasets (e.g., FairFace) from HuggingFace 
to build Fallen Forensic-Grade Bayesian calibration matrices.
"""

import os
import sys
import argparse
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

def ingest_dataset(dataset_name: str, split: str, output_dir: str, shard_size: int, max_images: int):
    print(f"[INGEST] Starting ingestion of {dataset_name} ({split})")
    print(f"[INGEST] Output directory: {output_dir}")
    print(f"[INGEST] Images per shard limit: {shard_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Enable streaming to prevent downloading the entire dataset into memory/disk at once
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"[INGEST] FATAL: Failed to load dataset '{dataset_name}': {e}")
        sys.exit(1)
        
    # We will shuffle the stream to ensure maximum demographic variance across shards
    # For streaming datasets, shuffle uses a buffer
    print(f"[INGEST] Applying streaming shuffle buffer (size=10000) for demographic variance...")
    try:
        shuffled_dataset = dataset.shuffle(buffer_size=10000, seed=42)
    except Exception as e:
        print(f"[INGEST] WARN: Shuffle not supported on this dataset stream, proceeding linearly: {e}")
        shuffled_dataset = dataset
    
    current_shard = 0
    current_shard_dir = os.path.join(output_dir, f"shard_{current_shard:04d}")
    os.makedirs(current_shard_dir, exist_ok=True)
    
    count = 0
    shard_count = 0
    
    for item in tqdm(shuffled_dataset, desc="Ingesting images", total=max_images if max_images else None):
        if max_images and count >= max_images:
            break
            
        try:
            # HuggingFace datasets usually return the image under an 'image' key
            if "image" in item:
                img = item["image"]
            elif "img" in item:
                img = item["img"]
            else:
                # Find the first PIL.Image feature
                img = None
                for val in item.values():
                    if isinstance(val, Image.Image):
                        img = val
                        break
                if img is None:
                    continue
                    
            if not isinstance(img, Image.Image):
                continue
                
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            # Create a unique filename. If the dataset has a person ID or filename, use it, otherwise generate one.
            filename = f"img_{count:08d}.jpg"
            out_path = os.path.join(current_shard_dir, filename)
            
            img.save(out_path, format="JPEG", quality=95)
            
            count += 1
            shard_count += 1
            
            # Rotate shards
            if shard_count >= shard_size:
                current_shard += 1
                current_shard_dir = os.path.join(output_dir, f"shard_{current_shard:04d}")
                os.makedirs(current_shard_dir, exist_ok=True)
                shard_count = 0
                
        except Exception as e:
            # Skip corrupted images
            continue
            
    print(f"\n[INGEST] Ingestion Complete.")
    print(f"[INGEST] Total images saved: {count}")
    print(f"[INGEST] Total shards generated: {current_shard + (1 if shard_count > 0 else 0)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Hyperscale Face Dataset from HuggingFace.")
    # Default to FairFace for demographic balance (Daubert compliance)
    parser.add_argument("--dataset_name", type=str, default="nlpub/fairface", help="HuggingFace dataset ID")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to stream")
    parser.add_argument("--output_dir", type=str, default="./datasets/hyperscale", help="Base directory for shards")
    parser.add_argument("--shard_size", type=int, default=10000, help="Number of images per shard directory")
    parser.add_argument("--max_images", type=int, default=1000000, help="Maximum total images to download")
    
    args = parser.parse_args()
    ingest_dataset(args.dataset_name, args.split, args.output_dir, args.shard_size, args.max_images)
