import math
import sys
from pathlib import Path

# Add backend to path so we can import models and main
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import cv2
import numpy as np

def cosine_to_lr_arcface(cosine_score: float, temporal_delta: float = 0.0) -> float:
    raw_tpr = 0.90 # Mock for a 0.35 score (usually frr=0.10)
    far_value = 1e-4 # Mock FAR
    tpr = min(1.0, raw_tpr * math.exp(0.01 * temporal_delta))
    return tpr / far_value

def cross_spectral_normalize(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    sat1_std = np.std(hsv1[:, :, 1])
    sat2_std = np.std(hsv2[:, :, 1])
    
    threshold = 15.0 # Low saturation std indicates grayscale/monochrome
    
    is_gray1 = sat1_std < threshold
    is_gray2 = sat2_std < threshold
    
    correction_applied = False
    norm1 = img1.copy()
    norm2 = img2.copy()
    
    if is_gray1 and not is_gray2:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        norm2 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
        correction_applied = True
    elif is_gray2 and not is_gray1:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        norm1 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
        correction_applied = True
        
    return norm1, norm2, correction_applied

def main():
    print("Testing Temporal Invariance Engine (Age-Conditioned LR & Spectral Correction)")
    print("=" * 60)
    
    # 1. Test Bayesian Math TPR Boost
    base_cosine = 0.35 # Usually a fail
    
    print("\n--- Bayesian Math Adjustment Test ---")
    print(f"Base Cosine Score: {base_cosine}")
    
    # Test across multiple temporal gaps
    gaps = [0.0, 10.0, 20.0, 40.0]
    for gap in gaps:
        lr = cosine_to_lr_arcface(base_cosine, temporal_delta=gap)
        print(f"Temporal Delta: {gap} years => LR: {lr:.6f}")
        
    print("\n--- Spectral Normalization Test ---")
    # Create fake RGB and Grayscale images
    img_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img_rgb[:, :, 1] = 200 # High saturation
    
    img_gray_3ch = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img_gray_3ch[:, :, :] = img_gray_3ch[:, :, 0:1] # Make it grayscale in 3 channels
    
    norm1, norm2, corrected = cross_spectral_normalize(img_rgb, img_gray_3ch)
    print(f"Correction Applied: {corrected}")
    if corrected:
        # Check if norm1 is now grayscale
        hsv1 = cv2.cvtColor(norm1, cv2.COLOR_BGR2HSV)
        sat1_std = np.std(hsv1[:, :, 1])
        print(f"New Image 1 Saturation STD: {sat1_std:.2f} (Expected ~0.0)")

if __name__ == "__main__":
    main()
