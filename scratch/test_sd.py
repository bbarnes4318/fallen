import sys
import os
import cv2
import numpy as np
import base64
import json

sys.path.insert(0, os.path.abspath('c:/Users/jimbo/OneDrive/Documents/facial/backend'))

from image_preprocessor import preprocess_for_mark_detection
import mark_detector

def test_structural_depression():
    img_path = r"C:\Users\jimbo\.gemini\antigravity\brain\17bfed72-deca-413e-860c-55de4eb4f60e\media__1778073825763.jpg"
    out_dir = "c:/Users/jimbo/OneDrive/Documents/facial/scratch/test_structural"
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path)
    prep = preprocess_for_mark_detection(img, None)
    
    aligned_crop = prep["images"]["mark_detector_input_bgr"]
    h, w = aligned_crop.shape[:2]
    
    skin_mask, _ = mark_detector._build_skin_mask((h, w, 3), None)
    
    lab = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    
    # 1. Blur to remove tiny pores
    L_blur = cv2.medianBlur(L, 7)
    
    # 2. Morphological black-hat (targets dark regions smaller than kernel size)
    k_size = 31 # larger kernel for broad craters
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    blackhat = cv2.morphologyEx(L_blur, cv2.MORPH_BLACKHAT, k)
    
    # 3. Threshold
    _, sd_thresh = cv2.threshold(blackhat, 8, 255, cv2.THRESH_BINARY)
    
    # Clean up
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(sd_thresh, cv2.MORPH_OPEN, k_clean)
    
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    out_img = aligned_crop.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 15:  # Filter small ones
            cv2.drawContours(out_img, [c], -1, (0, 255, 0), 1)
            
    cv2.imwrite(os.path.join(out_dir, "structural_contours.png"), out_img)
    cv2.imwrite(os.path.join(out_dir, "blackhat.png"), blackhat)
    cv2.imwrite(os.path.join(out_dir, "sd_thresh.png"), sd_thresh)
    print(f"Found {len(cnts)} raw contours")
    
if __name__ == "__main__":
    test_structural_depression()
