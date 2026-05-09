import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath('c:/Users/jimbo/OneDrive/Documents/facial/backend'))

from image_preprocessor import preprocess_for_mark_detection
import mark_detector

def test_structural_depression():
    img_path = r"C:\Users\jimbo\.gemini\antigravity\brain\17bfed72-deca-413e-860c-55de4eb4f60e\media__1778073825763.jpg"
    img = cv2.imread(img_path)
    prep = preprocess_for_mark_detection(img, None)
    
    aligned_crop = prep["images"]["mark_detector_input_bgr"]
    h, w = aligned_crop.shape[:2]
    
    skin_mask, _ = mark_detector._build_skin_mask((h, w, 3), None)
    gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
    
    lab = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    
    # Structural depression logic
    L_blur = cv2.medianBlur(L, 7)
    k_size = 31
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    blackhat = cv2.morphologyEx(L_blur, cv2.MORPH_BLACKHAT, k)
    
    # Extract gradient evidence as requested
    sx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    grad = np.sqrt(sx**2 + sy**2)
    
    _, sd_thresh = cv2.threshold(blackhat, 5, 255, cv2.THRESH_BINARY)
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(sd_thresh, cv2.MORPH_OPEN, k_clean)
    
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area > 10:
            desc = mark_detector._contour_to_descriptor(c, gray, skin_mask, h, w, "structural_depression", None, i, is_fallback=False)
            if desc:
                # Local gradient evidence
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [c], -1, 255, -1)
                
                # Dilate slightly to get the "rim"
                rim_mask = cv2.dilate(mask, np.ones((5,5), np.uint8)) - mask
                mean_rim_grad = cv2.mean(grad, mask=rim_mask)[0]
                
                if mean_rim_grad > 30 and desc["contrast_score"] > 0.4:
                    print(f"Area: {area:.1f}, Contrast: {desc['contrast_score']:.2f}, Rim Grad: {mean_rim_grad:.1f}")

if __name__ == "__main__":
    test_structural_depression()
