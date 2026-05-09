import sys
import os
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath('c:/Users/jimbo/OneDrive/Documents/facial/backend'))

from image_preprocessor import preprocess_for_mark_detection
import mark_detector

def print_areas_and_contrast(channels, gray, skin_mask, valid_mask, h, w, landmarks,
                             min_area, max_area, min_overlap, min_contrast, is_fallback=False):
    stats = {}
    idx = 0
    for ch_name, cnts in channels.items():
        stats[ch_name] = {"areas": [], "contrasts": []}
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            desc = mark_detector._contour_to_descriptor(cnt, gray, skin_mask, h, w, ch_name, landmarks, idx, is_fallback=is_fallback)
            if desc is None:
                continue
            stats[ch_name]["areas"].append(desc["area"])
            stats[ch_name]["contrasts"].append(desc["contrast_score"])
            idx += 1
            
    for ch_name, st in stats.items():
        print(f"\nChannel: {ch_name}")
        areas = st["areas"]
        contrasts = st["contrasts"]
        print(f"  Total passed area filter: {len(areas)}")
        if len(areas) > 0:
            print(f"  Mean area: {np.mean(areas):.2f}")
            print(f"  Median area: {np.median(areas):.2f}")
            print(f"  Mean contrast: {np.mean(contrasts):.2f}")
            print(f"  Median contrast: {np.median(contrasts):.2f}")
            print(f"  Contrast P10: {np.percentile(contrasts, 10):.2f}")
            print(f"  Contrast P50: {np.percentile(contrasts, 50):.2f}")
            print(f"  Contrast P90: {np.percentile(contrasts, 90):.2f}")

    return [], [], {"area_pass": 0, "shape_pass": 0, "region_pass": 0, "contrast_pass": 0}

mark_detector._filter_contours = print_areas_and_contrast

def main():
    img_path = r"C:\Users\jimbo\.gemini\antigravity\brain\17bfed72-deca-413e-860c-55de4eb4f60e\media__1778073825763.jpg"
    img = cv2.imread(img_path)
    prep = preprocess_for_mark_detection(img, None)

    mark_detector.detect_facial_marks(
        prep["images"]["mark_detector_input_bgr"],
        None,
        input_is_preprocessed=True
    )

if __name__ == "__main__":
    main()
