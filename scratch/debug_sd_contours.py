import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath('c:/Users/jimbo/OneDrive/Documents/facial/backend'))

from image_preprocessor import preprocess_for_mark_detection
import mark_detector

img_path = r"C:\Users\jimbo\.gemini\antigravity\brain\17bfed72-deca-413e-860c-55de4eb4f60e\media__1778073825763.jpg"
img = cv2.imread(img_path)
prep = preprocess_for_mark_detection(img, None)
h, w = prep["images"]["mark_detector_input_bgr"].shape[:2]

lab = cv2.cvtColor(prep["images"]["mark_detector_input_bgr"], cv2.COLOR_BGR2LAB)
L_enhanced = lab[:, :, 0]

L_blur = cv2.medianBlur(L_enhanced, 7)
k_size = 31
k_sd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
blackhat = cv2.morphologyEx(L_blur, cv2.MORPH_BLACKHAT, k_sd)
_, sd_thresh = cv2.threshold(blackhat, 5, 255, cv2.THRESH_BINARY)
k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
sd_cleaned = cv2.morphologyEx(sd_thresh, cv2.MORPH_OPEN, k_clean)
cnts, _ = cv2.findContours(sd_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

ch_min_area = max(10, mark_detector._MIN_AREA * 2.0)
ch_max_area = mark_detector._MAX_AREA * 10.0

passed_area = []
for cnt in cnts:
    area = cv2.contourArea(cnt)
    if area < ch_min_area or area > ch_max_area:
        continue
    passed_area.append(cnt)

print(f"Passed area: {len(passed_area)}")

# Run them through _contour_to_descriptor
gray = cv2.cvtColor(prep["images"]["mark_detector_input_bgr"], cv2.COLOR_BGR2GRAY)
skin_mask = np.ones((h, w), dtype=np.uint8) * 255
landmarks = None

passed_shape = []
for cnt in passed_area:
    desc = mark_detector._contour_to_descriptor(cnt, gray, skin_mask, h, w, "structural_depression", landmarks, 0, False)
    if desc is not None:
        passed_shape.append(desc)
        print("Passed shape! Area:", desc["area"], "Contrast:", desc["contrast_score"], "Salience:", desc["salience_score"])
    else:
        print("Failed shape!")

print(f"Passed shape: {len(passed_shape)}")
