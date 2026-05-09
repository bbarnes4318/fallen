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
gray = cv2.cvtColor(prep["images"]["mark_detector_input_bgr"], cv2.COLOR_BGR2GRAY)
skin_mask = prep["masks"]["skin_mask"]
valid_mask = skin_mask

channels, raw_masks = mark_detector._run_channels(
    prep["images"]["mark_detector_input_bgr"], gray, valid_mask, 
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), h, w, True
)

print(f"SD channels generated: {len(channels['structural_depression'])}")

marks, rejected, ft = mark_detector._filter_contours(
    channels, gray, skin_mask, valid_mask, h, w, None,
    mark_detector._MIN_AREA, mark_detector._MAX_AREA, mark_detector._MIN_OVERLAP_RATIO, mark_detector._MIN_CONTRAST
)

sd_marks = [m for m in marks if m.get("channel") == "structural_depression"]
sd_rej = [m for m in rejected if m.get("channel") == "structural_depression"]

print(f"SD passed filter: {len(sd_marks)}")
print(f"SD rejected filter: {len(sd_rej)}")

for m in sd_marks:
    print(f"Accepted SD - Area: {m['area']}, Contrast: {m['contrast_score']}, Salience: {m['salience_score']}")

for m in sd_rej:
    print(f"Rejected SD - Reason: {m['rejection_reason']}")
