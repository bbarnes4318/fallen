import sys
import os
import json
import numpy as np
import cv2
import math

sys.path.insert(0, os.path.abspath('c:/Users/jimbo/OneDrive/Documents/facial/backend'))

from image_preprocessor import preprocess_for_mark_detection
import mark_detector

original_filter_contours = mark_detector._filter_contours
original_contour_to_descriptor = mark_detector._contour_to_descriptor

def new_filter_contours(channels, gray, skin_mask, valid_mask, h, w, landmarks,
                        min_area, max_area, min_overlap, min_contrast, is_fallback=False):
    marks = []
    rejected = []
    used_mask = np.zeros((h, w), dtype=np.uint8)
    idx = 0
    t = {"area_pass": 0, "shape_pass": 0, "region_pass": 0, "contrast_pass": 0}

    for ch_name, cnts in channels.items():
        ch_min_contrast = min_contrast
        if ch_name in ("dark_lesion", "bright_scar", "texture_anomaly"):
            ch_min_contrast = max(0.4, min_contrast * 0.3)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            t["area_pass"] += 1

            desc = mark_detector._contour_to_descriptor(cnt, gray, skin_mask, h, w, ch_name, landmarks, idx, is_fallback=is_fallback)
            if desc is None:
                continue
            t["shape_pass"] += 1

            ix, iy = int(desc["centroid_px"][0]), int(desc["centroid_px"][1])
            if iy >= h or ix >= w:
                continue
            if used_mask[iy, ix] > 0:
                continue

            rej = None
            if skin_mask[iy, ix] == 0:
                rej = "outside_face_mask"
            if rej is None and (ix < mark_detector._BORDER_MARGIN or ix >= w - mark_detector._BORDER_MARGIN or
                                iy < mark_detector._BORDER_MARGIN or iy >= h - mark_detector._BORDER_MARGIN):
                rej = "border_artifact"
            if rej is None:
                mm = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mm, [cnt], -1, 255, -1)
                cp = np.count_nonzero(mm)
                if cp > 0:
                    op = np.count_nonzero(cv2.bitwise_and(mm, skin_mask))
                    if op / cp < min_overlap:
                        rej = f"insufficient_face_overlap ({op/cp:.2f})"
            if rej is None:
                t["region_pass"] += 1

            if rej is None and desc["contrast_score"] < ch_min_contrast:
                if desc["contrast_score"] >= ch_min_contrast * 0.5:
                    if desc["salience_score"] < 5.0:
                        rej = "low_contrast"
                    else:
                        desc["mark_type"] = "blemish"
                else:
                    rej = "low_contrast"

            if rej is not None:
                desc["rejection_reason"] = rej
                rd = {k: v for k, v in desc.items() if k != "contour"}
                rejected.append(rd)
                continue
            t["contrast_pass"] += 1
            cv2.drawContours(used_mask, [cnt], -1, 255, -1)
            desc["index"] = idx
            idx += 1
            marks.append(desc)

    return marks, rejected, t


def new_contour_to_descriptor(cnt, gray, skin_mask, h, w, channel, landmarks, mark_index, is_fallback=False):
    desc = original_contour_to_descriptor(cnt, gray, skin_mask, h, w, channel, landmarks, mark_index, is_fallback)
    if desc is not None:
        if channel in ("dark_lesion", "bright_scar", "texture_anomaly"):
            # Boost salience to prioritize structural anomalies
            desc["salience_score"] *= 4.0
            
            # Recalculate confidence with boosted salience
            salience = desc["salience_score"]
            conf = min(1.0, salience / 30.0) if not is_fallback else min(0.5, salience / 60.0)
            desc["confidence"] = float(conf)
            
    return desc

mark_detector._filter_contours = new_filter_contours
mark_detector._contour_to_descriptor = new_contour_to_descriptor

def main():
    img_path = r"C:\Users\jimbo\.gemini\antigravity\brain\17bfed72-deca-413e-860c-55de4eb4f60e\media__1778073825763.jpg"
    img = cv2.imread(img_path)
    prep = preprocess_for_mark_detection(img, None)

    marks, rejected, occ_mask, trace, overlays = mark_detector.detect_facial_marks(
        prep["images"]["mark_detector_input_bgr"],
        None,
        input_is_preprocessed=True
    )
    
    print(f"Accepted: {len(marks)}")
    for m in marks:
        if m["channel"] in ("dark_lesion", "texture_anomaly") and m["confidence"] > 0.5:
            print(f"FOUND CRATER: channel={m['channel']}, area={m['area']}, contrast={m['contrast_score']:.2f}, conf={m['confidence']:.2f}")

if __name__ == "__main__":
    main()
