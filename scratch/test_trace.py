import sys
import os
import json
import base64
import numpy as np
import cv2
import math

sys.path.insert(0, os.path.abspath('c:/Users/jimbo/OneDrive/Documents/facial/backend'))

from image_preprocessor import preprocess_for_mark_detection
import mark_detector

def main():
    img_path = r"C:\Users\jimbo\.gemini\antigravity\brain\17bfed72-deca-413e-860c-55de4eb4f60e\media__1778073825763.jpg"
    img = cv2.imread(img_path)
    prep = preprocess_for_mark_detection(img, None)

    os.environ["DEBUG_FORENSIC"] = "true"
    marks, rejected, occ_mask, trace, overlays = mark_detector.detect_facial_marks(
        prep["images"]["mark_detector_input_bgr"],
        None,
        input_is_preprocessed=True
    )
    
    print(f"Trace: {json.dumps(trace, indent=2)}")

if __name__ == "__main__":
    main()
