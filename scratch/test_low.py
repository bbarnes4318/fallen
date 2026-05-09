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

mark_detector._MIN_CONTRAST = 0.5

def main():
    img_path = r"C:\Users\jimbo\.gemini\antigravity\brain\17bfed72-deca-413e-860c-55de4eb4f60e\media__1778073825763.jpg"
    out_dir = "c:/Users/jimbo/OneDrive/Documents/facial/scratch/autopsy_output_low_contrast"
    os.makedirs(out_dir, exist_ok=True)
    
    img = cv2.imread(img_path)
    prep = preprocess_for_mark_detection(img, None)

    os.environ["DEBUG_FORENSIC"] = "true"
    marks, rejected, occ_mask, trace, overlays = mark_detector.detect_facial_marks(
        prep["images"]["mark_detector_input_bgr"],
        None,
        input_is_preprocessed=True
    )
    
    for k, b64 in overlays.items():
        if b64:
            b64_data = b64.split(",")[1]
            out_path = os.path.join(out_dir, f"05_overlay_{k}.png")
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(b64_data))
                
    report = {
        "trace": trace,
        "accepted_count": len(marks),
        "marks": [mark_detector.serialize_mark_descriptor(m) for m in marks]
    }
    
    with open(os.path.join(out_dir, "autopsy_report.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Autopsy low_contrast complete. Accepted: {len(marks)}")

if __name__ == "__main__":
    main()
