import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))

import pipeline_core

cal = pipeline_core.CALIBRATION
print(f"CALIBRATION is None: {cal is None}")
if cal:
    print(f"Keys: {cal.keys()}")
    if 'ensemble' in cal:
        print(f"Ensemble Thresholds: {list(cal['ensemble'].get('thresholds', {}).keys())[:5]}")
        print(f"Threshold count: {len(cal['ensemble'].get('thresholds', {}))}")
    if 'arcface' in cal:
        print(f"Arcface Thresholds: {list(cal['arcface'].get('thresholds', {}).keys())[:5]}")

print("score_to_lr_ensemble values:")
for score in [0.30, 0.40, 0.60, 0.80]:
    lr = pipeline_core.score_to_lr_ensemble(score)
    print(f"Score {score} -> LR {lr}")
