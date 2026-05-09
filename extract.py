import ast
import os

source_file = r'C:\Users\jimbo\OneDrive\Documents\facial\backend\main.py'
dest_file = r'C:\Users\jimbo\OneDrive\Documents\facial\backend\pipeline_core.py'

functions_to_extract = [
    'fetch_image_from_url',
    'apply_clahe',
    'align_face_crop',
    'extract_ensemble_embeddings',
    'compute_ensemble_similarity',
    'score_to_lr_ensemble',
    'evaluate_mark_veto_override',
    '_run_mark_evidence_pipeline',
    'estimate_age',
    'cross_spectral_normalize',
    'compute_image_hash',
    'calculate_cosine_similarity',
    'finite_or_none'
]

with open(source_file, 'r', encoding='utf-8') as f:
    source_code = f.read()

tree = ast.parse(source_code)

extracted_code_blocks = []

for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in functions_to_extract:
        # Get source code of the function
        start_lineno = node.lineno - 1
        end_lineno = node.end_lineno
        func_code = '\n'.join(source_code.splitlines()[start_lineno:end_lineno])
        
        # Add decorator if exists (e.g. @staticmethod) but these are top level.
        # However, ast.get_source_segment is safer in Python 3.8+
        func_code = ast.get_source_segment(source_code, node)
        extracted_code_blocks.append(func_code)

header = '''import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["DEEPFACE_HOME"] = "/app"

import cv2
import numpy as np
import urllib.request
import math
import hashlib
import time
from google.cloud import storage
from skimage.feature import local_binary_pattern
import mediapipe as mp
import onnxruntime as ort
from deepface import DeepFace

# Initialize ML models for the pipeline
print("Initializing ML models in pipeline_core...")
try:
    DeepFace.build_model("ArcFace")
    DeepFace.build_model("Facenet512")
except Exception as e:
    print(f"Warning: DeepFace initialization failed: {e}")

try:
    pad_session = ort.InferenceSession("models/MiniFASNetV2.onnx", providers=['CPUExecutionProvider'])
except Exception as e:
    pad_session = None
    print(f"Warning: pad_session initialization failed: {e}")

mp_face_mesh = mp.solutions.face_mesh

'''

with open(dest_file, 'w', encoding='utf-8') as f:
    f.write(header)
    for code in extracted_code_blocks:
        f.write(code + '\n\n')

print("Extraction complete.")
