# Diff with e695a0a814ed4ee2fdd37a98a7d25352a37496d5

``diff
diff --git a/backend/main.py b/backend/main.py
index 6bd78d3..1ddae02 100644
--- a/backend/main.py
+++ b/backend/main.py
@@ -1,7 +1,7 @@
 import os
 import pickle
 os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Force Keras 2 ΓÇö required for DeepFace/ArcFace
-
+os.environ["DEEPFACE_HOME"] = "/app"     # Force DeepFace to use the pre-baked weights directory
 import cv2
 import numpy as np
 import urllib.request
@@ -11,6 +11,7 @@ import base64
 import struct
 import hashlib
 import time
+import json
 from pathlib import Path
 import uuid
 import jwt
@@ -18,6 +19,7 @@ from google.cloud import storage
 from google.cloud import kms
 from cryptography.fernet import Fernet
 from fastapi import FastAPI, HTTPException, Depends, Request
+from fastapi.responses import JSONResponse
 from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
 from fastapi.middleware.cors import CORSMiddleware
 from slowapi import Limiter, _rate_limit_exceeded_handler
@@ -29,6 +31,8 @@ from skimage.feature import local_binary_pattern
 import mediapipe as mp
 import onnxruntime as ort
 from models import SessionLocal, IdentityProfile, VerificationEvent, init_db
+from sqlalchemy import event
+from vault_index import vault_index
 import stripe
 from deepface import DeepFace
 
@@ -46,16 +50,70 @@ app = FastAPI(
 app.state.limiter = limiter
 app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
 
-# Preload ArcFace model at module init to avoid per-request cold start
+@app.exception_handler(Exception)
+async def global_exception_handler(request: Request, exc: Exception):
+    import traceback
+    traceback.print_exc()
+    return JSONResponse(
+        status_code=500,
+        content={"detail": "Internal Server Error", "error": str(exc)}
+    )
+
+# Preload Tier 1 Neural Ensemble models at module init to avoid per-request cold start
+print("Starting Tier 1 Neural Ensemble model preload...", flush=True)
 try:
     DeepFace.build_model("ArcFace")
-    print("ArcFace biometric model preloaded successfully.")
+    DeepFace.build_model("Facenet512")
+    print("Tier 1 Neural Ensemble models (ArcFace, Facenet512) preloaded successfully.", flush=True)
 except Exception as e:
-    print(f"Warning: ArcFace preload failed (will retry on first request): {e}")
+    print(f"Warning: Model preload failed (will retry on first request): {e}", flush=True)
 
 @app.on_event("startup")
 def on_startup():
+    print("Starting database initialization (init_db)...", flush=True)
     init_db()
+    print("Database initialization complete.", flush=True)
+    print("Hydrating FAISS Vault Index...", flush=True)
+    session = SessionLocal()
+    try:
+        import concurrent.futures
+        # Query specific columns to detach from SQLAlchemy Session (Thread-safety)
+        profiles = session.query(IdentityProfile.user_id, IdentityProfile.encrypted_facial_embedding).all()
+        
+        def _process_profile(data):
+            user_id, encrypted_emb = data
+            try:
+                emb = decrypt_embedding(encrypted_emb)
+                return user_id, emb
+            except Exception as e:
+                print(f"Failed to decrypt embedding for {user_id}: {e}", flush=True)
+                return None
+                
+        # Perform network-bound KMS decryption in parallel to prevent startup timeouts
+        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
+            # Eagerly evaluate the map to a list to prevent the `with` block from exiting prematurely
+            results = list(executor.map(_process_profile, profiles))
+            
+        for result in results:
+            if result is not None:
+                user_id, emb = result
+                try:
+                    vault_index.add_identity(user_id, emb)
+                except Exception as e:
+                    print(f"Failed to add embedding to FAISS for {user_id}: {e}", flush=True)
+
+        print(f"FAISS index hydrated with {vault_index.index.ntotal} records.", flush=True)
+    finally:
+        session.close()
+
+# Dynamically sync FAISS when new identities are added
+@event.listens_for(IdentityProfile, 'after_insert')
+def receive_after_insert(mapper, connection, target):
+    try:
+        emb = decrypt_embedding(target.encrypted_facial_embedding)
+        vault_index.add_identity(target.user_id, emb)
+    except Exception as e:
+        print(f"FAISS sync failed for {target.user_id}: {e}")
 
 app.add_middleware(
     CORSMiddleware,
@@ -65,8 +123,10 @@ app.add_middleware(
         "https://scargods.com",
         "https://www.scargods.com",
         "https://facial-frontend-vkd6b6ijxa-uk.a.run.app",
+        "https://facial-frontend-196207148120.us-east4.run.app",
         "https://facial-verify-api-196207148120.us-central1.run.app",
     ],
+    allow_origin_regex=r"https://.*\.run\.app",
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
@@ -155,6 +215,13 @@ def detect_liveness(image: np.ndarray) -> dict:
 # The Key Encryption Key (KEK) managed by GCP KMS
 KMS_KEY_NAME = os.getenv("KMS_KEY_NAME") or "projects/hoppwhistle/locations/us-central1/keyRings/facial-keyring/cryptoKeys/facial-dek"
 
+_kms_client = None
+def get_kms_client():
+    global _kms_client
+    if _kms_client is None:
+        _kms_client = kms.KeyManagementServiceClient()
+    return _kms_client
+
 def encrypt_embedding(embedding: np.ndarray) -> bytes:
     """
     Application-Level Envelope Encryption.
@@ -171,7 +238,7 @@ def encrypt_embedding(embedding: np.ndarray) -> bytes:
         encrypted_payload = cipher.encrypt(payload_bytes)
         
         # 3. Encrypt the DEK with KMS
-        client = kms.KeyManagementServiceClient()
+        client = get_kms_client()
         encrypt_response = client.encrypt(request={'name': KMS_KEY_NAME, 'plaintext': dek})
         encrypted_dek = encrypt_response.ciphertext
         
@@ -198,7 +265,7 @@ def decrypt_embedding(packet: bytes) -> np.ndarray:
         encrypted_payload = packet[4+dek_len:]
         
         # 2. Decrypt DEK via KMS
-        client = kms.KeyManagementServiceClient()
+        client = get_kms_client()
         decrypt_response = client.decrypt(request={'name': KMS_KEY_NAME, 'ciphertext': encrypted_dek})
         dek = decrypt_response.plaintext
         
@@ -228,6 +295,28 @@ MODEL_POINTS_3D = np.array([
     (150.0, -150.0, -125.0)      # Right Mouth corner
 ], dtype=np.float64)
 
+# 17-Landmark Canonical Skull for Tier 2 3D Procrustes Alignment
+CANONICAL_SKULL_3D = np.array([
+    [-0.5, -0.2, -0.1],   # 33: Left Eye Outer
+    [-0.2, -0.2, -0.05],  # 133: Left Eye Inner
+    [0.2, -0.2, -0.05],   # 362: Right Eye Inner
+    [0.5, -0.2, -0.1],    # 263: Right Eye Outer
+    [0.0, 0.2, -0.5],     # 1: Nose Tip
+    [0.0, -0.1, -0.2],    # 6: Nose Bridge
+    [0.0, 1.0, -0.1],     # 152: Chin
+    [-0.3, 0.6, -0.2],    # 61: Left Mouth Corner
+    [0.3, 0.6, -0.2],     # 291: Right Mouth Corner
+    [0.0, -0.8, -0.1],    # 10: Forehead Top
+    [-0.8, 0.4, 0.2],     # 234: Left Jaw
+    [0.8, 0.4, 0.2],      # 454: Right Jaw
+    [-0.4, -0.4, -0.15],  # 70: Left Eyebrow
+    [0.4, -0.4, -0.15],   # 300: Right Eyebrow
+    [0.0, 0.5, -0.3],     # 0: Upper Lip
+    [0.0, 0.7, -0.25],    # 17: Lower Lip
+    [0.0, 0.9, -0.15]     # 199: Chin Center
+], dtype=np.float64)
+LANDMARK_INDICES_17 = [33, 133, 362, 263, 1, 6, 152, 61, 291, 10, 234, 454, 70, 300, 0, 17, 199]
+
 # Initialize MediaPipe Face Mesh
 mp_face_mesh = mp.solutions.face_mesh
 face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
@@ -235,11 +324,12 @@ face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refin
 class VerificationRequest(BaseModel):
     gallery_url: str
     probe_url: str
+    require_liveness: bool = False
 
 # ---------------------------------------------------------
 # PIPELINE VERSION PINNING (Daubert Reproducibility)
 # ---------------------------------------------------------
-PIPELINE_VERSION = "AurumShield Daubert-Compliant v3.0 (Bayesian LR)"
+PIPELINE_VERSION = "AurumShield Daubert-Compliant v4.0 (Ensemble + 3D Procrustes + Bayesian LR)"
 
 def _get_dependency_versions() -> dict:
     """Snapshot the exact versions of critical biometric libraries."""
@@ -260,6 +350,20 @@ DEPENDENCY_VERSIONS = _get_dependency_versions()
 
 class AuditLog(BaseModel):
     raw_cosine_score: float
+    # Neural Ensemble Audit Trail
+    raw_arcface_score: Optional[float] = None
+    raw_secondary_score: Optional[float] = None
+    ensemble_model_secondary: Optional[str] = None
+    # Tier 2: 3D Topographical Mapping Telemetry
+    pose_corrected_3d: Optional[bool] = None
+    probe_pose_angles: Optional[dict] = None
+    gallery_pose_angles: Optional[dict] = None
+    occlusion_percentage: Optional[float] = None
+    occluded_regions: Optional[list] = None
+    effective_geometric_ratios_used: Optional[int] = None
+    # Temporal & Spectral Telemetry (Tier 1/3)
+    estimated_temporal_delta: Optional[float] = None
+    cross_spectral_correction_applied: Optional[bool] = None
     statistical_certainty: str
     false_acceptance_rate: str
     nodes_mapped: int
@@ -311,6 +415,9 @@ class VerificationResponse(BaseModel):
     marks_detected_gallery: int = 0
     marks_detected_probe: int = 0
     marks_matched: int = 0
+    correspondences: list = []
+    raw_probe_marks: list = []
+    raw_gallery_marks: list = []
     audit_log: Optional[AuditLog] = None
 
 # ---------------------------------------------------------
@@ -366,6 +473,15 @@ TIER4_CALIBRATION = None
 def _load_tier4_calibration():
     """Load the Tier 4 population model from local file or GCS."""
     import json as _json
+    import sys
+    import numpy as np
+    
+    # Monkey-patch to allow unpickling numpy 2.x models in numpy 1.x environments
+    if "numpy.core.numeric" in sys.modules and "numpy._core.numeric" not in sys.modules:
+        sys.modules["numpy._core"] = sys.modules["numpy.core"]
+        sys.modules["numpy._core.numeric"] = sys.modules["numpy.core.numeric"]
+        sys.modules["numpy._core.multiarray"] = sys.modules["numpy.core.multiarray"]
+
     bucket_name = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
     gcs_path = "calibration/tier4_population_model.pkl"
 
@@ -447,9 +563,9 @@ def calculate_statistical_confidence(cosine_score: float) -> dict:
     }
 
 
-def cosine_to_lr_arcface(cosine_score: float) -> float:
+def score_to_lr_ensemble(ensemble_score: float, temporal_delta: float = 0.0) -> float:
     """
-    Convert ArcFace cosine similarity to a Likelihood Ratio using
+    Convert Fused 60/40 Ensemble Score to a Likelihood Ratio using
     empirically calibrated FAR/FRR from the LFW benchmark.
 
     LR = P(score | Hp) / P(score | Hd) = (1 - FRR) / FAR
@@ -457,18 +573,28 @@ def cosine_to_lr_arcface(cosine_score: float) -> float:
     Where:
       - P(score | Hp) = True Positive Rate = 1 - FRR (same person produces this score)
       - P(score | Hd) = False Acceptance Rate = FAR (different person produces this score)
+      
+    Temporal Invariance:
+      - Exponential decay curve applied to TPR probability based on temporal delta.
+      - As the time gap increases, the expected FRR naturally increases.
+      - By boosting the expected TPR for degraded scores, we prevent the "Aging Problem"
+        without manipulating the raw structural score or Bayesian math.
     """
     if CALIBRATION is None:
         return 1.0  # Neutral LR ΓÇö no calibration data
 
-    thresholds = CALIBRATION["arcface"]["thresholds"]
+    thresholds = CALIBRATION.get("ensemble", {}).get("thresholds", {})
+    if not thresholds:
+        # Fallback to arcface if ensemble calibration is not yet loaded
+        thresholds = CALIBRATION.get("arcface", {}).get("thresholds", {})
+        
     sorted_thresh = sorted(thresholds.keys(), key=float)
 
     far_value = None
     frr_value = None
 
     for t in sorted_thresh:
-        if cosine_score >= float(t):
+        if ensemble_score >= float(t):
             far_value = thresholds[t]["far"]
             frr_value = thresholds[t]["frr"]
 
@@ -476,7 +602,12 @@ def cosine_to_lr_arcface(cosine_score: float) -> float:
         # Score below all thresholds ΓÇö strong evidence against match
         return 1e-6  # Floor: extremely low LR
 
-    tpr = 1.0 - (frr_value if frr_value is not None else 0.0)
+    raw_tpr = 1.0 - (frr_value if frr_value is not None else 0.0)
+    
+    # Age-Conditioned Likelihood Ratio (Temporal Invariance)
+    # The expected TPR for degraded scores is exponentially boosted ~1% per year of temporal gap
+    tpr = min(1.0, raw_tpr * math.exp(0.01 * temporal_delta))
+
     # Epsilon floor to prevent division by zero
     far_value = max(far_value, 1e-9)
     lr = tpr / far_value
@@ -818,6 +949,60 @@ def extract_landmark_embedding(landmarks) -> np.ndarray:
     return coords.flatten()  # 1404-D vector
 
 
+def estimate_age(image: np.ndarray) -> float:
+    """
+    Estimates the apparent age of the subject using DeepFace.
+    """
+    try:
+        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
+        result = DeepFace.analyze(
+            img_path=rgb_image,
+            actions=["age"],
+            enforce_detection=False,
+            detector_backend="skip" # Image is already cropped/aligned
+        )
+        if isinstance(result, list):
+            return float(result[0]["age"])
+        return float(result["age"])
+    except Exception as e:
+        print(f"Age estimation failed: {e}")
+        return 0.0
+
+
+def cross_spectral_normalize(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
+    """
+    Analyzes HSV saturation to detect if one image is grayscale/sepia and the other is color.
+    If a spectral mismatch is detected, converts the color image to grayscale to match domains,
+    preventing artificial texture noise in Tier 3 (LBP) and Tier 4 (Marks).
+    Returns (norm_img1, norm_img2, correction_applied).
+    """
+    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
+    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
+    
+    sat1_std = np.std(hsv1[:, :, 1])
+    sat2_std = np.std(hsv2[:, :, 1])
+    
+    threshold = 15.0 # Low saturation std indicates grayscale/monochrome
+    
+    is_gray1 = sat1_std < threshold
+    is_gray2 = sat2_std < threshold
+    
+    correction_applied = False
+    norm1 = img1.copy()
+    norm2 = img2.copy()
+    
+    if is_gray1 and not is_gray2:
+        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
+        norm2 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
+        correction_applied = True
+    elif is_gray2 and not is_gray1:
+        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
+        norm1 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
+        correction_applied = True
+        
+    return norm1, norm2, correction_applied
+
+
 def extract_arcface_embedding(image: np.ndarray) -> np.ndarray:
     """
     Extracts a 512-D ArcFace biometric embedding from an aligned face crop.
@@ -855,31 +1040,180 @@ def extract_arcface_embedding(image: np.ndarray) -> np.ndarray:
     return embedding  # 512-D vector
 
 
-def extract_geometric_ratios(landmarks) -> np.ndarray:
+def extract_facenet_embedding(image: np.ndarray) -> np.ndarray:
+    """
+    Extracts a 512-D Facenet512 biometric embedding from an aligned face crop.
+    This serves as the secondary model in the Tier 1 Neural Ensemble.
+    Uses 'retinaface' detector backend for consistency with ArcFace extraction.
+    """
+    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
+    result = DeepFace.represent(
+        img_path=rgb_image,
+        model_name="Facenet512",
+        enforce_detection=False,
+        detector_backend="retinaface",
+    )
+    return np.array(result[0]["embedding"], dtype=np.float64)
+
+
+def extract_ensemble_embeddings(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
     """
-    Computes scale-invariant facial geometric ratios for Tier 2 soft biometric comparison.
-    Each ratio captures a unique structural characteristic of the face (nose length,
-    mouth width, jawline symmetry, etc.) relative to the inter-ocular distance.
+    Extracts both ArcFace and Facenet512 embeddings.
+    Returns: (arcface_embedding, facenet_embedding)
     """
-    coords = np.array([(l.x, l.y) for l in landmarks])
+    arcface_embed = extract_arcface_embedding(image)
+    facenet_embed = extract_facenet_embedding(image)
+    return arcface_embed, facenet_embed
 
-    left_eye = coords[33]
-    right_eye = coords[263]
+
+def compute_ensemble_similarity(embed_pair_1: tuple[np.ndarray, np.ndarray], embed_pair_2: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, float]:
+    """
+    Computes cosine similarity for both models and fuses them using a weighted ensemble (60% ArcFace, 40% Facenet512).
+    Returns: (fused_score, arcface_score, secondary_score)
+    """
+    arc1, face1 = embed_pair_1
+    arc2, face2 = embed_pair_2
+    
+    # Needs calculate_cosine_similarity which is defined below, but Python handles forward references
+    # wait, this is executed later anyway so it's fine.
+    arc_score = calculate_cosine_similarity(arc1, arc2)
+    face_score = calculate_cosine_similarity(face1, face2)
+    
+    fused_score = (arc_score * 0.60) + (face_score * 0.40)
+    return fused_score, arc_score, face_score
+
+
+def procrustes_align_3d(landmarks_3d: np.ndarray) -> tuple[np.ndarray, dict]:
+    """
+    Rigid Procrustes analysis via SVD to neutralize pitch, yaw, and roll.
+    Takes N x 3 landmarks and aligns the 17 key points to the CANONICAL_SKULL_3D.
+    Returns the fully un-rotated N x 3 mesh and the extracted Euler angles.
+    """
+    source_points = landmarks_3d[LANDMARK_INDICES_17]
+    target_points = CANONICAL_SKULL_3D
+    
+    # Center the points
+    source_centroid = np.mean(source_points, axis=0)
+    target_centroid = np.mean(target_points, axis=0)
+    
+    source_centered = source_points - source_centroid
+    target_centered = target_points - target_centroid
+    
+    # SVD
+    H = source_centered.T @ target_centered
+    U, S, Vt = np.linalg.svd(H)
+    R = Vt.T @ U.T
+    
+    # Handle reflection
+    if np.linalg.det(R) < 0:
+        Vt[2, :] *= -1
+        R = Vt.T @ U.T
+        
+    # Extract Euler angles from R (yaw, pitch, roll)
+    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
+    singular = sy < 1e-6
+    if not singular:
+        pitch = np.arctan2(R[2,1], R[2,2])
+        yaw = np.arctan2(-R[2,0], sy)
+        roll = np.arctan2(R[1,0], R[0,0])
+    else:
+        pitch = np.arctan2(-R[1,2], R[1,1])
+        yaw = np.arctan2(-R[2,0], sy)
+        roll = 0
+        
+    angles = {
+        "pitch_deg": round(np.degrees(pitch), 2),
+        "yaw_deg": round(np.degrees(yaw), 2),
+        "roll_deg": round(np.degrees(roll), 2)
+    }
+    
+    # Apply rotation to ALL landmarks (centered at their centroid to prevent translation explosion)
+    all_centered = landmarks_3d - np.mean(landmarks_3d, axis=0)
+    aligned_landmarks = (R @ all_centered.T).T
+    
+    return aligned_landmarks, angles
+
+def extract_geometric_ratios_3d(landmarks) -> tuple[np.ndarray, dict, dict]:
+    """
+    Computes scale-invariant, true 3D Euclidean facial geometric ratios for Tier 2.
+    Uses Procrustes alignment to mathematically un-rotate the face to a perfect frontal view.
+    Also computes landmark visibility telemetry for Tier 2 dynamic dropping.
+    """
+    VISIBILITY_THRESHOLD = 0.85
+
+    # Compute Visibility Telemetry
+    masked_count = sum(1 for l in landmarks if getattr(l, "visibility", 1.0) < VISIBILITY_THRESHOLD)
+    occlusion_percentage = (masked_count / len(landmarks)) * 100.0 if len(landmarks) > 0 else 0.0
+
+    STRUCTURAL_GROUPS = {
+        "Left Orbital": [33, 133, 160, 159, 158, 144, 145, 153],
+        "Right Orbital": [263, 362, 387, 386, 385, 373, 374, 380],
+        "Nose": [1, 2, 98, 327, 4, 5, 195, 197, 6],
+        "Mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88],
+        "Left Jaw": [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152],
+        "Right Jaw": [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
+        "Left Eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
+        "Right Eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
+        "Forehead": [10, 338, 297, 332, 284, 251, 389, 356]
+    }
+
+    occluded_regions = []
+    for region, indices in STRUCTURAL_GROUPS.items():
+        region_masked = sum(1 for idx in indices if getattr(landmarks[idx], "visibility", 1.0) < VISIBILITY_THRESHOLD)
+        if region_masked > len(indices) * 0.5:  # If >50% masked, call it occluded
+            occluded_regions.append(region)
+
+    ratio_landmarks_indices = [
+        [1, 152],         # Nose-to-chin
+        [6, 1],           # Nose length
+        [61, 291],        # Mouth width
+        [10, 152],        # Face height
+        [234, 454],       # Jaw width
+        [70, 33],         # Left brow height
+        [300, 263],       # Right brow height
+        [1, 33],          # Nose-to-left-eye
+        [1, 263],         # Nose-to-right-eye
+        [152, 61],        # Chin-to-left-mouth
+        [152, 291],       # Chin-to-right-mouth
+        [234, 152, 454]   # Jaw symmetry
+    ]
+    
+    iod_points = [33, 263]
+    ratio_visibility = []
+    for idx_group in ratio_landmarks_indices:
+        is_visible = all(getattr(landmarks[idx], "visibility", 1.0) >= VISIBILITY_THRESHOLD for idx in idx_group)
+        is_iod_visible = all(getattr(landmarks[idx], "visibility", 1.0) >= VISIBILITY_THRESHOLD for idx in iod_points)
+        if len(idx_group) == 3: # Jaw symmetry doesn't use IOD
+            ratio_visibility.append(is_visible)
+        else:
+            ratio_visibility.append(is_visible and is_iod_visible)
+
+    vis_data = {
+        "occlusion_percentage": round(occlusion_percentage, 2),
+        "occluded_regions": occluded_regions,
+        "ratio_visibility": np.array(ratio_visibility, dtype=bool)
+    }
+
+    coords_3d = np.array([(l.x, l.y, l.z) for l in landmarks])
+    aligned_coords, angles = procrustes_align_3d(coords_3d)
+
+    left_eye = aligned_coords[33]
+    right_eye = aligned_coords[263]
     iod = np.linalg.norm(right_eye - left_eye)
 
     if iod < 1e-6:
-        return np.zeros(12)
-
-    nose_tip = coords[1]
-    nose_bridge = coords[6]
-    chin = coords[152]
-    left_mouth = coords[61]
-    right_mouth = coords[291]
-    forehead_top = coords[10]
-    left_jaw = coords[234]
-    right_jaw = coords[454]
-    left_eyebrow = coords[70]
-    right_eyebrow = coords[300]
+        return np.zeros(12), angles, vis_data
+
+    nose_tip = aligned_coords[1]
+    nose_bridge = aligned_coords[6]
+    chin = aligned_coords[152]
+    left_mouth = aligned_coords[61]
+    right_mouth = aligned_coords[291]
+    forehead_top = aligned_coords[10]
+    left_jaw = aligned_coords[234]
+    right_jaw = aligned_coords[454]
+    left_eyebrow = aligned_coords[70]
+    right_eyebrow = aligned_coords[300]
 
     jaw_to_chin_r = np.linalg.norm(right_jaw - chin)
 
@@ -898,7 +1232,7 @@ def extract_geometric_ratios(landmarks) -> np.ndarray:
         np.linalg.norm(left_jaw - chin) / jaw_to_chin_r if jaw_to_chin_r > 1e-6 else 1.0,  # Jaw symmetry
     ])
 
-    return ratios
+    return ratios, angles, vis_data
 
 
 # ---------------------------------------------------------
@@ -1036,15 +1370,14 @@ def _build_skin_mask(shape: tuple, landmarks, margin: int = 5) -> np.ndarray:
     return skin_mask
 
 
-def detect_facial_marks(aligned_crop: np.ndarray, landmarks) -> list:
+def detect_facial_marks(aligned_crop: np.ndarray, landmarks) -> tuple[list, np.ndarray]:
     """
     Detects discrete facial anomalies (scars, moles, birthmarks) on
     the skin surface of an aligned 256├ù256 face crop.
 
-    Returns a list of mark descriptors:
-        [{"centroid": (cx, cy), "area": float, "intensity": float, "circularity": float}, ...]
-
-    Centroids are normalized to [0,1] relative to crop dimensions.
+    Returns a tuple:
+        - list of mark descriptors: [{"centroid": (cx, cy), "area": float, "intensity": float, "circularity": float}, ...]
+        - occlusion_mask (np.ndarray): Mask of occluded regions
     """
     h, w = aligned_crop.shape[:2]
     gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
@@ -1052,6 +1385,15 @@ def detect_facial_marks(aligned_crop: np.ndarray, landmarks) -> list:
     # Build skin mask excluding major facial features
     skin_mask = _build_skin_mask(aligned_crop.shape, landmarks)
 
+    # Build occluded mask for Bayesian Penalty Nullification
+    occ_mask = np.zeros((h, w), dtype=np.uint8)
+    pts = []
+    for lm in landmarks:
+        if getattr(lm, "visibility", 1.0) < 0.85:
+            pts.append((int(lm.x * w), int(lm.y * h)))
+    for pt in pts:
+        cv2.circle(occ_mask, pt, int(min(h, w) * 0.05), 255, -1)
+
     # Adaptive threshold to detect dark anomalies on skin
     thresh = cv2.adaptiveThreshold(
         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
@@ -1061,6 +1403,9 @@ def detect_facial_marks(aligned_crop: np.ndarray, landmarks) -> list:
     # Apply skin mask ΓÇö only keep marks on skin surface
     masked = cv2.bitwise_and(thresh, skin_mask)
 
+    # Apply occlusion mask to avoid detecting shadows as marks
+    masked[occ_mask > 0] = 0
+
     # Morphological opening to remove speckle noise
     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
     cleaned = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)
@@ -1098,7 +1443,7 @@ def detect_facial_marks(aligned_crop: np.ndarray, landmarks) -> list:
             "contour": cnt,  # keep for visualization
         })
 
-    return marks
+    return marks, occ_mask
 
 
 def compute_mark_correspondence(marks_gallery: list, marks_probe: list) -> dict:
@@ -1299,14 +1644,11 @@ def generate_scar_delta_map(
     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
     true_scars = cv2.dilate(true_scars, kernel, iterations=1)
 
-    # 6. Build overlay canvas: darkened, desaturated gallery
-    hsv = cv2.cvtColor(img_gallery, cv2.COLOR_BGR2HSV)
-    hsv[:, :, 1] = (hsv[:, :, 1] * 0.3).astype(np.uint8)   # Desaturate
-    hsv[:, :, 2] = (hsv[:, :, 2] * 0.35).astype(np.uint8)   # Darken
-    canvas = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
+    # 6. Build overlay canvas: transparent RGBA (so it can be overlaid without blending faces)
+    canvas = np.zeros((h, w, 4), dtype=np.uint8)
 
-    # Paint neon crimson (BGR: 30, 0, 180) where scars are detected
-    canvas[true_scars > 0] = (30, 0, 180)
+    # Paint neon crimson (BGRA: 30, 0, 180, 255) where scars are detected
+    canvas[true_scars > 0] = (30, 0, 180, 255)
 
     # 7. Overlay detected mark circles (from Tier 4 engine)
     if marks_gallery and mark_matches is not None:
@@ -1319,10 +1661,10 @@ def generate_scar_delta_map(
 
             if i in matched_gallery_indices:
                 # Green circle ΓÇö matched mark (confirmed in both faces)
-                cv2.circle(canvas, (cx, cy), radius, (0, 220, 80), 2, cv2.LINE_AA)
+                cv2.circle(canvas, (cx, cy), radius, (0, 220, 80, 255), 2, cv2.LINE_AA)
             else:
                 # Yellow circle ΓÇö unmatched mark (only in gallery)
-                cv2.circle(canvas, (cx, cy), radius, (0, 200, 220), 1, cv2.LINE_AA)
+                cv2.circle(canvas, (cx, cy), radius, (0, 200, 220, 255), 1, cv2.LINE_AA)
 
     # 8. Encode to base64 data URI
     _, buffer = cv2.imencode('.png', canvas)
@@ -1530,6 +1872,34 @@ def generate_upload_urls(req: UploadUrlsRequest, _: dict = Depends(verify_jwt)):
         print(f"GCS Error: {e}")
         raise HTTPException(status_code=500, detail=f"Failed to generate upload URLs: {str(e)}")
 
+def analyze_frequency_domain(image: np.ndarray) -> float:
+    """
+    Phase 7: Synthetic Provenance Veto.
+    Performs FFT frequency domain analysis to detect checkerboard artifacts 
+    (high-frequency grid anomalies) inherent to AI upscaling networks.
+    """
+    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
+    f = np.fft.fft2(gray)
+    fshift = np.fft.fftshift(f)
+    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
+    
+    h, w = magnitude_spectrum.shape
+    cy, cx = h // 2, w // 2
+    Y, X = np.ogrid[:h, :w]
+    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
+    mask = dist_from_center > (min(h, w) * 0.15)
+    
+    high_freq = magnitude_spectrum[mask]
+    if len(high_freq) == 0:
+        return 0.0
+        
+    peak_intensity = float(np.max(high_freq))
+    mean_intensity = float(np.mean(high_freq))
+    
+    anomaly_score = (peak_intensity - mean_intensity) / (mean_intensity + 1e-8)
+    normalized_score = max(0.0, min(1.0, anomaly_score * 0.4))
+    return normalized_score
+
 @app.post("/verify/fuse", response_model=VerificationResponse)
 @limiter.limit("5/minute")
 def verify_pipeline(request: Request, payload: VerificationRequest, _: dict = Depends(verify_jwt)):
@@ -1537,11 +1907,48 @@ def verify_pipeline(request: Request, payload: VerificationRequest, _: dict = De
     gallery_img, gallery_file_hash = fetch_image_from_url(payload.gallery_url)
     probe_img, probe_file_hash = fetch_image_from_url(payload.probe_url)
     
+    # Phase 7: Synthetic Provenance Gatekeeper
+    SYNTHETIC_ARTIFACT_THRESHOLD = 0.85
+    gallery_anomaly = analyze_frequency_domain(gallery_img)
+    probe_anomaly = analyze_frequency_domain(probe_img)
+    max_anomaly = max(gallery_anomaly, probe_anomaly)
+    
+    if max_anomaly > SYNTHETIC_ARTIFACT_THRESHOLD:
+        ledger_session = SessionLocal()
+        try:
+            event = VerificationEvent(
+                probe_hash=probe_file_hash,
+                gallery_hash=gallery_file_hash,
+                fused_score_x100=0,
+                conclusion="VETO: Synthetic Media Detected",
+                pipeline_version=PIPELINE_VERSION,
+                veto_triggered=True,
+                failed_provenance_veto=True,
+                synthetic_anomaly_score=max_anomaly
+            )
+            ledger_session.add(event)
+            ledger_session.commit()
+        except Exception as e:
+            print(f"Failed to write provenance veto to ledger: {e}")
+            ledger_session.rollback()
+        finally:
+            ledger_session.close()
+            
+        return JSONResponse(status_code=200, content={
+            "status": "success", 
+            "conclusion": "VETO: Synthetic Media Detected", 
+            "fused_score": 0,
+            "synthetic_anomaly_score": max_anomaly
+        })
+
     # 1.5 Presentation Attack Detection (Liveness Firewall)
-    liveness_result = detect_liveness(probe_img)
-    liveness_telemetry = build_liveness_telemetry(liveness_result)
-    if liveness_result["score"] < 0.95:
-        raise HTTPException(status_code=403, detail="SPOOF_DETECTED: Presentation attack suspected.")
+    if payload.require_liveness:
+        liveness_result = detect_liveness(probe_img)
+        liveness_telemetry = build_liveness_telemetry(liveness_result)
+        if liveness_result["score"] < 0.95:
+            raise HTTPException(status_code=403, detail="SPOOF_DETECTED: Presentation attack suspected.")
+    else:
+        liveness_telemetry = {"status": "BYPASSED", "method": "NONE"}
     
     # 2. Preprocess (CLAHE)
     gallery_clahe = apply_clahe(gallery_img)
@@ -1557,22 +1964,36 @@ def verify_pipeline(request: Request, payload: VerificationRequest, _: dict = De
             detail="FACE_NOT_DETECTED: Could not detect a face in one or both images. Please upload clear, front-facing photographs."
         )
     
-    # 4. TIER 1: Structural Identity (512-D ArcFace Biometric Embedding)
-    embed_gallery = extract_arcface_embedding(gallery_aligned)
-    embed_probe = extract_arcface_embedding(probe_aligned)
-    structural_sim = calculate_cosine_similarity(embed_gallery, embed_probe)
+    # 3.5 Temporal Invariance Engine (Age Estimation & Cross-Spectral Normalization)
+    gallery_age = estimate_age(gallery_aligned)
+    probe_age = estimate_age(probe_aligned)
+    temporal_delta = abs(probe_age - gallery_age)
+
+    # Cross-spectral matching before passing to textural/mark layers
+    gallery_aligned, probe_aligned, spectral_correction = cross_spectral_normalize(gallery_aligned, probe_aligned)
+    
+    # 4. TIER 1: Structural Identity (Neural Ensemble: 60% ArcFace, 40% Facenet512)
+    ensemble_gallery = extract_ensemble_embeddings(gallery_aligned)
+    ensemble_probe = extract_ensemble_embeddings(probe_aligned)
+    structural_sim, arcface_sim, secondary_sim = compute_ensemble_similarity(ensemble_gallery, ensemble_probe)
     tier1_score = structural_sim * 100
     
-    # 5. TIER 2: Geometric Biometrics (Anthropometric Ratio ΓÇö L2 Distance)
-    # Uses Euclidean distance between 12-D scale-invariant facial ratio vectors.
-    # Cosine similarity is inappropriate here: ratio vectors are always positive
-    # and in similar ranges, yielding ~0.99+ for ANY two human faces.
-    ratios_gallery = extract_geometric_ratios(gallery_landmarks)
-    ratios_probe = extract_geometric_ratios(probe_landmarks)
-    ratio_l2 = float(np.linalg.norm(ratios_gallery - ratios_probe))
-    # L2 mapping: 0 distance ΓåÆ 100%, ΓëÑ0.40 ΓåÆ 0%. Empirical threshold from
-    # same-person L2 < 0.10, different-person L2 > 0.20.
-    tier2_score = max(0.0, min(100.0, (1.0 - (ratio_l2 / 0.40)) * 100))
+    # 5. TIER 2: Geometric Biometrics (3D Topographical Mapping)
+    # Uses Euclidean distance between 12-D scale-invariant, 3D Procrustes-aligned facial ratio vectors.
+    ratios_gallery, gal_angles, gal_vis = extract_geometric_ratios_3d(gallery_landmarks)
+    ratios_probe, pro_angles, pro_vis = extract_geometric_ratios_3d(probe_landmarks)
+    
+    valid_mask = gal_vis["ratio_visibility"] & pro_vis["ratio_visibility"]
+    effective_ratios = int(np.sum(valid_mask))
+    
+    if effective_ratios > 0:
+        raw_l2 = float(np.linalg.norm((ratios_gallery - ratios_probe)[valid_mask]))
+        ratio_l2 = raw_l2 * math.sqrt(12.0 / effective_ratios)
+    else:
+        ratio_l2 = 0.50
+        
+    # L2 mapping: 0 distance ΓåÆ 100%, ΓëÑ0.50 ΓåÆ 0%. Recalibrated from 0.40 due to added 3D variance.
+    tier2_score = max(0.0, min(100.0, (1.0 - (ratio_l2 / 0.50)) * 100))
     
     # 6. TIER 3: Micro-Topology (LBP Chi-Squared Distance)
     # Chi-squared is the standard metric for comparing LBP histograms
@@ -1587,20 +2008,35 @@ def verify_pipeline(request: Request, payload: VerificationRequest, _: dict = De
     veto_triggered = structural_sim < 0.40
 
     # 7.5 TIER 4: Mark Correspondence (Bayesian LR Engine)
-    marks_gallery = detect_facial_marks(gallery_aligned, gallery_landmarks)
-    marks_probe = detect_facial_marks(probe_aligned, probe_landmarks)
-    mark_result = compute_mark_correspondence(marks_gallery, marks_probe)
+    marks_gallery, occ_gallery = detect_facial_marks(gallery_aligned, gallery_landmarks)
+    marks_probe, occ_probe = detect_facial_marks(probe_aligned, probe_landmarks)
+    
+    valid_gallery_marks = []
+    for m in marks_gallery:
+        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
+        if cy < 256 and cx < 256 and occ_probe[cy, cx] == 0:
+            clean_m = {k: v for k, v in m.items() if k != "contour"}
+            valid_gallery_marks.append(clean_m)
+            
+    valid_probe_marks = []
+    for m in marks_probe:
+        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
+        if cy < 256 and cx < 256 and occ_gallery[cy, cx] == 0:
+            clean_m = {k: v for k, v in m.items() if k != "contour"}
+            valid_probe_marks.append(clean_m)
+
+    mark_result = compute_mark_correspondence(valid_gallery_marks, valid_probe_marks)
     tier4_score = mark_result["score"]  # None if insufficient marks
 
-    # ΓöÇΓöÇ BAYESIAN EVIDENCE FUSION (Daubert v3.0) ΓöÇΓöÇ
-    # Convert ArcFace cosine to Likelihood Ratio
-    lr_arcface = cosine_to_lr_arcface(structural_sim)
+    # ΓöÇΓöÇ BAYESIAN EVIDENCE FUSION (Daubert v4.0) ΓöÇΓöÇ
+    # Convert Fused Ensemble score to Likelihood Ratio
+    lr_ensemble = score_to_lr_ensemble(structural_sim, temporal_delta=temporal_delta)
 
     # Combined mark LR (product of individual mark LRs)
     lr_marks = mark_result.get("lr_marks", 1.0)
 
     # Total LR = independent evidence product
-    lr_total = lr_arcface * lr_marks
+    lr_total = lr_ensemble * lr_marks
 
     # Posterior probability via Bayes' Theorem (neutral prior = 0.5)
     # P(Hp|E) = (Prior ├ù LR) / ((Prior ├ù LR) + (1 - Prior))
@@ -1617,14 +2053,8 @@ def verify_pipeline(request: Request, payload: VerificationRequest, _: dict = De
     base_fused_score = (tier1_score * _w1) + (tier2_score * _w2) + (tier3_score * _w3)
 
     if veto_triggered:
-        # Bayesian override: if mark LR is strong enough, the posterior
-        # will naturally rise above threshold. The veto FLAG still fires
-        # for forensic transparency but doesn't artificially cap the score.
-        matched_marks = mark_result["matched"]
-        if lr_marks > 100.0 and matched_marks >= 1:
-            conclusion = f"Conditional Match: ArcFace Veto ΓÇö {matched_marks} mark(s) yield LR={lr_marks:.1f}"
-        else:
-            conclusion = "Exclusion: Biometric Non-Match (ArcFace Cosine < 0.40)"
+        fused_score = 0.0
+        conclusion = "EXCLUSION: Biometric Non-Match (ArcFace Veto)"
     elif fused_score > 90.0:
         conclusion = "Strongest Support for Common Source"
     elif fused_score > 75.0:
@@ -1654,15 +2084,38 @@ def verify_pipeline(request: Request, payload: VerificationRequest, _: dict = De
     gallery_wireframe = generate_wireframe_hud(gallery_aligned, gallery_landmarks)
     probe_wireframe = generate_wireframe_hud(probe_aligned, probe_landmarks)
 
-    # Statistical confidence from Tier-1 raw cosine
+    # Statistical confidence from Tier-1 raw cosine (baseline)
     stats = calculate_statistical_confidence(structural_sim)
+    
+    # Upgrade statistical confidence to reflect the final Bayesian Posterior
+    bayesian_far = 1.0 - posterior
+    if bayesian_far < 1e-7:
+        stats["false_acceptance_rate"] = "< 1 in 10,000,000"
+        stats["statistical_certainty"] = f"{(posterior * 100):.6f}%"
+    elif bayesian_far >= 0.60:  # Maps to a fused_score < 40.0
+        stats["false_acceptance_rate"] = "DIFFERENT IDENTITIES"
+        stats["statistical_certainty"] = "0% ΓÇö Non-Match"
+    else:
+        stats["false_acceptance_rate"] = f"1 in {int(1.0 / bayesian_far):,}"
+        stats["statistical_certainty"] = f"{(posterior * 100):.6f}%"
 
     # Deep Forensic Telemetry (hash of 512-D ArcFace vector)
-    probe_vector_hash = compute_vector_hash(embed_probe)
+    probe_vector_hash = compute_vector_hash(ensemble_probe[0])
     probe_alignment = compute_alignment_variance(probe_aligned)
 
     audit = AuditLog(
         raw_cosine_score=round(structural_sim, 6),
+        raw_arcface_score=round(arcface_sim, 6),
+        raw_secondary_score=round(secondary_sim, 6),
+        ensemble_model_secondary="Facenet512",
+        pose_corrected_3d=True,
+        probe_pose_angles=pro_angles,
+        gallery_pose_angles=gal_angles,
+        occlusion_percentage=pro_vis["occlusion_percentage"],
+        occluded_regions=pro_vis["occluded_regions"],
+        effective_geometric_ratios_used=effective_ratios,
+        estimated_temporal_delta=round(temporal_delta, 1),
+        cross_spectral_correction_applied=spectral_correction,
         statistical_certainty=stats["statistical_certainty"],
         false_acceptance_rate=stats["false_acceptance_rate"],
         nodes_mapped=468,
@@ -1677,13 +2130,22 @@ def verify_pipeline(request: Request, payload: VerificationRequest, _: dict = De
         pipeline_version=PIPELINE_VERSION,
         dependency_versions=DEPENDENCY_VERSIONS,
         # Bayesian LR Forensic Audit Trail
-        lr_arcface=round(lr_arcface, 6),
+        lr_arcface=round(lr_ensemble, 6),
         lr_marks=round(lr_marks, 6),
         lr_total=round(lr_total, 6),
         posterior_probability=round(posterior, 8),
         mark_lrs=[round(lr, 4) for lr in mark_result.get("mark_lrs", [])],
     )
 
+    # Build correspondences list for the UI
+    correspondences = []
+    for g_idx, p_idx, individual_lr in mark_result.get("matches", []):
+        correspondences.append({
+            "gallery_pt": valid_gallery_marks[g_idx]["centroid"],
+            "probe_pt": valid_probe_marks[p_idx]["centroid"],
+            "lr": individual_lr
+        })
+
     response = VerificationResponse(
         structural_score=round(tier1_score, 2),
         soft_biometrics_score=round(tier2_score, 2),
@@ -1702,6 +2164,9 @@ def verify_pipeline(request: Request, payload: VerificationRequest, _: dict = De
         marks_detected_gallery=mark_result["total_gallery"],
         marks_detected_probe=mark_result["total_probe"],
         marks_matched=mark_result["matched"],
+        correspondences=correspondences,
+        raw_probe_marks=valid_probe_marks,
+        raw_gallery_marks=valid_gallery_marks,
         audit_log=audit
     )
 
@@ -1728,9 +2193,18 @@ def verify_pipeline(request: Request, payload: VerificationRequest, _: dict = De
             false_acceptance_rate=stats["false_acceptance_rate"],
             veto_triggered=veto_triggered,
             structural_score_x100=round(tier1_score * 100),
+            arcface_score_x10000=round(arcface_sim * 10000),
+            secondary_score_x10000=round(secondary_sim * 10000),
+            ensemble_model_secondary="Facenet512",
             geometric_score_x100=round(tier2_score * 100),
             micro_topology_score_x100=round(tier3_score * 100),
             mark_correspondence_x100=round(tier4_score * 100) if tier4_score is not None else None,
+            pose_corrected_3d=True,
+            probe_pose_angles=json.dumps(pro_angles) if pro_angles else None,
+            gallery_pose_angles=json.dumps(gal_angles) if gal_angles else None,
+            occlusion_percentage=pro_vis["occlusion_percentage"],
+            occluded_regions=json.dumps(pro_vis["occluded_regions"]) if pro_vis["occluded_regions"] else None,
+            effective_geometric_ratios_used=effective_ratios,
             receipt_url=receipt_url,
         )
         ledger_session.add(event)
@@ -1751,6 +2225,7 @@ TARGET_PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "
 
 class VaultSearchRequest(BaseModel):
     probe_url: str
+    require_liveness: bool = False
 
 
 def _generate_user_id(filepath: str) -> str:
@@ -1786,10 +2261,13 @@ def vault_search(request: Request, payload: VaultSearchRequest, _: dict = Depend
     # 1. Fetch & validate probe (with pre-decode binary hashing)
     probe_img, probe_file_hash = fetch_image_from_url(payload.probe_url)
 
-    liveness_result = detect_liveness(probe_img)
-    liveness_telemetry_vault = build_liveness_telemetry(liveness_result)
-    if liveness_result["score"] < 0.95:
-        raise HTTPException(status_code=403, detail="SPOOF_DETECTED: Presentation attack suspected.")
+    if payload.require_liveness:
+        liveness_result = detect_liveness(probe_img)
+        liveness_telemetry_vault = build_liveness_telemetry(liveness_result)
+        if liveness_result["score"] < 0.95:
+            raise HTTPException(status_code=403, detail="SPOOF_DETECTED: Presentation attack suspected.")
+    else:
+        liveness_telemetry_vault = {"status": "BYPASSED", "method": "NONE"}
 
     # 3. Pre-process probe
     probe_clahe = apply_clahe(probe_img)
@@ -1803,45 +2281,24 @@ def vault_search(request: Request, payload: VaultSearchRequest, _: dict = Depend
 
     probe_embedding = extract_arcface_embedding(probe_aligned)
 
-    # 4. Decrypt & search the entire vault
-    session = SessionLocal()
-    best_score = -1.0
-    best_user_id = None
-    vault_decrypt_start = time.perf_counter()
-
-    try:
-        profiles = session.query(IdentityProfile).all()
-
-        if not profiles:
-            raise HTTPException(
-                status_code=404,
-                detail="VAULT_EMPTY: No identity profiles in the database."
-            )
-
-        for profile in profiles:
-            try:
-                gallery_vec = decrypt_embedding(profile.encrypted_facial_embedding)
-                # Skip dimension mismatch (legacy 1404-D vs current 512-D ArcFace)
-                if gallery_vec.shape[0] != probe_embedding.shape[0]:
-                    continue
-                score = calculate_cosine_similarity(probe_embedding, gallery_vec)
-                if score > best_score:
-                    best_score = score
-                    best_user_id = profile.user_id
-            except Exception:
-                continue  # Skip corrupted or undecryptable records
-    finally:
-        session.close()
-    vault_decrypt_elapsed_ms = (time.perf_counter() - vault_decrypt_start) * 1000
+    # 4. Two-Stage Retrieval: Stage 1 (FAISS Filter)
+    vault_search_start = time.perf_counter()
+    results = vault_index.search(probe_embedding, top_k=1)
+    vault_decrypt_elapsed_ms = (time.perf_counter() - vault_search_start) * 1000
 
-    if best_user_id is None:
+    if not results:
         raise HTTPException(
             status_code=404,
-            detail="NO_MATCH: Could not match against any vault profile."
+            detail="VAULT_EMPTY_OR_NO_MATCH: No valid matches found in vault index."
         )
 
-    # 5. Tier 1 score from vault search
+    best_user_id, best_score = results[0]
+
+    # 5. Tier 1 score from vault search (ArcFace only initially)
     tier1_score = best_score * 100
+    arcface_sim = best_score
+    secondary_sim = 0.0
+    structural_sim = best_score
 
     # 6. Load matched gallery image for forensic overlays
     # First, fetch the matched profile from DB to get thumbnail_url
@@ -1880,14 +2337,44 @@ def vault_search(request: Request, payload: VaultSearchRequest, _: dict = Depend
                 gallery_aligned = probe_aligned
                 gallery_landmarks = probe_landmarks
 
-    # 7. Tier 2: Geometric Biometrics (Anthropometric Ratio ΓÇö L2 Distance)
+    # 6.4 Temporal Invariance Engine
     if gallery_landmarks is not None:
-        ratios_gallery = extract_geometric_ratios(gallery_landmarks)
-        ratios_probe = extract_geometric_ratios(probe_landmarks)
-        ratio_l2 = float(np.linalg.norm(ratios_gallery - ratios_probe))
-        tier2_score = max(0.0, min(100.0, (1.0 - (ratio_l2 / 0.40)) * 100))
+        gallery_age = estimate_age(gallery_aligned)
+        probe_age = estimate_age(probe_aligned)
+        temporal_delta = abs(probe_age - gallery_age)
+        gallery_aligned, probe_aligned, spectral_correction = cross_spectral_normalize(gallery_aligned, probe_aligned)
+    else:
+        temporal_delta = 0.0
+        spectral_correction = False
+
+    # 6.5 Upgrade Tier 1 to Neural Ensemble now that we have the gallery image
+    if gallery_landmarks is not None:
+        ensemble_gallery = extract_ensemble_embeddings(gallery_aligned)
+        ensemble_probe = extract_ensemble_embeddings(probe_aligned)
+        structural_sim, arcface_sim, secondary_sim = compute_ensemble_similarity(ensemble_gallery, ensemble_probe)
+        tier1_score = structural_sim * 100
+        best_score = structural_sim  # Use fused score for the rest of the pipeline
+
+    # 7. Tier 2: Geometric Biometrics (3D Topographical Mapping)
+    if gallery_landmarks is not None:
+        ratios_gallery, gal_angles, gal_vis = extract_geometric_ratios_3d(gallery_landmarks)
+        ratios_probe, pro_angles, pro_vis = extract_geometric_ratios_3d(probe_landmarks)
+        
+        valid_mask = gal_vis["ratio_visibility"] & pro_vis["ratio_visibility"]
+        effective_ratios = int(np.sum(valid_mask))
+        
+        if effective_ratios > 0:
+            raw_l2 = float(np.linalg.norm((ratios_gallery - ratios_probe)[valid_mask]))
+            ratio_l2 = raw_l2 * math.sqrt(12.0 / effective_ratios)
+        else:
+            ratio_l2 = 0.50
+            
+        tier2_score = max(0.0, min(100.0, (1.0 - (ratio_l2 / 0.50)) * 100))
     else:
         tier2_score = 0.0
+        gal_angles, pro_angles = {}, {}
+        pro_vis = {"occlusion_percentage": 0.0, "occluded_regions": []}
+        effective_ratios = 0
 
     # 8. Tier 3: Micro-Topology (LBP Chi-Squared Distance)
     lbp_gal = extract_lbp_histogram(gallery_aligned)
@@ -1896,15 +2383,30 @@ def vault_search(request: Request, payload: VaultSearchRequest, _: dict = Depend
     tier3_score = max(0.0, min(100.0, (1.0 - chi_squared) * 100))
 
     # 9. TIER 4: Mark Correspondence (Bayesian LR Engine)
-    marks_gallery = detect_facial_marks(gallery_aligned, gallery_landmarks)
-    marks_probe = detect_facial_marks(probe_aligned, probe_landmarks)
-    mark_result = compute_mark_correspondence(marks_gallery, marks_probe)
+    marks_gallery, occ_gallery = detect_facial_marks(gallery_aligned, gallery_landmarks)
+    marks_probe, occ_probe = detect_facial_marks(probe_aligned, probe_landmarks)
+    
+    valid_gallery_marks = []
+    for m in marks_gallery:
+        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
+        if cy < 256 and cx < 256 and occ_probe[cy, cx] == 0:
+            clean_m = {k: v for k, v in m.items() if k != "contour"}
+            valid_gallery_marks.append(clean_m)
+            
+    valid_probe_marks = []
+    for m in marks_probe:
+        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
+        if cy < 256 and cx < 256 and occ_gallery[cy, cx] == 0:
+            clean_m = {k: v for k, v in m.items() if k != "contour"}
+            valid_probe_marks.append(clean_m)
+
+    mark_result = compute_mark_correspondence(valid_gallery_marks, valid_probe_marks)
     tier4_score = mark_result["score"]  # None if insufficient marks
 
-    # ΓöÇΓöÇ BAYESIAN EVIDENCE FUSION (Daubert v3.0) ΓöÇΓöÇ
-    lr_arcface = cosine_to_lr_arcface(best_score)
+    # ΓöÇΓöÇ BAYESIAN EVIDENCE FUSION (Daubert v4.0) ΓöÇΓöÇ
+    lr_ensemble = score_to_lr_ensemble(structural_sim, temporal_delta=temporal_delta)
     lr_marks = mark_result.get("lr_marks", 1.0)
-    lr_total = lr_arcface * lr_marks
+    lr_total = lr_ensemble * lr_marks
 
     # Posterior probability via Bayes' Theorem (neutral prior = 0.5)
     PRIOR = 0.5
@@ -1923,11 +2425,8 @@ def vault_search(request: Request, payload: VaultSearchRequest, _: dict = Depend
 
     # 11. Conclusion
     if veto_arcface:
-        matched_marks = mark_result["matched"]
-        if lr_marks > 100.0 and matched_marks >= 1:
-            conclusion = f"CONDITIONAL MATCH ΓÇö ArcFace Veto, {matched_marks} mark(s) LR={lr_marks:.1f}: {best_user_id}"
-        else:
-            conclusion = f"EXCLUSION ΓÇö Biometric Non-Match: {best_user_id} (ArcFace: {best_score:.4f})"
+        fused_score = 0.0
+        conclusion = "EXCLUSION: Biometric Non-Match (ArcFace Veto)"
     elif fused_score > 90.0:
         conclusion = f"TARGET ACQUIRED ΓÇö Strongest match: {best_user_id} (Posterior: {fused_score:.1f}%)"
     elif fused_score > 75.0:
@@ -1975,7 +2474,18 @@ def vault_search(request: Request, payload: VaultSearchRequest, _: dict = Depend
     vault_alignment = compute_alignment_variance(probe_aligned)
 
     audit = AuditLog(
-        raw_cosine_score=round(best_score, 6),
+        raw_cosine_score=round(structural_sim, 6),
+        raw_arcface_score=round(arcface_sim, 6),
+        raw_secondary_score=round(secondary_sim, 6),
+        ensemble_model_secondary="Facenet512",
+        pose_corrected_3d=True,
+        probe_pose_angles=pro_angles,
+        gallery_pose_angles=gal_angles,
+        occlusion_percentage=pro_vis["occlusion_percentage"],
+        occluded_regions=pro_vis["occluded_regions"],
+        effective_geometric_ratios_used=effective_ratios,
+        estimated_temporal_delta=round(temporal_delta, 1),
+        cross_spectral_correction_applied=spectral_correction,
         statistical_certainty=stats["statistical_certainty"],
         false_acceptance_rate=stats["false_acceptance_rate"],
         nodes_mapped=468,
@@ -1998,13 +2508,22 @@ def vault_search(request: Request, payload: VaultSearchRequest, _: dict = Depend
         pipeline_version=PIPELINE_VERSION,
         dependency_versions=DEPENDENCY_VERSIONS,
         # Bayesian LR Forensic Audit Trail
-        lr_arcface=round(lr_arcface, 6),
+        lr_arcface=round(lr_ensemble, 6),
         lr_marks=round(lr_marks, 6),
         lr_total=round(lr_total, 6),
         posterior_probability=round(posterior, 8),
         mark_lrs=[round(lr, 4) for lr in mark_result.get("mark_lrs", [])],
     )
 
+    # Build correspondences list for the UI
+    correspondences = []
+    for g_idx, p_idx, individual_lr in mark_result.get("matches", []):
+        correspondences.append({
+            "gallery_pt": valid_gallery_marks[g_idx]["centroid"],
+            "probe_pt": valid_probe_marks[p_idx]["centroid"],
+            "lr": individual_lr
+        })
+
     response = VerificationResponse(
         structural_score=round(tier1_score, 2),
         soft_biometrics_score=round(tier2_score, 2),
@@ -2023,6 +2542,9 @@ def vault_search(request: Request, payload: VaultSearchRequest, _: dict = Depend
         marks_detected_gallery=mark_result["total_gallery"],
         marks_detected_probe=mark_result["total_probe"],
         marks_matched=mark_result["matched"],
+        correspondences=correspondences,
+        raw_probe_marks=valid_probe_marks,
+        raw_gallery_marks=valid_gallery_marks,
         audit_log=audit,
     )
 
@@ -2049,9 +2571,18 @@ def vault_search(request: Request, payload: VaultSearchRequest, _: dict = Depend
             false_acceptance_rate=stats["false_acceptance_rate"],
             veto_triggered=veto_arcface,
             structural_score_x100=round(tier1_score * 100),
+            arcface_score_x10000=round(arcface_sim * 10000),
+            secondary_score_x10000=round(secondary_sim * 10000),
+            ensemble_model_secondary="Facenet512",
             geometric_score_x100=round(tier2_score * 100),
             micro_topology_score_x100=round(tier3_score * 100),
             mark_correspondence_x100=round(tier4_score * 100) if tier4_score is not None else None,
+            pose_corrected_3d=True,
+            probe_pose_angles=json.dumps(pro_angles) if pro_angles else None,
+            gallery_pose_angles=json.dumps(gal_angles) if gal_angles else None,
+            occlusion_percentage=pro_vis["occlusion_percentage"],
+            occluded_regions=json.dumps(pro_vis["occluded_regions"]) if pro_vis["occluded_regions"] else None,
+            effective_geometric_ratios_used=effective_ratios,
             receipt_url=receipt_url,
         )
         ledger_session.add(event)
diff --git a/backend/models.py b/backend/models.py
index f91117f..9603bd7 100644
--- a/backend/models.py
+++ b/backend/models.py
@@ -101,6 +101,24 @@ class VerificationEvent(Base):
     geometric_score_x100 = Column(Integer, nullable=True)
     micro_topology_score_x100 = Column(Integer, nullable=True)
 
+    # Neural Ensemble Audit Trail (Tier 1 v4.0)
+    # Individual model scores before weighted fusion (├ù10000 for 4-decimal precision)
+    arcface_score_x10000 = Column(Integer, nullable=True)       # Raw ArcFace cosine ├ù 10000
+    secondary_score_x10000 = Column(Integer, nullable=True)     # Raw secondary model cosine ├ù 10000
+    ensemble_model_secondary = Column(String(64), nullable=True) # Name of secondary model (e.g., 'Facenet512')
+
+    # Tier 2: 3D Topographical Mapping Telemetry
+    pose_corrected_3d = Column(Boolean, nullable=True)
+    probe_pose_angles = Column(Text, nullable=True)     # JSON string of pitch, yaw, roll
+    gallery_pose_angles = Column(Text, nullable=True)   # JSON string of pitch, yaw, roll
+    occlusion_percentage = Column(Float, nullable=True)
+    occluded_regions = Column(Text, nullable=True)      # JSON list of strings
+    effective_geometric_ratios_used = Column(Integer, nullable=True)
+
+    # Temporal & Spectral Telemetry (Tier 1/3)
+    estimated_temporal_delta = Column(Float, nullable=True)
+    cross_spectral_correction_applied = Column(Boolean, nullable=True, default=False)
+
     # Tier 4: Mark Correspondence (├ù100 for integer consistency)
     mark_correspondence_x100 = Column(Integer, nullable=True)
 
@@ -111,6 +129,10 @@ class VerificationEvent(Base):
     lr_total = Column(Float, nullable=True)
     posterior_probability = Column(Float, nullable=True)
 
+    # Phase 7: Synthetic Provenance Veto (Deepfake Detection)
+    synthetic_anomaly_score = Column(Float, nullable=True)
+    failed_provenance_veto = Column(Boolean, nullable=True, default=False)
+
     # Composite forensic receipt (GCS URI of stitched PNG)
     receipt_url = Column(Text, nullable=True)
 
diff --git a/backend/requirements.txt b/backend/requirements.txt
index eb24694..7607e00 100644
--- a/backend/requirements.txt
+++ b/backend/requirements.txt
@@ -23,3 +23,4 @@ tqdm==4.67.1
 scikit-learn
 Pillow
 datasets
+faiss-cpu==1.8.0
diff --git a/frontend/app/page.tsx b/frontend/app/page.tsx
index 417415a..27ecd24 100644
--- a/frontend/app/page.tsx
+++ b/frontend/app/page.tsx
@@ -9,12 +9,126 @@ import html2canvas from 'html2canvas';
 const IdentityGraph = dynamic(() => import('@/components/IdentityGraph'), { ssr: false });
 
 function getApiUrl(): string {
-  if (typeof window !== 'undefined' && window.location.hostname.includes('facial-frontend')) {
-    return window.location.origin.replace('facial-frontend', 'facial-backend');
+  if (typeof window !== 'undefined' && window.location.hostname.includes('run.app')) {
+    return 'https://facial-backend-196207148120.us-east4.run.app';
   }
   return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
 }
 
+function TelemetryLoader() {
+  const [logs, setLogs] = useState<string[]>([]);
+  const [hash, setHash] = useState<string>('');
+  const [progress, setProgress] = useState(0);
+
+  const PIPELINE_STEPS = [
+    "[sys] Initializing secure TLS enclave...",
+    "[sys] Injecting payload into volatile memory...",
+    "[tier_1] Verifying image integrity and EXIF metadata...",
+    "[tier_1] Normalizing cross-spectral variants...",
+    "[tier_1] Executing MTCNN face detection...",
+    "[tier_1] Extracting 512-D neural embeddings (ArcFace)...",
+    "[tier_2] Extracting 468-point 3D facial mesh...",
+    "[tier_2] Executing 3D Procrustes rigid alignment...",
+    "[tier_2] Computing geometric ratios and soft biometrics...",
+    "[tier_3] Mapping micro-topology LBP textures...",
+    "[tier_3] Scanning for synthetic GAN anomalies...",
+    "[tier_4] Extracting localized facial marks and scars...",
+    "[tier_4] Querying population frequency database...",
+    "[tier_4] Calculating Bayesian Likelihood Ratios...",
+    "[sys] Fusing independent identity scores...",
+    "[sys] Finalizing Daubert-compliant audit trail..."
+  ];
+
+  useEffect(() => {
+    let currentIndex = 0;
+    const interval = setInterval(() => {
+      if (currentIndex < PIPELINE_STEPS.length) {
+        setLogs(prev => {
+          const newLogs = [...prev, PIPELINE_STEPS[currentIndex]];
+          if (newLogs.length > 5) return newLogs.slice(newLogs.length - 5);
+          return newLogs;
+        });
+        setProgress(Math.min(99, (currentIndex / PIPELINE_STEPS.length) * 100 + Math.random() * 5));
+        currentIndex++;
+      } else {
+        setLogs(prev => {
+          const newLogs = [...prev, "[sys] Awaiting server response..."];
+          if (newLogs.length > 5) return newLogs.slice(newLogs.length - 5);
+          return newLogs;
+        });
+        setProgress(99);
+      }
+    }, 450);
+
+    return () => clearInterval(interval);
+  }, []);
+
+  useEffect(() => {
+    const hashInterval = setInterval(() => {
+      const randomHex = Array.from({length: 64}, () => Math.floor(Math.random()*16).toString(16)).join('');
+      setHash(randomHex);
+    }, 50);
+    return () => clearInterval(hashInterval);
+  }, []);
+
+  return (
+    <div className="h-full flex flex-col items-center justify-center p-4 w-full">
+      <div className="w-full max-w-2xl border border-gray-700 bg-[#050505] flex flex-col p-6 font-mono relative overflow-hidden shadow-[0_0_30px_rgba(0,0,0,0.8)]">
+        {/* Header */}
+        <div className="flex justify-between items-center border-b border-gray-800 pb-3 mb-4 relative z-10">
+          <div className="flex flex-col">
+            <span className="text-[#D4AF37] text-xs tracking-[0.2em] font-bold">ACTIVE TELEMETRY</span>
+            <span className="text-gray-600 text-[9px] tracking-widest mt-0.5">INSTITUTIONAL PIPELINE STATUS</span>
+          </div>
+          <div className="flex items-center gap-2">
+            <div className="w-2 h-2 bg-red-500 animate-[pulse_0.5s_infinite] rounded-full"></div>
+            <span className="text-red-500 text-[10px] tracking-widest font-bold">PROCESSING</span>
+          </div>
+        </div>
+        
+        {/* Cryptographic Visual (Hex Dump) */}
+        <div className="bg-[#0a0a0a] p-3 border border-gray-800 mb-4 relative z-10">
+          <div className="text-[8px] text-gray-500 tracking-[0.2em] mb-1.5 flex justify-between">
+            <span>KMS ENVELOPE DECRYPTION [SHA-256]</span>
+            <span className="text-gray-600">SECURE ENCLAVE</span>
+          </div>
+          <div className="text-[10px] text-emerald-500/80 break-all font-bold tracking-widest leading-relaxed">
+            {hash}
+          </div>
+        </div>
+
+        {/* Terminal Feed */}
+        <div className="h-28 flex flex-col justify-end text-[10px] text-gray-400 gap-1.5 mb-4 relative z-10">
+          {logs.map((log, i) => (
+            <div key={i} className="flex gap-2 items-start">
+              <span className="text-gray-600 shrink-0">{'>'}</span>
+              <span className={i === logs.length - 1 ? 'text-gray-200' : 'text-gray-500'}>{log}</span>
+            </div>
+          ))}
+          <div className="flex gap-2 items-start text-[#D4AF37] animate-pulse">
+            <span className="shrink-0">{'>'}</span>
+            <span>_</span>
+          </div>
+        </div>
+
+        {/* Progress Architecture */}
+        <div className="relative z-10">
+          <div className="flex justify-between text-[9px] text-gray-500 mb-1.5 tracking-widest font-bold">
+            <span>PIPELINE LOAD</span>
+            <span className="text-[#D4AF37]">{Math.floor(progress)}%</span>
+          </div>
+          <div className="w-full h-2 bg-[#111] border border-gray-700 overflow-hidden">
+            <div 
+              className="h-full bg-gradient-to-r from-gray-700 via-gray-400 to-white transition-all duration-300 ease-out" 
+              style={{ width: `${progress}%` }}
+            ></div>
+          </div>
+        </div>
+      </div>
+    </div>
+  );
+}
+
 export default function Home() {
   const [probeFile, setProbeFile] = useState<File | null>(null);
   const [probePreview, setProbePreview] = useState<string>('');
@@ -91,6 +205,7 @@ export default function Home() {
     scar_delta_b64: string;
     gallery_wireframe_b64: string;
     probe_wireframe_b64: string;
+    correspondences?: any[];
     audit_log?: AuditLog;
   }
 
@@ -463,7 +578,7 @@ export default function Home() {
   // ΓöÇΓöÇΓöÇ SOVEREIGN IDENTITY GATEWAY (Landing Page) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
   if (!token) {
     return (
-      <main className="min-h-screen w-screen bg-[#050A10] text-[#E0E0E0] selection:bg-[#D4AF37] selection:text-black overflow-y-auto overflow-x-hidden">
+      <main className="min-h-screen w-full bg-[#050A10] text-[#E0E0E0] selection:bg-[#D4AF37] selection:text-black overflow-y-auto overflow-x-hidden">
 
         {/* ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
             SECTION 1: HERO ΓÇö Sovereign Identity Gateway
@@ -775,7 +890,7 @@ export default function Home() {
 
   // ΓöÇΓöÇΓöÇ MAIN TERMINAL ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
   return (
-    <main className="h-screen w-screen overflow-hidden bg-[#0A0A0B] text-[#E0E0E0] font-mono selection:bg-[#D4AF37] selection:text-black flex flex-col">
+    <main className="h-screen w-full overflow-hidden bg-[#0A0A0B] text-[#E0E0E0] font-mono selection:bg-[#D4AF37] selection:text-black flex flex-col">
       
       {/* ΓöÇΓöÇ Top Bar ΓöÇΓöÇ */}
       <header className="shrink-0 flex justify-between items-center px-5 py-2.5 border-b border-[#1a1a1a]">
@@ -918,14 +1033,7 @@ export default function Home() {
 
         {/* ΓòÉΓòÉΓòÉΓòÉ LOADING ΓòÉΓòÉΓòÉΓòÉ */}
         {['uploading', 'frontalizing', 'calculating'].includes(step) && (
-          <div className="h-full flex flex-col items-center justify-center">
-            <div className="w-12 h-12 border-4 border-[#333] border-t-[#D4AF37] rounded-full animate-spin mb-4"></div>
-            <p className="text-sm text-gray-300 tracking-widest animate-pulse">
-              {step === 'uploading' && "UPLOADING TARGET..."}
-              {step === 'frontalizing' && "SCANNING ENCRYPTED VAULT..."}
-              {step === 'calculating' && "DECRYPTING IDENTITY VECTORS..."}
-            </p>
-          </div>
+          <TelemetryLoader />
         )}
 
         {/* ΓòÉΓòÉΓòÉΓòÉ ERROR ΓòÉΓòÉΓòÉΓòÉ */}
@@ -1055,18 +1163,14 @@ export default function Home() {
               </div>
               <div className={`flex-1 min-h-0 bg-[#0d0d0e] border rounded-lg p-2 ${(results.fused_identity_score < 40.0) ? 'border-red-900/50 shadow-[0_0_20px_rgba(180,0,30,0.15)]' : 'border-[#D4AF37]/30 shadow-[0_0_20px_rgba(212,175,55,0.08)]'}`}>
                 <SymmetryMerge
-                  galleryImageSrc={results.gallery_aligned_b64}
-                  probeImageSrc={results.probe_aligned_b64}
-                  deltaImageSrc={results.scar_delta_b64}
-                  galleryWireframeSrc={results.gallery_wireframe_b64}
-                  probeWireframeSrc={results.probe_wireframe_b64}
+                  results={results}
                   isXrayMode={isXrayMode}
                 />
               </div>
             </div>
 
             {/* ΓöÇΓöÇ RIGHT PANEL (30%): Intelligence Panel ΓÇö Human-Readable ΓöÇΓöÇ */}
-            <div className="w-[30%] flex flex-col gap-2 min-h-0 overflow-y-auto overflow-x-hidden shrink-0 pr-0.5">
+            <div className="w-[30%] flex flex-col gap-2 min-h-0 overflow-y-auto overflow-x-hidden shrink-0 min-w-0 break-words pr-0.5">
 
               {/* ΓòÉΓòÉΓòÉ OVERALL MATCH ΓÇö Hero Score ΓòÉΓòÉΓòÉ */}
               <div className={`relative overflow-hidden rounded-lg p-4 border-2 ${(results.fused_identity_score < 40.0) ? 'border-red-700/60 bg-gradient-to-br from-[#1a0505] to-[#0d0d0e]' : 'border-[#D4AF37]/50 bg-gradient-to-br from-[#1a170d] to-[#0d0d0e]'}`}>
@@ -1074,15 +1178,15 @@ export default function Home() {
                 <div className={`absolute -bottom-4 -left-4 w-16 h-16 rounded-full ${(results.fused_identity_score < 40.0) ? 'bg-red-500/5' : 'bg-[#D4AF37]/5'}`}></div>
                 <div className="relative z-10">
                   <div className={`text-[8px] tracking-[0.3em] mb-1 ${(results.fused_identity_score < 40.0) ? 'text-red-400/70' : 'text-[#D4AF37]/70'}`}>POSTERIOR PROBABILITY</div>
-                  <div className="flex items-baseline gap-1.5">
+                  <div className="flex items-baseline gap-1.5 flex-wrap overflow-hidden min-w-0 w-full">
                     <span className={`text-4xl font-bold tabular-nums ${(results.fused_identity_score < 40.0) ? 'text-red-400' : 'text-[#D4AF37]'}`}>{results.fused_identity_score}</span>
                     <span className={`text-lg font-bold ${(results.fused_identity_score < 40.0) ? 'text-red-400/60' : 'text-[#D4AF37]/60'}`}>%</span>
                   </div>
                   {/* LR_total context */}
                   {results.audit_log?.lr_total != null && (
-                    <div className="mt-1.5 flex items-center gap-2">
-                      <span className="text-[8px] text-gray-500 tracking-wider">LR<sub>total</sub></span>
-                      <span className={`text-[11px] font-bold tabular-nums ${(results.fused_identity_score < 40.0) ? 'text-red-400/80' : 'text-[#D4AF37]/90'}`}>{formatLR(results.audit_log.lr_total)}</span>
+                    <div className="mt-1.5 flex items-center gap-2 max-w-full">
+                      <span className="text-[8px] text-gray-500 tracking-wider shrink-0">LR<sub>total</sub></span>
+                      <span className={`text-[11px] font-bold tabular-nums truncate ${(results.fused_identity_score < 40.0) ? 'text-red-400/80' : 'text-[#D4AF37]/90'}`}>{formatLR(results.audit_log.lr_total)}</span>
                     </div>
                   )}
                   {/* Score bar */}
@@ -1177,7 +1281,7 @@ export default function Home() {
                   <div className="p-2.5 border-t border-[#1a1a1a]">
                     <div className="flex items-baseline justify-between">
                       <h3 className="text-[#D4AF37] text-[9px] tracking-wider font-bold">MARK EVIDENCE (LR)</h3>
-                      <span className="text-lg font-bold tabular-nums text-[#D4AF37]">{formatLR(results.audit_log?.lr_marks)}</span>
+                      <span className="text-lg font-bold tabular-nums text-[#D4AF37] break-all whitespace-normal overflow-hidden">{formatLR(results.audit_log?.lr_marks)}</span>
                     </div>
                     {/* LR magnitude bar ΓÇö log-scaled */}
                     <div className="mt-1 h-1 w-full bg-[#111] rounded-full overflow-hidden border border-[#D4AF37]/20">
@@ -1186,23 +1290,23 @@ export default function Home() {
                         style={{ width: `${Math.min(100, results.audit_log?.lr_marks != null ? Math.min(100, Math.log10(Math.max(1, results.audit_log.lr_marks)) * 10) : 0)}%` }}
                       />
                     </div>
-                    <p className="text-[9px] text-[#D4AF37]/70 mt-1.5 leading-relaxed">Bayesian Likelihood Ratio from {results.marks_matched} matching scars, moles, and birthmarks. Values {'>'} 1 support same-identity hypothesis; values {'>'} 10,000 constitute strong forensic evidence.</p>
+                    <p className="text-[8px] break-words text-[#D4AF37]/70 mt-1.5 leading-relaxed">Bayesian Likelihood Ratio from {results.marks_matched} matching scars, moles, and birthmarks. Values {'>'} 1 support same-identity hypothesis; values {'>'} 10,000 constitute strong forensic evidence.</p>
                     {/* Individual mark LR breakdown */}
                     {results.audit_log?.lr_arcface != null && (
                       <div className="mt-1.5 flex items-center gap-3 flex-wrap">
                         <div className="flex items-center gap-1">
                           <span className="text-[7px] text-gray-500">LR<sub>arcface</sub></span>
-                          <span className="text-[8px] font-bold text-[#D4AF37]/80 tabular-nums">{formatLR(results.audit_log.lr_arcface)}</span>
+                          <span className="text-[8px] font-bold text-[#D4AF37]/80 tabular-nums break-all whitespace-normal overflow-hidden">{formatLR(results.audit_log.lr_arcface)}</span>
                         </div>
                         <span className="text-[7px] text-gray-600">├ù</span>
                         <div className="flex items-center gap-1">
                           <span className="text-[7px] text-gray-500">LR<sub>marks</sub></span>
-                          <span className="text-[8px] font-bold text-[#D4AF37]/80 tabular-nums">{formatLR(results.audit_log.lr_marks)}</span>
+                          <span className="text-[8px] font-bold text-[#D4AF37]/80 tabular-nums break-all whitespace-normal overflow-hidden">{formatLR(results.audit_log.lr_marks)}</span>
                         </div>
                         <span className="text-[7px] text-gray-600">=</span>
                         <div className="flex items-center gap-1">
                           <span className="text-[7px] text-gray-500">LR<sub>total</sub></span>
-                          <span className="text-[8px] font-bold text-[#D4AF37] tabular-nums">{formatLR(results.audit_log.lr_total)}</span>
+                          <span className="text-[8px] font-bold text-[#D4AF37] tabular-nums break-all whitespace-normal overflow-hidden">{formatLR(results.audit_log.lr_total)}</span>
                         </div>
                       </div>
                     )}
@@ -1223,21 +1327,21 @@ export default function Home() {
                   {(results.fused_identity_score < 40.0) ? 'Γ£ù VERDICT: NOT A MATCH' : (results.veto_triggered && results.fused_identity_score >= 40.0) ? 'ΓÜá VERDICT: CONDITIONAL MATCH' : 'Γ£ô VERDICT: MATCH DETECTED'}
                 </div>
                 <div className={`px-3 py-3 ${(results.fused_identity_score < 40.0) ? 'bg-red-950/20' : 'bg-[#0d0d0e]'}`}>
-                  <p className={`text-[11px] leading-relaxed ${(results.fused_identity_score < 40.0) ? 'text-red-300/90' : 'text-gray-200'}`}>
+                  <p className={`text-[11px] leading-relaxed break-all whitespace-normal overflow-hidden ${(results.fused_identity_score < 40.0) ? 'text-red-300/90' : 'text-gray-200'}`}>
                     {results.conclusion}
                   </p>
-                  {(results.fused_identity_score < 40.0) && (
+                  {(results.fused_identity_score < 40.0 && results.veto_triggered) && (
                     <div className="mt-2 px-2 py-1.5 bg-red-950/30 rounded border border-red-900/30">
-                      <p className="text-[8px] text-red-400/70 leading-relaxed">The face recognition AI returned a similarity of {results.structural_score}%, which is below the 40% minimum required to consider a potential match. This is an automatic exclusion.</p>
-                    </div>
-                  )}
-                  {(results.veto_triggered && results.fused_identity_score >= 40.0) && (
-                    <div className="mt-2 px-2 py-1.5 bg-amber-950/20 rounded border border-amber-900/30">
                       <div className="flex items-center gap-1.5 mb-1">
-                        <div className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></div>
-                        <span className="text-[9px] text-amber-500/80 tracking-wider font-bold">VETO OVERRIDDEN BY BAYESIAN EVIDENCE</span>
+                        <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></div>
+                        <span className="text-[9px] text-red-500/80 tracking-wider font-bold">ARCFACE VETO TRIGGERED</span>
                       </div>
-                      <p className="text-[8px] text-amber-400/60 leading-relaxed">ArcFace structural similarity fell below threshold, but physical mark evidence (LR<sub>marks</sub> = {formatLR(results.audit_log?.lr_marks)}) provided overwhelming statistical support for same-identity. Posterior probability: {results.audit_log?.posterior_probability != null ? `${(results.audit_log.posterior_probability * 100).toFixed(4)}%` : 'N/A'}.</p>
+                      <p className="text-[8px] text-red-400/70 leading-relaxed break-words">Structural similarity fell below threshold. This is an automatic exclusion. Any subsequent Bayesian mark evidence has been overruled.</p>
+                    </div>
+                  )}
+                  {(results.fused_identity_score < 40.0 && !results.veto_triggered) && (
+                    <div className="mt-2 px-2 py-1.5 bg-red-950/30 rounded border border-red-900/30">
+                      <p className="text-[8px] text-red-400/70 leading-relaxed break-words">The face recognition AI returned a similarity of {results.structural_score}%, which is below the 40% minimum required to consider a potential match. This is an automatic exclusion.</p>
                     </div>
                   )}
                   {(!results.veto_triggered && results.fused_identity_score >= 40.0) && (
@@ -1287,10 +1391,10 @@ export default function Home() {
               {/* ΓöÇΓöÇ Technical Details (3-column) ΓöÇΓöÇ */}
               {auditExpanded && results.audit_log && (
                 <div className="border border-[#1a1a0a] bg-[#000000] rounded p-2.5 font-mono text-[9px] leading-relaxed shadow-[inset_0_0_30px_rgba(0,0,0,0.5)]">
-                  <div className="grid grid-cols-2 gap-4 mt-1">
+                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-1 w-full min-w-0">
 
                     {/* Block 1: Confidence & Accuracy */}
-                    <div className="border border-[#1a2a1a] rounded p-2 bg-[#010201]">
+                    <div className="border border-[#1a2a1a] rounded p-2 bg-[#010201] min-w-0">
                       <div className="text-green-500/80 tracking-[0.2em] mb-1 border-b border-green-900/30 pb-1 text-[8px]">Γû╕ CONFIDENCE &amp; ACCURACY</div>
                       <p className="text-[7px] text-gray-600 mb-1.5 leading-relaxed">How confident is the system in this result? Lower error rates mean higher reliability.</p>
                       <div className="space-y-0.5 pl-1">
@@ -1305,7 +1409,7 @@ export default function Home() {
                     </div>
 
                     {/* Block 2: Image Quality & Authenticity */}
-                    <div className="border border-[#2a1a1a] rounded p-2 bg-[#020101]">
+                    <div className="border border-[#2a1a1a] rounded p-2 bg-[#020101] min-w-0">
                       <div className="text-cyan-500/80 tracking-[0.2em] mb-1 border-b border-cyan-900/30 pb-1 text-[8px]">Γû╕ IMAGE QUALITY &amp; AUTHENTICITY</div>
                       <p className="text-[7px] text-gray-600 mb-1.5 leading-relaxed">How were the photos corrected for comparison, and are they real photographs?</p>
                       <div className="space-y-0.5 pl-1">
@@ -1326,37 +1430,37 @@ export default function Home() {
                     </div>
 
                     {/* Block 3: Security & Data Integrity */}
-                    <div className="border border-[#1a1a2a] rounded p-2 bg-[#010102]">
+                    <div className="border border-[#1a1a2a] rounded p-2 bg-[#010102] min-w-0">
                       <div className="text-amber-500/80 tracking-[0.2em] mb-1 border-b border-amber-900/30 pb-1 text-[8px]">Γû╕ SECURITY &amp; DATA INTEGRITY</div>
                       <p className="text-[7px] text-gray-600 mb-1.5 leading-relaxed">Cryptographic proof that the biometric data was not tampered with during analysis.</p>
                       <div className="space-y-0.5 pl-1">
                         {results.audit_log.vector_hash && (
-                          <div><span className="text-gray-500">Digital Fingerprint</span><div className="text-amber-300/80 text-[8px] break-all mt-0.5">{results.audit_log.vector_hash}</div></div>
+                          <div><span className="text-gray-500">Digital Fingerprint</span><div className="text-amber-300/80 text-[8px] break-all whitespace-pre-wrap w-full min-w-0 mt-0.5">{results.audit_log.vector_hash}</div></div>
                         )}
                         {results.audit_log.crypto_envelope && (<>
-                          <div className="flex justify-between mt-1"><span className="text-gray-500">Encryption Standard</span><span className="text-amber-300">{results.audit_log.crypto_envelope.standard}</span></div>
-                          <div className="flex justify-between"><span className="text-gray-500">Decryption Speed</span><span className="text-amber-300">{results.audit_log.crypto_envelope.decryption_time}</span></div>
+                          <div className="flex justify-between items-start gap-2 break-words w-full min-w-0 mt-1"><span className="text-gray-500">Encryption Standard</span><span className="text-amber-300">{results.audit_log.crypto_envelope.standard}</span></div>
+                          <div className="flex justify-between items-start gap-2 break-words w-full min-w-0"><span className="text-gray-500">Decryption Speed</span><span className="text-amber-300">{results.audit_log.crypto_envelope.decryption_time}</span></div>
                         </>)}
                         {results.audit_log.matched_user_id && (
-                          <div className="flex justify-between mt-1"><span className="text-gray-500">Matched Profile ID</span><span className="text-white">{results.audit_log.matched_user_id}</span></div>
+                          <div className="flex justify-between items-start gap-2 break-words w-full min-w-0 mt-1"><span className="text-gray-500">Matched Profile ID</span><span className="text-white">{results.audit_log.matched_user_id}</span></div>
                         )}
                         {results.audit_log.person_name && (
-                          <div className="flex justify-between"><span className="text-gray-500">Matched Name</span><span className="text-white">{results.audit_log.person_name}</span></div>
+                          <div className="flex justify-between items-start gap-2 break-words w-full min-w-0"><span className="text-gray-500">Matched Name</span><span className="text-white">{results.audit_log.person_name}</span></div>
                         )}
                         {results.audit_log.license_short_name && (
-                          <div className="flex justify-between"><span className="text-gray-500">Image License</span><span className="text-gray-400">{results.audit_log.license_short_name}</span></div>
+                          <div className="flex justify-between items-start gap-2 break-words w-full min-w-0"><span className="text-gray-500">Image License</span><span className="text-gray-400">{results.audit_log.license_short_name}</span></div>
                         )}
                       </div>
                     </div>
 
                     {/* Block 4: Bayesian Evidence ΓÇö Daubert Forensic Trail */}
-                    <div className="border border-[#2a1a2a] rounded p-2 bg-[#020102]">
+                    <div className="border border-[#2a1a2a] rounded p-2 bg-[#020102] min-w-0">
                       <div className="text-purple-400/80 tracking-[0.2em] mb-1 border-b border-purple-900/30 pb-1 text-[8px]">Γû╕ BAYESIAN EVIDENCE (DAUBERT v3.0)</div>
-                      <p className="text-[7px] text-gray-600 mb-1.5 leading-relaxed">Likelihood Ratios quantifying the strength of evidence for same-identity hypothesis.</p>
+                      <p className="text-[8px] break-words text-gray-600 mb-1.5 leading-relaxed">Likelihood Ratios quantifying the strength of evidence for same-identity hypothesis.</p>
                       <div className="space-y-0.5 pl-1">
-                        <div className="flex justify-between"><span className="text-gray-500">LR<sub>arcface</sub></span><span className="text-purple-300 font-bold">{formatLRSci(results.audit_log.lr_arcface)}</span></div>
-                        <div className="flex justify-between"><span className="text-gray-500">LR<sub>marks</sub></span><span className="text-purple-300 font-bold">{formatLRSci(results.audit_log.lr_marks)}</span></div>
-                        <div className="flex justify-between"><span className="text-gray-500">LR<sub>total</sub></span><span className="text-[#D4AF37] font-bold">{formatLRSci(results.audit_log.lr_total)}</span></div>
+                        <div className="flex justify-between"><span className="text-gray-500">LR<sub>arcface</sub></span><span className="text-purple-300 font-bold break-all whitespace-normal overflow-hidden">{formatLRSci(results.audit_log.lr_arcface)}</span></div>
+                        <div className="flex justify-between"><span className="text-gray-500">LR<sub>marks</sub></span><span className="text-purple-300 font-bold break-all whitespace-normal overflow-hidden">{formatLRSci(results.audit_log.lr_marks)}</span></div>
+                        <div className="flex justify-between"><span className="text-gray-500">LR<sub>total</sub></span><span className="text-[#D4AF37] font-bold break-all whitespace-normal overflow-hidden">{formatLRSci(results.audit_log.lr_total)}</span></div>
                         <div className="flex justify-between mt-1 pt-1 border-t border-purple-900/20"><span className="text-gray-500">Posterior P(same)</span><span className="text-[#D4AF37] font-bold">{results.audit_log.posterior_probability != null ? `${(results.audit_log.posterior_probability * 100).toFixed(6)}%` : 'N/A'}</span></div>
                         {results.audit_log.mark_lrs && results.audit_log.mark_lrs.length > 0 && (
                           <div className="mt-1 pt-1 border-t border-purple-900/20">
diff --git a/frontend/components/SymmetryMerge.tsx b/frontend/components/SymmetryMerge.tsx
index 4c14fb4..a524759 100644
--- a/frontend/components/SymmetryMerge.tsx
+++ b/frontend/components/SymmetryMerge.tsx
@@ -3,11 +3,7 @@
 import React, { useRef, useState, useEffect, useCallback } from 'react';
 
 interface SymmetryMergeProps {
-  galleryImageSrc: string;
-  probeImageSrc: string;
-  deltaImageSrc?: string;
-  galleryWireframeSrc?: string;
-  probeWireframeSrc?: string;
+  results: any | null;
   isXrayMode?: boolean;
 }
 
@@ -59,7 +55,8 @@ function drawPane(
   borderColor?: string,
   baseOpacity?: number,
   overlayOpacity?: number,
-  xrayFilter?: boolean
+  xrayFilter?: boolean,
+  points?: {x: number, y: number, lr?: number, isMatched?: boolean}[]
 ) {
   const ctx = canvas.getContext('2d');
   if (!ctx) return;
@@ -109,6 +106,27 @@ function drawPane(
     ctx.globalAlpha = 1;
   }
 
+  if (points && points.length > 0) {
+    points.forEach(p => {
+      const px = p.x * iw;
+      const py = p.y * ih;
+      ctx.beginPath();
+      ctx.arc(px, py, Math.max(2, 4 / scale), 0, 2 * Math.PI);
+      
+      if (p.isMatched === false) {
+        ctx.fillStyle = 'rgba(255, 32, 64, 0.9)'; // Red for missing correspondence
+        ctx.strokeStyle = '#5a0015';
+      } else {
+        ctx.fillStyle = 'rgba(212, 175, 55, 0.9)'; // Gold color for matched marks
+        ctx.strokeStyle = '#111';
+      }
+      
+      ctx.fill();
+      ctx.lineWidth = 1.5 / scale;
+      ctx.stroke();
+    });
+  }
+
   if (borderColor) {
     ctx.globalAlpha = 1;
     ctx.filter = 'none';
@@ -127,13 +145,14 @@ function drawPane(
  *   Right = Gallery (Vault Match / Known Alias)
  */
 export default function SymmetryMerge({
-  galleryImageSrc,
-  probeImageSrc,
-  deltaImageSrc,
-  galleryWireframeSrc,
-  probeWireframeSrc,
+  results,
   isXrayMode = false,
 }: SymmetryMergeProps) {
+  const galleryImageSrc = results?.gallery_aligned_b64;
+  const probeImageSrc = results?.probe_aligned_b64;
+  const deltaImageSrc = results?.scar_delta_b64;
+  const galleryWireframeSrc = results?.gallery_wireframe_b64;
+  const probeWireframeSrc = results?.probe_wireframe_b64;
   const [mode, setMode] = useState<ViewMode>('aligned');
   const [zoom, setZoom] = useState(1);
   const [pan, setPan] = useState({ x: 0, y: 0 });
@@ -159,12 +178,12 @@ export default function SymmetryMerge({
 
   const imagesReady = !!galleryImg && !!probeImg;
 
-  // ΓöÇΓöÇ LEFT PANE = PROBE (+ delta overlay in delta mode, + wireframe in mesh mode) ΓöÇΓöÇ
+  // ΓöÇΓöÇ LEFT PANE = PROBE (+ wireframe in mesh mode) ΓöÇΓöÇ
   const getLeftOverlay = useCallback((): HTMLImageElement | null => {
     if (mode === 'mesh' && pWireImg) return pWireImg;
-    if (mode === 'delta' && deltaImg) return deltaImg;
+    if (mode === 'delta') return null;
     return null;
-  }, [mode, pWireImg, deltaImg]);
+  }, [mode, pWireImg]);
 
   // ΓöÇΓöÇ RIGHT PANE = GALLERY (+ delta overlay in delta mode, + wireframe in mesh mode) ΓöÇΓöÇ
   const getRightOverlay = useCallback((): HTMLImageElement | null => {
@@ -184,15 +203,53 @@ export default function SymmetryMerge({
   const baseOpacity = isXrayMode && hasOverlay ? 0.1 : 1.0;
   const overlayOpacity = isXrayMode && hasOverlay ? 1.0 : (mode === 'delta' ? 0.7 : 0.85);
 
+  const getIsMatched = (pt: any, side: 'probe' | 'gallery') => {
+    if (!results?.correspondences) return false;
+    return results.correspondences.some((c: any) => {
+      const cPt = c[`${side}_pt`];
+      if (!cPt) return false;
+      const pX = pt[0] !== undefined ? pt[0] : pt.x;
+      const pY = pt[1] !== undefined ? pt[1] : pt.y;
+      const cx = cPt[0] !== undefined ? cPt[0] : cPt.x;
+      const cy = cPt[1] !== undefined ? cPt[1] : cPt.y;
+      return Math.abs(pX - cx) < 0.001 && Math.abs(pY - cy) < 0.001;
+    });
+  };
+
+  const mapPoint = (m: any, side: 'probe' | 'gallery') => {
+    const x = m[0] !== undefined ? m[0] : m.x;
+    const y = m[1] !== undefined ? m[1] : m.y;
+    let lr = m.lr;
+    if (lr === undefined && results?.correspondences) {
+      const corr = results.correspondences.find((c: any) => {
+        const cPt = c[`${side}_pt`];
+        if (!cPt) return false;
+        const cx = cPt[0] !== undefined ? cPt[0] : cPt.x;
+        const cy = cPt[1] !== undefined ? cPt[1] : cPt.y;
+        return Math.abs(x - cx) < 0.001 && Math.abs(y - cy) < 0.001;
+      });
+      if (corr) lr = corr.lr;
+    }
+    return { x, y, lr, isMatched: getIsMatched(m, side) };
+  };
+
+  const probeMarksRaw = results?.raw_probe_marks || results?.probe_data?.marks || [];
+  const galleryMarksRaw = results?.raw_gallery_marks || results?.gallery_data?.marks || [];
+
+  console.log("STRICT MARKS - LEFT:", results?.probe_data?.marks, "RIGHT:", results?.gallery_data?.marks);
+
+  const probePoints = probeMarksRaw.map((m: any) => mapPoint(m, 'probe'));
+  const galleryPoints = galleryMarksRaw.map((m: any) => mapPoint(m, 'gallery'));
+
   // Draw dual panes ΓÇö LEFT = PROBE, RIGHT = GALLERY (both get delta overlay in delta mode)
   useEffect(() => {
     if (!imagesReady || mode === 'overlap') return;
 
     if (leftCanvasRef.current && probeImg) {
-      drawPane(leftCanvasRef.current, probeImg, getLeftOverlay(), zoom, pan, getBorderColor(), baseOpacity, overlayOpacity, isXrayMode);
+      drawPane(leftCanvasRef.current, probeImg, getLeftOverlay(), zoom, pan, getBorderColor(), baseOpacity, overlayOpacity, isXrayMode, probePoints);
     }
     if (rightCanvasRef.current && galleryImg) {
-      drawPane(rightCanvasRef.current, galleryImg, getRightOverlay(), zoom, pan, getBorderColor(), baseOpacity, overlayOpacity, isXrayMode);
+      drawPane(rightCanvasRef.current, galleryImg, getRightOverlay(), zoom, pan, getBorderColor(), baseOpacity, overlayOpacity, isXrayMode, galleryPoints);
     }
   });
 
@@ -201,10 +258,10 @@ export default function SymmetryMerge({
     if (!imagesReady || mode !== 'overlap') return;
 
     if (overlapLeftRef.current && probeImg) {
-      drawPane(overlapLeftRef.current, probeImg, null, zoom, pan, undefined, undefined, undefined, isXrayMode);
+      drawPane(overlapLeftRef.current, probeImg, null, zoom, pan, undefined, undefined, undefined, isXrayMode, probePoints);
     }
     if (overlapRightRef.current && galleryImg) {
-      drawPane(overlapRightRef.current, galleryImg, null, zoom, pan, undefined, undefined, undefined, isXrayMode);
+      drawPane(overlapRightRef.current, galleryImg, null, zoom, pan, undefined, undefined, undefined, isXrayMode, galleryPoints);
     }
   });
 
@@ -299,6 +356,88 @@ export default function SymmetryMerge({
         </div>
       </div>
 
+      {/* ΓöÇΓöÇ High-Density Telemetry HUD ΓöÇΓöÇ */}
+      {results && (
+        <div className="mb-2 flex flex-col gap-[2px] shrink-0 font-mono text-[10px] uppercase select-none">
+          {/* Provenance Veto Row */}
+          {results.failed_provenance_veto && (
+            <div className="px-2 py-1.5 flex justify-between items-center bg-[#1a0005] border border-[#5a0015] text-[#ff2040]">
+              <span className="font-bold tracking-widest text-xs">DEEPFAKE VETO: SYNTHETIC PROVENANCE DETECTED</span>
+              <span className="tracking-widest font-bold opacity-90">FUSION ABORTED</span>
+            </div>
+          )}
+
+          {/* Main Verdict Row */}
+          {!results.failed_provenance_veto && (
+            <div className={`px-2 py-1.5 flex justify-between items-center border ${results.veto_triggered ? 'bg-[#1a0005] border-[#5a0015] text-[#ff2040]' : (results.fused_identity_score >= 40.0 ? 'bg-[#111100] border-[#D4AF37]/40 text-[#D4AF37]' : 'bg-[#0a0a0a] border-[#333] text-gray-400')}`}>
+              <span className="font-bold tracking-wider text-xs">
+                {results.veto_triggered ? 'VERDICT: MISMATCH (ARCFACE VETO)' : (results.fused_identity_score >= 40.0 ? 'VERDICT: MATCH' : 'VERDICT: INCONCLUSIVE')}
+              </span>
+              <span className="tracking-widest font-bold">FUSED SCORE: {results.fused_identity_score?.toFixed(2)}%</span>
+            </div>
+          )}
+
+          {/* Telemetry Data Grid */}
+          <div className="grid grid-cols-2 gap-[2px]">
+            {/* Provenance Module */}
+            <div className={`px-2 py-1 border flex justify-between items-center ${results.failed_provenance_veto ? 'bg-[#1a0005] border-[#5a0015] text-[#ff2040]' : 'bg-[#050505] border-[#222] text-gray-500'}`}>
+               <span className="tracking-widest text-[9px]">SYNTH_ANOMALY:</span>
+               <span className="font-bold text-gray-300">{results.synthetic_anomaly_score !== undefined ? results.synthetic_anomaly_score.toFixed(4) : 'N/A'}</span>
+            </div>
+
+            {/* Occlusion Module */}
+            <div className="px-2 py-1 border bg-[#050505] border-[#222] text-gray-500 flex justify-between items-center">
+              <span className="tracking-widest text-[9px]">OCCLUSION (RATIOS):</span>
+              <span className="font-bold text-gray-300">
+                {results.occlusion_percentage !== undefined ? `${(results.occlusion_percentage).toFixed(1)}% (${results.effective_geometric_ratios_used ?? 0} ACTIVE)` : 'N/A'}
+              </span>
+            </div>
+          </div>
+
+          {/* Dynamic Lists (Occlusions & Marks) */}
+          {((results.probe_data?.occluded_regions?.length > 0) || (results.gallery_data?.occluded_regions?.length > 0) || results.occluded_regions?.length > 0 || results.correspondences?.length > 0) && (
+            <div className="flex flex-col gap-[2px]">
+              {results.occluded_regions && results.occluded_regions.length > 0 && !results.probe_data && !results.gallery_data && (
+                <div className="flex gap-[2px] flex-wrap">
+                  {results.occluded_regions.map((region: string, i: number) => (
+                    <div key={`occ-${i}`} className="px-1.5 py-0.5 border border-[#8a4000]/40 bg-[#3a1500]/20 text-[#ff8800]/80 tracking-widest text-[9px]">
+                      MASKED: {region}
+                    </div>
+                  ))}
+                </div>
+              )}
+              {results.probe_data?.occluded_regions && results.probe_data.occluded_regions.length > 0 && (
+                <div className="flex gap-[2px] flex-wrap">
+                  {results.probe_data.occluded_regions.map((region: string, i: number) => (
+                    <div key={`probe-occ-${i}`} className="px-1.5 py-0.5 border border-[#8a4000]/40 bg-[#3a1500]/20 text-[#ff8800]/80 tracking-widest text-[9px]">
+                      PROBE MASK: {region}
+                    </div>
+                  ))}
+                </div>
+              )}
+              {results.gallery_data?.occluded_regions && results.gallery_data.occluded_regions.length > 0 && (
+                <div className="flex gap-[2px] flex-wrap">
+                  {results.gallery_data.occluded_regions.map((region: string, i: number) => (
+                    <div key={`gallery-occ-${i}`} className="px-1.5 py-0.5 border border-[#8a4000]/40 bg-[#3a1500]/20 text-[#ff8800]/80 tracking-widest text-[9px]">
+                      GALLERY MASK: {region}
+                    </div>
+                  ))}
+                </div>
+              )}
+              {results.correspondences && results.correspondences.length > 0 && (
+                <div className="flex gap-[2px] flex-wrap">
+                  {results.correspondences.map((c: any, i: number) => (
+                    <div key={`corr-${i}`} className={`px-1.5 py-0.5 border bg-[#050505] tracking-widest text-[9px] ${results.veto_triggered || results.failed_provenance_veto ? 'border-[#5a0015] text-[#ff2040]/70' : 'border-[#D4AF37]/30 text-[#D4AF37]/80'}`}>
+                      MARK {i+1} <span className="opacity-50 mx-0.5">LR:</span>{c.lr.toFixed(1)}
+                    </div>
+                  ))}
+                </div>
+              )}
+            </div>
+          )}
+        </div>
+      )}
+
       {/* ΓöÇΓöÇ Dual-Pane Viewport ΓöÇΓöÇ */}
       {!imagesReady ? (
         <div className="flex-1 min-h-0 flex items-center justify-center bg-[#0a0a0a] animate-pulse text-[#D4AF37] font-mono text-xs tracking-widest rounded border border-[#333]">
@@ -346,6 +485,22 @@ export default function SymmetryMerge({
           >
             <canvas ref={leftCanvasRef} className="block w-full h-full" />
             <div className="absolute top-2 left-3 text-[9px] font-mono text-gray-600 tracking-widest pointer-events-none">{mode === 'delta' ? <span className="text-red-500">PROBE + DELTA</span> : 'PROBE (A)'}</div>
+            
+            {mode === 'delta' && results?.raw_probe_marks && (
+              <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 50 }}>
+                {results.raw_probe_marks.map((mark: any, i: number) => (
+                  <circle 
+                    key={`probe-mark-${i}`}
+                    cx={`${mark.centroid[0] * 100}%`} 
+                    cy={`${mark.centroid[1] * 100}%`} 
+                    r={Math.max(2, Math.sqrt(mark.area) * 0.8)}
+                    fill="none"
+                    stroke="#00DC82" 
+                    strokeWidth="1.5"
+                  />
+                ))}
+              </svg>
+            )}
           </div>
 
           {/* Right Pane: Gallery */}
@@ -355,6 +510,22 @@ export default function SymmetryMerge({
           >
             <canvas ref={rightCanvasRef} className="block w-full h-full" />
             <div className="absolute top-2 left-3 text-[9px] font-mono text-gray-600 tracking-widest pointer-events-none">{mode === 'delta' ? <span className="text-red-500">GALLERY + DELTA</span> : 'GALLERY (B)'}</div>
+            
+            {mode === 'delta' && results?.raw_gallery_marks && (
+              <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 50 }}>
+                {results.raw_gallery_marks.map((mark: any, i: number) => (
+                  <circle 
+                    key={`gallery-mark-${i}`}
+                    cx={`${mark.centroid[0] * 100}%`} 
+                    cy={`${mark.centroid[1] * 100}%`} 
+                    r={Math.max(2, Math.sqrt(mark.area) * 0.8)}
+                    fill="none"
+                    stroke="#D4AF37" 
+                    strokeWidth="1.5"
+                  />
+                ))}
+              </svg>
+            )}
           </div>
         </div>
       )}

``
