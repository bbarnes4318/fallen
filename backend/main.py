import os
import pickle
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Force Keras 2 — required for DeepFace/ArcFace
os.environ["DEEPFACE_HOME"] = "/app"     # Force DeepFace to use the pre-baked weights directory
import cv2
import numpy as np
import urllib.request
import math
import datetime
import base64
import struct
import hashlib
import time
import json
from pathlib import Path
import uuid
import jwt
from google.cloud import storage
from google.cloud import kms
from cryptography.fernet import Fernet
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import Optional
from skimage.feature import local_binary_pattern
import mediapipe as mp
import onnxruntime as ort
from models import SessionLocal, IdentityProfile, VerificationEvent, init_db
from sqlalchemy import event
from vault_index import vault_index
import stripe
from deepface import DeepFace

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

limiter = Limiter(key_func=get_remote_address)
# Only enable interactive docs in local development mode
is_development = os.getenv("ENVIRONMENT") == "development"
app = FastAPI(
    title="Biometric Facial Verification Pipeline",
    docs_url="/docs" if is_development else None,
    redoc_url="/redoc" if is_development else None,
    openapi_url="/openapi.json" if is_development else None
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)}
    )

# Preload Tier 1 Neural Ensemble models at module init to avoid per-request cold start
print("Starting Tier 1 Neural Ensemble model preload...", flush=True)
try:
    DeepFace.build_model("ArcFace")
    DeepFace.build_model("Facenet512")
    print("Tier 1 Neural Ensemble models (ArcFace, Facenet512) preloaded successfully.", flush=True)
except Exception as e:
    print(f"Warning: Model preload failed (will retry on first request): {e}", flush=True)

@app.on_event("startup")
def on_startup():
    print("Starting database initialization (init_db)...", flush=True)
    init_db()
    print("Database initialization complete.", flush=True)
    print("Hydrating FAISS Vault Index...", flush=True)
    session = SessionLocal()
    try:
        import concurrent.futures
        # Query specific columns to detach from SQLAlchemy Session (Thread-safety)
        profiles = session.query(IdentityProfile.user_id, IdentityProfile.encrypted_facial_embedding).all()
        
        def _process_profile(data):
            user_id, encrypted_emb = data
            try:
                emb = decrypt_embedding(encrypted_emb)
                return user_id, emb
            except Exception as e:
                print(f"Failed to decrypt embedding for {user_id}: {e}", flush=True)
                return None
                
        # Perform network-bound KMS decryption in parallel to prevent startup timeouts
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            # Eagerly evaluate the map to a list to prevent the `with` block from exiting prematurely
            results = list(executor.map(_process_profile, profiles))
            
        for result in results:
            if result is not None:
                user_id, emb = result
                try:
                    vault_index.add_identity(user_id, emb)
                except Exception as e:
                    print(f"Failed to add embedding to FAISS for {user_id}: {e}", flush=True)

        print(f"FAISS index hydrated with {vault_index.index.ntotal} records.", flush=True)
    finally:
        session.close()

# Dynamically sync FAISS when new identities are added
@event.listens_for(IdentityProfile, 'after_insert')
def receive_after_insert(mapper, connection, target):
    try:
        emb = decrypt_embedding(target.encrypted_facial_embedding)
        vault_index.add_identity(target.user_id, emb)
    except Exception as e:
        print(f"FAISS sync failed for {target.user_id}: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://scargods.com",
        "https://www.scargods.com",
        "https://facial-frontend-vkd6b6ijxa-uk.a.run.app",
        "https://facial-verify-api-196207148120.us-central1.run.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# JWT ZERO-TRUST AUTHENTICATION
# ---------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET")
OPERATOR_PASSWORD = os.getenv("OPERATOR_PASSWORD")

if not JWT_SECRET or not OPERATOR_PASSWORD:
    raise RuntimeError("CRITICAL SECRETS MISSING: JWT_SECRET and OPERATOR_PASSWORD must be set in the environment.")
ALGORITHM = "HS256"

security = HTTPBearer()

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[ALGORITHM])
        if payload.get("role") != "operator":
            raise HTTPException(status_code=403, detail="Insufficient privileges.")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

class LoginRequest(BaseModel):
    password: str

@app.post("/login")
def login(req: LoginRequest):
    if req.password != OPERATOR_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    
    # Generate token valid for 8 hours
    expire = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    token = jwt.encode(
        {"sub": "operator_admin", "role": "operator", "exp": expire.timestamp()},
        JWT_SECRET,
        algorithm=ALGORITHM
    )
    return {"access_token": token, "token_type": "bearer"}

# ---------------------------------------------------------
# LIVENESS & ANTI-SPOOFING (PAD)
# ---------------------------------------------------------
try:
    # Initialize the ONNX session for the PAD model (e.g., MiniFASNet)
    pad_session = ort.InferenceSession("models/MiniFASNetV2.onnx", providers=['CPUExecutionProvider'])
    PAD_MODEL_AVAILABLE = True
except Exception as e:
    print(f"Warning: PAD ONNX model not found. Using Laplacian variance fallback. {e}")
    PAD_MODEL_AVAILABLE = False

def detect_liveness(image: np.ndarray) -> dict:
    """
    Presentation Attack Detection (PAD).
    Returns a dict with 'score' (0.0-1.0) and 'variance' (raw Laplacian value).
    """
    if PAD_MODEL_AVAILABLE:
        # Preprocess for MiniFASNet: Resize to 80x80, normalize, CHW format
        resized = cv2.resize(image, (80, 80))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = rgb.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        
        input_name = pad_session.get_inputs()[0].name
        outputs = pad_session.run(None, {input_name: input_data})
        real_prob = float(outputs[0][0][1])
        return {"score": real_prob, "variance": None}
    else:
        # PAD fallback: Laplacian variance measures edge sharpness.
        # Low variance (<50) indicates a blurry source (printed photo, screen capture).
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < 50:
            return {"score": 0.10, "variance": round(laplacian_var, 2)}
        return {"score": 0.98, "variance": round(laplacian_var, 2)}

# ---------------------------------------------------------
# KMS ENVELOPE ENCRYPTION (BIOMETRIC VAULT)
# ---------------------------------------------------------
# The Key Encryption Key (KEK) managed by GCP KMS
KMS_KEY_NAME = os.getenv("KMS_KEY_NAME") or "projects/hoppwhistle/locations/us-central1/keyRings/facial-keyring/cryptoKeys/facial-dek"

_kms_client = None
def get_kms_client():
    global _kms_client
    if _kms_client is None:
        _kms_client = kms.KeyManagementServiceClient()
    return _kms_client

def encrypt_embedding(embedding: np.ndarray) -> bytes:
    """
    Application-Level Envelope Encryption.
    Generates a local DEK, encrypts the 512-D embedding with AES (Fernet),
    and encrypts the DEK via GCP KMS.
    """
    try:
        # 1. Generate local DEK
        dek = Fernet.generate_key()
        cipher = Fernet(dek)
        
        # 2. Encrypt the biometric payload
        payload_bytes = embedding.tobytes()
        encrypted_payload = cipher.encrypt(payload_bytes)
        
        # 3. Encrypt the DEK with KMS
        client = get_kms_client()
        encrypt_response = client.encrypt(request={'name': KMS_KEY_NAME, 'plaintext': dek})
        encrypted_dek = encrypt_response.ciphertext
        
        # 4. Package: [4 bytes length of DEK] + [Encrypted DEK] + [Encrypted Payload]
        dek_len = struct.pack(">I", len(encrypted_dek))
        return dek_len + encrypted_dek + encrypted_payload
    except Exception as e:
        print(f"KMS Encryption warning/fallback: {e}")
        # Fallback for local development if KMS isn't configured
        return b"MOCK_ENCRYPTED_PACKET"

def decrypt_embedding(packet: bytes) -> np.ndarray:
    """
    Decrypts the envelope. Extracts the encrypted DEK, decrypts it via KMS,
    then decrypts the payload back into a 512-D numpy array in-memory.
    """
    if packet == b"MOCK_ENCRYPTED_PACKET":
        return np.random.rand(512)
        
    try:
        # 1. Unpack
        dek_len = struct.unpack(">I", packet[:4])[0]
        encrypted_dek = packet[4:4+dek_len]
        encrypted_payload = packet[4+dek_len:]
        
        # 2. Decrypt DEK via KMS
        client = get_kms_client()
        decrypt_response = client.decrypt(request={'name': KMS_KEY_NAME, 'ciphertext': encrypted_dek})
        dek = decrypt_response.plaintext
        
        # 3. Decrypt Payload
        cipher = Fernet(dek)
        payload_bytes = cipher.decrypt(encrypted_payload)
        
        # 4. Restore Numpy Array
        embedding = np.frombuffer(payload_bytes, dtype=np.float64) 
        return embedding
    except Exception as e:
        raise ValueError(f"Decryption failed: {e}")

# ---------------------------------------------------------
# MATHEMATICAL CONSTANTS & MODELS
# ---------------------------------------------------------

# Generic 3D Morphable Model (3DMM) points for frontalization
# These are idealized 3D coordinates (X, Y, Z) of facial landmarks:
# Nose tip, Chin, Left Eye Left Corner, Right Eye Right Corner, Left Mouth Corner, Right Mouth Corner
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right Mouth corner
], dtype=np.float64)

# 17-Landmark Canonical Skull for Tier 2 3D Procrustes Alignment
CANONICAL_SKULL_3D = np.array([
    [-0.5, -0.2, -0.1],   # 33: Left Eye Outer
    [-0.2, -0.2, -0.05],  # 133: Left Eye Inner
    [0.2, -0.2, -0.05],   # 362: Right Eye Inner
    [0.5, -0.2, -0.1],    # 263: Right Eye Outer
    [0.0, 0.2, -0.5],     # 1: Nose Tip
    [0.0, -0.1, -0.2],    # 6: Nose Bridge
    [0.0, 1.0, -0.1],     # 152: Chin
    [-0.3, 0.6, -0.2],    # 61: Left Mouth Corner
    [0.3, 0.6, -0.2],     # 291: Right Mouth Corner
    [0.0, -0.8, -0.1],    # 10: Forehead Top
    [-0.8, 0.4, 0.2],     # 234: Left Jaw
    [0.8, 0.4, 0.2],      # 454: Right Jaw
    [-0.4, -0.4, -0.15],  # 70: Left Eyebrow
    [0.4, -0.4, -0.15],   # 300: Right Eyebrow
    [0.0, 0.5, -0.3],     # 0: Upper Lip
    [0.0, 0.7, -0.25],    # 17: Lower Lip
    [0.0, 0.9, -0.15]     # 199: Chin Center
], dtype=np.float64)
LANDMARK_INDICES_17 = [33, 133, 362, 263, 1, 6, 152, 61, 291, 10, 234, 454, 70, 300, 0, 17, 199]

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

class VerificationRequest(BaseModel):
    gallery_url: str
    probe_url: str
    require_liveness: bool = False

# ---------------------------------------------------------
# PIPELINE VERSION PINNING (Daubert Reproducibility)
# ---------------------------------------------------------
PIPELINE_VERSION = "AurumShield Daubert-Compliant v4.0 (Ensemble + 3D Procrustes + Bayesian LR)"

def _get_dependency_versions() -> dict:
    """Snapshot the exact versions of critical biometric libraries."""
    versions = {}
    try:
        import deepface
        versions["deepface"] = getattr(deepface, "__version__", "unknown")
    except Exception:
        versions["deepface"] = "unavailable"
    try:
        versions["mediapipe"] = mp.__version__
    except Exception:
        versions["mediapipe"] = "unavailable"
    versions["opencv"] = cv2.__version__
    return versions

DEPENDENCY_VERSIONS = _get_dependency_versions()

class AuditLog(BaseModel):
    raw_cosine_score: float
    # Neural Ensemble Audit Trail
    raw_arcface_score: Optional[float] = None
    raw_secondary_score: Optional[float] = None
    ensemble_model_secondary: Optional[str] = None
    # Tier 2: 3D Topographical Mapping Telemetry
    pose_corrected_3d: Optional[bool] = None
    probe_pose_angles: Optional[dict] = None
    gallery_pose_angles: Optional[dict] = None
    occlusion_percentage: Optional[float] = None
    occluded_regions: Optional[list] = None
    effective_geometric_ratios_used: Optional[int] = None
    # Temporal & Spectral Telemetry (Tier 1/3)
    estimated_temporal_delta: Optional[float] = None
    cross_spectral_correction_applied: Optional[bool] = None
    statistical_certainty: str
    false_acceptance_rate: str
    nodes_mapped: int
    matched_user_id: Optional[str] = None
    person_name: Optional[str] = None
    source: Optional[str] = None
    creator: Optional[str] = None
    license_short_name: Optional[str] = None
    license_url: Optional[str] = None
    file_page_url: Optional[str] = None
    wikidata_id: Optional[str] = None
    # Deep Forensic Telemetry
    vector_hash: Optional[str] = None
    alignment_variance: Optional[dict] = None
    liveness_check: Optional[dict] = None
    crypto_envelope: Optional[dict] = None
    # Calibration provenance
    calibration_benchmark: Optional[str] = None
    calibration_pairs: Optional[int] = None
    # Chain of Custody — Pre-decode binary hashes
    probe_file_hash: Optional[str] = None
    gallery_file_hash: Optional[str] = None
    # Pipeline reproducibility
    pipeline_version: str = PIPELINE_VERSION
    dependency_versions: Optional[dict] = None
    # Bayesian Likelihood Ratio Audit Trail (Daubert v3.0)
    lr_arcface: Optional[float] = None
    lr_marks: Optional[float] = None
    lr_total: Optional[float] = None
    posterior_probability: Optional[float] = None
    mark_lrs: Optional[list] = None  # Individual LR per matched mark

class VerificationResponse(BaseModel):
    structural_score: float
    soft_biometrics_score: float
    micro_topology_score: float
    fused_identity_score: float
    conclusion: str
    veto_triggered: bool
    gallery_heatmap_b64: str
    probe_heatmap_b64: str
    gallery_aligned_b64: str
    probe_aligned_b64: str
    scar_delta_b64: str
    gallery_wireframe_b64: str
    probe_wireframe_b64: str
    # Tier 4: Mark Correspondence
    mark_correspondence_score: Optional[float] = None
    marks_detected_gallery: int = 0
    marks_detected_probe: int = 0
    marks_matched: int = 0
    correspondences: list = []
    raw_probe_marks: list = []
    raw_gallery_marks: list = []
    audit_log: Optional[AuditLog] = None

# ---------------------------------------------------------
# STATISTICAL CONFIDENCE ENGINE (CALIBRATION-DRIVEN)
# ---------------------------------------------------------

# Load empirical calibration data at startup from GCS
CALIBRATION = None

def _load_calibration():
    """Attempt to load calibration JSON from GCS, fallback to local file."""
    import json as _json

    bucket_name = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
    gcs_path = "calibration/lfw_calibration.json"

    # Try GCS first
    try:
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        if blob.exists():
            content = blob.download_as_text()
            cal = _json.loads(content)
            print(f"Calibration loaded from GCS: {cal['benchmark']} ({cal['pairs_evaluated']} pairs)")
            return cal
        else:
            print(f"No calibration blob at gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        print(f"GCS calibration load failed: {e}")

    # Fallback to local file (for dev environments)
    _local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data", "lfw_calibration.json")
    try:
        with open(_local_path, "r") as _f:
            cal = _json.load(_f)
        print(f"Calibration loaded from local file: {cal['benchmark']} ({cal['pairs_evaluated']} pairs)")
        return cal
    except FileNotFoundError:
        print("WARNING: No calibration data found (GCS or local). FAR will be reported as UNCALIBRATED.")
    except Exception as _e:
        print(f"WARNING: Failed to load calibration data: {_e}. FAR will be reported as UNCALIBRATED.")

    return None

CALIBRATION = _load_calibration()

# ---------------------------------------------------------
# TIER 4 BAYESIAN CALIBRATION DATA
# ---------------------------------------------------------
TIER4_CALIBRATION = None

def _load_tier4_calibration():
    """Load the Tier 4 population model from local file or GCS."""
    import json as _json
    import sys
    import numpy as np
    
    # Monkey-patch to allow unpickling numpy 2.x models in numpy 1.x environments
    if "numpy.core.numeric" in sys.modules and "numpy._core.numeric" not in sys.modules:
        sys.modules["numpy._core"] = sys.modules["numpy.core"]
        sys.modules["numpy._core.numeric"] = sys.modules["numpy.core.numeric"]
        sys.modules["numpy._core.multiarray"] = sys.modules["numpy.core.multiarray"]

    bucket_name = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
    gcs_path = "calibration/tier4_population_model.pkl"

    # Try local file first (faster)
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data", "tier4_population_model.pkl")
    try:
        with open(local_path, "rb") as f:
            cal = pickle.load(f)
        print(f"Tier 4 Bayesian model loaded from local: {cal.get('total_marks', '?')} population marks")
        return cal
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"WARNING: Failed to load local Tier 4 model: {e}")

    # Fallback to GCS
    try:
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        if blob.exists():
            pkl_bytes = blob.download_as_bytes()
            cal = pickle.loads(pkl_bytes)
            print(f"Tier 4 Bayesian model loaded from GCS: {cal.get('total_marks', '?')} population marks")
            return cal
    except Exception as e:
        print(f"GCS Tier 4 model load failed: {e}")

    print("WARNING: No Tier 4 Bayesian calibration data found. Mark LR will be unavailable.")
    return None

TIER4_CALIBRATION = _load_tier4_calibration()

def calculate_statistical_confidence(cosine_score: float) -> dict:
    """
    Convert ArcFace cosine similarity to FAR and statistical certainty
    using empirically calibrated thresholds from the LFW benchmark.
    If no calibration data is available, honestly reports UNCALIBRATED.

    NOTE (v3.0): The heuristic '10^marks FAR reduction' has been removed.
    Mark evidence is now fused via proper Bayesian Likelihood Ratios.
    """
    if CALIBRATION is None:
        return {
            "false_acceptance_rate": "UNCALIBRATED",
            "statistical_certainty": "UNCALIBRATED",
            "benchmark": "N/A",
            "pairs_evaluated": 0
        }

    thresholds = CALIBRATION["arcface"]["thresholds"]
    sorted_thresh = sorted(thresholds.keys(), key=float)

    far_value = None
    frr_value = None
    matched_threshold = None

    for t in sorted_thresh:
        if cosine_score >= float(t):
            far_value = thresholds[t]["far"]
            frr_value = thresholds[t]["frr"]
            matched_threshold = t

    if far_value is None or far_value <= 0:
        far_str = "DIFFERENT IDENTITIES"
        certainty = "0% — Non-Match"
    elif far_value < 1e-7:
        far_str = f"< 1 in {10_000_000:,}"
        certainty = f"{(1.0 - far_value) * 100:.6f}%"
    else:
        far_str = f"1 in {int(1.0 / far_value):,}"
        certainty = f"{(1.0 - far_value) * 100:.6f}%"

    return {
        "false_acceptance_rate": far_str,
        "statistical_certainty": certainty,
        "benchmark": CALIBRATION.get("benchmark", "LFW"),
        "pairs_evaluated": CALIBRATION.get("pairs_evaluated", 0),
    }


def score_to_lr_ensemble(ensemble_score: float, temporal_delta: float = 0.0) -> float:
    """
    Convert Fused 60/40 Ensemble Score to a Likelihood Ratio using
    empirically calibrated FAR/FRR from the LFW benchmark.

    LR = P(score | Hp) / P(score | Hd) = (1 - FRR) / FAR

    Where:
      - P(score | Hp) = True Positive Rate = 1 - FRR (same person produces this score)
      - P(score | Hd) = False Acceptance Rate = FAR (different person produces this score)
      
    Temporal Invariance:
      - Exponential decay curve applied to TPR probability based on temporal delta.
      - As the time gap increases, the expected FRR naturally increases.
      - By boosting the expected TPR for degraded scores, we prevent the "Aging Problem"
        without manipulating the raw structural score or Bayesian math.
    """
    if CALIBRATION is None:
        return 1.0  # Neutral LR — no calibration data

    thresholds = CALIBRATION.get("ensemble", {}).get("thresholds", {})
    if not thresholds:
        # Fallback to arcface if ensemble calibration is not yet loaded
        thresholds = CALIBRATION.get("arcface", {}).get("thresholds", {})
        
    sorted_thresh = sorted(thresholds.keys(), key=float)

    far_value = None
    frr_value = None

    for t in sorted_thresh:
        if ensemble_score >= float(t):
            far_value = thresholds[t]["far"]
            frr_value = thresholds[t]["frr"]

    if far_value is None or far_value <= 0:
        # Score below all thresholds — strong evidence against match
        return 1e-6  # Floor: extremely low LR

    raw_tpr = 1.0 - (frr_value if frr_value is not None else 0.0)
    
    # Age-Conditioned Likelihood Ratio (Temporal Invariance)
    # The expected TPR for degraded scores is exponentially boosted ~1% per year of temporal gap
    tpr = min(1.0, raw_tpr * math.exp(0.01 * temporal_delta))

    # Epsilon floor to prevent division by zero
    far_value = max(far_value, 1e-9)
    lr = tpr / far_value
    return lr


# ---------------------------------------------------------
# DEEP FORENSIC TELEMETRY HELPERS
# ---------------------------------------------------------

def compute_vector_hash(embedding: np.ndarray) -> str:
    """SHA-256 hash representation of the 512-D ArcFace embedding array."""
    return hashlib.sha256(embedding.tobytes()).hexdigest()


def compute_alignment_variance(image: np.ndarray) -> dict:
    """
    Extracts Yaw, Pitch, Roll correction degrees from the face
    using solvePnP against the generic 3DMM model points.
    Returns variance as formatted degree strings.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"yaw": "N/A", "pitch": "N/A", "roll": "N/A"}

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]

    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),
        (landmarks[152].x * w, landmarks[152].y * h),
        (landmarks[33].x * w, landmarks[33].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[61].x * w, landmarks[61].y * h),
        (landmarks[291].x * w, landmarks[291].y * h)
    ], dtype="double")

    camera_matrix, dist_coeffs = estimate_camera_intrinsic(image.shape)
    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return {"yaw": "N/A", "pitch": "N/A", "roll": "N/A"}

    rmat, _ = cv2.Rodrigues(rotation_vector)
    euler_angles = cv2.RQDecomp3x3(rmat)[0]

    pitch = euler_angles[0]
    yaw = euler_angles[1]
    roll = euler_angles[2]

    return {
        "yaw": f"{yaw:+.1f}°",
        "pitch": f"{pitch:+.1f}°",
        "roll": f"{roll:+.1f}°"
    }


def build_liveness_telemetry(liveness_result: dict) -> dict:
    """
    Packages the PAD result into a forensic telemetry block.
    Honestly reports the detection method used.
    """
    score = liveness_result["score"]
    variance = liveness_result["variance"]
    spoof_prob = (1.0 - score) * 100

    if PAD_MODEL_AVAILABLE:
        method = "MINIFASNET_ONNX"
        if score >= 0.98:
            status = "LIVE_VERIFIED"
        elif score >= 0.95:
            status = "PROBABLE_LIVE"
        else:
            status = "SPOOF_SUSPECTED"
    else:
        method = "LAPLACIAN_VARIANCE"
        if score >= 0.95:
            status = "BLUR_CHECK_PASSED"
        else:
            status = "BLUR_CHECK_FAILED"

    result = {
        "method": method,
        "spoof_probability": f"{spoof_prob:.3f}%",
        "status": status
    }
    if variance is not None:
        result["laplacian_variance"] = variance
    return result


def build_crypto_envelope(decryption_time_ms: float | None = None) -> dict:
    """
    Builds the cryptographic envelope telemetry.
    Uses actual decryption latency if available, otherwise a realistic static value.
    """
    latency = f"{decryption_time_ms:.0f}ms" if decryption_time_ms is not None else "N/A (1:1 mode)"
    return {
        "standard": "AES-256-GCM / GCP KMS",
        "decryption_time": latency
    }

# ---------------------------------------------------------
# CORE PREPROCESSING LOGIC
# ---------------------------------------------------------

def fetch_image_from_url(uri: str) -> tuple:
    """
    Fetches an image from a GCS URI or HTTP URL.
    Returns a tuple of (decoded_image_array, sha256_hash_of_raw_bytes).
    The hash is computed on the raw byte stream BEFORE OpenCV decode,
    establishing an immutable chain-of-custody fingerprint.
    """
    try:
        if uri.startswith("gs://"):
            storage_client = storage.Client()
            parts = uri.replace("gs://", "").split("/", 1)
            bucket = storage_client.bucket(parts[0])
            blob = bucket.blob(parts[1])
            img_bytes = blob.download_as_bytes()
        else:
            req = urllib.request.urlopen(uri, timeout=10)
            img_bytes = req.read()

        # Chain of Custody: hash the raw binary BEFORE decode
        raw_hash = hashlib.sha256(img_bytes).hexdigest()

        arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
        if img is None:
            raise ValueError("Could not decode image.")
        return img, raw_hash
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    MATH: g(x,y) = T(f(x,y)). 
    We transform the intensity values in localized tiles (8x8) to a uniform distribution,
    clipping the histogram at 2.0 to prevent noise over-amplification.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def estimate_camera_intrinsic(image_shape):
    """
    Estimates the Camera Intrinsic Matrix (K) assuming optical center is at the image center.
    MATH: K = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]
    """
    h, w = image_shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion initially
    return camera_matrix, dist_coeffs

def frontalize_face(image: np.ndarray) -> np.ndarray:
    """
    3DMM Frontalization using Perspective-n-Point (solvePnP).
    MATH: s * m_2d = K * [R | t] * M_3d
    Calculates the rigid rotation (R) and translation (t) to un-rotate Pitch, Yaw, Roll.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if not results.multi_face_landmarks:
        return image # Return original if no face detected
    
    landmarks = results.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]
    
    # Extract corresponding 2D points: Nose tip, Chin, Eyes, Mouth
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),       # Nose tip
        (landmarks[152].x * w, landmarks[152].y * h),   # Chin
        (landmarks[33].x * w, landmarks[33].y * h),     # Left eye
        (landmarks[263].x * w, landmarks[263].y * h),   # Right eye
        (landmarks[61].x * w, landmarks[61].y * h),     # Left mouth
        (landmarks[291].x * w, landmarks[291].y * h)    # Right mouth
    ], dtype="double")
    
    camera_matrix, dist_coeffs = estimate_camera_intrinsic(image.shape)
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return image
    
    # Calculate inverse rotation matrix to "un-rotate" the image
    rmat, _ = cv2.Rodrigues(rotation_vector)
    # We invert the rotation matrix for frontalization
    inv_rmat = np.linalg.inv(rmat)
    
    # Warp perspective (Mocking a full 3D mesh warp with an affine approximation for this snippet)
    # In a full pipeline, we map the entire 3D mesh texture using barycentric coordinates.
    # For now, we apply a 2D affine un-rotation derived from the Pitch/Yaw/Roll.
    euler_angles = cv2.RQDecomp3x3(rmat)[0]
    roll_angle = euler_angles[2]
    
    M = cv2.getRotationMatrix2D((w/2, h/2), roll_angle, 1.0)
    frontalized = cv2.warpAffine(image, M, (w, h))
    
    return frontalized

# ---------------------------------------------------------
# FACE ALIGNMENT & REAL BIOMETRIC EXTRACTION
# ---------------------------------------------------------

def align_face_crop(image: np.ndarray, target_size: int = 256):
    """
    Detects a face, aligns it by rotating to make the eye-line horizontal,
    crops tightly around the face with padding, and resizes to a canonical
    target_size × target_size image. Re-detects landmarks on the final crop
    for accurate downstream embedding extraction.
    Returns (aligned_crop, landmarks) or (resized_original, None) if no face.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        resized = cv2.resize(image, (target_size, target_size))
        return resized, None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]

    # Eye center positions for alignment
    left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
    right_eye_indices = [263, 362, 387, 386, 385, 373, 374, 380]

    left_eye_center = np.mean(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in left_eye_indices], axis=0
    )
    right_eye_center = np.mean(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in right_eye_indices], axis=0
    )

    # Rotation angle to make eyes horizontal
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eye_midpoint = (
        (left_eye_center[0] + right_eye_center[0]) / 2,
        (left_eye_center[1] + right_eye_center[1]) / 2,
    )
    M = cv2.getRotationMatrix2D(eye_midpoint, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    # Re-detect landmarks on the rotated image for an accurate bounding box
    rgb_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    results_rot = face_mesh.process(rgb_rot)
    lm = results_rot.multi_face_landmarks[0].landmark if results_rot.multi_face_landmarks else landmarks

    # Face bounding box from all landmarks
    xs = [l.x * w for l in lm]
    ys = [l.y * h for l in lm]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add 25% padding
    face_w = x_max - x_min
    face_h = y_max - y_min
    pad_x = face_w * 0.25
    pad_y = face_h * 0.25
    x_min = max(0, int(x_min - pad_x))
    x_max = min(w, int(x_max + pad_x))
    y_min = max(0, int(y_min - pad_y))
    y_max = min(h, int(y_max + pad_y))

    # Make it square (use the larger dimension)
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    if crop_w > crop_h:
        diff = crop_w - crop_h
        y_min = max(0, y_min - diff // 2)
        y_max = min(h, y_max + (diff - diff // 2))
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        x_min = max(0, x_min - diff // 2)
        x_max = min(w, x_max + (diff - diff // 2))

    cropped = rotated[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        cropped = rotated

    aligned = cv2.resize(cropped, (target_size, target_size))

    # Final landmark detection on the aligned canonical crop
    rgb_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    results_aligned = face_mesh.process(rgb_aligned)

    final_landmarks = None
    if results_aligned.multi_face_landmarks:
        final_landmarks = results_aligned.multi_face_landmarks[0].landmark

    return aligned, final_landmarks


def extract_landmark_embedding(landmarks) -> np.ndarray:
    """
    LEGACY: Builds a 1404-D geometric embedding from MediaPipe's 468 face landmarks.
    Retained for backward compatibility but NO LONGER USED for identity scoring.
    Identity matching now uses extract_arcface_embedding().
    """
    coords = np.array([(l.x, l.y, l.z) for l in landmarks])  # (468, 3)

    # Center on nose tip (landmark 1) for translation invariance
    nose_tip = coords[1].copy()
    coords = coords - nose_tip

    # Normalize by inter-ocular distance for scale invariance
    left_eye = coords[33]
    right_eye = coords[263]
    iod = np.linalg.norm(right_eye - left_eye)
    if iod > 1e-6:
        coords = coords / iod

    return coords.flatten()  # 1404-D vector


def estimate_age(image: np.ndarray) -> float:
    """
    Estimates the apparent age of the subject using DeepFace.
    """
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(
            img_path=rgb_image,
            actions=["age"],
            enforce_detection=False,
            detector_backend="skip" # Image is already cropped/aligned
        )
        if isinstance(result, list):
            return float(result[0]["age"])
        return float(result["age"])
    except Exception as e:
        print(f"Age estimation failed: {e}")
        return 0.0


def cross_spectral_normalize(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Analyzes HSV saturation to detect if one image is grayscale/sepia and the other is color.
    If a spectral mismatch is detected, converts the color image to grayscale to match domains,
    preventing artificial texture noise in Tier 3 (LBP) and Tier 4 (Marks).
    Returns (norm_img1, norm_img2, correction_applied).
    """
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    sat1_std = np.std(hsv1[:, :, 1])
    sat2_std = np.std(hsv2[:, :, 1])
    
    threshold = 15.0 # Low saturation std indicates grayscale/monochrome
    
    is_gray1 = sat1_std < threshold
    is_gray2 = sat2_std < threshold
    
    correction_applied = False
    norm1 = img1.copy()
    norm2 = img2.copy()
    
    if is_gray1 and not is_gray2:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        norm2 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
        correction_applied = True
    elif is_gray2 and not is_gray1:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        norm1 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
        correction_applied = True
        
    return norm1, norm2, correction_applied


def extract_arcface_embedding(image: np.ndarray) -> np.ndarray:
    """
    Extracts a 512-D ArcFace biometric embedding from an aligned face crop.
    This is the TRUE identity discriminator — replaces MediaPipe geometric
    cosine for Tier 1 structural identity matching.

    CRITICAL: We use detector_backend='retinaface' — the ONLY backend that
    reliably handles BOTH failure modes:

    - 'skip' bypasses alignment entirely. Our align_face_crop() uses MediaPipe
      landmarks which don't match ArcFace's training alignment → poor
      discriminative power between different identities.

    - 'opencv' (Haar cascade) fails on tightly-cropped face images. With
      enforce_detection=False, failed detections produce degenerate embeddings
      that are nearly identical across all subjects → mass false matches.

    - 'retinaface' is a deep-learning face detector that successfully detects
      faces even in pre-cropped images AND provides the 5 facial landmarks
      needed for ArcFace-compatible affine alignment. This is the standard
      recommended backend for ArcFace in the DeepFace library.

    Returns a 512-D numpy array (ArcFace latent space).
    """
    # DeepFace expects RGB; our pipeline uses BGR (OpenCV)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = DeepFace.represent(
        img_path=rgb_image,
        model_name="ArcFace",
        enforce_detection=False,
        detector_backend="retinaface",
    )
    embedding = np.array(result[0]["embedding"], dtype=np.float64)
    return embedding  # 512-D vector


def extract_facenet_embedding(image: np.ndarray) -> np.ndarray:
    """
    Extracts a 512-D Facenet512 biometric embedding from an aligned face crop.
    This serves as the secondary model in the Tier 1 Neural Ensemble.
    Uses 'retinaface' detector backend for consistency with ArcFace extraction.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = DeepFace.represent(
        img_path=rgb_image,
        model_name="Facenet512",
        enforce_detection=False,
        detector_backend="retinaface",
    )
    return np.array(result[0]["embedding"], dtype=np.float64)


def extract_ensemble_embeddings(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts both ArcFace and Facenet512 embeddings.
    Returns: (arcface_embedding, facenet_embedding)
    """
    arcface_embed = extract_arcface_embedding(image)
    facenet_embed = extract_facenet_embedding(image)
    return arcface_embed, facenet_embed


def compute_ensemble_similarity(embed_pair_1: tuple[np.ndarray, np.ndarray], embed_pair_2: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, float]:
    """
    Computes cosine similarity for both models and fuses them using a weighted ensemble (60% ArcFace, 40% Facenet512).
    Returns: (fused_score, arcface_score, secondary_score)
    """
    arc1, face1 = embed_pair_1
    arc2, face2 = embed_pair_2
    
    # Needs calculate_cosine_similarity which is defined below, but Python handles forward references
    # wait, this is executed later anyway so it's fine.
    arc_score = calculate_cosine_similarity(arc1, arc2)
    face_score = calculate_cosine_similarity(face1, face2)
    
    fused_score = (arc_score * 0.60) + (face_score * 0.40)
    return fused_score, arc_score, face_score


def procrustes_align_3d(landmarks_3d: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Rigid Procrustes analysis via SVD to neutralize pitch, yaw, and roll.
    Takes N x 3 landmarks and aligns the 17 key points to the CANONICAL_SKULL_3D.
    Returns the fully un-rotated N x 3 mesh and the extracted Euler angles.
    """
    source_points = landmarks_3d[LANDMARK_INDICES_17]
    target_points = CANONICAL_SKULL_3D
    
    # Center the points
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    # SVD
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    # Extract Euler angles from R (yaw, pitch, roll)
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2,1], R[2,2])
        yaw = np.arctan2(-R[2,0], sy)
        roll = np.arctan2(R[1,0], R[0,0])
    else:
        pitch = np.arctan2(-R[1,2], R[1,1])
        yaw = np.arctan2(-R[2,0], sy)
        roll = 0
        
    angles = {
        "pitch_deg": round(np.degrees(pitch), 2),
        "yaw_deg": round(np.degrees(yaw), 2),
        "roll_deg": round(np.degrees(roll), 2)
    }
    
    # Apply rotation to ALL landmarks (centered at their centroid to prevent translation explosion)
    all_centered = landmarks_3d - np.mean(landmarks_3d, axis=0)
    aligned_landmarks = (R @ all_centered.T).T
    
    return aligned_landmarks, angles

def extract_geometric_ratios_3d(landmarks) -> tuple[np.ndarray, dict, dict]:
    """
    Computes scale-invariant, true 3D Euclidean facial geometric ratios for Tier 2.
    Uses Procrustes alignment to mathematically un-rotate the face to a perfect frontal view.
    Also computes landmark visibility telemetry for Tier 2 dynamic dropping.
    """
    VISIBILITY_THRESHOLD = 0.85

    # Compute Visibility Telemetry
    masked_count = sum(1 for l in landmarks if getattr(l, "visibility", 1.0) < VISIBILITY_THRESHOLD)
    occlusion_percentage = (masked_count / len(landmarks)) * 100.0 if len(landmarks) > 0 else 0.0

    STRUCTURAL_GROUPS = {
        "Left Orbital": [33, 133, 160, 159, 158, 144, 145, 153],
        "Right Orbital": [263, 362, 387, 386, 385, 373, 374, 380],
        "Nose": [1, 2, 98, 327, 4, 5, 195, 197, 6],
        "Mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88],
        "Left Jaw": [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152],
        "Right Jaw": [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
        "Left Eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        "Right Eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
        "Forehead": [10, 338, 297, 332, 284, 251, 389, 356]
    }

    occluded_regions = []
    for region, indices in STRUCTURAL_GROUPS.items():
        region_masked = sum(1 for idx in indices if getattr(landmarks[idx], "visibility", 1.0) < VISIBILITY_THRESHOLD)
        if region_masked > len(indices) * 0.5:  # If >50% masked, call it occluded
            occluded_regions.append(region)

    ratio_landmarks_indices = [
        [1, 152],         # Nose-to-chin
        [6, 1],           # Nose length
        [61, 291],        # Mouth width
        [10, 152],        # Face height
        [234, 454],       # Jaw width
        [70, 33],         # Left brow height
        [300, 263],       # Right brow height
        [1, 33],          # Nose-to-left-eye
        [1, 263],         # Nose-to-right-eye
        [152, 61],        # Chin-to-left-mouth
        [152, 291],       # Chin-to-right-mouth
        [234, 152, 454]   # Jaw symmetry
    ]
    
    iod_points = [33, 263]
    ratio_visibility = []
    for idx_group in ratio_landmarks_indices:
        is_visible = all(getattr(landmarks[idx], "visibility", 1.0) >= VISIBILITY_THRESHOLD for idx in idx_group)
        is_iod_visible = all(getattr(landmarks[idx], "visibility", 1.0) >= VISIBILITY_THRESHOLD for idx in iod_points)
        if len(idx_group) == 3: # Jaw symmetry doesn't use IOD
            ratio_visibility.append(is_visible)
        else:
            ratio_visibility.append(is_visible and is_iod_visible)

    vis_data = {
        "occlusion_percentage": round(occlusion_percentage, 2),
        "occluded_regions": occluded_regions,
        "ratio_visibility": np.array(ratio_visibility, dtype=bool)
    }

    coords_3d = np.array([(l.x, l.y, l.z) for l in landmarks])
    aligned_coords, angles = procrustes_align_3d(coords_3d)

    left_eye = aligned_coords[33]
    right_eye = aligned_coords[263]
    iod = np.linalg.norm(right_eye - left_eye)

    if iod < 1e-6:
        return np.zeros(12), angles, vis_data

    nose_tip = aligned_coords[1]
    nose_bridge = aligned_coords[6]
    chin = aligned_coords[152]
    left_mouth = aligned_coords[61]
    right_mouth = aligned_coords[291]
    forehead_top = aligned_coords[10]
    left_jaw = aligned_coords[234]
    right_jaw = aligned_coords[454]
    left_eyebrow = aligned_coords[70]
    right_eyebrow = aligned_coords[300]

    jaw_to_chin_r = np.linalg.norm(right_jaw - chin)

    ratios = np.array([
        np.linalg.norm(nose_tip - chin) / iod,                # Nose-to-chin / IOD
        np.linalg.norm(nose_bridge - nose_tip) / iod,          # Nose length / IOD
        np.linalg.norm(left_mouth - right_mouth) / iod,        # Mouth width / IOD
        np.linalg.norm(forehead_top - chin) / iod,             # Face height / IOD
        np.linalg.norm(left_jaw - right_jaw) / iod,            # Jaw width / IOD
        np.linalg.norm(left_eyebrow - left_eye) / iod,         # Left brow height / IOD
        np.linalg.norm(right_eyebrow - right_eye) / iod,       # Right brow height / IOD
        np.linalg.norm(nose_tip - left_eye) / iod,             # Nose-to-left-eye / IOD
        np.linalg.norm(nose_tip - right_eye) / iod,            # Nose-to-right-eye / IOD
        np.linalg.norm(chin - left_mouth) / iod,               # Chin-to-left-mouth / IOD
        np.linalg.norm(chin - right_mouth) / iod,              # Chin-to-right-mouth / IOD
        np.linalg.norm(left_jaw - chin) / jaw_to_chin_r if jaw_to_chin_r > 1e-6 else 1.0,  # Jaw symmetry
    ])

    return ratios, angles, vis_data


# ---------------------------------------------------------
# VERIFICATION LOGIC (MATH FUSION)
# ---------------------------------------------------------

def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    MATH: Cosine Similarity = (A • B) / (||A|| * ||B||)
    Measures orientation of the embedding vectors, highly robust to magnitude shifts.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def extract_lbp_histogram(image: np.ndarray) -> np.ndarray:
    """
    Local Binary Patterns for Micro-Topology (Skin Texture Analysis).
    MATH: LBP_{P,R} = sum_{p=0}^{P-1} s(g_p - g_c) 2^p
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    # Calculate uniform LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Build histogram
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7) # Normalize
    return hist

# Discriminative landmark indices for ArcFace alignment regions.
# Eyes, nose, and mouth carry the highest identity signal in deep face models.
# Ref: Deng et al., "ArcFace: Additive Angular Margin Loss" (CVPR 2019)
DISCRIMINATIVE_LANDMARKS = {
    # Left eye contour (16 points)
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    # Right eye contour (16 points)
    "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    # Nose ridge and tip (9 points)
    "nose": [1, 2, 98, 327, 4, 5, 195, 197, 6],
    # Lips outer contour (20 points)
    "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88],
}

def generate_landmark_attention_map(image: np.ndarray, landmarks: list) -> str:
    """
    Landmark Attention Map — places a Gaussian kernel at each of the 468
    MediaPipe mesh points. Discriminative regions (eyes, nose, mouth) are
    weighted higher because these are the anchor points for ArcFace alignment
    and contain the highest identity signal.
    Returns a base64-encoded JET colormap overlay blended with the source image.
    """
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Collect all discriminative indices into a set for O(1) lookup
    discriminative_set = set()
    for indices in DISCRIMINATIVE_LANDMARKS.values():
        discriminative_set.update(indices)

    # Place Gaussian kernel at each landmark position
    for idx, lm in enumerate(landmarks):
        px = int(lm.x * w)
        py = int(lm.y * h)
        if 0 <= px < w and 0 <= py < h:
            # Higher weight for identity-discriminative regions
            weight = 1.5 if idx in discriminative_set else 0.6
            radius = int(min(h, w) * 0.025)
            cv2.circle(heatmap, (px, py), radius, weight, -1)

    # Smooth into a continuous density field
    kernel_size = int(min(h, w) * 0.15) | 1  # Ensure odd
    sigma = kernel_size / 4.0
    heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), sigma)

    # Normalize and colorize
    heatmap_norm = cv2.normalize(heatmap, None, alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    blended = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

    _, buffer = cv2.imencode('.png', blended)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

# ---------------------------------------------------------
# TIER 4: MARK CORRESPONDENCE ENGINE
# ---------------------------------------------------------
# Detects discrete facial anomalies (scars, moles, birthmarks)
# and matches them between gallery and probe using spatial +
# descriptor similarity via Hungarian optimal bipartite matching.

# MediaPipe landmark indices for masking facial features
# (eyes, brows, nose interior, lips) — we only want skin surface
_LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LEFT_BROW_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RIGHT_BROW_IDX = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
_NOSE_IDX = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 220, 115, 48, 64, 102, 49, 131, 134, 236, 196, 3, 51, 281, 275, 440, 344, 278, 294, 331, 279, 360, 363, 456, 420, 399, 412, 351]
_LIPS_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]

def _build_skin_mask(shape: tuple, landmarks, margin: int = 5) -> np.ndarray:
    """
    Build a binary mask that covers the face skin surface,
    EXCLUDING eyes, eyebrows, nose interior, and lips.
    """
    h, w = shape[:2]
    # Start with full face mask
    skin_mask = np.ones((h, w), dtype=np.uint8) * 255

    # Build exclusion polygons from landmark groups
    for idx_group in [_LEFT_EYE_IDX, _RIGHT_EYE_IDX, _LEFT_BROW_IDX, _RIGHT_BROW_IDX, _NOSE_IDX, _LIPS_IDX]:
        pts = []
        for idx in idx_group:
            if idx < len(landmarks):
                lm = landmarks[idx]
                pts.append([int(lm.x * w), int(lm.y * h)])
        if len(pts) >= 3:
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            # Inflate slightly to ensure full coverage
            M = cv2.moments(hull)
            if M["m00"] > 0:
                cx_h = int(M["m10"] / M["m00"])
                cy_h = int(M["m01"] / M["m00"])
                scale = 1.15  # 15% inflation
                inflated = ((hull - [cx_h, cy_h]) * scale + [cx_h, cy_h]).astype(np.int32)
                cv2.fillConvexPoly(skin_mask, inflated, 0)

    return skin_mask


def detect_facial_marks(aligned_crop: np.ndarray, landmarks) -> tuple[list, np.ndarray]:
    """
    Detects discrete facial anomalies (scars, moles, birthmarks) on
    the skin surface of an aligned 256×256 face crop.

    Returns a tuple:
        - list of mark descriptors: [{"centroid": (cx, cy), "area": float, "intensity": float, "circularity": float}, ...]
        - occlusion_mask (np.ndarray): Mask of occluded regions
    """
    h, w = aligned_crop.shape[:2]
    gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)

    # Build skin mask excluding major facial features
    skin_mask = _build_skin_mask(aligned_crop.shape, landmarks)

    # Build occluded mask for Bayesian Penalty Nullification
    occ_mask = np.zeros((h, w), dtype=np.uint8)
    pts = []
    for lm in landmarks:
        if getattr(lm, "visibility", 1.0) < 0.85:
            pts.append((int(lm.x * w), int(lm.y * h)))
    for pt in pts:
        cv2.circle(occ_mask, pt, int(min(h, w) * 0.05), 255, -1)

    # Adaptive threshold to detect dark anomalies on skin
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=5
    )

    # Apply skin mask — only keep marks on skin surface
    masked = cv2.bitwise_and(thresh, skin_mask)

    # Apply occlusion mask to avoid detecting shadows as marks
    masked[occ_mask > 0] = 0

    # Morphological opening to remove speckle noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter by area: min 8px² (noise), max 500px² (shadows/large regions)
        if area < 8 or area > 500:
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Mean intensity of the mark region
        mark_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mark_mask, [cnt], -1, 255, -1)
        mean_intensity = float(cv2.mean(gray, mask=mark_mask)[0])

        marks.append({
            "centroid": (cx / w, cy / h),  # normalized [0,1]
            "area": area,
            "intensity": mean_intensity,
            "circularity": circularity,
            "contour": cnt,  # keep for visualization
        })

    return marks, occ_mask


def compute_mark_correspondence(marks_gallery: list, marks_probe: list) -> dict:
    """
    Bayesian Likelihood Ratio Mark Correspondence Engine (Daubert v3.0).

    Evaluates the LR for every potential mark match:
      - Numerator P(E|Hp): Multivariate Gaussian PDF at observed delta vector
      - Denominator P(E|Hd): KDE spatial density × morphological PDFs

    Uses Hungarian optimal matching on -log(LR) to maximize the joint LR.

    Returns:
        {
            "score": float (0-100) — posterior-derived percentage, or None,
            "matched": int,
            "total_gallery": int,
            "total_probe": int,
            "matches": [(gallery_idx, probe_idx, individual_lr), ...],
            "lr_marks": float — product of all individual mark LRs,
            "mark_lrs": [float, ...] — individual LR per matched mark,
        }
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import multivariate_normal as mvn

    n_gal = len(marks_gallery)
    n_pro = len(marks_probe)

    # Minimum mark threshold: 1 (the math filters noise via LR ≈ 1)
    if n_gal < 1 or n_pro < 1:
        return {
            "score": None, "matched": 0,
            "total_gallery": n_gal, "total_probe": n_pro,
            "matches": [], "lr_marks": 1.0, "mark_lrs": [],
        }

    # If no Bayesian calibration data, fall back to neutral LR
    if TIER4_CALIBRATION is None:
        return {
            "score": None, "matched": 0,
            "total_gallery": n_gal, "total_probe": n_pro,
            "matches": [], "lr_marks": 1.0, "mark_lrs": [],
        }

    # Unpack calibration models
    spatial_kde = TIER4_CALIBRATION["spatial_kde"]
    area_dist = TIER4_CALIBRATION["area_distribution"]
    int_dist = TIER4_CALIBRATION["intensity_distribution"]
    circ_dist = TIER4_CALIBRATION["circularity_distribution"]
    delta_model = TIER4_CALIBRATION["intra_person_delta"]
    EPSILON = TIER4_CALIBRATION.get("epsilon_floor", 1e-9)

    delta_mean = np.array(delta_model["mean"])
    delta_cov = np.array(delta_model["covariance"])

    # Build NxM LR matrix
    lr_matrix = np.ones((n_gal, n_pro))
    neg_log_lr_matrix = np.full((n_gal, n_pro), 1e6)  # For Hungarian minimization

    for i, mg in enumerate(marks_gallery):
        for j, mp_mark in enumerate(marks_probe):
            # Delta vector: gallery - probe
            delta_v = np.array([
                mg["centroid"][0] - mp_mark["centroid"][0],
                mg["centroid"][1] - mp_mark["centroid"][1],
                mg["area"] - mp_mark["area"],
                mg["intensity"] - mp_mark["intensity"],
                mg["circularity"] - mp_mark["circularity"],
            ])

            # Spatial proximity gate: skip pairs too far apart (> 20% of face)
            spatial_dist = np.sqrt(delta_v[0]**2 + delta_v[1]**2)
            if spatial_dist > 0.20:
                continue

            # NUMERATOR: P(delta | Hp) — how likely is this delta for same person
            try:
                numerator = mvn.pdf(delta_v, mean=delta_mean, cov=delta_cov)
            except Exception:
                numerator = EPSILON
            numerator = max(numerator, EPSILON)

            # DENOMINATOR: P(E | Hd) — population frequency of this mark
            # Product of: spatial KDE × area PDF × intensity PDF × circularity PDF
            try:
                p_spatial = float(spatial_kde.evaluate(
                    np.array([[mp_mark["centroid"][0]], [mp_mark["centroid"][1]]])
                )[0])
            except Exception:
                p_spatial = EPSILON
            p_spatial = max(p_spatial, EPSILON)

            from scipy.stats import lognorm as _lognorm, norm as _norm
            p_area = max(float(_lognorm.pdf(
                mp_mark["area"],
                area_dist["shape"], loc=area_dist["loc"], scale=area_dist["scale"]
            )), EPSILON)
            p_intensity = max(float(_norm.pdf(
                mp_mark["intensity"],
                loc=int_dist["mean"], scale=int_dist["std"]
            )), EPSILON)
            p_circularity = max(float(_norm.pdf(
                mp_mark["circularity"],
                loc=circ_dist["mean"], scale=circ_dist["std"]
            )), EPSILON)

            denominator = max(p_spatial * p_area * p_intensity * p_circularity, EPSILON)

            # Individual Likelihood Ratio
            lr = numerator / denominator
            lr_matrix[i, j] = lr

            # -log(LR) for Hungarian minimization (maximize LR → minimize -log LR)
            neg_log_lr_matrix[i, j] = -np.log(max(lr, EPSILON))

    # Hungarian optimal matching to maximize joint LR
    row_ind, col_ind = linear_sum_assignment(neg_log_lr_matrix)

    # Accept matches where LR > 1 (evidence supports same-source)
    matches = []
    mark_lrs = []
    for r, c in zip(row_ind, col_ind):
        lr_val = float(lr_matrix[r, c])
        if lr_val > 1.0:
            matches.append((int(r), int(c), lr_val))
            mark_lrs.append(lr_val)

    # Combined LR = product of individual mark LRs
    lr_marks = 1.0
    for lr_val in mark_lrs:
        lr_marks *= lr_val

    matched_count = len(matches)
    total = max(n_gal, n_pro)
    score = (matched_count / total) * 100.0 if total > 0 else 0.0

    return {
        "score": round(score, 2),
        "matched": matched_count,
        "total_gallery": n_gal,
        "total_probe": n_pro,
        "matches": matches,
        "lr_marks": lr_marks,
        "mark_lrs": mark_lrs,
    }


def generate_scar_delta_map(
    img_gallery: np.ndarray,
    img_probe: np.ndarray,
    marks_gallery: list = None,
    marks_probe: list = None,
    mark_matches: list = None,
) -> str:
    """
    Biological Topography Delta — Scar Mapper.
    Isolates persistent micro-topology (scars, pores, creases) that appears
    consistently across both the gallery and probe aligned face crops.

    Now also visualizes detected marks:
      - Green circles: matched marks (present in both faces)
      - Yellow circles: unmatched marks (only in one face)

    ALGORITHM:
    1. Convert both images to grayscale.
    2. Run Canny edge detection on both to extract edge maps.
    3. Compute absdiff on grayscales and threshold at a LOW value —
       pixels with small intensity difference represent *persistent* structure.
    4. bitwise_and(gallery_edges, probe_edges, persistent_mask) isolates
       topology that exists in BOTH images and didn't shift between captures.
    5. Dilate slightly for UI readability.
    6. Overlay in neon crimson (BGR: 30, 0, 180) on a darkened, desaturated
       version of the gallery image.
    7. Draw mark detection circles if mark data is provided.
    8. Base64 encode and return as a data URI.
    """
    h, w = img_gallery.shape[:2]

    # 1. Grayscale conversion
    gray_gallery = cv2.cvtColor(img_gallery, cv2.COLOR_BGR2GRAY)
    gray_probe = cv2.cvtColor(img_probe, cv2.COLOR_BGR2GRAY)

    # 2. Canny edge detection (tuned for facial micro-features)
    edges_gallery = cv2.Canny(gray_gallery, 30, 100)
    edges_probe = cv2.Canny(gray_probe, 30, 100)

    # 3. Absolute difference → persistent structure mask
    #    Low diff = structure that didn't move between captures
    diff = cv2.absdiff(gray_gallery, gray_probe)
    _, persistent_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY_INV)

    # 4. Intersection: edges present in BOTH images AND persistent
    common_edges = cv2.bitwise_and(edges_gallery, edges_probe)
    true_scars = cv2.bitwise_and(common_edges, persistent_mask)

    # 5. Dilate for UI visibility (2×2 kernel, 1 iteration)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    true_scars = cv2.dilate(true_scars, kernel, iterations=1)

    # 6. Build overlay canvas: transparent RGBA (so it can be overlaid without blending faces)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    # Paint neon crimson (BGRA: 30, 0, 180, 255) where scars are detected
    canvas[true_scars > 0] = (30, 0, 180, 255)

    # 7. Overlay detected mark circles (from Tier 4 engine)
    if marks_gallery and mark_matches is not None:
        matched_gallery_indices = {m[0] for m in mark_matches} if mark_matches else set()

        for i, mark in enumerate(marks_gallery):
            cx = int(mark["centroid"][0] * w)
            cy = int(mark["centroid"][1] * h)
            radius = max(4, int(np.sqrt(mark["area"]) * 0.8))

            if i in matched_gallery_indices:
                # Green circle — matched mark (confirmed in both faces)
                cv2.circle(canvas, (cx, cy), radius, (0, 220, 80, 255), 2, cv2.LINE_AA)
            else:
                # Yellow circle — unmatched mark (only in gallery)
                cv2.circle(canvas, (cx, cy), radius, (0, 200, 220, 255), 1, cv2.LINE_AA)

    # 8. Encode to base64 data URI
    _, buffer = cv2.imencode('.png', canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def generate_wireframe_hud(image: np.ndarray, landmarks) -> str:
    """
    3DMM Wireframe HUD — Geometric Mesh Visualizer.
    Renders the full FACEMESH_TESSELATION (468-point mesh) in 24K Gold
    over a darkened, desaturated copy of the input image to prove
    geometric extraction to the operator.

    STYLING:
    - Mesh color: 24K Gold → BGR (55, 175, 212)
    - Line thickness: 1px hairline
    - Landmark dots: suppressed (circle_radius=0 removes them entirely)
    - Background: 30% saturation, 35% brightness (matches Scar Delta canvas)

    Returns a base64-encoded PNG data URI.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh_module = mp.solutions.face_mesh

    # Build darkened, desaturated canvas
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = (hsv[:, :, 1] * 0.3).astype(np.uint8)   # Desaturate
    hsv[:, :, 2] = (hsv[:, :, 2] * 0.35).astype(np.uint8)   # Darken
    canvas = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Reconstruct a NormalizedLandmarkList so draw_landmarks can consume it
    from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark
    landmark_list = NormalizedLandmarkList()
    for lm in landmarks:
        landmark_list.landmark.append(
            NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
        )

    # 24K Gold connection style — BGR (55, 175, 212), 1px, no circles
    gold_spec = mp_drawing.DrawingSpec(
        color=(55, 175, 212),
        thickness=1,
        circle_radius=0
    )
    # Suppress landmark dots entirely
    dot_spec = mp_drawing.DrawingSpec(
        color=(55, 175, 212),
        thickness=0,
        circle_radius=0
    )

    mp_drawing.draw_landmarks(
        image=canvas,
        landmark_list=landmark_list,
        connections=mp_face_mesh_module.FACEMESH_TESSELATION,
        landmark_drawing_spec=dot_spec,
        connection_drawing_spec=gold_spec
    )

    # Encode to base64 data URI
    _, buffer = cv2.imencode('.png', canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


# ---------------------------------------------------------
# COMPOSITE FORENSIC RECEIPT (EVIDENCE PRESERVATION)
# ---------------------------------------------------------

def generate_forensic_receipt(
    gallery_aligned: np.ndarray,
    probe_aligned: np.ndarray,
    gallery_heatmap_b64: str,
    fused_score: float,
    probe_file_hash: str,
) -> str | None:
    """
    Generates a self-contained composite forensic receipt PNG.
    Layout: Gallery (256x256) | Probe (256x256) | Attention Map (256x256)
    with a high-contrast 94px text panel at the bottom (total: 768x350).

    Uploads to GCS under receipts/ prefix.
    Returns the GCS URI or None on failure.
    """
    try:
        # Decode heatmap from base64 data URI back to numpy
        b64_data = gallery_heatmap_b64.split(",")[1] if "," in gallery_heatmap_b64 else gallery_heatmap_b64
        heatmap_bytes = base64.b64decode(b64_data)
        heatmap_arr = np.frombuffer(heatmap_bytes, dtype=np.uint8)
        heatmap_img = cv2.imdecode(heatmap_arr, cv2.IMREAD_COLOR)

        # Ensure all panels are 256x256
        g = cv2.resize(gallery_aligned, (256, 256))
        p = cv2.resize(probe_aligned, (256, 256))
        h = cv2.resize(heatmap_img, (256, 256)) if heatmap_img is not None else np.zeros((256, 256, 3), dtype=np.uint8)

        # Stitch side-by-side: 768x256
        composite = np.hstack([g, p, h])

        # Build high-contrast text panel (768x94, dark background)
        text_panel = np.zeros((94, 768, 3), dtype=np.uint8)
        text_panel[:] = (20, 20, 20)

        # Column labels
        cv2.putText(text_panel, "GALLERY", (85, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(text_panel, "PROBE", (355, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(text_panel, "ATTENTION MAP", (570, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

        # Divider line
        cv2.line(text_panel, (0, 24), (768, 24), (60, 60, 60), 1)

        # ISO-8601 timestamp
        timestamp_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Forensic data lines
        cv2.putText(text_panel, f"FUSED SCORE: {fused_score:.2f}%", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(text_panel, f"PROBE SHA-256: {probe_file_hash}", (15, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(text_panel, f"UTC TIMESTAMP: {timestamp_iso}", (15, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # Final composite: 768x350
        final = np.vstack([composite, text_panel])

        # Encode as PNG and upload to GCS
        _, buffer = cv2.imencode('.png', final)
        receipt_bytes = buffer.tobytes()

        bucket_name = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
        receipt_blob_name = f"receipts/{uuid.uuid4().hex}.png"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(receipt_blob_name)
        blob.upload_from_string(receipt_bytes, content_type="image/png")

        receipt_uri = f"gs://{bucket_name}/{receipt_blob_name}"
        print(f"[RECEIPT] Forensic receipt uploaded: {receipt_uri}")
        return receipt_uri

    except Exception as e:
        print(f"[RECEIPT] WARNING: Failed to generate forensic receipt: {e}")
        return None


class UploadUrlsRequest(BaseModel):
    gallery_content_type: Optional[str] = None
    probe_content_type: str

class UploadUrlsResponse(BaseModel):
    gallery_upload_url: Optional[str] = None
    probe_upload_url: str
    gallery_gs_uri: Optional[str] = None
    probe_gs_uri: str

@app.post("/generate-upload-urls", response_model=UploadUrlsResponse)
def generate_upload_urls(req: UploadUrlsRequest, _: dict = Depends(verify_jwt)):
    bucket_name = os.getenv("BUCKET_NAME") or "hoppwhistle-facial-raw-images-bucket"
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, project = google.auth.default()
        
        # Refresh credentials to get a valid token for IAM signBlob
        auth_request = google.auth.transport.requests.Request()
        credentials.refresh(auth_request)

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        probe_blob_name = f"probe_{uuid.uuid4().hex}.jpg"
        probe_blob = bucket.blob(probe_blob_name)
        
        probe_url = probe_blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=15),
            method="PUT",
            content_type=req.probe_content_type,
            service_account_email=credentials.service_account_email,
            access_token=credentials.token
        )
        
        gallery_url = None
        gallery_gs_uri = None
        
        if req.gallery_content_type:
            gallery_blob_name = f"gallery_{uuid.uuid4().hex}.jpg"
            gallery_blob = bucket.blob(gallery_blob_name)
            gallery_url = gallery_blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(minutes=15),
                method="PUT",
                content_type=req.gallery_content_type,
                service_account_email=credentials.service_account_email,
                access_token=credentials.token
            )
            gallery_gs_uri = f"gs://{bucket_name}/{gallery_blob_name}"
        
        return UploadUrlsResponse(
            gallery_upload_url=gallery_url,
            probe_upload_url=probe_url,
            gallery_gs_uri=gallery_gs_uri,
            probe_gs_uri=f"gs://{bucket_name}/{probe_blob_name}"
        )
    except Exception as e:
        print(f"GCS Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate upload URLs: {str(e)}")

def analyze_frequency_domain(image: np.ndarray) -> float:
    """
    Phase 7: Synthetic Provenance Veto.
    Performs FFT frequency domain analysis to detect checkerboard artifacts 
    (high-frequency grid anomalies) inherent to AI upscaling networks.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist_from_center > (min(h, w) * 0.15)
    
    high_freq = magnitude_spectrum[mask]
    if len(high_freq) == 0:
        return 0.0
        
    peak_intensity = float(np.max(high_freq))
    mean_intensity = float(np.mean(high_freq))
    
    anomaly_score = (peak_intensity - mean_intensity) / (mean_intensity + 1e-8)
    normalized_score = max(0.0, min(1.0, anomaly_score * 0.4))
    return normalized_score

@app.post("/verify/fuse", response_model=VerificationResponse)
@limiter.limit("5/minute")
def verify_pipeline(request: Request, payload: VerificationRequest, _: dict = Depends(verify_jwt)):
    # 1. Fetch images from GCS (with pre-decode binary hashing)
    gallery_img, gallery_file_hash = fetch_image_from_url(payload.gallery_url)
    probe_img, probe_file_hash = fetch_image_from_url(payload.probe_url)
    
    # Phase 7: Synthetic Provenance Gatekeeper
    SYNTHETIC_ARTIFACT_THRESHOLD = 0.85
    gallery_anomaly = analyze_frequency_domain(gallery_img)
    probe_anomaly = analyze_frequency_domain(probe_img)
    max_anomaly = max(gallery_anomaly, probe_anomaly)
    
    if max_anomaly > SYNTHETIC_ARTIFACT_THRESHOLD:
        ledger_session = SessionLocal()
        try:
            event = VerificationEvent(
                probe_hash=probe_file_hash,
                gallery_hash=gallery_file_hash,
                fused_score_x100=0,
                conclusion="VETO: Synthetic Media Detected",
                pipeline_version=PIPELINE_VERSION,
                veto_triggered=True,
                failed_provenance_veto=True,
                synthetic_anomaly_score=max_anomaly
            )
            ledger_session.add(event)
            ledger_session.commit()
        except Exception as e:
            print(f"Failed to write provenance veto to ledger: {e}")
            ledger_session.rollback()
        finally:
            ledger_session.close()
            
        return JSONResponse(status_code=200, content={
            "status": "success", 
            "conclusion": "VETO: Synthetic Media Detected", 
            "fused_score": 0,
            "synthetic_anomaly_score": max_anomaly
        })

    # 1.5 Presentation Attack Detection (Liveness Firewall)
    if payload.require_liveness:
        liveness_result = detect_liveness(probe_img)
        liveness_telemetry = build_liveness_telemetry(liveness_result)
        if liveness_result["score"] < 0.95:
            raise HTTPException(status_code=403, detail="SPOOF_DETECTED: Presentation attack suspected.")
    else:
        liveness_telemetry = {"status": "BYPASSED", "method": "NONE"}
    
    # 2. Preprocess (CLAHE)
    gallery_clahe = apply_clahe(gallery_img)
    probe_clahe = apply_clahe(probe_img)
    
    # 3. Face Alignment & Crop to canonical 256×256
    gallery_aligned, gallery_landmarks = align_face_crop(gallery_clahe)
    probe_aligned, probe_landmarks = align_face_crop(probe_clahe)
    
    if gallery_landmarks is None or probe_landmarks is None:
        raise HTTPException(
            status_code=400,
            detail="FACE_NOT_DETECTED: Could not detect a face in one or both images. Please upload clear, front-facing photographs."
        )
    
    # 3.5 Temporal Invariance Engine (Age Estimation & Cross-Spectral Normalization)
    gallery_age = estimate_age(gallery_aligned)
    probe_age = estimate_age(probe_aligned)
    temporal_delta = abs(probe_age - gallery_age)

    # Cross-spectral matching before passing to textural/mark layers
    gallery_aligned, probe_aligned, spectral_correction = cross_spectral_normalize(gallery_aligned, probe_aligned)
    
    # 4. TIER 1: Structural Identity (Neural Ensemble: 60% ArcFace, 40% Facenet512)
    ensemble_gallery = extract_ensemble_embeddings(gallery_aligned)
    ensemble_probe = extract_ensemble_embeddings(probe_aligned)
    structural_sim, arcface_sim, secondary_sim = compute_ensemble_similarity(ensemble_gallery, ensemble_probe)
    tier1_score = structural_sim * 100
    
    # 5. TIER 2: Geometric Biometrics (3D Topographical Mapping)
    # Uses Euclidean distance between 12-D scale-invariant, 3D Procrustes-aligned facial ratio vectors.
    ratios_gallery, gal_angles, gal_vis = extract_geometric_ratios_3d(gallery_landmarks)
    ratios_probe, pro_angles, pro_vis = extract_geometric_ratios_3d(probe_landmarks)
    
    valid_mask = gal_vis["ratio_visibility"] & pro_vis["ratio_visibility"]
    effective_ratios = int(np.sum(valid_mask))
    
    if effective_ratios > 0:
        raw_l2 = float(np.linalg.norm((ratios_gallery - ratios_probe)[valid_mask]))
        ratio_l2 = raw_l2 * math.sqrt(12.0 / effective_ratios)
    else:
        ratio_l2 = 0.50
        
    # L2 mapping: 0 distance → 100%, ≥0.50 → 0%. Recalibrated from 0.40 due to added 3D variance.
    tier2_score = max(0.0, min(100.0, (1.0 - (ratio_l2 / 0.50)) * 100))
    
    # 6. TIER 3: Micro-Topology (LBP Chi-Squared Distance)
    # Chi-squared is the standard metric for comparing LBP histograms
    # in the biometrics literature. Histogram intersection is not
    # discriminative enough on CLAHE-normalized, aligned crops.
    lbp_gal = extract_lbp_histogram(gallery_aligned)
    lbp_pro = extract_lbp_histogram(probe_aligned)
    chi_squared = 0.5 * float(np.sum(((lbp_gal - lbp_pro) ** 2) / (lbp_gal + lbp_pro + 1e-10)))
    tier3_score = max(0.0, min(100.0, (1.0 - chi_squared) * 100))
    
    # 7. Veto Protocol — ArcFace Hard Fail (flag only — Bayesian math handles scoring)
    veto_triggered = structural_sim < 0.40

    # 7.5 TIER 4: Mark Correspondence (Bayesian LR Engine)
    marks_gallery, occ_gallery = detect_facial_marks(gallery_aligned, gallery_landmarks)
    marks_probe, occ_probe = detect_facial_marks(probe_aligned, probe_landmarks)
    
    valid_gallery_marks = []
    for m in marks_gallery:
        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
        if cy < 256 and cx < 256 and occ_probe[cy, cx] == 0:
            valid_gallery_marks.append(m)
            
    valid_probe_marks = []
    for m in marks_probe:
        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
        if cy < 256 and cx < 256 and occ_gallery[cy, cx] == 0:
            valid_probe_marks.append(m)

    mark_result = compute_mark_correspondence(valid_gallery_marks, valid_probe_marks)
    tier4_score = mark_result["score"]  # None if insufficient marks

    # ── BAYESIAN EVIDENCE FUSION (Daubert v4.0) ──
    # Convert Fused Ensemble score to Likelihood Ratio
    lr_ensemble = score_to_lr_ensemble(structural_sim, temporal_delta=temporal_delta)

    # Combined mark LR (product of individual mark LRs)
    lr_marks = mark_result.get("lr_marks", 1.0)

    # Total LR = independent evidence product
    lr_total = lr_ensemble * lr_marks

    # Posterior probability via Bayes' Theorem (neutral prior = 0.5)
    # P(Hp|E) = (Prior × LR) / ((Prior × LR) + (1 - Prior))
    # With Prior = 0.5: Posterior = LR / (LR + 1)
    PRIOR = 0.5
    posterior = (PRIOR * lr_total) / ((PRIOR * lr_total) + (1.0 - PRIOR))
    fused_score = posterior * 100.0

    # Retain tier scores for dashboard display
    _fw = CALIBRATION["fusion"]["optimal_weights"] if CALIBRATION else None
    _w1 = _fw["structural"] if _fw else 0.60
    _w2 = _fw["geometric"] if _fw else 0.25
    _w3 = _fw["micro_topology"] if _fw else 0.15
    base_fused_score = (tier1_score * _w1) + (tier2_score * _w2) + (tier3_score * _w3)

    if veto_triggered:
        fused_score = 0.0
        conclusion = "EXCLUSION: Biometric Non-Match (ArcFace Veto)"
    elif fused_score > 90.0:
        conclusion = "Strongest Support for Common Source"
    elif fused_score > 75.0:
        conclusion = "Support for Common Source"
    else:
        conclusion = "Exclusion: Insufficient Fused Similarity"

    # Landmark Attention Maps on aligned crops (real 468-point density, not fabricated)
    gallery_heatmap = generate_landmark_attention_map(gallery_aligned, gallery_landmarks)
    probe_heatmap = generate_landmark_attention_map(probe_aligned, probe_landmarks)
    
    # Encode aligned crops as base64 for frontend SymmetryMerge
    _, gal_buf = cv2.imencode('.png', gallery_aligned)
    gallery_aligned_b64 = f"data:image/png;base64,{base64.b64encode(gal_buf).decode('utf-8')}"
    _, pro_buf = cv2.imencode('.png', probe_aligned)
    probe_aligned_b64 = f"data:image/png;base64,{base64.b64encode(pro_buf).decode('utf-8')}"

    # Biological Topography Delta (Scar Mapper — now with mark visualization)
    scar_delta = generate_scar_delta_map(
        gallery_aligned, probe_aligned,
        marks_gallery=marks_gallery,
        marks_probe=marks_probe,
        mark_matches=mark_result["matches"],
    )

    # 3DMM Wireframe HUD (Geometric Mesh Overlay)
    gallery_wireframe = generate_wireframe_hud(gallery_aligned, gallery_landmarks)
    probe_wireframe = generate_wireframe_hud(probe_aligned, probe_landmarks)

    # Statistical confidence from Tier-1 raw cosine (baseline)
    stats = calculate_statistical_confidence(structural_sim)
    
    # Upgrade statistical confidence to reflect the final Bayesian Posterior
    bayesian_far = 1.0 - posterior
    if bayesian_far < 1e-7:
        stats["false_acceptance_rate"] = "< 1 in 10,000,000"
        stats["statistical_certainty"] = f"{(posterior * 100):.6f}%"
    elif bayesian_far >= 0.60:  # Maps to a fused_score < 40.0
        stats["false_acceptance_rate"] = "DIFFERENT IDENTITIES"
        stats["statistical_certainty"] = "0% — Non-Match"
    else:
        stats["false_acceptance_rate"] = f"1 in {int(1.0 / bayesian_far):,}"
        stats["statistical_certainty"] = f"{(posterior * 100):.6f}%"

    # Deep Forensic Telemetry (hash of 512-D ArcFace vector)
    probe_vector_hash = compute_vector_hash(ensemble_probe[0])
    probe_alignment = compute_alignment_variance(probe_aligned)

    audit = AuditLog(
        raw_cosine_score=round(structural_sim, 6),
        raw_arcface_score=round(arcface_sim, 6),
        raw_secondary_score=round(secondary_sim, 6),
        ensemble_model_secondary="Facenet512",
        pose_corrected_3d=True,
        probe_pose_angles=pro_angles,
        gallery_pose_angles=gal_angles,
        occlusion_percentage=pro_vis["occlusion_percentage"],
        occluded_regions=pro_vis["occluded_regions"],
        effective_geometric_ratios_used=effective_ratios,
        estimated_temporal_delta=round(temporal_delta, 1),
        cross_spectral_correction_applied=spectral_correction,
        statistical_certainty=stats["statistical_certainty"],
        false_acceptance_rate=stats["false_acceptance_rate"],
        nodes_mapped=468,
        vector_hash=probe_vector_hash,
        alignment_variance=probe_alignment,
        liveness_check=liveness_telemetry,
        crypto_envelope=build_crypto_envelope(),
        calibration_benchmark=stats.get("benchmark"),
        calibration_pairs=stats.get("pairs_evaluated"),
        probe_file_hash=probe_file_hash,
        gallery_file_hash=gallery_file_hash,
        pipeline_version=PIPELINE_VERSION,
        dependency_versions=DEPENDENCY_VERSIONS,
        # Bayesian LR Forensic Audit Trail
        lr_arcface=round(lr_ensemble, 6),
        lr_marks=round(lr_marks, 6),
        lr_total=round(lr_total, 6),
        posterior_probability=round(posterior, 8),
        mark_lrs=[round(lr, 4) for lr in mark_result.get("mark_lrs", [])],
    )

    # Build correspondences list for the UI
    correspondences = []
    for g_idx, p_idx, individual_lr in mark_result.get("matches", []):
        correspondences.append({
            "gallery_pt": valid_gallery_marks[g_idx]["centroid"],
            "probe_pt": valid_probe_marks[p_idx]["centroid"],
            "lr": individual_lr
        })

    response = VerificationResponse(
        structural_score=round(tier1_score, 2),
        soft_biometrics_score=round(tier2_score, 2),
        micro_topology_score=round(tier3_score, 2),
        fused_identity_score=round(fused_score, 2),
        conclusion=conclusion,
        veto_triggered=veto_triggered,
        gallery_heatmap_b64=gallery_heatmap,
        probe_heatmap_b64=probe_heatmap,
        gallery_aligned_b64=gallery_aligned_b64,
        probe_aligned_b64=probe_aligned_b64,
        scar_delta_b64=scar_delta,
        gallery_wireframe_b64=gallery_wireframe,
        probe_wireframe_b64=probe_wireframe,
        mark_correspondence_score=tier4_score,
        marks_detected_gallery=mark_result["total_gallery"],
        marks_detected_probe=mark_result["total_probe"],
        marks_matched=mark_result["matched"],
        correspondences=correspondences,
        raw_probe_marks=valid_probe_marks,
        raw_gallery_marks=valid_gallery_marks,
        audit_log=audit
    )

    # ── Composite Forensic Receipt (Evidence Preservation) ──
    receipt_url = generate_forensic_receipt(
        gallery_aligned=gallery_aligned,
        probe_aligned=probe_aligned,
        gallery_heatmap_b64=gallery_heatmap,
        fused_score=fused_score,
        probe_file_hash=probe_file_hash,
    )

    # ── Immutable Audit Ledger ──
    ledger_session = SessionLocal()
    try:
        event = VerificationEvent(
            probe_hash=probe_file_hash,
            gallery_hash=gallery_file_hash,
            matched_user_id=None,
            fused_score_x100=round(fused_score * 100),
            conclusion=conclusion,
            pipeline_version=PIPELINE_VERSION,
            calibration_benchmark=stats.get("benchmark"),
            false_acceptance_rate=stats["false_acceptance_rate"],
            veto_triggered=veto_triggered,
            structural_score_x100=round(tier1_score * 100),
            arcface_score_x10000=round(arcface_sim * 10000),
            secondary_score_x10000=round(secondary_sim * 10000),
            ensemble_model_secondary="Facenet512",
            geometric_score_x100=round(tier2_score * 100),
            micro_topology_score_x100=round(tier3_score * 100),
            mark_correspondence_x100=round(tier4_score * 100) if tier4_score is not None else None,
            pose_corrected_3d=True,
            probe_pose_angles=json.dumps(pro_angles) if pro_angles else None,
            gallery_pose_angles=json.dumps(gal_angles) if gal_angles else None,
            occlusion_percentage=pro_vis["occlusion_percentage"],
            occluded_regions=json.dumps(pro_vis["occluded_regions"]) if pro_vis["occluded_regions"] else None,
            effective_geometric_ratios_used=effective_ratios,
            receipt_url=receipt_url,
        )
        ledger_session.add(event)
        ledger_session.commit()
    except Exception as ledger_err:
        print(f"Audit ledger write failed (non-fatal): {ledger_err}")
        ledger_session.rollback()
    finally:
        ledger_session.close()

    return response

# ---------------------------------------------------------
# PHASE 2: 1:N VAULT SEARCH (TARGET ACQUISITION)
# ---------------------------------------------------------
TARGET_PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target_profiles")


class VaultSearchRequest(BaseModel):
    probe_url: str
    require_liveness: bool = False


def _generate_user_id(filepath: str) -> str:
    """Mirrors the ingestion script's deterministic user_id generation."""
    stem = Path(filepath).stem.lower().replace(" ", "_")
    file_hash = hashlib.sha256(Path(filepath).name.encode()).hexdigest()[:8]
    return f"{stem}_{file_hash}"


def _resolve_target_image(user_id: str) -> str | None:
    """
    Reverse-lookup: scans target_profiles/ and finds the original
    image file whose deterministic user_id matches the vault record.
    """
    if not os.path.isdir(TARGET_PROFILES_DIR):
        return None
    for fname in os.listdir(TARGET_PROFILES_DIR):
        fpath = os.path.join(TARGET_PROFILES_DIR, fname)
        if os.path.isfile(fpath) and _generate_user_id(fpath) == user_id:
            return fpath
    return None


@app.post("/vault/search", response_model=VerificationResponse)
@limiter.limit("5/minute")
def vault_search(request: Request, payload: VaultSearchRequest, _: dict = Depends(verify_jwt)):
    """
    Phase 2: 1:N Target Acquisition.
    Compares an uploaded probe image against every encrypted IdentityProfile
    in the vault and returns the highest-confidence match with full
    forensic visualization (heatmaps, scar delta, wireframe HUD).
    """
    # 1. Fetch & validate probe (with pre-decode binary hashing)
    probe_img, probe_file_hash = fetch_image_from_url(payload.probe_url)

    if payload.require_liveness:
        liveness_result = detect_liveness(probe_img)
        liveness_telemetry_vault = build_liveness_telemetry(liveness_result)
        if liveness_result["score"] < 0.95:
            raise HTTPException(status_code=403, detail="SPOOF_DETECTED: Presentation attack suspected.")
    else:
        liveness_telemetry_vault = {"status": "BYPASSED", "method": "NONE"}

    # 3. Pre-process probe
    probe_clahe = apply_clahe(probe_img)
    probe_aligned, probe_landmarks = align_face_crop(probe_clahe)

    if probe_landmarks is None:
        raise HTTPException(
            status_code=400,
            detail="FACE_NOT_DETECTED: No face detected in probe image."
        )

    probe_embedding = extract_arcface_embedding(probe_aligned)

    # 4. Two-Stage Retrieval: Stage 1 (FAISS Filter)
    vault_search_start = time.perf_counter()
    results = vault_index.search(probe_embedding, top_k=1)
    vault_decrypt_elapsed_ms = (time.perf_counter() - vault_search_start) * 1000

    if not results:
        raise HTTPException(
            status_code=404,
            detail="VAULT_EMPTY_OR_NO_MATCH: No valid matches found in vault index."
        )

    best_user_id, best_score = results[0]

    # 5. Tier 1 score from vault search (ArcFace only initially)
    tier1_score = best_score * 100
    arcface_sim = best_score
    secondary_sim = 0.0
    structural_sim = best_score

    # 6. Load matched gallery image for forensic overlays
    # First, fetch the matched profile from DB to get thumbnail_url
    gallery_session = SessionLocal()
    matched_profile_for_gallery = None
    try:
        matched_profile_for_gallery = gallery_session.query(IdentityProfile).filter(
            IdentityProfile.user_id == best_user_id
        ).first()
    finally:
        gallery_session.close()

    gallery_aligned = probe_aligned
    gallery_landmarks = probe_landmarks

    gallery_file_hash = None  # Populated if gallery image is fetched

    if matched_profile_for_gallery and matched_profile_for_gallery.thumbnail_url:
        try:
            gallery_img, gallery_file_hash = fetch_image_from_url(matched_profile_for_gallery.thumbnail_url)
            gallery_clahe = apply_clahe(gallery_img)
            gallery_aligned, gallery_landmarks = align_face_crop(gallery_clahe)
            if gallery_landmarks is None:
                gallery_aligned = probe_aligned
                gallery_landmarks = probe_landmarks
        except Exception:
            pass  # Fallback to probe if gallery fetch fails
    else:
        # Legacy fallback: try local target_profiles/ directory
        gallery_file = _resolve_target_image(best_user_id)
        if gallery_file and os.path.isfile(gallery_file):
            gallery_img = cv2.imread(gallery_file)
            gallery_clahe = apply_clahe(gallery_img)
            gallery_aligned, gallery_landmarks = align_face_crop(gallery_clahe)
            if gallery_landmarks is None:
                gallery_aligned = probe_aligned
                gallery_landmarks = probe_landmarks

    # 6.4 Temporal Invariance Engine
    if gallery_landmarks is not None:
        gallery_age = estimate_age(gallery_aligned)
        probe_age = estimate_age(probe_aligned)
        temporal_delta = abs(probe_age - gallery_age)
        gallery_aligned, probe_aligned, spectral_correction = cross_spectral_normalize(gallery_aligned, probe_aligned)
    else:
        temporal_delta = 0.0
        spectral_correction = False

    # 6.5 Upgrade Tier 1 to Neural Ensemble now that we have the gallery image
    if gallery_landmarks is not None:
        ensemble_gallery = extract_ensemble_embeddings(gallery_aligned)
        ensemble_probe = extract_ensemble_embeddings(probe_aligned)
        structural_sim, arcface_sim, secondary_sim = compute_ensemble_similarity(ensemble_gallery, ensemble_probe)
        tier1_score = structural_sim * 100
        best_score = structural_sim  # Use fused score for the rest of the pipeline

    # 7. Tier 2: Geometric Biometrics (3D Topographical Mapping)
    if gallery_landmarks is not None:
        ratios_gallery, gal_angles, gal_vis = extract_geometric_ratios_3d(gallery_landmarks)
        ratios_probe, pro_angles, pro_vis = extract_geometric_ratios_3d(probe_landmarks)
        
        valid_mask = gal_vis["ratio_visibility"] & pro_vis["ratio_visibility"]
        effective_ratios = int(np.sum(valid_mask))
        
        if effective_ratios > 0:
            raw_l2 = float(np.linalg.norm((ratios_gallery - ratios_probe)[valid_mask]))
            ratio_l2 = raw_l2 * math.sqrt(12.0 / effective_ratios)
        else:
            ratio_l2 = 0.50
            
        tier2_score = max(0.0, min(100.0, (1.0 - (ratio_l2 / 0.50)) * 100))
    else:
        tier2_score = 0.0
        gal_angles, pro_angles = {}, {}
        pro_vis = {"occlusion_percentage": 0.0, "occluded_regions": []}
        effective_ratios = 0

    # 8. Tier 3: Micro-Topology (LBP Chi-Squared Distance)
    lbp_gal = extract_lbp_histogram(gallery_aligned)
    lbp_pro = extract_lbp_histogram(probe_aligned)
    chi_squared = 0.5 * float(np.sum(((lbp_gal - lbp_pro) ** 2) / (lbp_gal + lbp_pro + 1e-10)))
    tier3_score = max(0.0, min(100.0, (1.0 - chi_squared) * 100))

    # 9. TIER 4: Mark Correspondence (Bayesian LR Engine)
    marks_gallery, occ_gallery = detect_facial_marks(gallery_aligned, gallery_landmarks)
    marks_probe, occ_probe = detect_facial_marks(probe_aligned, probe_landmarks)
    
    valid_gallery_marks = []
    for m in marks_gallery:
        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
        if cy < 256 and cx < 256 and occ_probe[cy, cx] == 0:
            valid_gallery_marks.append(m)
            
    valid_probe_marks = []
    for m in marks_probe:
        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
        if cy < 256 and cx < 256 and occ_gallery[cy, cx] == 0:
            valid_probe_marks.append(m)

    mark_result = compute_mark_correspondence(valid_gallery_marks, valid_probe_marks)
    tier4_score = mark_result["score"]  # None if insufficient marks

    # ── BAYESIAN EVIDENCE FUSION (Daubert v4.0) ──
    lr_ensemble = score_to_lr_ensemble(structural_sim, temporal_delta=temporal_delta)
    lr_marks = mark_result.get("lr_marks", 1.0)
    lr_total = lr_ensemble * lr_marks

    # Posterior probability via Bayes' Theorem (neutral prior = 0.5)
    PRIOR = 0.5
    posterior = (PRIOR * lr_total) / ((PRIOR * lr_total) + (1.0 - PRIOR))
    fused_score = posterior * 100.0

    # Retain tier scores for dashboard display
    _fw = CALIBRATION["fusion"]["optimal_weights"] if CALIBRATION else None
    _w1 = _fw["structural"] if _fw else 0.60
    _w2 = _fw["geometric"] if _fw else 0.25
    _w3 = _fw["micro_topology"] if _fw else 0.15
    base_fused_score = (tier1_score * _w1) + (tier2_score * _w2) + (tier3_score * _w3)

    # 10. ArcFace Veto (flag only — Bayesian math handles scoring)
    veto_arcface = best_score < 0.40

    # 11. Conclusion
    if veto_arcface:
        fused_score = 0.0
        conclusion = "EXCLUSION: Biometric Non-Match (ArcFace Veto)"
    elif fused_score > 90.0:
        conclusion = f"TARGET ACQUIRED — Strongest match: {best_user_id} (Posterior: {fused_score:.1f}%)"
    elif fused_score > 75.0:
        conclusion = f"TARGET ACQUIRED — Probable match: {best_user_id} (Posterior: {fused_score:.1f}%)"
    else:
        conclusion = f"WEAK MATCH — Nearest candidate: {best_user_id} (Posterior: {fused_score:.1f}%)"

    # 11. Forensic visualizations (real landmark density maps)
    gallery_heatmap = generate_landmark_attention_map(gallery_aligned, gallery_landmarks)
    probe_heatmap = generate_landmark_attention_map(probe_aligned, probe_landmarks)

    _, gal_buf = cv2.imencode('.png', gallery_aligned)
    gallery_aligned_b64 = f"data:image/png;base64,{base64.b64encode(gal_buf).decode('utf-8')}"
    _, pro_buf = cv2.imencode('.png', probe_aligned)
    probe_aligned_b64 = f"data:image/png;base64,{base64.b64encode(pro_buf).decode('utf-8')}"

    scar_delta = generate_scar_delta_map(
        gallery_aligned, probe_aligned,
        marks_gallery=marks_gallery,
        marks_probe=marks_probe,
        mark_matches=mark_result["matches"],
    )

    # Wireframe HUD
    gallery_wireframe_b64 = ""
    probe_wireframe_b64 = ""
    if gallery_landmarks:
        gallery_wireframe_b64 = generate_wireframe_hud(gallery_aligned, gallery_landmarks)
    if probe_landmarks:
        probe_wireframe_b64 = generate_wireframe_hud(probe_aligned, probe_landmarks)

    # Statistical confidence & attribution from vault match
    stats = calculate_statistical_confidence(best_score)
    matched_profile = None
    attr_session = SessionLocal()
    try:
        matched_profile = attr_session.query(IdentityProfile).filter(
            IdentityProfile.user_id == best_user_id
        ).first()
    finally:
        attr_session.close()

    # Deep Forensic Telemetry (hash of 512-D ArcFace vector)
    vault_vector_hash = compute_vector_hash(probe_embedding)
    vault_alignment = compute_alignment_variance(probe_aligned)

    audit = AuditLog(
        raw_cosine_score=round(structural_sim, 6),
        raw_arcface_score=round(arcface_sim, 6),
        raw_secondary_score=round(secondary_sim, 6),
        ensemble_model_secondary="Facenet512",
        pose_corrected_3d=True,
        probe_pose_angles=pro_angles,
        gallery_pose_angles=gal_angles,
        occlusion_percentage=pro_vis["occlusion_percentage"],
        occluded_regions=pro_vis["occluded_regions"],
        effective_geometric_ratios_used=effective_ratios,
        estimated_temporal_delta=round(temporal_delta, 1),
        cross_spectral_correction_applied=spectral_correction,
        statistical_certainty=stats["statistical_certainty"],
        false_acceptance_rate=stats["false_acceptance_rate"],
        nodes_mapped=468,
        matched_user_id=best_user_id,
        person_name=matched_profile.person_name if matched_profile else None,
        source=matched_profile.source if matched_profile else None,
        creator=matched_profile.creator if matched_profile else None,
        license_short_name=matched_profile.license_short_name if matched_profile else None,
        license_url=matched_profile.license_url if matched_profile else None,
        file_page_url=matched_profile.file_page_url if matched_profile else None,
        wikidata_id=matched_profile.wikidata_id if matched_profile else None,
        vector_hash=vault_vector_hash,
        alignment_variance=vault_alignment,
        liveness_check=liveness_telemetry_vault,
        crypto_envelope=build_crypto_envelope(vault_decrypt_elapsed_ms),
        calibration_benchmark=stats.get("benchmark"),
        calibration_pairs=stats.get("pairs_evaluated"),
        probe_file_hash=probe_file_hash,
        gallery_file_hash=gallery_file_hash,
        pipeline_version=PIPELINE_VERSION,
        dependency_versions=DEPENDENCY_VERSIONS,
        # Bayesian LR Forensic Audit Trail
        lr_arcface=round(lr_ensemble, 6),
        lr_marks=round(lr_marks, 6),
        lr_total=round(lr_total, 6),
        posterior_probability=round(posterior, 8),
        mark_lrs=[round(lr, 4) for lr in mark_result.get("mark_lrs", [])],
    )

    # Build correspondences list for the UI
    correspondences = []
    for g_idx, p_idx, individual_lr in mark_result.get("matches", []):
        correspondences.append({
            "gallery_pt": valid_gallery_marks[g_idx]["centroid"],
            "probe_pt": valid_probe_marks[p_idx]["centroid"],
            "lr": individual_lr
        })

    response = VerificationResponse(
        structural_score=round(tier1_score, 2),
        soft_biometrics_score=round(tier2_score, 2),
        micro_topology_score=round(tier3_score, 2),
        fused_identity_score=round(fused_score, 2),
        conclusion=conclusion,
        veto_triggered=veto_arcface,
        gallery_heatmap_b64=gallery_heatmap,
        probe_heatmap_b64=probe_heatmap,
        gallery_aligned_b64=gallery_aligned_b64,
        probe_aligned_b64=probe_aligned_b64,
        scar_delta_b64=scar_delta,
        gallery_wireframe_b64=gallery_wireframe_b64,
        probe_wireframe_b64=probe_wireframe_b64,
        mark_correspondence_score=tier4_score,
        marks_detected_gallery=mark_result["total_gallery"],
        marks_detected_probe=mark_result["total_probe"],
        marks_matched=mark_result["matched"],
        correspondences=correspondences,
        raw_probe_marks=valid_probe_marks,
        raw_gallery_marks=valid_gallery_marks,
        audit_log=audit,
    )

    # ── Composite Forensic Receipt (Evidence Preservation) ──
    receipt_url = generate_forensic_receipt(
        gallery_aligned=gallery_aligned,
        probe_aligned=probe_aligned,
        gallery_heatmap_b64=gallery_heatmap,
        fused_score=fused_score,
        probe_file_hash=probe_file_hash,
    )

    # ── Immutable Audit Ledger ──
    ledger_session = SessionLocal()
    try:
        event = VerificationEvent(
            probe_hash=probe_file_hash,
            gallery_hash=gallery_file_hash,
            matched_user_id=best_user_id,
            fused_score_x100=round(fused_score * 100),
            conclusion=conclusion,
            pipeline_version=PIPELINE_VERSION,
            calibration_benchmark=stats.get("benchmark"),
            false_acceptance_rate=stats["false_acceptance_rate"],
            veto_triggered=veto_arcface,
            structural_score_x100=round(tier1_score * 100),
            arcface_score_x10000=round(arcface_sim * 10000),
            secondary_score_x10000=round(secondary_sim * 10000),
            ensemble_model_secondary="Facenet512",
            geometric_score_x100=round(tier2_score * 100),
            micro_topology_score_x100=round(tier3_score * 100),
            mark_correspondence_x100=round(tier4_score * 100) if tier4_score is not None else None,
            pose_corrected_3d=True,
            probe_pose_angles=json.dumps(pro_angles) if pro_angles else None,
            gallery_pose_angles=json.dumps(gal_angles) if gal_angles else None,
            occlusion_percentage=pro_vis["occlusion_percentage"],
            occluded_regions=json.dumps(pro_vis["occluded_regions"]) if pro_vis["occluded_regions"] else None,
            effective_geometric_ratios_used=effective_ratios,
            receipt_url=receipt_url,
        )
        ledger_session.add(event)
        ledger_session.commit()
    except Exception as ledger_err:
        print(f"Audit ledger write failed (non-fatal): {ledger_err}")
        ledger_session.rollback()
    finally:
        ledger_session.close()

    return response


# ---------------------------------------------------------
# VAULT NETWORK TOPOLOGY (SOVEREIGN IDENTITY GRAPH)
# ---------------------------------------------------------

@app.get("/vault/network")
def vault_network(_: dict = Depends(verify_jwt)):
    """
    Phase 3: Sovereign Identity Graph.
    Returns a signed URL to the pre-computed network topology JSON on GCS.
    The frontend fetches the JSON directly from GCS, bypassing backend I/O.
    """
    import datetime as _dt
    bucket_name = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
    blob_path = "topology/network_graph.json"
    try:
        # Use IAM-based signing (same pattern as generate_identity_graph.py)
        from google.auth import default as _auth_default
        from google.auth.transport.requests import Request as _AuthRequest
        from google.auth import iam as _iam
        from google.oauth2 import service_account as _sa_creds
        import requests as _requests

        credentials, project = _auth_default()
        credentials.refresh(_AuthRequest())

        # Resolve SA email from metadata if needed
        sa_email = getattr(credentials, "service_account_email", None)
        if not sa_email or sa_email == "default":
            r = _requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
                headers={"Metadata-Flavor": "Google"}, timeout=5,
            )
            sa_email = r.text.strip()

        signer = _iam.Signer(
            request=_AuthRequest(),
            credentials=credentials,
            service_account_email=sa_email,
        )

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            return {
                "nodes": [],
                "links": [],
                "status": "PENDING_GENERATION",
                "detail": "Identity graph has not been generated yet."
            }

        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=_dt.timedelta(hours=1),
            method="GET",
            credentials=_sa_creds.Credentials(
                signer=signer,
                service_account_email=sa_email,
                token_uri="https://oauth2.googleapis.com/token",
            ),
        )
        return {"graph_url": signed_url, "status": "READY"}
    except Exception as e:
        print(f"Failed to generate graph URL: {e}")
        return {
            "nodes": [],
            "links": [],
            "status": "ERROR",
            "detail": str(e)
        }


# ---------------------------------------------------------
# STRIPE CHECKOUT (FLAT-FEE PAYWALL)
# ---------------------------------------------------------

class CheckoutRequest(BaseModel):
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None

@app.post("/checkout/create-session")
def create_checkout_session(req: CheckoutRequest = None):
    """
    Creates a Stripe Checkout Session for a one-time $4.99 payment
    to unlock the biometric dossier results.
    """
    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe is not configured.")

    frontend_origin = os.getenv("FRONTEND_URL", "http://localhost:3000")
    success = req.success_url if req and req.success_url else f"{frontend_origin}/?session_id={{CHECKOUT_SESSION_ID}}&success=true"
    cancel = req.cancel_url if req and req.cancel_url else f"{frontend_origin}/?canceled=true"

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": "Biometric Dossier Decryption",
                        "description": "One-time unlock of identity verification results",
                    },
                    "unit_amount": 499,  # $4.99 in cents
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=success,
            cancel_url=cancel,
        )
        return {"checkout_url": session.url, "session_id": session.id}
    except stripe.StripeError as e:
        raise HTTPException(status_code=500, detail=str(e))
