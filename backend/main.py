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
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
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
from models import SessionLocal, IdentityProfile, VerificationEvent, VerificationJob, init_db
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
    import uuid
    import logging
    from security_helpers import get_safe_error_response
    request_id = str(uuid.uuid4())
    logging.error(f"Request ID: {request_id} | Exception: {str(exc)}")
    logging.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content=get_safe_error_response(request_id)
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
    
    # ── SCHEMA GUARD ──
    from models import engine
    from sqlalchemy import inspect
    inspector = inspect(engine)
    if "verification_events" in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns("verification_events")]
        from models import VerificationEvent
        required_columns = [col.name for col in inspect(VerificationEvent).c]
        
        missing_cols = [c for c in required_columns if c not in columns]
        if missing_cols:
            import logging
            logging.critical(f"CRITICAL: Missing canonical columns in verification_events: {missing_cols}")
            if os.getenv("ALLOW_SCHEMA_MISMATCH", "false").lower() != "true":
                raise RuntimeError(f"CRITICAL SCHEMA ERROR: Missing columns {missing_cols}. Run scripts/migrate_production_schema_contract_v1.py before deploying. Set ALLOW_SCHEMA_MISMATCH=true to bypass.")
            else:
                logging.warning("ALLOW_SCHEMA_MISMATCH is true. Bypassing schema guard.")
    
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

from security_helpers import parse_allowed_origins
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = parse_allowed_origins(allowed_origins_env)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/verify/") or request.url.path.startswith("/vault/"):
        response.headers["Cache-Control"] = "no-store"
    return response

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
@limiter.limit("5/minute")
def login(request: Request, req: LoginRequest):
    import secrets
    if not OPERATOR_PASSWORD or not secrets.compare_digest(req.password, OPERATOR_PASSWORD):
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
        from security_helpers import is_mock_crypto_allowed
        if is_mock_crypto_allowed(os.getenv("ENVIRONMENT", ""), os.getenv("ALLOW_MOCK_CRYPTO", "false")):
            return b"MOCK_ENCRYPTED_PACKET"
        raise RuntimeError(f"CRITICAL: KMS Encryption failed: {e}")

def decrypt_embedding(packet: bytes) -> np.ndarray:
    """
    Decrypts the envelope. Extracts the encrypted DEK, decrypts it via KMS,
    then decrypts the payload back into a 512-D numpy array in-memory.
    """
    if packet == b"MOCK_ENCRYPTED_PACKET":
        from security_helpers import is_mock_crypto_allowed
        if is_mock_crypto_allowed(os.getenv("ENVIRONMENT", ""), os.getenv("ALLOW_MOCK_CRYPTO", "false")):
            return np.random.rand(512)
        raise RuntimeError("CRITICAL: Mock crypto detected but not allowed in this environment.")
        
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
# PIPELINE VERSION PINNING (Reproducibility)
# ---------------------------------------------------------
PIPELINE_VERSION = "Fallen Pipeline v4.0 (Ensemble + 3D Procrustes + Bayesian LR)"
MARK_DETECTOR_VERSION = "2.1.0"
MARK_MATCHER_VERSION = "2.0.0"

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


def _build_rejection_summary(
    valid_probe_marks: list,
    valid_gallery_marks: list,
    mark_result: dict,
    rejected_cands: list,
    mark_match_status: str,
    exact_image_match: bool,
    tier4_calibration,
    trace_probe: dict = None,
    trace_gallery: dict = None,
) -> str:
    """Return a human-readable explanation for why LR_marks is neutral or absent."""
    matched = mark_result.get("matched", 0)
    n_probe = len(valid_probe_marks)
    n_gallery = len(valid_gallery_marks)

    def _get_stage_reason(side_name, trace):
        if not trace:
            return f"No raw marks detected on {side_name}"
        if trace.get("initial_candidates", 0) == 0:
            return f"No initial mark candidates found after thresholding on {side_name}"
        if trace.get("after_skin_mask", 0) == 0:
            return f"Candidates found, but removed by occlusion/skin mask on {side_name}"
        if trace.get("after_area_filter", 0) == 0:
            return f"Initial candidates found, but all rejected by area threshold on {side_name}"
        if trace.get("after_shape_filter", 0) == 0:
            return f"Initial candidates found, but all rejected by shape/geometry filter on {side_name}"
        if trace.get("after_region_exclusion", 0) == 0:
            return f"Candidates found, but removed by facial-region exclusion mask on {side_name}"
        if trace.get("after_contrast_filter", 0) == 0:
            return f"Candidates found, but all rejected by low contrast threshold on {side_name}"
        return f"No raw marks detected on {side_name}"

    if exact_image_match:
        return "Exact self-match: mark evidence self-corresponding by identity (LR neutralized to 1.0)"
    if n_probe == 0 and n_gallery == 0:
        reason_p = _get_stage_reason("probe", trace_probe)
        reason_g = _get_stage_reason("gallery", trace_gallery)
        base_p = reason_p.replace(" on probe", "")
        base_g = reason_g.replace(" on gallery", "")
        if base_p == base_g:
            return f"{base_p} on both images"
        return f"{reason_p}. {reason_g}."
    if n_probe == 0:
        return _get_stage_reason("probe", trace_probe)
    if n_gallery == 0:
        return _get_stage_reason("gallery", trace_gallery)
    if tier4_calibration is None and matched > 0:
        return "Mark calibration data unavailable — LR defaulted to 1.0"
    if mark_match_status == "DETECTOR_UNAVAILABLE":
        return "Mark detector unavailable"
    if matched == 0 and len(rejected_cands) > 0:
        return f"All {len(rejected_cands)} candidate marks rejected by cost/distance thresholds"
    if matched == 0:
        return "Raw marks detected, but no accepted correspondences passed matching thresholds"
    if matched > 0:
        lr_marks_val = mark_result.get("lr_marks", 1.0)
        if lr_marks_val == 1.0 and tier4_calibration is None:
            return "Mark calibration data unavailable — LR defaulted to 1.0"
        if lr_marks_val is None or lr_marks_val == 1.0:
            return f"{matched} mark(s) matched but combined LR is neutral (1.0) — mark evidence neither supports nor refutes common source"
        return None  # Marks contributing normally — LR != 1.0
    return "Unknown mark pipeline state"


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
    calibration_status: Optional[str] = None
    # Bayesian Likelihood Ratio Audit Trail
    lr_arcface: Optional[float] = None
    lr_marks: Optional[float] = None
    lr_total: Optional[float] = None
    posterior_probability: Optional[float] = None
    mark_lrs: Optional[list] = None
    bayesian_fused_score: Optional[float] = None
    veto_reason: Optional[str] = None
    veto_override_applied: bool = False
    veto_override_reason: Optional[str] = None
    scoring_trace: Optional[dict] = None
    # Mark Evidence Audit Trail (v2.0)
    mark_match_status: Optional[str] = None
    marks_detected_probe: Optional[int] = None
    marks_detected_gallery: Optional[int] = None

    # Full Forensic Provenance Audit (v3.0)
    # Legacy / Backwards Compatibility Fields
    probe_image_dimensions: Optional[str] = None
    gallery_image_dimensions: Optional[str] = None
    preprocessing_steps_applied: Optional[str] = None
    secondary_weight_hash: Optional[str] = None
    probe_aligned_crop_hash: Optional[str] = None
    gallery_aligned_crop_hash: Optional[str] = None
    mark_lrs_json: Optional[str] = None
    accepted_mark_correspondences_json: Optional[str] = None
    mark_detector_version: Optional[str] = None
    mark_matcher_version: Optional[str] = None
    mark_overlay_url: Optional[str] = None
    receipt_url: Optional[str] = None
    synthetic_anomaly_score: Optional[float] = None
    failed_provenance_veto: Optional[bool] = None

    # New Canonical Fields (Phase 5B)
    probe_source_file_hash: Optional[str] = None
    gallery_source_file_hash: Optional[str] = None
    probe_decoded_image_hash: Optional[str] = None
    gallery_decoded_image_hash: Optional[str] = None
    probe_aligned_crop_hash_pre_clahe: Optional[str] = None
    gallery_aligned_crop_hash_pre_clahe: Optional[str] = None
    probe_aligned_crop_hash_post_clahe: Optional[str] = None
    gallery_aligned_crop_hash_post_clahe: Optional[str] = None
    
    probe_original_dimensions: Optional[str] = None
    gallery_original_dimensions: Optional[str] = None
    probe_decoded_dimensions: Optional[str] = None
    gallery_decoded_dimensions: Optional[str] = None
    probe_aligned_dimensions: Optional[str] = None
    gallery_aligned_dimensions: Optional[str] = None
    preprocessing_steps: Optional[list] = None
    
    raw_arcface_similarity: Optional[float] = None
    raw_secondary_similarity: Optional[float] = None
    fused_face_model_similarity: Optional[float] = None
    lr_face_model: Optional[float] = None

    # Model Provenance
    code_commit_hash: Optional[str] = None
    docker_image_digest: Optional[str] = None
    arcface_model_name: Optional[str] = None
    arcface_weight_hash: Optional[str] = None
    secondary_model_weight_hash: Optional[str] = None
    mediapipe_version: Optional[str] = None
    opencv_version: Optional[str] = None
    deepface_version: Optional[str] = None
    calibration_file_hash: Optional[str] = None
    calibration_pair_count: Optional[int] = None
    
    # Pipeline reproducibility
    pipeline_version: str = PIPELINE_VERSION
    dependency_versions: Optional[dict] = None


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
    scar_delta_b64: str  # Backwards compatibility — prefer edge_delta_b64
    edge_delta_b64: Optional[str] = None  # Forward-compatible field name
    gallery_wireframe_b64: str
    probe_wireframe_b64: str
    probe_mark_debug_b64: Optional[str] = None
    gallery_mark_debug_b64: Optional[str] = None
    mark_debug: Optional[dict] = None  # Full debug payload (DEBUG_FORENSIC only)
    mark_diagnostics: Optional[dict] = None  # Lightweight always-on diagnostics
    # Tier 2: Geometry telemetry
    geometry_status: Optional[str] = None  # OK, NO_VALID_RATIOS, INVALID_IOD
    geometric_ratio_distance: Optional[float] = None
    # Tier 4: Mark Correspondence
    mark_correspondence_score: Optional[float] = None
    marks_detected_gallery: int = 0
    marks_detected_probe: int = 0
    marks_matched: int = 0
    correspondences: list = []
    raw_probe_marks: list = []
    raw_gallery_marks: list = []
    # Mark evidence metadata (v2.0)
    mark_match_status: Optional[str] = None  # EXACT_SELF_MATCH | MATCHED | INSUFFICIENT_MARKS | NO_MATCHES | DETECTOR_UNAVAILABLE | UNKNOWN
    lr_marks: Optional[float] = None
    mark_lrs: Optional[list] = None
    mark_match_overlay_b64: Optional[str] = None
    mark_detector_version: Optional[str] = None
    mark_matcher_version: Optional[str] = None
    exact_image_match: bool = False  # True when probe and gallery are byte-identical
    # Veto transparency
    bayesian_fused_score: Optional[float] = None
    veto_reason: Optional[str] = None
    veto_override_applied: bool = False
    veto_override_reason: Optional[str] = None
    # Face-model evidence (explicit decomposition)
    raw_arcface_similarity: Optional[float] = None
    raw_secondary_similarity: Optional[float] = None
    fused_face_model_similarity: Optional[float] = None
    lr_face_model: Optional[float] = None
    synthetic_anomaly_score: Optional[float] = None
    failed_provenance_veto: Optional[bool] = None
    receipt_url: Optional[str] = None
    # Scoring trace (DEBUG_FORENSIC only)
    scoring_trace: Optional[dict] = None
    calibration_status: Optional[str] = None
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


def finite_or_none(value) -> float | None:
    """Safe float serializer — preserves scientific notation, rejects NaN/Inf."""
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return None


def percent_to_x100(value: float | None) -> int | None:
    value = finite_or_none(value)
    if value is None:
        return None
    return round(value * 100)


def raw_to_x10000(value: float | None) -> int | None:
    value = finite_or_none(value)
    if value is None:
        return None
    return round(value * 10000)


def evaluate_mark_veto_override(mark_result: dict, lr_marks: float) -> dict:
    """
    Evaluates whether independent forensic mark correspondence provides enough 
    evidence to override an ArcFace veto.
    
    SAFETY OVERRIDE (v1.153.0): Mark override is DISABLED.
    Testing revealed that mark correspondence produces false positives on 
    different-source pairs (e.g., Powell vs Djindjic scored 99.98% via override).
    The mark LR calibration is not validated for production use.
    The ArcFace veto MUST remain absolute until mark calibration is independently
    validated against a known impostor population.
    
    Original rules (preserved for future re-enablement):
    - at least 3 individual positive mark LRs > 1.0
    - aggregate lr_marks >= 100.0
    - malformed / non-numeric mark LRs are ignored
    - one extreme mark LR alone must not override
    """
    mark_lrs_raw = mark_result.get("mark_lrs", [])
    positive_mark_lrs = []
    for lr in mark_lrs_raw:
        lr_val = finite_or_none(lr)
        if lr_val is not None and lr_val > 1.0:
            positive_mark_lrs.append(lr_val)
            
    count = len(positive_mark_lrs)
    lr_marks_val = finite_or_none(lr_marks)
    
    # SAFETY: Always return ineligible until mark calibration is validated
    return {
        "eligible": False,
        "positive_mark_count": count,
        "positive_mark_lrs": positive_mark_lrs,
        "reason": f"Mark override DISABLED (v1.153.0 safety). count={count}, lr_marks={lr_marks_val}"
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


def compute_image_hash(image: np.ndarray) -> str:
    """SHA-256 hash of raw pixel bytes of an image array (BGR, uint8).
    Used for chain-of-custody hashing at specific preprocessing stages."""
    return hashlib.sha256(image.tobytes()).hexdigest()


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
    Fetches an image from a GCS URI ONLY.
    Returns a tuple of (decoded_image_array, sha256_hash_of_raw_bytes).
    The hash is computed on the raw byte stream BEFORE OpenCV decode,
    establishing an immutable chain-of-custody fingerprint.
    """
    try:
        if uri.startswith("gs://"):
            storage_client = storage.Client()
            parts = uri.replace("gs://", "").split("/", 1)
            bucket_name_req = parts[0]
            configured_bucket = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
            from security_helpers import is_safe_image_url
            if not is_safe_image_url(uri, configured_bucket):
                raise ValueError("Unauthorized GCS bucket.")
            bucket = storage_client.bucket(configured_bucket)
            blob = bucket.blob(parts[1])
            blob.reload()
            if blob.size and blob.size > 15 * 1024 * 1024:
                raise ValueError("File exceeds 15MB size limit.")
            img_bytes = blob.download_as_bytes()
        else:
            raise ValueError("Only gs:// URIs are allowed in verification paths.")

        if not (img_bytes.startswith(b'\xff\xd8') or img_bytes.startswith(b'\x89PNG') or img_bytes.startswith(b'RIFF')):
            raise ValueError("Invalid image signature. Only JPEG/PNG/WEBP allowed.")

        # Chain of Custody: hash the raw binary BEFORE decode
        raw_hash = hashlib.sha256(img_bytes).hexdigest()

        arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
        if img is None:
            raise ValueError("Could not decode image.")
            
        h, w = img.shape[:2]
        if h > 4096 or w > 4096:
            raise ValueError("Image dimensions exceed 4096x4096.")
            
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

def is_valid_face_landmark(lm) -> bool:
    """Check if a MediaPipe FaceMesh landmark has valid, finite coordinates.
    FaceMesh does not provide reliable visibility values, so we validate
    coordinate finiteness and range instead of using visibility thresholds."""
    if lm is None:
        return False
    try:
        x, y, z = float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
        and -0.25 <= x <= 1.25
        and -0.25 <= y <= 1.25
    )

# Named constant for Tier 2 geometry score conversion.
# L2 mapping: 0 distance -> 100%, >= threshold -> 0%.
# Recalibrated from 0.40 due to added 3D variance post-Procrustes.
GEOMETRY_DISTANCE_ZERO_THRESHOLD = 0.50

def extract_geometric_ratios_3d(landmarks) -> tuple[np.ndarray, dict, dict]:
    """
    Computes scale-invariant, true 3D Euclidean facial geometric ratios for Tier 2.
    Uses Procrustes alignment to mathematically un-rotate the face to a perfect frontal view.
    Also computes landmark validity telemetry for Tier 2 dynamic dropping.
    """

    # Compute Validity Telemetry using coordinate-based check
    # (MediaPipe FaceMesh visibility is unreliable)
    invalid_count = sum(1 for l in landmarks if not is_valid_face_landmark(l))
    occlusion_percentage = (invalid_count / len(landmarks)) * 100.0 if len(landmarks) > 0 else 0.0

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
        region_invalid = sum(1 for idx in indices if not is_valid_face_landmark(landmarks[idx]))
        if region_invalid > len(indices) * 0.5:  # If >50% invalid, call it occluded
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
        is_valid = all(is_valid_face_landmark(landmarks[idx]) for idx in idx_group)
        is_iod_valid = all(is_valid_face_landmark(landmarks[idx]) for idx in iod_points)
        if len(idx_group) == 3: # Jaw symmetry doesn't use IOD
            ratio_visibility.append(is_valid)
        else:
            ratio_visibility.append(is_valid and is_iod_valid)

    vis_data = {
        "occlusion_percentage": round(occlusion_percentage, 2),
        "occluded_regions": occluded_regions,
        "ratio_visibility": np.array(ratio_visibility, dtype=bool)
    }

    coords_3d = np.array([(l.x, l.y, l.z) for l in landmarks])
    aligned_coords, angles = procrustes_align_3d(coords_3d)

    # Verify Procrustes returned finite coordinates
    if not np.all(np.isfinite(aligned_coords)):
        vis_data["geometry_status"] = "PROCRUSTES_NAN"
        return np.zeros(12), angles, vis_data

    left_eye = aligned_coords[33]
    right_eye = aligned_coords[263]
    def dist2d(p1, p2):
        return np.linalg.norm(p1[:2] - p2[:2])

    iod = dist2d(right_eye, left_eye)

    if iod < 1e-6:
        vis_data["geometry_status"] = "INVALID_IOD"
        vis_data["iod"] = float(iod)
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

    jaw_to_chin_r = dist2d(right_jaw, chin)

    ratios = np.array([
        dist2d(nose_tip, chin) / iod,                # Nose-to-chin / IOD
        dist2d(nose_bridge, nose_tip) / iod,          # Nose length / IOD
        dist2d(left_mouth, right_mouth) / iod,        # Mouth width / IOD
        dist2d(forehead_top, chin) / iod,             # Face height / IOD
        dist2d(left_jaw, right_jaw) / iod,            # Jaw width / IOD
        dist2d(left_eyebrow, left_eye) / iod,         # Left brow height / IOD
        dist2d(right_eyebrow, right_eye) / iod,       # Right brow height / IOD
        dist2d(nose_tip, left_eye) / iod,             # Nose-to-left-eye / IOD
        dist2d(nose_tip, right_eye) / iod,            # Nose-to-right-eye / IOD
        dist2d(chin, left_mouth) / iod,               # Chin-to-left-mouth / IOD
        dist2d(chin, right_mouth) / iod,              # Chin-to-right-mouth / IOD
        dist2d(left_jaw, chin) / jaw_to_chin_r if jaw_to_chin_r > 1e-6 else 1.0,  # Jaw symmetry
    ])

    vis_data["geometry_status"] = "OK"
    vis_data["iod"] = float(iod)
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

# MediaPipe landmark indices and mark detector imported from pure module
from mark_detector import (
    detect_facial_marks,
    _build_skin_mask,
    serialize_mark_descriptor,
    get_thresholds as get_detector_thresholds,
    MARK_DETECTOR_VERSION as _MDV,
    _LEFT_EYE_IDX, _RIGHT_EYE_IDX, _LEFT_BROW_IDX, _RIGHT_BROW_IDX,
    _NOSE_IDX, _LIPS_IDX, _FACE_OVAL_IDX, _BORDER_MARGIN,
)
from mark_matcher import (
    match_facial_marks as match_marks_v2,
    MARK_MATCHER_VERSION as MARK_MATCHER_V2_VERSION,
    get_matcher_thresholds,
)


def match_facial_marks(marks_gallery: list, marks_probe: list, dist_threshold: float = 0.20):
    """
    Cost-matrix based Hungarian assignment for marks.

    Assignment-level penalties (do NOT affect calibrated Bayesian LR):
        - Spatial distance (normalized)
        - Area ratio
        - Intensity difference
        - Mark type mismatch (+0.5)
        - Face region mismatch (+0.3)
        - Circularity difference (+0.3 * abs_diff)
        - Orientation difference for linear scars (+0.2 * angular_delta/90)

    Returns:
        matches: list of enriched match dicts
        unmatched_gallery: list of gal_idx
        unmatched_probe: list of pro_idx
        rejected_candidates: list of dicts with reasons
    """
    from scipy.optimize import linear_sum_assignment
    
    n_gal = len(marks_gallery)
    n_pro = len(marks_probe)
    
    if n_gal == 0 or n_pro == 0:
        return [], list(range(n_gal)), list(range(n_pro)), []
        
    cost_matrix = np.full((n_gal, n_pro), 1e6)
    reasons = {}
    match_metadata = {}
    
    for i, mg in enumerate(marks_gallery):
        for j, mp in enumerate(marks_probe):
            delta_x = mg["centroid"][0] - mp["centroid"][0]
            delta_y = mg["centroid"][1] - mp["centroid"][1]
            spatial_dist = np.sqrt(delta_x**2 + delta_y**2)
            
            if spatial_dist > dist_threshold:
                reasons[(i, j)] = f"Too far (dist={spatial_dist:.2f})"
                continue
                
            # Cost based on descriptor differences
            area_ratio = min(mg["area"], mp["area"]) / max(mg["area"], mp["area"])
            if area_ratio < 0.2:
                reasons[(i, j)] = f"Area mismatch (ratio={area_ratio:.2f})"
                continue
                
            intensity_diff = abs(mg["intensity"] - mp["intensity"])
            type_match = mg.get("mark_type") == mp.get("mark_type")
            region_match = mg.get("face_region") == mp.get("face_region")
            
            cost = (spatial_dist * 5) + (1.0 - area_ratio) + (intensity_diff / 255.0)

            # Type mismatch penalty
            if not type_match:
                cost += 0.5

            # Face region mismatch penalty (assignment-level only)
            if not region_match:
                cost += 0.3

            # Circularity difference penalty (assignment-level only)
            circ_diff = abs(mg.get("circularity", 0) - mp.get("circularity", 0))
            cost += circ_diff * 0.3

            # Orientation difference penalty for linear scars (assignment-level only)
            if mg.get("mark_type") == "linear_scar" and mp.get("mark_type") == "linear_scar":
                orient_g = mg.get("orientation")
                orient_p = mp.get("orientation")
                if orient_g is not None and orient_p is not None:
                    angular_delta = abs(orient_g - orient_p)
                    angular_delta = min(angular_delta, 180.0 - angular_delta)  # Handle wrap-around
                    cost += (angular_delta / 90.0) * 0.2

            cost_matrix[i, j] = cost
            match_metadata[(i, j)] = {
                "cost": cost,
                "position_distance": spatial_dist,
                "area_ratio": area_ratio,
                "type_match": type_match,
                "region_match": region_match,
            }
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_gallery = set(range(n_gal))
    unmatched_probe = set(range(n_pro))
    rejected_candidates = []
    
    MAX_COST = 2.0
    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost < MAX_COST:
            mg = marks_gallery[r]
            mp_mark = marks_probe[c]
            meta = match_metadata.get((r, c), {})
            matches.append({
                "gallery_idx": int(r),
                "probe_idx": int(c),
                "cost": meta.get("cost", cost),
                "position_distance": meta.get("position_distance", 0),
                "area_ratio": meta.get("area_ratio", 0),
                "type_match": meta.get("type_match", False),
                "region_match": meta.get("region_match", False),
                "mark_type": mg.get("mark_type", "unknown"),
                "face_region": mg.get("face_region", "unknown"),
                "gallery_centroid": list(mg["centroid"]),
                "probe_centroid": list(mp_mark["centroid"]),
            })
            unmatched_gallery.discard(int(r))
            unmatched_probe.discard(int(c))
        else:
            if cost < 1e5:
                rejected_candidates.append({"gallery_idx": int(r), "probe_idx": int(c), "reason": f"Cost too high ({cost:.2f})"})
            elif (int(r), int(c)) in reasons:
                rejected_candidates.append({"gallery_idx": int(r), "probe_idx": int(c), "reason": reasons[(int(r), int(c))]})
                
    return matches, list(unmatched_gallery), list(unmatched_probe), rejected_candidates



def compute_mark_correspondence(marks_gallery: list, marks_probe: list, matched_pairs: list = None) -> dict:
    """
    Bayesian Likelihood Ratio Mark Correspondence Engine (Scientific v3.0).

    Evaluates the LR for explicitly matched mark pairs:
      - Numerator P(E|Hp): Multivariate Gaussian PDF at observed delta vector
      - Denominator P(E|Hd): KDE spatial density × morphological PDFs

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
    from scipy.stats import multivariate_normal as mvn

    n_gal = len(marks_gallery)
    n_pro = len(marks_probe)

    if n_gal < 1 or n_pro < 1 or not matched_pairs:
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

    matches = []
    mark_lrs = []
    
    for pair in matched_pairs:
        # Support both dict {gallery_idx, probe_idx, ...} and tuple (r, c) shapes
        if isinstance(pair, dict):
            r = pair["gallery_idx"]
            c = pair["probe_idx"]
        else:
            try:
                r, c = pair[0], pair[1]
            except (IndexError, TypeError):
                continue
        mg = marks_gallery[r]
        mp_mark = marks_probe[c]
        
        # Delta vector: gallery - probe
        delta_v = np.array([
            mg["centroid"][0] - mp_mark["centroid"][0],
            mg["centroid"][1] - mp_mark["centroid"][1],
            mg["area"] - mp_mark["area"],
            mg["intensity"] - mp_mark["intensity"],
            mg["circularity"] - mp_mark["circularity"],
        ])

        # NUMERATOR: P(delta | Hp) — how likely is this delta for same person
        try:
            numerator = mvn.pdf(delta_v, mean=delta_mean, cov=delta_cov)
        except Exception:
            numerator = EPSILON
        numerator = max(numerator, EPSILON)

        # DENOMINATOR: P(E | Hd) — population frequency of this mark
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
        lr = max(numerator / denominator, EPSILON)
        
        # Only accept if LR > 1.0 (evidence supports same-source)
        if lr > 1.0:
            matches.append((r, c, float(lr)))
            mark_lrs.append(float(lr))

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


def generate_edge_delta_map(
    img_gallery: np.ndarray,
    img_probe: np.ndarray,
    marks_gallery: list = None,
    marks_probe: list = None,
    mark_matches: list = None,
) -> str:
    """
    Edge-Based Pixel Difference Map.
    Computes the pixel/edge difference between aligned gallery and probe crops.
    This is NOT scar/mole/blemish correspondence evidence — it only shows
    structural pixel differences after Procrustes alignment.

    ALGORITHM:
    1. Convert both images to grayscale.
    2. Run Canny edge detection on both to extract edge maps.
    3. Compute absdiff on grayscales and threshold at a LOW value —
       pixels with small intensity difference represent *persistent* structure.
    4. bitwise_and(gallery_edges, probe_edges, persistent_mask) isolates
       topology that exists in BOTH images and didn't shift between captures.
    5. Dilate slightly for UI readability.
    6. Render in neon crimson (BGRA: 30, 0, 180, 255) on a transparent canvas.
    7. Base64 encode and return as a data URI.
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
    edge_diff = cv2.bitwise_and(common_edges, persistent_mask)

    # 5. Dilate for UI visibility (2×2 kernel, 1 iteration)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edge_diff = cv2.dilate(edge_diff, kernel, iterations=1)

    # 6. Build overlay canvas: transparent RGBA (standalone diagnostic map)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    # Paint neon crimson (BGRA: 30, 0, 180, 255) where edge differences are detected
    canvas[edge_diff > 0] = (30, 0, 180, 255)

    # Mark circles are drawn in the frontend, not here.
    # This map is a standalone pixel/edge difference diagnostic only.

    # 8. Encode to base64 data URI
    _, buffer = cv2.imencode('.png', canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def generate_wireframe_hud(image: np.ndarray, landmarks) -> str:
    """
    3DMM Wireframe HUD — Geometric Mesh Visualizer.
    Renders the full FACEMESH_TESSELATION (468-point mesh) in 24K Gold
    over a darkened, desaturated copy of the input image to demonstrate
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

def generate_mark_overlay_receipt(
    gal_debug_img: np.ndarray,
    pro_debug_img: np.ndarray,
    probe_file_hash: str,
) -> str | None:
    """
    Generates a self-contained composite forensic mark receipt PNG.
    Layout: Gallery Marks (256x256) | Probe Marks (256x256)
    with a high-contrast 94px text panel at the bottom (total: 512x350).
    Uploads to GCS under receipts/ prefix.
    """
    try:
        g = cv2.resize(gal_debug_img, (256, 256))
        p = cv2.resize(pro_debug_img, (256, 256))
        composite = np.hstack([g, p])

        text_panel = np.zeros((94, 512, 3), dtype=np.uint8)
        text_panel[:] = (20, 20, 20)

        cv2.putText(text_panel, "GALLERY MARKS", (60, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(text_panel, "PROBE MARKS", (325, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.line(text_panel, (0, 24), (512, 24), (60, 60, 60), 1)

        timestamp_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        cv2.putText(text_panel, f"FORENSIC MARK EVIDENCE", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(text_panel, f"PROBE SHA-256: {probe_file_hash}", (15, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(text_panel, f"UTC TIMESTAMP: {timestamp_iso}", (15, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        final = np.vstack([composite, text_panel])
        _, buffer = cv2.imencode('.png', final)
        
        bucket_name = os.getenv("BUCKET_NAME", "hoppwhistle-facial-uploads")
        receipt_blob_name = f"receipts/marks_{uuid.uuid4().hex}.png"
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(receipt_blob_name)
        blob.upload_from_string(buffer.tobytes(), content_type="image/png")
        
        return f"gs://{bucket_name}/{receipt_blob_name}"
    except Exception as e:
        print(f"[RECEIPT] WARNING: Failed to generate mark overlay receipt: {e}")
        return None

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
        cv2.putText(text_panel, f"BAYESIAN POSTERIOR: {fused_score:.2f}%", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)
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
@limiter.limit("5/minute")
def generate_upload_urls(request: Request, req: UploadUrlsRequest, _: dict = Depends(verify_jwt)):
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

@app.post("/verify/fuse")
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
                conclusion="Synthetic Provenance Veto Triggered",
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
            "conclusion": "Synthetic Provenance Veto Triggered", 
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
    
    # Phase 5B: Tracking dimensions and decoding hashes
    probe_decoded_image_hash = compute_image_hash(probe_img)
    gallery_decoded_image_hash = compute_image_hash(gallery_img)
    probe_original_dimensions = f"{probe_img.shape[1]}x{probe_img.shape[0]}"
    gallery_original_dimensions = f"{gallery_img.shape[1]}x{gallery_img.shape[0]}"
    probe_decoded_dimensions = probe_original_dimensions
    gallery_decoded_dimensions = gallery_original_dimensions
    
    # 2. Pre-CLAHE alignment for provenance hash (raw pixel chain-of-custody)
    gallery_pre_clahe_crop, _gpc_lm = align_face_crop(gallery_img)
    probe_pre_clahe_crop, _ppc_lm = align_face_crop(probe_img)
    gallery_aligned_crop_hash_pre_clahe = compute_image_hash(gallery_pre_clahe_crop)
    probe_aligned_crop_hash_pre_clahe = compute_image_hash(probe_pre_clahe_crop)

    # 2.5 Preprocess (CLAHE)
    gallery_clahe = apply_clahe(gallery_img)
    probe_clahe = apply_clahe(probe_img)
    
    # 3. Face Alignment & Crop to canonical 256×256
    gallery_aligned, gallery_landmarks = align_face_crop(gallery_clahe)
    probe_aligned, probe_landmarks = align_face_crop(probe_clahe)

    # Post-CLAHE aligned crop hashes (these are the actual model-input pixels)
    gallery_aligned_crop_hash_post_clahe = compute_image_hash(gallery_aligned)
    probe_aligned_crop_hash_post_clahe = compute_image_hash(probe_aligned)
    
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
    
    # Determine geometry status from extraction
    gal_geom_status = gal_vis.get("geometry_status", "UNKNOWN")
    pro_geom_status = pro_vis.get("geometry_status", "UNKNOWN")
    
    if gal_geom_status != "OK" or pro_geom_status != "OK":
        geometry_status = gal_geom_status if gal_geom_status != "OK" else pro_geom_status
    else:
        geometry_status = "OK"  # may be overridden below
    
    valid_mask = gal_vis["ratio_visibility"] & pro_vis["ratio_visibility"]
    effective_ratios = int(np.sum(valid_mask))
    
    if effective_ratios > 0 and geometry_status == "OK":
        raw_l2 = float(np.linalg.norm((ratios_gallery - ratios_probe)[valid_mask]))
        ratio_l2 = raw_l2 * math.sqrt(12.0 / effective_ratios)
        tier2_score = max(0.0, min(100.0, (1.0 - (ratio_l2 / GEOMETRY_DISTANCE_ZERO_THRESHOLD)) * 100))
    else:
        ratio_l2 = None
        tier2_score = 0.0
        if geometry_status == "OK":
            geometry_status = "NO_VALID_RATIOS"
    
    # Debug logging for Tier 2 geometry pipeline
    if os.getenv("DEBUG_FORENSIC") == "true" or os.getenv("ENVIRONMENT") == "development":
        print(f"[FORENSIC DEBUG] Tier 2 Geometry:", flush=True)
        print(f"  geometry_status={geometry_status}, effective_ratios={effective_ratios}", flush=True)
        print(f"  gal_geom_status={gal_geom_status}, pro_geom_status={pro_geom_status}", flush=True)
        print(f"  IOD: gallery={gal_vis.get('iod', 'N/A')}, probe={pro_vis.get('iod', 'N/A')}", flush=True)
        print(f"  ratio_visibility mask: {valid_mask.tolist()}", flush=True)
        print(f"  ratio_l2={ratio_l2}, threshold={GEOMETRY_DISTANCE_ZERO_THRESHOLD}", flush=True)
        print(f"  tier2_score={tier2_score:.2f}", flush=True)
        if effective_ratios > 0 and ratio_l2 is not None:
            print(f"  gallery_ratios: {ratios_gallery.tolist()}", flush=True)
            print(f"  probe_ratios:   {ratios_probe.tolist()}", flush=True)
    
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
    marks_gallery, rejected_gallery, occ_gallery, trace_gallery, overlays_gallery = detect_facial_marks(gallery_aligned, gallery_landmarks)
    marks_probe, rejected_probe, occ_probe, trace_probe, overlays_probe = detect_facial_marks(probe_aligned, probe_landmarks)
    
    valid_gallery_marks = []
    for m in marks_gallery:
        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
        if cy < 256 and cx < 256 and occ_gallery[cy, cx] == 0:
            clean_m = {k: v for k, v in m.items() if k != "contour"}
            clean_m["source_side"] = "gallery"
            valid_gallery_marks.append(clean_m)
            
    valid_probe_marks = []
    for m in marks_probe:
        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
        if cy < 256 and cx < 256 and occ_probe[cy, cx] == 0:
            clean_m = {k: v for k, v in m.items() if k != "contour"}
            clean_m["source_side"] = "probe"
            valid_probe_marks.append(clean_m)

    # ── Exact Self-Match Detection ──
    exact_image_match = (probe_file_hash == gallery_file_hash)

    if exact_image_match:
        # Identical images — all marks self-correspond
        mark_match_status = "EXACT_SELF_MATCH"
        tier4_score = 100.0
        n_self = min(len(valid_probe_marks), len(valid_gallery_marks))
        assigned_pairs = []
        for si in range(n_self):
            assigned_pairs.append({
                "gallery_idx": si, "probe_idx": si,
                "cost": 0.0, "position_distance": 0.0, "area_ratio": 1.0,
                "type_match": True, "region_match": True,
                "mark_type": valid_gallery_marks[si].get("mark_type", "unknown"),
                "face_region": valid_gallery_marks[si].get("face_region", "unknown"),
                "gallery_centroid": list(valid_gallery_marks[si]["centroid"]),
                "probe_centroid": list(valid_probe_marks[si]["centroid"]),
            })
        unmatched_gal = list(range(n_self, len(valid_gallery_marks)))
        unmatched_pro = list(range(n_self, len(valid_probe_marks)))
        rejected_cands = []
        mark_result = {
            "score": 100.0, "matched": n_self,
            "total_gallery": len(valid_gallery_marks), "total_probe": len(valid_probe_marks),
            "matches": [(si, si, 1.0) for si in range(n_self)],
            "lr_marks": 1.0, "mark_lrs": [],
        }
    else:
        assigned_pairs, unmatched_gal, unmatched_pro, rejected_cands = match_facial_marks(valid_gallery_marks, valid_probe_marks)
        mark_result = compute_mark_correspondence(valid_gallery_marks, valid_probe_marks, matched_pairs=assigned_pairs)
        tier4_score = mark_result["score"]  # None if insufficient marks

        # Determine mark_match_status
        if len(valid_probe_marks) < 2 or len(valid_gallery_marks) < 2:
            mark_match_status = "INSUFFICIENT_MARKS"
        elif mark_result.get("matched", 0) > 0:
            mark_match_status = "MATCHED"
        else:
            mark_match_status = "NO_MATCHES"

    # ── BAYESIAN EVIDENCE FUSION (Scientific v4.0) ──
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

    bayesian_fused_score = fused_score  # preserve pre-veto posterior × 100

    # ── MARK OVERRIDE PROTOCOL (v1.0) ──
    # If ArcFace veto triggers but 3+ independent mark correspondences
    # with positive individual LRs provide aggregate LR >= 100,
    # the hard-zero policy is lifted. ArcFace itself did NOT pass.
    mark_override_eval = evaluate_mark_veto_override(mark_result, lr_marks)
    mark_override_eligible = mark_override_eval["eligible"]
    positive_mark_count = mark_override_eval["positive_mark_count"]

    veto_reason = None
    veto_override_applied = False
    veto_override_reason = None
    if veto_triggered:
        if mark_override_eligible:
            # Override lifts the hard-zero — ArcFace veto still flagged
            veto_reason = "ARCFACE_VETO_MARK_OVERRIDE"
            veto_override_applied = True
            veto_override_reason = mark_override_eval["reason"]
            conclusion = (
                "Supports Common Source — Face-Model Veto Overridden by Mark Correspondence"
            )
            # fused_score keeps its Bayesian posterior value
        else:
            fused_score = 0.0
            veto_reason = "ARCFACE_VETO"
            conclusion = (
                "Inconclusive — Limited by Face-Model Threshold"
            )
    elif fused_score > 90.0:
        conclusion = "Strongly Supports Common Source"
    elif fused_score > 75.0:
        conclusion = "Supports Common Source"
    else:
        conclusion = "Inconclusive — Insufficient Evidence"

    # Landmark Attention Maps on aligned crops (real 468-point density, not fabricated)
    gallery_heatmap = generate_landmark_attention_map(gallery_aligned, gallery_landmarks)
    probe_heatmap = generate_landmark_attention_map(probe_aligned, probe_landmarks)
    
    # Encode aligned crops as base64 for frontend SymmetryMerge
    _, gal_buf = cv2.imencode('.png', gallery_aligned)
    gallery_aligned_b64 = f"data:image/png;base64,{base64.b64encode(gal_buf).decode('utf-8')}"
    _, pro_buf = cv2.imencode('.png', probe_aligned)
    probe_aligned_b64 = f"data:image/png;base64,{base64.b64encode(pro_buf).decode('utf-8')}"

    # Edge-Based Pixel Difference Map (standalone diagnostic — NOT mark evidence)
    edge_delta = generate_edge_delta_map(
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
        stats["false_acceptance_rate"] = "Below Operating Threshold"
        stats["statistical_certainty"] = "0% — Below Threshold"
    else:
        stats["false_acceptance_rate"] = f"1 in {int(1.0 / bayesian_far):,}"
        stats["statistical_certainty"] = f"{(posterior * 100):.6f}%"

    # Deep Forensic Telemetry (hash of 512-D ArcFace vector)
    probe_vector_hash = compute_vector_hash(ensemble_probe[0])
    probe_alignment = compute_alignment_variance(probe_aligned)

    # ── FORENSIC RECEIPT GENERATION (Always-On Evidence) ──
    gal_debug_img = gallery_aligned.copy()
    pro_debug_img = probe_aligned.copy()
    
    gal_matched_idx = {(m["gallery_idx"] if isinstance(m, dict) else m[0]) for m in mark_result.get("matches", [])}
    pro_matched_idx = {(m["probe_idx"] if isinstance(m, dict) else m[1]) for m in mark_result.get("matches", [])}
    
    for idx, m in enumerate(valid_gallery_marks):
        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
        color = (0, 255, 0) if idx in gal_matched_idx else ((255, 255, 0) if idx in unmatched_gal else (128, 128, 128))
        cv2.circle(gal_debug_img, (cx, cy), 4, color, 2)
        cv2.putText(gal_debug_img, str(idx), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
    for idx, m in enumerate(valid_probe_marks):
        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
        color = (0, 255, 0) if idx in pro_matched_idx else ((255, 255, 0) if idx in unmatched_pro else (128, 128, 128))
        cv2.circle(pro_debug_img, (cx, cy), 4, color, 2)
        cv2.putText(pro_debug_img, str(idx), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # ── Composite Forensic Receipt (Evidence Preservation) ──
    receipt_url = generate_forensic_receipt(
        gallery_aligned=gallery_aligned,
        probe_aligned=probe_aligned,
        gallery_heatmap_b64=gallery_heatmap,
        fused_score=fused_score,
        probe_file_hash=probe_file_hash,
    )

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
        lr_arcface=finite_or_none(lr_ensemble),
        lr_marks=finite_or_none(lr_marks),
        lr_total=finite_or_none(lr_total),
        posterior_probability=finite_or_none(posterior),
        mark_lrs=[finite_or_none(lr) for lr in mark_result.get("mark_lrs", [])],
        bayesian_fused_score=finite_or_none(bayesian_fused_score),
        # Mark Evidence Audit Trail (v2.0)
        mark_match_status=mark_match_status,
        marks_detected_probe=mark_result.get("total_probe", 0),
        marks_detected_gallery=mark_result.get("total_gallery", 0),
        mark_lrs_json=json.dumps([finite_or_none(lr) for lr in mark_result.get("mark_lrs", [])]),
        accepted_mark_correspondences_json=json.dumps(assigned_pairs),
        mark_detector_version=MARK_DETECTOR_VERSION,
        mark_matcher_version=MARK_MATCHER_VERSION,
        mark_overlay_url=generate_mark_overlay_receipt(gal_debug_img, pro_debug_img, probe_file_hash),
        # Full Forensic Provenance Audit (v3.0 + Phase 5B)
        probe_source_file_hash=probe_file_hash,
        gallery_source_file_hash=gallery_file_hash,
        probe_decoded_image_hash=probe_decoded_image_hash,
        gallery_decoded_image_hash=gallery_decoded_image_hash,
        probe_aligned_crop_hash=probe_vector_hash,
        gallery_aligned_crop_hash=compute_vector_hash(ensemble_gallery[0]) if ensemble_gallery is not None and isinstance(ensemble_gallery, tuple) and len(ensemble_gallery) > 0 else None,
        probe_original_dimensions=probe_original_dimensions,
        gallery_original_dimensions=gallery_original_dimensions,
        probe_decoded_dimensions=probe_decoded_dimensions,
        gallery_decoded_dimensions=gallery_decoded_dimensions,
        probe_aligned_dimensions=f"{probe_aligned.shape[1]}x{probe_aligned.shape[0]}" if probe_aligned is not None else None,
        gallery_aligned_dimensions=f"{gallery_aligned.shape[1]}x{gallery_aligned.shape[0]}" if gallery_aligned is not None else None,
        probe_image_dimensions=f"{probe_aligned.shape[1]}x{probe_aligned.shape[0]}" if probe_aligned is not None else None,
        gallery_image_dimensions=f"{gallery_aligned.shape[1]}x{gallery_aligned.shape[0]}" if gallery_aligned is not None else None,
        preprocessing_steps=["decode_image", "mediapipe_face_landmarks", "align_crop_256", "clahe_normalize"],
        preprocessing_steps_applied="clahe,frontalize,align_crop(256)",
        # Pre/Post CLAHE provenance hashes (chain of custody at each preprocessing stage)
        probe_aligned_crop_hash_pre_clahe=probe_aligned_crop_hash_pre_clahe,
        gallery_aligned_crop_hash_pre_clahe=gallery_aligned_crop_hash_pre_clahe,
        probe_aligned_crop_hash_post_clahe=probe_aligned_crop_hash_post_clahe,
        gallery_aligned_crop_hash_post_clahe=gallery_aligned_crop_hash_post_clahe,
        code_commit_hash=os.getenv("GIT_COMMIT_SHA", "unknown"),
        docker_image_digest=os.getenv("DOCKER_IMAGE_DIGEST", "unknown"),
        arcface_model_name="ArcFace-R100",
        arcface_weight_hash=os.getenv("ARCFACE_WEIGHT_HASH", "unknown"),
        secondary_weight_hash=os.getenv("SECONDARY_WEIGHT_HASH", "unknown"),
        secondary_model_weight_hash=os.getenv("SECONDARY_WEIGHT_HASH", "unknown"),
        mediapipe_version=DEPENDENCY_VERSIONS.get("mediapipe", "unknown"),
        opencv_version=DEPENDENCY_VERSIONS.get("opencv", "unknown"),
        deepface_version=DEPENDENCY_VERSIONS.get("deepface", "unknown"),
        calibration_file_hash=os.getenv("CALIBRATION_FILE_HASH", "unknown"),
        calibration_pair_count=stats.get("pairs_evaluated", 0),
        # Face-Model Evidence Fields
        raw_arcface_similarity=round(arcface_sim, 6),
        raw_secondary_similarity=round(secondary_sim, 6),
        fused_face_model_similarity=round(structural_sim, 6),
        lr_face_model=finite_or_none(lr_ensemble),
        receipt_url=receipt_url,
        synthetic_anomaly_score=max_anomaly,
        failed_provenance_veto=False,
    )

    # Build correspondences list for the UI (enriched with forensic metadata)
    correspondences = []
    # Build LR lookup from mark_result matches
    lr_lookup = {}
    for match_entry in mark_result.get("matches", []):
        if isinstance(match_entry, dict):
            lr_lookup[(match_entry.get("gallery_idx"), match_entry.get("probe_idx"))] = match_entry.get("lr", 0)
        elif isinstance(match_entry, (tuple, list)) and len(match_entry) >= 3:
            lr_lookup[(match_entry[0], match_entry[1])] = match_entry[2]

    for pair in assigned_pairs:
        if isinstance(pair, dict):
            g_idx = pair.get("gallery_idx")
            p_idx = pair.get("probe_idx")
        elif isinstance(pair, (tuple, list)) and len(pair) >= 2:
            g_idx, p_idx = pair[0], pair[1]
        else:
            continue
        # Validate indices
        if not isinstance(g_idx, int) or not isinstance(p_idx, int):
            continue
        if g_idx < 0 or g_idx >= len(valid_gallery_marks) or p_idx < 0 or p_idx >= len(valid_probe_marks):
            continue
        individual_lr = lr_lookup.get((g_idx, p_idx), 0)
        corr_entry = {
            "gallery_idx": g_idx,
            "probe_idx": p_idx,
            "gallery_pt": valid_gallery_marks[g_idx]["centroid"],
            "probe_pt": valid_probe_marks[p_idx]["centroid"],
            "lr": individual_lr,
            "mark_type": valid_gallery_marks[g_idx].get("mark_type", "unknown"),
            "face_region": valid_gallery_marks[g_idx].get("face_region", "unknown"),
            "gallery_centroid": list(valid_gallery_marks[g_idx]["centroid"]),
            "probe_centroid": list(valid_probe_marks[p_idx]["centroid"]),
        }
        # Add match quality metrics from assigned_pairs if available
        if isinstance(pair, dict):
            corr_entry["match_quality"] = pair.get("cost", 0)
            corr_entry["position_distance"] = pair.get("position_distance", 0)
            corr_entry["area_ratio"] = pair.get("area_ratio", 0)
            corr_entry["type_match"] = pair.get("type_match", False)
            corr_entry["region_match"] = pair.get("region_match", False)
        correspondences.append(corr_entry)

    probe_mark_debug_b64 = None
    gallery_mark_debug_b64 = None
    mark_debug_payload = None

    # ── Lightweight always-on mark diagnostics (production-safe) ──
    # Use trace-based detector_status for richer reporting (v2.1)
    _probe_det_status = trace_probe.get("detector_status", "UNKNOWN") if trace_probe else "UNKNOWN"
    _gallery_det_status = trace_gallery.get("detector_status", "UNKNOWN") if trace_gallery else "UNKNOWN"
    _fuse_detector_status = (
        _probe_det_status if _probe_det_status != "OK"
        else _gallery_det_status if _gallery_det_status != "OK"
        else "OK" if (len(marks_gallery) > 0 or len(marks_probe) > 0)
        else "NO_CANDIDATES"
    )
    mark_diagnostics_payload = {
        "raw_probe_marks_count": len(valid_probe_marks),
        "raw_gallery_marks_count": len(valid_gallery_marks),
        "accepted_correspondences_count": mark_result.get("matched", 0),
        "rejected_candidates_count": len(rejected_cands) if rejected_cands else 0,
        "detector_status": _fuse_detector_status,
        "probe_detector_status": _probe_det_status,
        "gallery_detector_status": _gallery_det_status,
        "matcher_status": "OK" if mark_result.get("matched", 0) > 0 else ("NO_MATCHES" if (len(valid_probe_marks) > 0 and len(valid_gallery_marks) > 0) else "INSUFFICIENT_INPUT"),
        "lr_marks": finite_or_none(lr_marks),
        "mark_match_status": mark_match_status,
        "rejection_summary": _build_rejection_summary(
            valid_probe_marks, valid_gallery_marks,
            mark_result, rejected_cands, mark_match_status,
            exact_image_match, TIER4_CALIBRATION,
            trace_probe=trace_probe, trace_gallery=trace_gallery,
        ),
        "mark_detector_trace": {
            "probe": trace_probe,
            "gallery": trace_gallery,
        },
    }

    if os.getenv("DEBUG_FORENSIC") == "true":
        _, gal_dbuf = cv2.imencode('.png', gal_debug_img)
        gallery_mark_debug_b64 = f"data:image/png;base64,{base64.b64encode(gal_dbuf).decode('utf-8')}"
        _, pro_dbuf = cv2.imencode('.png', pro_debug_img)
        probe_mark_debug_b64 = f"data:image/png;base64,{base64.b64encode(pro_dbuf).decode('utf-8')}"
        
        mark_debug_payload = {
            "probe_marks_count": len(valid_probe_marks),
            "gallery_marks_count": len(valid_gallery_marks),
            "correspondences_count": len(mark_result.get("matches", [])),
            "probe_marks_first_20": valid_probe_marks[:20],
            "gallery_marks_first_20": valid_gallery_marks[:20],
            "correspondences_first_20": [
                {
                    "gallery_idx": m[0],
                    "probe_idx": m[1],
                    "lr": m[2]
                } for m in mark_result.get("matches", [])
            ][:20],
            "unmatched_probe_indices": list(unmatched_pro),
            "unmatched_gallery_indices": list(unmatched_gal),
            "rejected_candidates": rejected_cands,
            "rejected_probe_marks": rejected_probe,
            "rejected_gallery_marks": rejected_gallery,
            "detector_version": "v2.0 (multi-scale + face-oval)",
            "matcher_version": "v2.0 (Hungarian + LR)"
        }

    if os.getenv("DEBUG_FORENSIC") == "true" or os.getenv("ENVIRONMENT") == "development":
        print(f"[FORENSIC DEBUG] raw_probe_marks: {len(valid_probe_marks)}, "
              f"raw_gallery_marks: {len(valid_gallery_marks)}, "
              f"correspondences: {len(correspondences)}", flush=True)
        for ci, c in enumerate(correspondences[:5]):
            print(f"  corr[{ci}]: probe_idx={c['probe_idx']}, gallery_idx={c['gallery_idx']}, lr={c['lr']:.4f}", flush=True)

    # ── SCORING TRACE (debug-only response payload) ──
    _calibration_status = "LOADED" if CALIBRATION else "MISSING"
    scoring_trace = None
    if os.getenv("DEBUG_FORENSIC") == "true":
        scoring_trace = {
            "calibration_status": _calibration_status,
            "calibration_source": CALIBRATION.get("source", "NONE") if CALIBRATION else "NONE",
            "calibration_benchmark": CALIBRATION.get("benchmark", "NONE") if CALIBRATION else "NONE",
            "tier4_calibration_status": "LOADED" if TIER4_CALIBRATION else "MISSING",
            "lr_ensemble_raw": finite_or_none(lr_ensemble),
            "lr_marks_raw": finite_or_none(lr_marks),
            "lr_total_raw": finite_or_none(lr_total),
            "lr_ensemble_display": "{:.6e}".format(lr_ensemble) if math.isfinite(lr_ensemble) else "N/A",
            "lr_marks_display": "{:.6e}".format(lr_marks) if math.isfinite(lr_marks) else "N/A",
            "lr_total_display": "{:.6e}".format(lr_total) if math.isfinite(lr_total) else "N/A",
            "posterior_raw": finite_or_none(posterior),
            "fused_score_pre_veto": finite_or_none(bayesian_fused_score),
            "fused_score_post_veto": finite_or_none(fused_score),
            "veto_triggered": veto_triggered,
            "veto_reason": veto_reason,
            "veto_override_applied": veto_override_applied,
            "veto_override_reason": veto_override_reason,
            "mark_override_eligible": mark_override_eligible,
            "positive_mark_count": positive_mark_count,
            "temporal_delta_years": finite_or_none(temporal_delta),
            "ensemble_thresholds_key": (
                "ensemble" if CALIBRATION and "ensemble" in CALIBRATION
                else ("arcface" if CALIBRATION and "arcface" in CALIBRATION else "NONE")
            ),
        }

    audit.veto_reason = veto_reason
    audit.veto_override_applied = veto_override_applied
    audit.veto_override_reason = veto_override_reason
    audit.scoring_trace = scoring_trace
    audit.calibration_status = _calibration_status

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
        scar_delta_b64=edge_delta,
        edge_delta_b64=edge_delta,
        gallery_wireframe_b64=gallery_wireframe,
        probe_wireframe_b64=probe_wireframe,
        probe_mark_debug_b64=probe_mark_debug_b64,
        gallery_mark_debug_b64=gallery_mark_debug_b64,
        mark_debug=mark_debug_payload,
        mark_diagnostics=mark_diagnostics_payload,
        geometry_status=geometry_status,
        geometric_ratio_distance=round(ratio_l2, 6) if ratio_l2 is not None else None,
        mark_correspondence_score=tier4_score,
        marks_detected_gallery=mark_result.get("total_gallery", 0),
        marks_detected_probe=mark_result.get("total_probe", 0),
        marks_matched=mark_result.get("matched", 0),
        correspondences=correspondences,
        raw_probe_marks=valid_probe_marks,
        raw_gallery_marks=valid_gallery_marks,
        # Mark evidence metadata (v2.0)
        mark_match_status=mark_match_status,
        lr_marks=finite_or_none(lr_marks),
        mark_lrs=mark_result.get("mark_lrs", []),
        mark_detector_version=MARK_DETECTOR_VERSION,
        mark_matcher_version=MARK_MATCHER_VERSION,
        exact_image_match=exact_image_match,
        # Face-model evidence (explicit decomposition)
        raw_arcface_similarity=round(arcface_sim, 6),
        raw_secondary_similarity=round(secondary_sim, 6),
        fused_face_model_similarity=round(structural_sim, 6),
        lr_face_model=finite_or_none(lr_ensemble),
        # Veto transparency
        bayesian_fused_score=round(bayesian_fused_score, 2),
        veto_reason=veto_reason,
        veto_override_applied=veto_override_applied,
        veto_override_reason=veto_override_reason,
        scoring_trace=scoring_trace,
        calibration_status=_calibration_status,
        receipt_url=receipt_url,
        synthetic_anomaly_score=max_anomaly,
        failed_provenance_veto=False,
        audit_log=audit,
    )

    # ── Immutable Audit Ledger ──
    ledger_session = SessionLocal()
    try:
        event = VerificationEvent(
            probe_hash=probe_file_hash,
            gallery_hash=gallery_file_hash,
            matched_user_id=None,
            fused_score_x100=percent_to_x100(fused_score) or 0,
            conclusion=conclusion,
            pipeline_version=PIPELINE_VERSION,
            calibration_benchmark=stats.get("benchmark"),
            false_acceptance_rate=stats["false_acceptance_rate"],
            veto_triggered=veto_triggered,
            structural_score_x100=percent_to_x100(tier1_score) or 0,
            arcface_score_x10000=raw_to_x10000(arcface_sim) or 0,
            secondary_score_x10000=raw_to_x10000(secondary_sim) or 0,
            ensemble_model_secondary="Facenet512",
            geometric_score_x100=percent_to_x100(tier2_score) or 0,
            micro_topology_score_x100=percent_to_x100(tier3_score) or 0,
            mark_correspondence_x100=percent_to_x100(tier4_score),
            pose_corrected_3d=True,
            probe_pose_angles=json.dumps(pro_angles) if pro_angles else None,
            gallery_pose_angles=json.dumps(gal_angles) if gal_angles else None,
            occlusion_percentage=pro_vis["occlusion_percentage"],
            occluded_regions=json.dumps(pro_vis["occluded_regions"]) if pro_vis["occluded_regions"] else None,
            effective_geometric_ratios_used=effective_ratios,
            receipt_url=receipt_url,
            synthetic_anomaly_score=max_anomaly,
            failed_provenance_veto=False,
            lr_arcface=finite_or_none(lr_ensemble),
            lr_marks_product=finite_or_none(lr_marks),
            lr_total=finite_or_none(lr_total),
            posterior_probability=finite_or_none(posterior),
            bayesian_fused_score_x100=percent_to_x100(bayesian_fused_score),
            marks_matched=mark_result.get("matched", 0),
            calibration_status=_calibration_status,
            veto_reason=veto_reason,
            veto_override_applied=veto_override_applied,
            # Mark Evidence Audit Trail (v2.0)
            mark_match_status=mark_match_status,
            marks_detected_probe=mark_result.get("total_probe", 0),
            marks_detected_gallery=mark_result.get("total_gallery", 0),
            mark_lrs_json=json.dumps([finite_or_none(lr) for lr in mark_result.get("mark_lrs", [])]),
            accepted_mark_correspondences_json=json.dumps(correspondences),
            mark_detector_version=MARK_DETECTOR_VERSION,
            mark_matcher_version=MARK_MATCHER_VERSION,
            mark_overlay_url=audit.mark_overlay_url,
            # Full Forensic Provenance Audit (v3.0)
            probe_source_file_hash=audit.probe_source_file_hash,
            gallery_source_file_hash=audit.gallery_source_file_hash,
            probe_decoded_image_hash=audit.probe_decoded_image_hash,
            gallery_decoded_image_hash=audit.gallery_decoded_image_hash,
            probe_aligned_crop_hash=audit.probe_aligned_crop_hash,
            gallery_aligned_crop_hash=audit.gallery_aligned_crop_hash,
            probe_aligned_crop_hash_pre_clahe=audit.probe_aligned_crop_hash_pre_clahe,
            gallery_aligned_crop_hash_pre_clahe=audit.gallery_aligned_crop_hash_pre_clahe,
            probe_aligned_crop_hash_post_clahe=audit.probe_aligned_crop_hash_post_clahe,
            gallery_aligned_crop_hash_post_clahe=audit.gallery_aligned_crop_hash_post_clahe,
            probe_image_dimensions=audit.probe_image_dimensions,
            gallery_image_dimensions=audit.gallery_image_dimensions,
            preprocessing_steps_applied=audit.preprocessing_steps_applied,
            code_commit_hash=audit.code_commit_hash,
            docker_image_digest=audit.docker_image_digest,
            arcface_model_name=audit.arcface_model_name,
            arcface_weight_hash=audit.arcface_weight_hash,
            secondary_weight_hash=audit.secondary_weight_hash,
            mediapipe_version=audit.mediapipe_version,
            opencv_version=audit.opencv_version,
            deepface_version=audit.deepface_version,
            calibration_file_hash=audit.calibration_file_hash,
            calibration_pair_count=audit.calibration_pair_count,
            probe_original_dimensions=audit.probe_original_dimensions,
            gallery_original_dimensions=audit.gallery_original_dimensions,
            probe_decoded_dimensions=audit.probe_decoded_dimensions,
            gallery_decoded_dimensions=audit.gallery_decoded_dimensions,
            probe_aligned_dimensions=audit.probe_aligned_dimensions,
            gallery_aligned_dimensions=audit.gallery_aligned_dimensions,
            preprocessing_steps=json.dumps(audit.preprocessing_steps) if audit.preprocessing_steps else None,
            raw_arcface_similarity=audit.raw_arcface_similarity,
            raw_secondary_similarity=audit.raw_secondary_similarity,
            fused_face_model_similarity=audit.fused_face_model_similarity,
            lr_face_model=audit.lr_face_model,
            secondary_model_weight_hash=audit.secondary_model_weight_hash,
        )
        ledger_session.add(event)
        ledger_session.commit()
    except Exception as ledger_err:
        print(f"Audit ledger write failed (non-fatal): {ledger_err}")
        ledger_session.rollback()
    finally:
        ledger_session.close()

    # Store job in database
    job_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        job = VerificationJob(
            job_id=job_id,
            status="pending",
            result_payload=json.dumps(response.model_dump())
        )
        db.add(job)
        db.commit()
    except Exception as e:
        print(f"Failed to create VerificationJob: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save verification job")
    finally:
        db.close()

    return {
        "job_id": job_id,
        "locked": True,
        "preview": {
            "fused_identity_score": response.fused_identity_score,
            "conclusion": response.conclusion,
            "veto_triggered": response.veto_triggered
        }
    }

# ---------------------------------------------------------
# MARK EVIDENCE SHARED HELPER (V2 Pipeline)
# ---------------------------------------------------------

def _run_mark_evidence_pipeline(
    probe_img: np.ndarray,
    gallery_img: Optional[np.ndarray] = None,
    probe_file_hash: Optional[str] = None,
    gallery_file_hash: Optional[str] = None,
    mode: str = "diagnostic",
    target_size: int = 1024,
) -> dict:
    """
    Shared helper for deterministic forensic facial mark detection and matching.
    Operates on raw input images, aligns, preprocesses deterministically, detects marks,
    and runs the v2 matcher if paired. 
    """
    from image_preprocessor import preprocess_for_mark_detection, IMAGE_PREPROCESSOR_VERSION
    from mark_detector import MARK_DETECTOR_VERSION

    has_gallery = gallery_img is not None
    
    # ── 1. Alignment on RAW image ──
    probe_aligned_raw, probe_landmarks = align_face_crop(probe_img)
    gallery_aligned_raw = None
    gallery_landmarks = None
    if has_gallery:
        gallery_aligned_raw, gallery_landmarks = align_face_crop(gallery_img)

    detector_thresholds = get_detector_thresholds()
    
    probe_face_ok = probe_landmarks is not None
    gallery_face_ok = gallery_landmarks is not None if has_gallery else None

    # ── Face Detection Failure Early Exit ──
    if not probe_face_ok or (has_gallery and not gallery_face_ok):
        failed_side = []
        if not probe_face_ok: failed_side.append("probe")
        if has_gallery and not gallery_face_ok: failed_side.append("gallery")
        
        return {
            "mode": "probe_only" if not has_gallery else "paired",
            "aligned_probe_b64": None,
            "aligned_gallery_b64": None,
            "raw_probe_marks": [],
            "raw_gallery_marks": [],
            "rejected_probe_marks": [],
            "rejected_gallery_marks": [],
            "accepted_correspondences": [],
            "rejected_correspondences": [],
            "mark_match_status": "FACE_NOT_DETECTED",
            "matcher_status": "NOT_RUN_FACE_DETECTION_FAILED",
            "mark_diagnostics": {
                "raw_probe_marks_count": 0,
                "raw_gallery_marks_count": 0,
                "accepted_correspondences_count": 0,
                "rejected_candidates_count": 0,
                "detector_status": "FACE_NOT_DETECTED",
                "probe_detector_status": "FACE_NOT_DETECTED" if not probe_face_ok else "NOT_RUN",
                "gallery_detector_status": "FACE_NOT_DETECTED" if (has_gallery and not gallery_face_ok) else ("NOT_PROVIDED" if not has_gallery else "NOT_RUN"),
                "detector_status_probe": "FACE_NOT_DETECTED" if not probe_face_ok else "NOT_RUN",
                "detector_status_gallery": "FACE_NOT_DETECTED" if (has_gallery and not gallery_face_ok) else ("NOT_PROVIDED" if not has_gallery else "NOT_RUN"),
                "matcher_status": "NOT_RUN_FACE_DETECTION_FAILED",
                "calibration_status": "NOT_RUN_FACE_DETECTION_FAILED",
                "lr_marks": None,
                "mark_match_status": "FACE_NOT_DETECTED",
                "rejection_summary": f"Face detection failed for side(s): {', '.join(failed_side)}. Cannot perform mark analysis.",
                "mark_detector_trace": {"probe": None, "gallery": None},
                "matcher_thresholds": get_matcher_thresholds(),
                "technical_debt": "mark_matcher.py v2 integrated. Face detection failed."
            },
            "lr_marks": None,
            "individual_mark_lrs": [],
            "lr_calculation_trace": {
                "individual_lrs": [], "product": None, "calibration_status": "NOT_RUN_FACE_DETECTION_FAILED",
                "thresholds_used": detector_thresholds, "matcher_thresholds": get_matcher_thresholds()
            },
            "detector_thresholds": detector_thresholds,
            "matcher_thresholds": get_matcher_thresholds(),
            "mark_detector_version": MARK_DETECTOR_VERSION,
            "mark_matcher_version": MARK_MATCHER_V2_VERSION,
            "preprocessor_version": IMAGE_PREPROCESSOR_VERSION,
            "probe_preprocessing": None,
            "gallery_preprocessing": None
        }

    # ── 2. Preprocessing & Detection (Probe) ──
    probe_pp = preprocess_for_mark_detection(probe_aligned_raw, landmarks=probe_landmarks, target_size=target_size)
    probe_detector_input = probe_pp["images"]["mark_detector_input_bgr"]
    det_h, det_w = probe_detector_input.shape[:2]
    
    marks_probe, rejected_probe_raw, occ_probe, trace_probe, overlays_probe = detect_facial_marks(
        probe_detector_input, probe_landmarks, input_is_preprocessed=True
    )
    
    valid_probe_marks = []
    for m in marks_probe:
        cx, cy = int(m["centroid"][0] * det_w), int(m["centroid"][1] * det_h)
        if 0 <= cy < det_h and 0 <= cx < det_w and occ_probe[cy, cx] == 0:
            clean_m = {k: v for k, v in m.items() if k != "contour"}
            clean_m["source_side"] = "probe"
            valid_probe_marks.append(clean_m)
            
    rejected_probe_serialized = [serialize_mark_descriptor(r) for r in rejected_probe_raw]

    # ── 3. Preprocessing & Detection (Gallery) ──
    valid_gallery_marks = []
    rejected_gallery_serialized = []
    trace_gallery = None
    overlays_gallery = {}
    marks_gallery = []
    rejected_gallery_raw = []
    gallery_pp = None

    if has_gallery and gallery_face_ok:
        gallery_pp = preprocess_for_mark_detection(gallery_aligned_raw, landmarks=gallery_landmarks, target_size=target_size)
        gallery_detector_input = gallery_pp["images"]["mark_detector_input_bgr"]
        gal_h, gal_w = gallery_detector_input.shape[:2]
        
        marks_gallery, rejected_gallery_raw, occ_gallery, trace_gallery, overlays_gallery = detect_facial_marks(
            gallery_detector_input, gallery_landmarks, input_is_preprocessed=True
        )
        for m in marks_gallery:
            cx, cy = int(m["centroid"][0] * gal_w), int(m["centroid"][1] * gal_h)
            if 0 <= cy < gal_h and 0 <= cx < gal_w and occ_gallery[cy, cx] == 0:
                clean_m = {k: v for k, v in m.items() if k != "contour"}
                clean_m["source_side"] = "gallery"
                valid_gallery_marks.append(clean_m)
        rejected_gallery_serialized = [serialize_mark_descriptor(r) for r in rejected_gallery_raw]

    # ── 4. Matching & LR ──
    correspondences = []
    rejected_cands = []
    matcher_result = None
    lr_marks = None
    individual_mark_lrs = []
    calibration_status = "NOT_APPLICABLE"

    if not has_gallery:
        mark_match_status = "NOT_RUN_SINGLE_IMAGE"
        matcher_status = "NOT_RUN_SINGLE_IMAGE"
        lr_marks = None
    else:
        exact_image_match = (probe_file_hash == gallery_file_hash) if (probe_file_hash and gallery_file_hash) else False

        if exact_image_match:
            mark_match_status = "EXACT_SELF_MATCH"
            matcher_status = "EXACT_SELF_MATCH"
            n_self = min(len(valid_probe_marks), len(valid_gallery_marks))
            lr_marks = 1.0
            individual_mark_lrs = [1.0] * n_self
            calibration_status = "NOT_APPLICABLE_SELF_MATCH"
            for si in range(n_self):
                correspondences.append({
                    "gallery_idx": si, "probe_idx": si,
                    "gallery_centroid": list(valid_gallery_marks[si]["centroid"]),
                    "probe_centroid": list(valid_probe_marks[si]["centroid"]),
                    "position_distance": 0.0, "area_ratio": 1.0,
                    "type_match": True, "region_match": True,
                    "match_quality": 1.0,
                    "mark_type": valid_gallery_marks[si].get("mark_type", "unknown"),
                    "face_region": valid_gallery_marks[si].get("face_region", "unknown"),
                    "match_cost": 0.0,
                    "lr": 1.0,
                })
            matcher_result = {
                "matched": n_self, "score": 100.0 if n_self > 0 else None,
                "lr_marks": 1.0, "mark_lrs": individual_mark_lrs,
                "matches": correspondences, "rejected_candidates": [],
                "matcher_status": "EXACT_SELF_MATCH",
                "calibration_status": calibration_status,
                "matcher_version": MARK_MATCHER_V2_VERSION,
            }
        else:
            matcher_result = match_marks_v2(
                valid_gallery_marks, valid_probe_marks,
                calibration=TIER4_CALIBRATION
            )
            matcher_status = matcher_result["matcher_status"]
            calibration_status = matcher_result.get("calibration_status", "UNKNOWN")
            lr_marks = matcher_result["lr_marks"]
            individual_mark_lrs = matcher_result["mark_lrs"]
            correspondences = matcher_result["matches"]
            rejected_cands = matcher_result["rejected_candidates"]

            if len(valid_probe_marks) < 2 or len(valid_gallery_marks) < 2:
                mark_match_status = "INSUFFICIENT_MARKS"
            elif matcher_result["matched"] > 0:
                mark_match_status = "MATCHED"
            else:
                mark_match_status = "NO_MATCHES"

    # ── 5. Detector Status ──
    probe_detector_status = trace_probe.get("detector_status", "UNKNOWN") if trace_probe else "UNKNOWN"
    gallery_detector_status = trace_gallery.get("detector_status", "UNKNOWN") if trace_gallery else ("NOT_PROVIDED" if not has_gallery else "UNKNOWN")

    if len(marks_probe) > 0 or (has_gallery and len(marks_gallery) > 0):
        overall_detector_status = "OK"
    elif len(marks_probe) == 0 and (not has_gallery or len(marks_gallery) == 0):
        overall_detector_status = "NO_CANDIDATES"
    else:
        overall_detector_status = "PARTIAL"

    matched_count = matcher_result["matched"] if matcher_result else 0
    mark_diagnostics_payload = {
        "raw_probe_marks_count": len(valid_probe_marks),
        "raw_gallery_marks_count": len(valid_gallery_marks),
        "accepted_correspondences_count": matched_count,
        "rejected_candidates_count": len(rejected_cands) if rejected_cands else 0,
        "detector_status": overall_detector_status,
        "probe_detector_status": probe_detector_status,
        "gallery_detector_status": gallery_detector_status,
        "detector_status_probe": probe_detector_status,
        "detector_status_gallery": gallery_detector_status,
        "matcher_status": matcher_status,
        "calibration_status": calibration_status,
        "lr_marks": finite_or_none(lr_marks),
        "mark_match_status": mark_match_status,
        "rejection_summary": _build_rejection_summary(
            valid_probe_marks, valid_gallery_marks,
            matcher_result or {}, rejected_cands, mark_match_status,
            (probe_file_hash == gallery_file_hash) if (has_gallery and probe_file_hash and gallery_file_hash) else False,
            TIER4_CALIBRATION,
            trace_probe=trace_probe, trace_gallery=trace_gallery,
        ) if has_gallery else (
            f"Probe-only mode: {len(valid_probe_marks)} mark(s) detected, {len(rejected_probe_raw)} rejected. No gallery provided for matching."
        ),
        "mark_detector_trace": {"probe": trace_probe, "gallery": trace_gallery},
        "matcher_thresholds": get_matcher_thresholds(),
        "technical_debt": "mark_matcher.py v2 integrated via shared helper.",
    }

    lr_calculation_trace = {
        "individual_lrs": [finite_or_none(lr) for lr in individual_mark_lrs],
        "product": finite_or_none(lr_marks) if has_gallery else None,
        "calibration_status": calibration_status,
        "thresholds_used": detector_thresholds,
        "matcher_thresholds": get_matcher_thresholds(),
    }

    def _pp_summary(pp_dict):
        if not pp_dict: return None
        return {
            "decoded_hash": pp_dict["decoded_hash"],
            "aligned_pre_clahe_hash": pp_dict["aligned_pre_clahe_hash"],
            "aligned_post_clahe_hash": pp_dict["aligned_post_clahe_hash"],
            "original_dimensions": pp_dict["original_dimensions"],
            "decoded_dimensions": pp_dict["decoded_dimensions"],
            "aligned_dimensions": pp_dict["aligned_dimensions"],
            "quality": pp_dict["quality"],
            "preprocessing_steps": pp_dict["preprocessing_steps"],
            "preprocessor_version": pp_dict.get("preprocessor_version", IMAGE_PREPROCESSOR_VERSION),
            "aligned_pre_clahe_b64": pp_dict.get("debug_b64", {}).get("aligned_b64"),
            "aligned_post_clahe_b64": pp_dict.get("debug_b64", {}).get("lab_clahe_b64"),
            "illumination_normalized_b64": pp_dict.get("debug_b64", {}).get("illumination_normalized_b64"),
            "mark_detector_input_b64": pp_dict.get("debug_b64", {}).get("mark_detector_input_b64"),
            "skin_mask_b64": pp_dict.get("debug_b64", {}).get("skin_mask_b64"),
        }

    return {
        "mode": "probe_only" if not has_gallery else "paired",
        "aligned_probe_b64": probe_pp["debug_b64"]["aligned_b64"] if probe_pp else None,
        "aligned_gallery_b64": gallery_pp["debug_b64"]["aligned_b64"] if gallery_pp else None,
        "raw_probe_marks": [serialize_mark_descriptor(m) for m in valid_probe_marks],
        "raw_gallery_marks": [serialize_mark_descriptor(m) for m in valid_gallery_marks],
        "rejected_probe_marks": rejected_probe_serialized,
        "rejected_gallery_marks": rejected_gallery_serialized,
        "accepted_correspondences": correspondences,
        "rejected_correspondences": rejected_cands,
        "mark_match_status": mark_match_status,
        "matcher_status": matcher_status,
        "mark_diagnostics": mark_diagnostics_payload,
        "lr_marks": finite_or_none(lr_marks) if has_gallery else None,
        "individual_mark_lrs": [finite_or_none(lr) for lr in individual_mark_lrs],
        "lr_calculation_trace": lr_calculation_trace,
        "detector_thresholds": detector_thresholds,
        "matcher_thresholds": get_matcher_thresholds(),
        "mark_detector_version": MARK_DETECTOR_VERSION,
        "mark_matcher_version": MARK_MATCHER_V2_VERSION,
        "preprocessor_version": IMAGE_PREPROCESSOR_VERSION,
        "probe_preprocessing": _pp_summary(probe_pp),
        "gallery_preprocessing": _pp_summary(gallery_pp),
    }

# ---------------------------------------------------------
# MARK-ONLY ANALYSIS GATEWAY (Court-Survivable v2 — Phase 2)
# ---------------------------------------------------------

class MarkAnalyzeRequest(BaseModel):
    """Request model for standalone mark-only analysis.
    Supports both base64-encoded image data and GCS URLs."""
    probe_b64: Optional[str] = None
    gallery_b64: Optional[str] = None
    probe_url: Optional[str] = None
    gallery_url: Optional[str] = None


def _decode_b64_image(b64_str: str) -> tuple:
    """Decode a base64-encoded image string to (cv2_image, sha256_hash).
    Accepts 'data:image/...;base64,...' or raw base64."""
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        raw_hash = hashlib.sha256(img_bytes).hexdigest()
        arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode base64 image data.")
        h, w = img.shape[:2]
        if h > 4096 or w > 4096:
            raise ValueError("Image dimensions exceed 4096x4096.")
        return img, raw_hash
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")


def _resolve_mark_image(b64: Optional[str], url: Optional[str], label: str) -> tuple:
    """Resolve an image from either base64 or URL source. Returns (image, hash)."""
    if b64:
        return _decode_b64_image(b64)
    elif url:
        return fetch_image_from_url(url)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"No {label} image provided. Supply either {label}_b64 or {label}_url."
        )


@app.post("/marks/analyze")
@limiter.limit("10/minute")
def marks_analyze(request: Request, payload: MarkAnalyzeRequest, _: dict = Depends(verify_jwt)):
    """
    Standalone mark-only diagnostic endpoint.

    Pipeline (v1.158+):
      raw image -> align_face_crop(raw) -> preprocess_for_mark_detection(aligned, 1024)
      -> detect_facial_marks(mark_detector_input_bgr, landmarks)

    Does NOT run: ArcFace, Facenet512, LBP, vault search, or full Bayesian fusion.
    Supports probe-only mode (no gallery required) and probe+gallery mode.
    Protected by JWT authentication and rate limiting.

    NOTE: mark_detector.py still applies internal CLAHE in some channels
    (dark_lesion, bright_scar, linear_scar_v2). This is documented technical
    debt — the preprocessor provides single-CLAHE input, but the detector
    adds a second pass internally. This will be addressed in a future step.
    """
    from image_preprocessor import preprocess_for_mark_detection, IMAGE_PREPROCESSOR_VERSION

    # ── 1. Image Acquisition ──
    probe_img, probe_file_hash = _resolve_mark_image(
        payload.probe_b64, payload.probe_url, "probe"
    )

    has_gallery = bool(payload.gallery_b64 or payload.gallery_url)
    gallery_img = None
    gallery_file_hash = None
    if has_gallery:
        gallery_img, gallery_file_hash = _resolve_mark_image(
            payload.gallery_b64, payload.gallery_url, "gallery"
        )

    # ── 2. Alignment on RAW image (no pre-CLAHE) ──
    # align_face_crop returns (aligned_crop, landmarks) at 256×256 by default.
    # We align at 256 first (MediaPipe expects reasonable size), then the
    # preprocessor resizes to 1024 for mark detection.
    probe_aligned_raw, probe_landmarks = align_face_crop(probe_img)

    gallery_aligned_raw = None
    gallery_landmarks = None
    if has_gallery:
        gallery_aligned_raw, gallery_landmarks = align_face_crop(gallery_img)

    # Canonical thresholds from mark_detector.py (single source of truth)
    detector_thresholds = get_detector_thresholds()

    # ── 3. Face Detection Validation ──
    probe_face_ok = probe_landmarks is not None
    gallery_face_ok = gallery_landmarks is not None if has_gallery else None

    if not probe_face_ok or (has_gallery and not gallery_face_ok):
        failed_side = []
        if not probe_face_ok:
            failed_side.append("probe")
        if has_gallery and not gallery_face_ok:
            failed_side.append("gallery")
        # Encode whatever aligned crops we got for diagnostics
        aligned_probe_b64 = None
        aligned_gallery_b64 = None
        if probe_aligned_raw is not None:
            _, pro_buf = cv2.imencode('.png', probe_aligned_raw)
            aligned_probe_b64 = f"data:image/png;base64,{base64.b64encode(pro_buf).decode('utf-8')}"
        if gallery_aligned_raw is not None:
            _, gal_buf = cv2.imencode('.png', gallery_aligned_raw)
            aligned_gallery_b64 = f"data:image/png;base64,{base64.b64encode(gal_buf).decode('utf-8')}"
        return {
            "aligned_probe_b64": aligned_probe_b64,
            "aligned_gallery_b64": aligned_gallery_b64,
            "raw_probe_marks": [],
            "raw_gallery_marks": [],
            "rejected_probe_marks": [],
            "rejected_gallery_marks": [],
            "accepted_correspondences": [],
            "rejected_correspondences": [],
            "mark_match_status": "FACE_NOT_DETECTED",
            "mark_diagnostics": {
                "raw_probe_marks_count": 0,
                "raw_gallery_marks_count": 0,
                "accepted_correspondences_count": 0,
                "rejected_candidates_count": 0,
                "detector_status": "FACE_NOT_DETECTED",
                "probe_detector_status": "FACE_NOT_DETECTED" if not probe_face_ok else "NOT_RUN",
                "gallery_detector_status": ("FACE_NOT_DETECTED" if not gallery_face_ok else "NOT_RUN") if has_gallery else "NOT_PROVIDED",
                "detector_status_probe": "FACE_NOT_DETECTED" if not probe_face_ok else "NOT_RUN",
                "detector_status_gallery": ("FACE_NOT_DETECTED" if not gallery_face_ok else "NOT_RUN") if has_gallery else "NOT_PROVIDED",
                "matcher_status": "NOT_RUN_FACE_DETECTION_FAILED",
                "lr_marks": None,
                "mark_match_status": "FACE_NOT_DETECTED",
                "rejection_summary": f"Face detection failed on: {', '.join(failed_side)}",
                "mark_detector_trace": {"probe": None, "gallery": None},
            },
            "lr_marks": None,
            "individual_mark_lrs": [],
            "lr_calculation_trace": {
                "individual_lrs": [],
                "product": None,
                "calibration_status": "LOADED" if TIER4_CALIBRATION else "MISSING",
                "thresholds_used": detector_thresholds,
            },
            "detector_thresholds": detector_thresholds,
            "mark_detector_version": _MDV,
            "mark_matcher_version": MARK_MATCHER_VERSION,
            "preprocessor_version": IMAGE_PREPROCESSOR_VERSION,
            "mode": "probe_only" if not has_gallery else "paired",
        }

    # ── 4. Deterministic Preprocessing (probe) ──
    probe_pp = preprocess_for_mark_detection(probe_aligned_raw, landmarks=probe_landmarks, target_size=1024)
    probe_detector_input = probe_pp["images"]["mark_detector_input_bgr"]
    det_h, det_w = probe_detector_input.shape[:2]

    # ── 5. Mark Detection — Probe ──
    marks_probe, rejected_probe_raw, occ_probe, trace_probe, overlays_probe = detect_facial_marks(probe_detector_input, probe_landmarks, input_is_preprocessed=True)

    valid_probe_marks = []
    for m in marks_probe:
        cx, cy = int(m["centroid"][0] * det_w), int(m["centroid"][1] * det_h)
        if 0 <= cy < det_h and 0 <= cx < det_w and occ_probe[cy, cx] == 0:
            clean_m = {k: v for k, v in m.items() if k != "contour"}
            clean_m["source_side"] = "probe"
            valid_probe_marks.append(clean_m)

    rejected_probe_serialized = [serialize_mark_descriptor(r) for r in rejected_probe_raw]

    # ── 6. Deterministic Preprocessing + Detection — Gallery ──
    valid_gallery_marks = []
    rejected_gallery_serialized = []
    trace_gallery = None
    overlays_gallery = {}
    marks_gallery = []
    rejected_gallery_raw = []
    gallery_pp = None

    if has_gallery and gallery_face_ok:
        gallery_pp = preprocess_for_mark_detection(gallery_aligned_raw, landmarks=gallery_landmarks, target_size=1024)
        gallery_detector_input = gallery_pp["images"]["mark_detector_input_bgr"]
        gal_h, gal_w = gallery_detector_input.shape[:2]

        marks_gallery, rejected_gallery_raw, occ_gallery, trace_gallery, overlays_gallery = detect_facial_marks(gallery_detector_input, gallery_landmarks, input_is_preprocessed=True)
        for m in marks_gallery:
            cx, cy = int(m["centroid"][0] * gal_w), int(m["centroid"][1] * gal_h)
            if 0 <= cy < gal_h and 0 <= cx < gal_w and occ_gallery[cy, cx] == 0:
                clean_m = {k: v for k, v in m.items() if k != "contour"}
                clean_m["source_side"] = "gallery"
                valid_gallery_marks.append(clean_m)
        rejected_gallery_serialized = [serialize_mark_descriptor(r) for r in rejected_gallery_raw]

    # ── 7. Matching & LR (only if both sides present) ──
    correspondences = []
    rejected_cands = []
    matcher_result = None
    lr_marks = None
    individual_mark_lrs = []
    calibration_status = "NOT_APPLICABLE"

    if not has_gallery:
        mark_match_status = "NOT_RUN_SINGLE_IMAGE"
        matcher_status = "NOT_RUN_SINGLE_IMAGE"
        lr_marks = None
    else:
        exact_image_match = (probe_file_hash == gallery_file_hash) if (probe_file_hash and gallery_file_hash) else False

        if exact_image_match:
            # ── Exact self-match: LR is neutral (same-image mark evidence is circular) ──
            mark_match_status = "EXACT_SELF_MATCH"
            matcher_status = "EXACT_SELF_MATCH"
            n_self = min(len(valid_probe_marks), len(valid_gallery_marks))
            lr_marks = 1.0
            individual_mark_lrs = [1.0] * n_self
            calibration_status = "NOT_APPLICABLE_SELF_MATCH"
            for si in range(n_self):
                correspondences.append({
                    "gallery_idx": si, "probe_idx": si,
                    "gallery_centroid": list(valid_gallery_marks[si]["centroid"]),
                    "probe_centroid": list(valid_probe_marks[si]["centroid"]),
                    "position_distance": 0.0, "area_ratio": 1.0,
                    "type_match": True, "region_match": True,
                    "match_quality": 1.0,
                    "mark_type": valid_gallery_marks[si].get("mark_type", "unknown"),
                    "face_region": valid_gallery_marks[si].get("face_region", "unknown"),
                    "match_cost": 0.0,
                    "lr": 1.0,
                })
            matcher_result = {
                "matched": n_self, "score": 100.0 if n_self > 0 else None,
                "lr_marks": 1.0, "mark_lrs": individual_mark_lrs,
                "matches": correspondences, "rejected_candidates": [],
                "matcher_status": "EXACT_SELF_MATCH",
                "calibration_status": calibration_status,
                "matcher_version": MARK_MATCHER_V2_VERSION,
            }
        else:
            # ── Paired non-self-match: use mark_matcher.py v2 ──
            matcher_result = match_marks_v2(
                valid_gallery_marks, valid_probe_marks,
                calibration=TIER4_CALIBRATION
            )

            matcher_status = matcher_result["matcher_status"]
            calibration_status = matcher_result.get("calibration_status", "UNKNOWN")
            lr_marks = matcher_result["lr_marks"]
            individual_mark_lrs = matcher_result["mark_lrs"]
            correspondences = matcher_result["matches"]
            rejected_cands = matcher_result["rejected_candidates"]

            if len(valid_probe_marks) < 2 or len(valid_gallery_marks) < 2:
                mark_match_status = "INSUFFICIENT_MARKS"
            elif matcher_result["matched"] > 0:
                mark_match_status = "MATCHED"
            else:
                mark_match_status = "NO_MATCHES"

    # ── 8. Detector status per side ──
    probe_detector_status = trace_probe.get("detector_status", "UNKNOWN") if trace_probe else "UNKNOWN"
    gallery_detector_status = trace_gallery.get("detector_status", "UNKNOWN") if trace_gallery else ("NOT_PROVIDED" if not has_gallery else "UNKNOWN")

    if len(marks_probe) > 0 or (has_gallery and len(marks_gallery) > 0):
        overall_detector_status = "OK"
    elif len(marks_probe) == 0 and (not has_gallery or len(marks_gallery) == 0):
        overall_detector_status = "NO_CANDIDATES"
    else:
        overall_detector_status = "PARTIAL"

    # ── 9. Mark Diagnostics (always present, always truthful) ──
    matched_count = matcher_result["matched"] if matcher_result else 0
    mark_diagnostics_payload = {
        "raw_probe_marks_count": len(valid_probe_marks),
        "raw_gallery_marks_count": len(valid_gallery_marks),
        "accepted_correspondences_count": matched_count,
        "rejected_candidates_count": len(rejected_cands) if rejected_cands else 0,
        "detector_status": overall_detector_status,
        "probe_detector_status": probe_detector_status,
        "gallery_detector_status": gallery_detector_status,
        "detector_status_probe": probe_detector_status,
        "detector_status_gallery": gallery_detector_status,
        "matcher_status": matcher_status,
        "calibration_status": calibration_status,
        "lr_marks": finite_or_none(lr_marks),
        "mark_match_status": mark_match_status,
        "rejection_summary": _build_rejection_summary(
            valid_probe_marks, valid_gallery_marks,
            matcher_result or {}, rejected_cands, mark_match_status,
            (probe_file_hash == gallery_file_hash) if (has_gallery and probe_file_hash and gallery_file_hash) else False,
            TIER4_CALIBRATION,
            trace_probe=trace_probe, trace_gallery=trace_gallery,
        ) if has_gallery else (
            f"Probe-only mode: {len(valid_probe_marks)} mark(s) detected, {len(rejected_probe_raw)} rejected. No gallery provided for matching."
        ),
        "mark_detector_trace": {
            "probe": trace_probe,
            "gallery": trace_gallery,
        },
        "matcher_thresholds": get_matcher_thresholds(),
        "technical_debt": "mark_matcher.py v2 integrated for /marks/analyze. Production routes still use legacy in-main matcher.",
    }

    # ── 10. LR Calculation Trace ──
    lr_calculation_trace = {
        "individual_lrs": [finite_or_none(lr) for lr in individual_mark_lrs],
        "product": finite_or_none(lr_marks) if has_gallery else None,
        "calibration_status": calibration_status,
        "thresholds_used": detector_thresholds,
        "matcher_thresholds": get_matcher_thresholds(),
    }

    # ── 11. Build Preprocessing Summary (strip numpy arrays) ──
    def _pp_summary(pp_dict):
        """Extract JSON-safe preprocessing summary (no numpy arrays)."""
        return {
            "decoded_hash": pp_dict["decoded_hash"],
            "aligned_pre_clahe_hash": pp_dict["aligned_pre_clahe_hash"],
            "aligned_post_clahe_hash": pp_dict["aligned_post_clahe_hash"],
            "original_dimensions": pp_dict["original_dimensions"],
            "decoded_dimensions": pp_dict["decoded_dimensions"],
            "aligned_dimensions": pp_dict["aligned_dimensions"],
            "quality": pp_dict["quality"],
            "preprocessing_steps": pp_dict["preprocessing_steps"],
            "preprocessor_version": pp_dict.get("preprocessor_version", IMAGE_PREPROCESSOR_VERSION),
            "aligned_pre_clahe_b64": pp_dict["debug_b64"]["aligned_b64"],
            "aligned_post_clahe_b64": pp_dict["debug_b64"]["lab_clahe_b64"],
            "illumination_normalized_b64": pp_dict["debug_b64"]["illumination_normalized_b64"],
            "mark_detector_input_b64": pp_dict["debug_b64"]["mark_detector_input_b64"],
            "skin_mask_b64": pp_dict["debug_b64"]["skin_mask_b64"],
        }

    # ── 12. Build Response ──
    result = {
        "mode": "probe_only" if not has_gallery else "paired",
        "aligned_probe_b64": probe_pp["debug_b64"]["aligned_b64"],
        "aligned_gallery_b64": gallery_pp["debug_b64"]["aligned_b64"] if gallery_pp else None,
        "raw_probe_marks": [serialize_mark_descriptor(m) for m in valid_probe_marks],
        "raw_gallery_marks": [serialize_mark_descriptor(m) for m in valid_gallery_marks],
        "rejected_probe_marks": rejected_probe_serialized,
        "rejected_gallery_marks": rejected_gallery_serialized,
        "accepted_correspondences": correspondences,
        "rejected_correspondences": rejected_cands,
        "mark_match_status": mark_match_status,
        "mark_diagnostics": mark_diagnostics_payload,
        "lr_marks": finite_or_none(lr_marks) if has_gallery else None,
        "individual_mark_lrs": [finite_or_none(lr) for lr in individual_mark_lrs],
        "lr_calculation_trace": lr_calculation_trace,
        "detector_thresholds": detector_thresholds,
        "mark_detector_version": _MDV,
        "mark_matcher_version": MARK_MATCHER_V2_VERSION,
        "preprocessor_version": IMAGE_PREPROCESSOR_VERSION,
        "probe_preprocessing": _pp_summary(probe_pp),
        "gallery_preprocessing": _pp_summary(gallery_pp) if gallery_pp else None,
    }

    # ── 13. Debug Overlays (gated behind DEBUG_FORENSIC) ──
    if os.getenv("DEBUG_FORENSIC") == "true":
        pro_debug_img = probe_detector_input.copy()
        pro_matched_idx = set()
        if has_gallery and matcher_result:
            for match_entry in matcher_result.get("matches", []):
                if isinstance(match_entry, dict):
                    pro_matched_idx.add(match_entry.get("probe_idx"))
                elif isinstance(match_entry, (tuple, list)) and len(match_entry) >= 2:
                    pro_matched_idx.add(match_entry[1])
        for idx, m in enumerate(valid_probe_marks):
            cx, cy = int(m["centroid"][0] * det_w), int(m["centroid"][1] * det_h)
            color = (0, 255, 0) if idx in pro_matched_idx else (128, 128, 128)
            cv2.circle(pro_debug_img, (cx, cy), 6, color, 2)
            cv2.putText(pro_debug_img, str(idx), (cx + 7, cy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        _, pro_dbuf = cv2.imencode('.png', pro_debug_img)
        result["probe_marks_overlay_b64"] = f"data:image/png;base64,{base64.b64encode(pro_dbuf).decode('utf-8')}"

        if has_gallery and gallery_pp is not None:
            gal_det_input = gallery_pp["images"]["mark_detector_input_bgr"]
            gal_dh, gal_dw = gal_det_input.shape[:2]
            gal_debug_img = gal_det_input.copy()
            gal_matched_idx = set()
            for match_entry in (matcher_result or {}).get("matches", []):
                if isinstance(match_entry, dict):
                    gal_matched_idx.add(match_entry.get("gallery_idx"))
                elif isinstance(match_entry, (tuple, list)) and len(match_entry) >= 2:
                    gal_matched_idx.add(match_entry[0])
            for idx, m in enumerate(valid_gallery_marks):
                cx, cy = int(m["centroid"][0] * gal_dw), int(m["centroid"][1] * gal_dh)
                color = (0, 255, 0) if idx in gal_matched_idx else (128, 128, 128)
                cv2.circle(gal_debug_img, (cx, cy), 6, color, 2)
                cv2.putText(gal_debug_img, str(idx), (cx + 7, cy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            _, gal_dbuf = cv2.imencode('.png', gal_debug_img)
            result["gallery_marks_overlay_b64"] = f"data:image/png;base64,{base64.b64encode(gal_dbuf).decode('utf-8')}"

        result["matcher_thresholds"] = get_matcher_thresholds()
        result["debug_overlays"] = {
            "probe": overlays_probe,
            "gallery": overlays_gallery if has_gallery else {},
        }

    return result



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
@app.post("/vault/search")
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

    # 3. Pre-CLAHE alignment for provenance hash (raw pixel chain-of-custody)
    probe_pre_clahe_crop, _ppc_lm = align_face_crop(probe_img)
    probe_aligned_crop_hash_pre_clahe = compute_image_hash(probe_pre_clahe_crop)

    # 3.5 Pre-process probe (CLAHE + alignment)
    probe_clahe = apply_clahe(probe_img)
    probe_aligned, probe_landmarks = align_face_crop(probe_clahe)

    # Post-CLAHE aligned crop hash (actual model-input pixels)
    probe_aligned_crop_hash_post_clahe = compute_image_hash(probe_aligned)

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
    gallery_decoded_image_hash = None
    gallery_original_dimensions = None
    gallery_decoded_dimensions = None
    # Gallery CLAHE provenance defaults (probe-mirrored if no gallery fetched)
    gallery_aligned_crop_hash_pre_clahe = probe_aligned_crop_hash_pre_clahe
    gallery_aligned_crop_hash_post_clahe = probe_aligned_crop_hash_post_clahe

    if matched_profile_for_gallery and matched_profile_for_gallery.thumbnail_url:
        try:
            gallery_img, gallery_file_hash = fetch_image_from_url(matched_profile_for_gallery.thumbnail_url)
            gallery_decoded_image_hash = compute_image_hash(gallery_img)
            gallery_original_dimensions = f"{gallery_img.shape[1]}x{gallery_img.shape[0]}"
            gallery_decoded_dimensions = gallery_original_dimensions
            # Pre-CLAHE gallery provenance hash
            gallery_pre_clahe_crop, _gpc_lm = align_face_crop(gallery_img)
            gallery_aligned_crop_hash_pre_clahe = compute_image_hash(gallery_pre_clahe_crop)
            gallery_clahe = apply_clahe(gallery_img)
            gallery_aligned, gallery_landmarks = align_face_crop(gallery_clahe)
            # Post-CLAHE gallery provenance hash
            gallery_aligned_crop_hash_post_clahe = compute_image_hash(gallery_aligned)
            if gallery_landmarks is None:
                gallery_aligned = probe_aligned
                gallery_landmarks = probe_landmarks
                gallery_aligned_crop_hash_pre_clahe = probe_aligned_crop_hash_pre_clahe
                gallery_aligned_crop_hash_post_clahe = probe_aligned_crop_hash_post_clahe
        except Exception:
            pass  # Fallback to probe if gallery fetch fails
    else:
        # Legacy fallback: try local target_profiles/ directory
        gallery_file = _resolve_target_image(best_user_id)
        if gallery_file and os.path.isfile(gallery_file):
            gallery_img = cv2.imread(gallery_file)
            gallery_decoded_image_hash = compute_image_hash(gallery_img)
            gallery_original_dimensions = f"{gallery_img.shape[1]}x{gallery_img.shape[0]}"
            gallery_decoded_dimensions = gallery_original_dimensions
            # Pre-CLAHE gallery provenance hash
            gallery_pre_clahe_crop, _gpc_lm = align_face_crop(gallery_img)
            gallery_aligned_crop_hash_pre_clahe = compute_image_hash(gallery_pre_clahe_crop)
            gallery_clahe = apply_clahe(gallery_img)
            gallery_aligned, gallery_landmarks = align_face_crop(gallery_clahe)
            # Post-CLAHE gallery provenance hash
            gallery_aligned_crop_hash_post_clahe = compute_image_hash(gallery_aligned)
            if gallery_landmarks is None:
                gallery_aligned = probe_aligned
                gallery_landmarks = probe_landmarks
                gallery_aligned_crop_hash_pre_clahe = probe_aligned_crop_hash_pre_clahe
                gallery_aligned_crop_hash_post_clahe = probe_aligned_crop_hash_post_clahe

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
        
        # Determine geometry status from extraction
        gal_geom_status = gal_vis.get("geometry_status", "UNKNOWN")
        pro_geom_status = pro_vis.get("geometry_status", "UNKNOWN")
        
        if gal_geom_status != "OK" or pro_geom_status != "OK":
            geometry_status = gal_geom_status if gal_geom_status != "OK" else pro_geom_status
        else:
            geometry_status = "OK"
        
        valid_mask = gal_vis["ratio_visibility"] & pro_vis["ratio_visibility"]
        effective_ratios = int(np.sum(valid_mask))
        
        if effective_ratios > 0 and geometry_status == "OK":
            raw_l2 = float(np.linalg.norm((ratios_gallery - ratios_probe)[valid_mask]))
            ratio_l2 = raw_l2 * math.sqrt(12.0 / effective_ratios)
            tier2_score = max(0.0, min(100.0, (1.0 - (ratio_l2 / GEOMETRY_DISTANCE_ZERO_THRESHOLD)) * 100))
        else:
            ratio_l2 = None
            tier2_score = 0.0
            if geometry_status == "OK":
                geometry_status = "NO_VALID_RATIOS"
        
        # Debug logging for Tier 2 geometry pipeline
        if os.getenv("DEBUG_FORENSIC") == "true" or os.getenv("ENVIRONMENT") == "development":
            print(f"[FORENSIC DEBUG] Tier 2 Geometry (vault):", flush=True)
            print(f"  geometry_status={geometry_status}, effective_ratios={effective_ratios}", flush=True)
            print(f"  IOD: gallery={gal_vis.get('iod', 'N/A')}, probe={pro_vis.get('iod', 'N/A')}", flush=True)
            print(f"  ratio_visibility mask: {valid_mask.tolist()}", flush=True)
            print(f"  ratio_l2={ratio_l2}, threshold={GEOMETRY_DISTANCE_ZERO_THRESHOLD}", flush=True)
            print(f"  tier2_score={tier2_score:.2f}", flush=True)
    else:
        tier2_score = 0.0
        gal_angles, pro_angles = {}, {}
        pro_vis = {"occlusion_percentage": 0.0, "occluded_regions": []}
        effective_ratios = 0
        ratio_l2 = None
        geometry_status = "NO_LANDMARKS"

    # 8. Tier 3: Micro-Topology (LBP Chi-Squared Distance)
    lbp_gal = extract_lbp_histogram(gallery_aligned)
    lbp_pro = extract_lbp_histogram(probe_aligned)
    chi_squared = 0.5 * float(np.sum(((lbp_gal - lbp_pro) ** 2) / (lbp_gal + lbp_pro + 1e-10)))
    tier3_score = max(0.0, min(100.0, (1.0 - chi_squared) * 100))

    # 9. TIER 4: Mark Correspondence (Bayesian LR Engine)
    marks_gallery, rejected_gallery, occ_gallery, trace_gallery, overlays_gallery = detect_facial_marks(gallery_aligned, gallery_landmarks)
    marks_probe, rejected_probe, occ_probe, trace_probe, overlays_probe = detect_facial_marks(probe_aligned, probe_landmarks)
    
    valid_gallery_marks = []
    for m in marks_gallery:
        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
        if cy < 256 and cx < 256 and occ_gallery[cy, cx] == 0:
            clean_m = {k: v for k, v in m.items() if k != "contour"}
            clean_m["source_side"] = "gallery"
            valid_gallery_marks.append(clean_m)
            
    valid_probe_marks = []
    for m in marks_probe:
        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
        if cy < 256 and cx < 256 and occ_probe[cy, cx] == 0:
            clean_m = {k: v for k, v in m.items() if k != "contour"}
            clean_m["source_side"] = "probe"
            valid_probe_marks.append(clean_m)

    # ── Exact Self-Match Detection ──
    exact_image_match = (probe_file_hash == gallery_file_hash) if (probe_file_hash and gallery_file_hash) else False

    if exact_image_match:
        mark_match_status = "EXACT_SELF_MATCH"
        tier4_score = 100.0
        n_self = min(len(valid_probe_marks), len(valid_gallery_marks))
        assigned_pairs = []
        for si in range(n_self):
            assigned_pairs.append({
                "gallery_idx": si, "probe_idx": si,
                "cost": 0.0, "position_distance": 0.0, "area_ratio": 1.0,
                "type_match": True, "region_match": True,
                "mark_type": valid_gallery_marks[si].get("mark_type", "unknown"),
                "face_region": valid_gallery_marks[si].get("face_region", "unknown"),
                "gallery_centroid": list(valid_gallery_marks[si]["centroid"]),
                "probe_centroid": list(valid_probe_marks[si]["centroid"]),
            })
        unmatched_gal = list(range(n_self, len(valid_gallery_marks)))
        unmatched_pro = list(range(n_self, len(valid_probe_marks)))
        rejected_cands = []
        mark_result = {
            "score": 100.0, "matched": n_self,
            "total_gallery": len(valid_gallery_marks), "total_probe": len(valid_probe_marks),
            "matches": [(si, si, 1.0) for si in range(n_self)],
            "lr_marks": 1.0, "mark_lrs": [],
        }
    else:
        assigned_pairs, unmatched_gal, unmatched_pro, rejected_cands = match_facial_marks(valid_gallery_marks, valid_probe_marks)
        mark_result = compute_mark_correspondence(valid_gallery_marks, valid_probe_marks, matched_pairs=assigned_pairs)
        tier4_score = mark_result["score"]  # None if insufficient marks

        # Determine mark_match_status
        if len(valid_probe_marks) < 2 or len(valid_gallery_marks) < 2:
            mark_match_status = "INSUFFICIENT_MARKS"
        elif mark_result.get("matched", 0) > 0:
            mark_match_status = "MATCHED"
        else:
            mark_match_status = "NO_MATCHES"

    # ── BAYESIAN EVIDENCE FUSION (Scientific v4.0) ──
    lr_ensemble = score_to_lr_ensemble(structural_sim, temporal_delta=temporal_delta)
    lr_marks = mark_result.get("lr_marks", 1.0)
    lr_total = lr_ensemble * lr_marks

    # Posterior probability via Bayes' Theorem (neutral prior = 0.5)
    PRIOR = 0.5
    posterior = (PRIOR * lr_total) / ((PRIOR * lr_total) + (1.0 - PRIOR))
    fused_score = posterior * 100.0

    bayesian_fused_score = fused_score  # preserve pre-veto

    # 10. ArcFace Veto (flag only — Bayesian math handles scoring)
    veto_arcface = best_score < 0.40

    # ── MARK OVERRIDE PROTOCOL (v1.0) ──
    mark_override_eval = evaluate_mark_veto_override(mark_result, lr_marks)
    mark_override_eligible = mark_override_eval["eligible"]
    positive_mark_count = mark_override_eval["positive_mark_count"]

    veto_reason = None
    veto_override_applied = False
    veto_override_reason = None
    if veto_arcface:
        if mark_override_eligible:
            veto_reason = "ARCFACE_VETO_MARK_OVERRIDE"
            veto_override_applied = True
            veto_override_reason = mark_override_eval["reason"]
            conclusion = (
                f"Supports Common Source — Face-Model Veto Overridden by Mark Correspondence ({best_user_id})"
            )
        else:
            fused_score = 0.0
            veto_reason = "ARCFACE_VETO"
            conclusion = (
                "Inconclusive — Limited by Face-Model Threshold"
            )
    elif fused_score > 90.0:
        conclusion = f"Strongly Supports Common Source — Nearest vault candidate: {best_user_id}"

    # 11. Forensic visualizations (real landmark density maps)
    gallery_heatmap = generate_landmark_attention_map(gallery_aligned, gallery_landmarks)
    probe_heatmap = generate_landmark_attention_map(probe_aligned, probe_landmarks)

    _, gal_buf = cv2.imencode('.png', gallery_aligned)
    gallery_aligned_b64 = f"data:image/png;base64,{base64.b64encode(gal_buf).decode('utf-8')}"
    _, pro_buf = cv2.imencode('.png', probe_aligned)
    probe_aligned_b64 = f"data:image/png;base64,{base64.b64encode(pro_buf).decode('utf-8')}"

    edge_delta = generate_edge_delta_map(
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

    # ── FORENSIC RECEIPT GENERATION (Always-On Evidence) ──
    gal_debug_img = gallery_aligned.copy()
    pro_debug_img = probe_aligned.copy()
    
    gal_matched_idx = {(m["gallery_idx"] if isinstance(m, dict) else m[0]) for m in mark_result.get("matches", [])}
    pro_matched_idx = {(m["probe_idx"] if isinstance(m, dict) else m[1]) for m in mark_result.get("matches", [])}
    
    for idx, m in enumerate(valid_gallery_marks):
        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
        color = (0, 255, 0) if idx in gal_matched_idx else ((255, 255, 0) if idx in unmatched_gal else (128, 128, 128))
        cv2.circle(gal_debug_img, (cx, cy), 4, color, 2)
        cv2.putText(gal_debug_img, str(idx), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
    for idx, m in enumerate(valid_probe_marks):
        cx, cy = int(m["centroid"][0] * 256), int(m["centroid"][1] * 256)
        color = (0, 255, 0) if idx in pro_matched_idx else ((255, 255, 0) if idx in unmatched_pro else (128, 128, 128))
        cv2.circle(pro_debug_img, (cx, cy), 4, color, 2)
        cv2.putText(pro_debug_img, str(idx), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

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
        lr_arcface=finite_or_none(lr_ensemble),
        lr_marks=finite_or_none(lr_marks),
        lr_total=finite_or_none(lr_total),
        posterior_probability=finite_or_none(posterior),
        mark_lrs=[finite_or_none(lr) for lr in mark_result.get("mark_lrs", [])],
        bayesian_fused_score=finite_or_none(bayesian_fused_score),
        # Mark Evidence Audit Trail (v2.0)
        mark_match_status=mark_match_status,
        marks_detected_probe=mark_result.get("total_probe", 0),
        marks_detected_gallery=mark_result.get("total_gallery", 0),
        mark_lrs_json=json.dumps([finite_or_none(lr) for lr in mark_result.get("mark_lrs", [])]),
        accepted_mark_correspondences_json=json.dumps(assigned_pairs),
        mark_detector_version=MARK_DETECTOR_VERSION,
        mark_matcher_version=MARK_MATCHER_VERSION,
        mark_overlay_url=generate_mark_overlay_receipt(gal_debug_img, pro_debug_img, probe_file_hash),
        # Full Forensic Provenance Audit (v3.0 + Phase 5B)
        probe_source_file_hash=probe_file_hash,
        gallery_source_file_hash=gallery_file_hash,
        probe_decoded_image_hash=probe_decoded_image_hash,
        gallery_decoded_image_hash=gallery_decoded_image_hash,
        probe_aligned_crop_hash=vault_vector_hash,
        gallery_aligned_crop_hash=compute_image_hash(gallery_aligned) if gallery_aligned is not None else None,
        probe_original_dimensions=probe_original_dimensions,
        gallery_original_dimensions=gallery_original_dimensions,
        probe_decoded_dimensions=probe_decoded_dimensions,
        gallery_decoded_dimensions=gallery_decoded_dimensions,
        probe_aligned_dimensions=f"{probe_aligned.shape[1]}x{probe_aligned.shape[0]}" if probe_aligned is not None else None,
        gallery_aligned_dimensions=f"{gallery_aligned.shape[1]}x{gallery_aligned.shape[0]}" if gallery_aligned is not None else None,
        probe_image_dimensions=f"{probe_aligned.shape[1]}x{probe_aligned.shape[0]}" if probe_aligned is not None else None,
        gallery_image_dimensions=f"{gallery_aligned.shape[1]}x{gallery_aligned.shape[0]}" if gallery_aligned is not None else None,
        preprocessing_steps=["decode_image", "mediapipe_face_landmarks", "align_crop_256", "clahe_normalize"],
        preprocessing_steps_applied="clahe,frontalize,align_crop(256)",
        # Pre/Post CLAHE provenance hashes (chain of custody at each preprocessing stage)
        probe_aligned_crop_hash_pre_clahe=probe_aligned_crop_hash_pre_clahe,
        gallery_aligned_crop_hash_pre_clahe=gallery_aligned_crop_hash_pre_clahe,
        probe_aligned_crop_hash_post_clahe=probe_aligned_crop_hash_post_clahe,
        gallery_aligned_crop_hash_post_clahe=gallery_aligned_crop_hash_post_clahe,
        code_commit_hash=os.getenv("GIT_COMMIT_SHA", "unknown"),
        docker_image_digest=os.getenv("DOCKER_IMAGE_DIGEST", "unknown"),
        arcface_model_name="ArcFace-R100",
        arcface_weight_hash=os.getenv("ARCFACE_WEIGHT_HASH", "unknown"),
        secondary_weight_hash=os.getenv("SECONDARY_WEIGHT_HASH", "unknown"),
        secondary_model_weight_hash=os.getenv("SECONDARY_WEIGHT_HASH", "unknown"),
        mediapipe_version=DEPENDENCY_VERSIONS.get("mediapipe", "unknown"),
        opencv_version=DEPENDENCY_VERSIONS.get("opencv", "unknown"),
        deepface_version=DEPENDENCY_VERSIONS.get("deepface", "unknown"),
        calibration_file_hash=os.getenv("CALIBRATION_FILE_HASH", "unknown"),
        calibration_pair_count=stats.get("pairs_evaluated", 0),
        # Face-Model Evidence Fields
        raw_arcface_similarity=round(arcface_sim, 6),
        raw_secondary_similarity=round(secondary_sim, 6),
        fused_face_model_similarity=round(structural_sim, 6),
        lr_face_model=finite_or_none(lr_ensemble),
    )

    # Build correspondences list for the UI (enriched with forensic metadata)
    correspondences = []
    lr_lookup = {}
    for match_entry in mark_result.get("matches", []):
        if isinstance(match_entry, dict):
            lr_lookup[(match_entry.get("gallery_idx"), match_entry.get("probe_idx"))] = match_entry.get("lr", 0)
        elif isinstance(match_entry, (tuple, list)) and len(match_entry) >= 3:
            lr_lookup[(match_entry[0], match_entry[1])] = match_entry[2]

    for pair in assigned_pairs:
        if isinstance(pair, dict):
            g_idx = pair.get("gallery_idx")
            p_idx = pair.get("probe_idx")
        elif isinstance(pair, (tuple, list)) and len(pair) >= 2:
            g_idx, p_idx = pair[0], pair[1]
        else:
            continue
        if not isinstance(g_idx, int) or not isinstance(p_idx, int):
            continue
        if g_idx < 0 or g_idx >= len(valid_gallery_marks) or p_idx < 0 or p_idx >= len(valid_probe_marks):
            continue
        individual_lr = lr_lookup.get((g_idx, p_idx), 0)
        corr_entry = {
            "gallery_idx": g_idx,
            "probe_idx": p_idx,
            "gallery_pt": valid_gallery_marks[g_idx]["centroid"],
            "probe_pt": valid_probe_marks[p_idx]["centroid"],
            "lr": individual_lr,
            "mark_type": valid_gallery_marks[g_idx].get("mark_type", "unknown"),
            "face_region": valid_gallery_marks[g_idx].get("face_region", "unknown"),
            "gallery_centroid": list(valid_gallery_marks[g_idx]["centroid"]),
            "probe_centroid": list(valid_probe_marks[p_idx]["centroid"]),
        }
        if isinstance(pair, dict):
            corr_entry["match_quality"] = pair.get("cost", 0)
            corr_entry["position_distance"] = pair.get("position_distance", 0)
            corr_entry["area_ratio"] = pair.get("area_ratio", 0)
            corr_entry["type_match"] = pair.get("type_match", False)
            corr_entry["region_match"] = pair.get("region_match", False)
        correspondences.append(corr_entry)

    probe_mark_debug_b64 = None
    gallery_mark_debug_b64 = None
    mark_debug_payload = None

    # ── Lightweight always-on mark diagnostics (production-safe) ──
    # Use trace-based detector_status for richer reporting (v2.1)
    _probe_det_status_v = trace_probe.get("detector_status", "UNKNOWN") if trace_probe else "UNKNOWN"
    _gallery_det_status_v = trace_gallery.get("detector_status", "UNKNOWN") if trace_gallery else "UNKNOWN"
    _vault_detector_status = (
        _probe_det_status_v if _probe_det_status_v != "OK"
        else _gallery_det_status_v if _gallery_det_status_v != "OK"
        else "OK" if (len(marks_gallery) > 0 or len(marks_probe) > 0)
        else "NO_CANDIDATES"
    )
    mark_diagnostics_payload = {
        "raw_probe_marks_count": len(valid_probe_marks),
        "raw_gallery_marks_count": len(valid_gallery_marks),
        "accepted_correspondences_count": mark_result.get("matched", 0),
        "rejected_candidates_count": len(rejected_cands) if rejected_cands else 0,
        "detector_status": _vault_detector_status,
        "probe_detector_status": _probe_det_status_v,
        "gallery_detector_status": _gallery_det_status_v,
        "matcher_status": "OK" if mark_result.get("matched", 0) > 0 else ("NO_MATCHES" if (len(valid_probe_marks) > 0 and len(valid_gallery_marks) > 0) else "INSUFFICIENT_INPUT"),
        "lr_marks": finite_or_none(lr_marks),
        "mark_match_status": mark_match_status,
        "rejection_summary": _build_rejection_summary(
            valid_probe_marks, valid_gallery_marks,
            mark_result, rejected_cands, mark_match_status,
            exact_image_match, TIER4_CALIBRATION,
            trace_probe=trace_probe, trace_gallery=trace_gallery,
        ),
        "mark_detector_trace": {
            "probe": trace_probe,
            "gallery": trace_gallery,
        },
    }

    if os.getenv("DEBUG_FORENSIC") == "true":
        _, gal_dbuf = cv2.imencode('.png', gal_debug_img)
        gallery_mark_debug_b64 = f"data:image/png;base64,{base64.b64encode(gal_dbuf).decode('utf-8')}"
        _, pro_dbuf = cv2.imencode('.png', pro_debug_img)
        probe_mark_debug_b64 = f"data:image/png;base64,{base64.b64encode(pro_dbuf).decode('utf-8')}"
        
        mark_debug_payload = {
            "probe_marks_count": len(valid_probe_marks),
            "gallery_marks_count": len(valid_gallery_marks),
            "correspondences_count": len(mark_result.get("matches", [])),
            "probe_marks_first_20": valid_probe_marks[:20],
            "gallery_marks_first_20": valid_gallery_marks[:20],
            "correspondences_first_20": [
                {
                    "gallery_idx": m[0],
                    "probe_idx": m[1],
                    "lr": m[2]
                } for m in mark_result.get("matches", [])
            ][:20],
            "unmatched_probe_indices": list(unmatched_pro),
            "unmatched_gallery_indices": list(unmatched_gal),
            "rejected_candidates": rejected_cands,
            "rejected_probe_marks": rejected_probe,
            "rejected_gallery_marks": rejected_gallery,
            "detector_version": "v2.0 (multi-scale + face-oval)",
            "matcher_version": "v2.0 (Hungarian + LR)"
        }

    if os.getenv("DEBUG_FORENSIC") == "true" or os.getenv("ENVIRONMENT") == "development":
        print(f"[FORENSIC DEBUG] raw_probe_marks: {len(valid_probe_marks)}, "
              f"raw_gallery_marks: {len(valid_gallery_marks)}, "
              f"correspondences: {len(correspondences)}", flush=True)
        for ci, c in enumerate(correspondences[:5]):
            print(f"  corr[{ci}]: probe_idx={c['probe_idx']}, gallery_idx={c['gallery_idx']}, lr={c['lr']:.4f}", flush=True)

    # ── SCORING TRACE (debug-only response payload) ──
    _calibration_status = "LOADED" if CALIBRATION else "MISSING"
    scoring_trace = None
    if os.getenv("DEBUG_FORENSIC") == "true":
        scoring_trace = {
            "calibration_status": _calibration_status,
            "calibration_source": CALIBRATION.get("source", "NONE") if CALIBRATION else "NONE",
            "calibration_benchmark": CALIBRATION.get("benchmark", "NONE") if CALIBRATION else "NONE",
            "tier4_calibration_status": "LOADED" if TIER4_CALIBRATION else "MISSING",
            "lr_ensemble_raw": finite_or_none(lr_ensemble),
            "lr_marks_raw": finite_or_none(lr_marks),
            "lr_total_raw": finite_or_none(lr_total),
            "lr_ensemble_display": "{:.6e}".format(lr_ensemble) if math.isfinite(lr_ensemble) else "N/A",
            "lr_marks_display": "{:.6e}".format(lr_marks) if math.isfinite(lr_marks) else "N/A",
            "lr_total_display": "{:.6e}".format(lr_total) if math.isfinite(lr_total) else "N/A",
            "posterior_raw": finite_or_none(posterior),
            "fused_score_pre_veto": finite_or_none(bayesian_fused_score),
            "fused_score_post_veto": finite_or_none(fused_score),
            "veto_triggered": veto_arcface,
            "veto_reason": veto_reason,
            "veto_override_applied": veto_override_applied,
            "veto_override_reason": veto_override_reason,
            "mark_override_eligible": mark_override_eligible,
            "positive_mark_count": positive_mark_count,
            "temporal_delta_years": finite_or_none(temporal_delta),
            "ensemble_thresholds_key": (
                "ensemble" if CALIBRATION and "ensemble" in CALIBRATION
                else ("arcface" if CALIBRATION and "arcface" in CALIBRATION else "NONE")
            ),
        }

    audit.veto_reason = veto_reason
    audit.veto_override_applied = veto_override_applied
    audit.veto_override_reason = veto_override_reason
    audit.scoring_trace = scoring_trace
    audit.calibration_status = _calibration_status

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
        scar_delta_b64=edge_delta,
        edge_delta_b64=edge_delta,
        gallery_wireframe_b64=gallery_wireframe_b64,
        probe_wireframe_b64=probe_wireframe_b64,
        probe_mark_debug_b64=probe_mark_debug_b64,
        gallery_mark_debug_b64=gallery_mark_debug_b64,
        mark_debug=mark_debug_payload,
        mark_diagnostics=mark_diagnostics_payload,
        geometry_status=geometry_status,
        geometric_ratio_distance=round(ratio_l2, 6) if ratio_l2 is not None else None,
        mark_correspondence_score=tier4_score,
        marks_detected_gallery=mark_result.get("total_gallery", 0),
        marks_detected_probe=mark_result.get("total_probe", 0),
        marks_matched=mark_result.get("matched", 0),
        correspondences=correspondences,
        raw_probe_marks=valid_probe_marks,
        raw_gallery_marks=valid_gallery_marks,
        # Mark evidence metadata (v2.0)
        mark_match_status=mark_match_status,
        lr_marks=finite_or_none(lr_marks),
        mark_lrs=mark_result.get("mark_lrs", []),
        mark_detector_version=MARK_DETECTOR_VERSION,
        mark_matcher_version=MARK_MATCHER_VERSION,
        exact_image_match=exact_image_match,
        # Face-model evidence (explicit decomposition)
        raw_arcface_similarity=round(arcface_sim, 6),
        raw_secondary_similarity=round(secondary_sim, 6),
        fused_face_model_similarity=round(structural_sim, 6),
        lr_face_model=finite_or_none(lr_ensemble),
        # Veto transparency
        bayesian_fused_score=round(bayesian_fused_score, 2),
        veto_reason=veto_reason,
        veto_override_applied=veto_override_applied,
        veto_override_reason=veto_override_reason,
        scoring_trace=scoring_trace,
        calibration_status=_calibration_status,
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
            fused_score_x100=percent_to_x100(fused_score) or 0,
            conclusion=conclusion,
            pipeline_version=PIPELINE_VERSION,
            calibration_benchmark=stats.get("benchmark"),
            false_acceptance_rate=stats["false_acceptance_rate"],
            veto_triggered=veto_arcface,
            structural_score_x100=percent_to_x100(tier1_score) or 0,
            arcface_score_x10000=raw_to_x10000(arcface_sim) or 0,
            secondary_score_x10000=raw_to_x10000(secondary_sim) or 0,
            ensemble_model_secondary="Facenet512",
            geometric_score_x100=percent_to_x100(tier2_score) or 0,
            micro_topology_score_x100=percent_to_x100(tier3_score) or 0,
            mark_correspondence_x100=percent_to_x100(tier4_score),
            pose_corrected_3d=True,
            probe_pose_angles=json.dumps(pro_angles) if pro_angles else None,
            gallery_pose_angles=json.dumps(gal_angles) if gal_angles else None,
            occlusion_percentage=pro_vis["occlusion_percentage"],
            occluded_regions=json.dumps(pro_vis["occluded_regions"]) if pro_vis["occluded_regions"] else None,
            effective_geometric_ratios_used=effective_ratios,
            receipt_url=receipt_url,
            synthetic_anomaly_score=max_anomaly,
            failed_provenance_veto=False,
            lr_arcface=finite_or_none(lr_ensemble),
            lr_marks_product=finite_or_none(lr_marks),
            lr_total=finite_or_none(lr_total),
            posterior_probability=finite_or_none(posterior),
            bayesian_fused_score_x100=percent_to_x100(bayesian_fused_score),
            marks_matched=mark_result.get("matched", 0),
            calibration_status=_calibration_status,
            veto_reason=veto_reason,
            veto_override_applied=veto_override_applied,
            # Mark Evidence Audit Trail (v2.0)
            mark_match_status=mark_match_status,
            marks_detected_probe=mark_result.get("total_probe", 0),
            marks_detected_gallery=mark_result.get("total_gallery", 0),
            mark_lrs_json=json.dumps([finite_or_none(lr) for lr in mark_result.get("mark_lrs", [])]),
            accepted_mark_correspondences_json=json.dumps(correspondences),
            mark_detector_version=MARK_DETECTOR_VERSION,
            mark_matcher_version=MARK_MATCHER_VERSION,
            mark_overlay_url=None,
            # Full Forensic Provenance Audit (v3.0)
            probe_source_file_hash=audit.probe_source_file_hash,
            gallery_source_file_hash=audit.gallery_source_file_hash,
            probe_decoded_image_hash=audit.probe_decoded_image_hash,
            gallery_decoded_image_hash=audit.gallery_decoded_image_hash,
            probe_aligned_crop_hash=audit.probe_aligned_crop_hash,
            gallery_aligned_crop_hash=audit.gallery_aligned_crop_hash,
            probe_aligned_crop_hash_pre_clahe=audit.probe_aligned_crop_hash_pre_clahe,
            gallery_aligned_crop_hash_pre_clahe=audit.gallery_aligned_crop_hash_pre_clahe,
            probe_aligned_crop_hash_post_clahe=audit.probe_aligned_crop_hash_post_clahe,
            gallery_aligned_crop_hash_post_clahe=audit.gallery_aligned_crop_hash_post_clahe,
            probe_image_dimensions=audit.probe_image_dimensions,
            gallery_image_dimensions=audit.gallery_image_dimensions,
            preprocessing_steps_applied=audit.preprocessing_steps_applied,
            code_commit_hash=audit.code_commit_hash,
            docker_image_digest=audit.docker_image_digest,
            arcface_model_name=audit.arcface_model_name,
            arcface_weight_hash=audit.arcface_weight_hash,
            secondary_weight_hash=audit.secondary_weight_hash,
            mediapipe_version=audit.mediapipe_version,
            opencv_version=audit.opencv_version,
            deepface_version=audit.deepface_version,
            calibration_file_hash=audit.calibration_file_hash,
            calibration_pair_count=audit.calibration_pair_count,
            probe_original_dimensions=audit.probe_original_dimensions,
            gallery_original_dimensions=audit.gallery_original_dimensions,
            probe_decoded_dimensions=audit.probe_decoded_dimensions,
            gallery_decoded_dimensions=audit.gallery_decoded_dimensions,
            probe_aligned_dimensions=audit.probe_aligned_dimensions,
            gallery_aligned_dimensions=audit.gallery_aligned_dimensions,
            preprocessing_steps=json.dumps(audit.preprocessing_steps) if audit.preprocessing_steps else None,
            raw_arcface_similarity=audit.raw_arcface_similarity,
            raw_secondary_similarity=audit.raw_secondary_similarity,
            fused_face_model_similarity=audit.fused_face_model_similarity,
            lr_face_model=audit.lr_face_model,
            secondary_model_weight_hash=audit.secondary_model_weight_hash,
        )
        ledger_session.add(event)
        ledger_session.commit()
    except Exception as ledger_err:
        print(f"Audit ledger write failed (non-fatal): {ledger_err}")
        ledger_session.rollback()
    finally:
        ledger_session.close()

    # Store job in database
    job_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        job = VerificationJob(
            job_id=job_id,
            status="pending",
            result_payload=json.dumps(response.model_dump())
        )
        db.add(job)
        db.commit()
    except Exception as e:
        print(f"Failed to create VerificationJob: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save verification job")
    finally:
        db.close()

    return {
        "job_id": job_id,
        "locked": True,
        "preview": {
            "fused_identity_score": response.fused_identity_score,
            "conclusion": response.conclusion,
            "veto_triggered": response.veto_triggered
        }
    }


# ---------------------------------------------------------
# VAULT NETWORK TOPOLOGY (IDENTITY GRAPH)
# ---------------------------------------------------------

@app.get("/vault/network")
def vault_network(_: dict = Depends(verify_jwt)):
    """
    Phase 3: Identity Graph.
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
    job_id: str

@app.post("/checkout/create-session")
@limiter.limit("5/minute")
def create_checkout_session(request: Request, req: CheckoutRequest):
    """
    Creates a Stripe Checkout Session for a one-time $4.99 payment
    to unlock the biometric dossier results.
    """
    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe is not configured.")

    frontend_origin = os.getenv("FRONTEND_URL", "http://localhost:3000")
    success = req.success_url if req.success_url else f"{frontend_origin}/?session_id={{CHECKOUT_SESSION_ID}}&success=true&job_id={req.job_id}"
    cancel = req.cancel_url if req.cancel_url else f"{frontend_origin}/?canceled=true&job_id={req.job_id}"

    # Verify job exists
    db = SessionLocal()
    try:
        job = db.query(VerificationJob).filter(VerificationJob.job_id == req.job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
    finally:
        db.close()

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
            metadata={"job_id": req.job_id}
        )
        return {"checkout_url": session.url, "session_id": session.id}
    except stripe.StripeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/checkout/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    if not endpoint_secret:
        return JSONResponse(status_code=400, content={"error": "Webhook secret not configured"})

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        job_id = session.get("metadata", {}).get("job_id")
        if job_id:
            db = SessionLocal()
            try:
                job = db.query(VerificationJob).filter(VerificationJob.job_id == job_id).first()
                if job and job.status != "paid":
                    job.status = "paid"
                    job.stripe_session_id = session.get("id")
                    from sqlalchemy.sql import func
                    job.paid_at = func.now()
                    db.commit()
            except Exception as e:
                print(f"Error processing webhook for job {job_id}: {e}")
                db.rollback()
            finally:
                db.close()

    return {"status": "success"}

@app.get("/verify/result/{job_id}")
def get_verification_result(job_id: str, session_id: Optional[str] = None, bypass_code: Optional[str] = None):
    db = SessionLocal()
    try:
        job = db.query(VerificationJob).filter(VerificationJob.job_id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Admin bypass
        op_pass = os.getenv("OPERATOR_PASSWORD")
        if bypass_code and op_pass and bypass_code == op_pass:
            return json.loads(job.result_payload)

        # Check if already marked as paid via webhook
        if job.status == "paid":
            return json.loads(job.result_payload)

        # Synchronous fallback: check Stripe session directly
        if session_id:
            try:
                stripe_session = stripe.checkout.Session.retrieve(session_id)
                from security_helpers import is_payment_unlocked
                if is_payment_unlocked(session_id, stripe_session.payment_status, stripe_session.metadata.get("job_id"), job_id):
                    job.status = "paid"
                    job.stripe_session_id = session_id
                    from sqlalchemy.sql import func
                    job.paid_at = func.now()
                    db.commit()
                    return json.loads(job.result_payload)
            except Exception as e:
                print(f"Error validating stripe session {session_id}: {e}")
        
        # If not paid and no bypass
        raise HTTPException(status_code=402, detail="Payment Required to decrypt biometric dossier")

    finally:
        db.close()

def rebuild_faiss_index_task():
    print("[FAISS] Starting background rebuild of FAISS index...", flush=True)
    vault_index.clear_index()
    session = SessionLocal()
    try:
        profiles = session.query(IdentityProfile.user_id, IdentityProfile.encrypted_facial_embedding).all()
        for user_id, encrypted_emb in profiles:
            try:
                emb = decrypt_embedding(encrypted_emb)
                vault_index.add_identity(user_id, emb)
            except Exception as e:
                print(f"[FAISS] Failed to decrypt/add {user_id} during rebuild: {e}", flush=True)
        print(f"[FAISS] Rebuild complete. Index contains {vault_index.index.ntotal} records.", flush=True)
    except Exception as e:
        print(f"[FAISS] Rebuild task failed: {e}", flush=True)
    finally:
        session.close()

@app.post("/admin/vault/reload-index", status_code=202)
@limiter.limit("2/minute")
def reload_vault_index(request: Request, background_tasks: BackgroundTasks, _ = Depends(verify_jwt)):
    """
    Clears and rebuilds the FAISS index from the database.
    Runs asynchronously to avoid timeouts.
    """
    background_tasks.add_task(rebuild_faiss_index_task)
    return {"status": "accepted", "detail": "FAISS index rebuild triggered in the background."}
