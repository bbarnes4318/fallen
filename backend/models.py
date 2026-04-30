from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, Boolean, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
import os

# ---------------------------------------------------------
# DATABASE CONNECTION SETUP
# ---------------------------------------------------------
# GCP Cloud SQL PostgreSQL connection string format
# Example: postgresql+psycopg2://<DB_USER>:<DB_PASS>@/<DB_NAME>?host=/cloudsql/<PROJECT_ID>:<REGION>:<INSTANCE_NAME>
DB_USER = os.getenv("DB_USER", "facial_app_user")
DB_PASS = os.getenv("DB_PASS", "SuperSecretPassword123!")
DB_NAME = os.getenv("DB_NAME", "facial_db")
CLOUD_SQL_CONNECTION_NAME = os.getenv("CLOUD_SQL_CONNECTION_NAME", "hoppwhistle:us-central1:facial-db-instance")

# For local development via Cloud SQL Auth Proxy, host is typically 127.0.0.1
# For Cloud Run Services/Jobs, host is a unix socket: /cloudsql/<instance>
# For direct connection (crawler), set DATABASE_URL env var
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@127.0.0.1:5432/{DB_NAME}"
    if os.getenv("K_SERVICE") or os.getenv("CLOUD_RUN_JOB"):  # Cloud Run Service or Job
        DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@/{DB_NAME}?host=/cloudsql/{CLOUD_SQL_CONNECTION_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------------------------------------------
# SCHEMAS (ENCRYPTED)
# ---------------------------------------------------------

class IdentityProfile(Base):
    """
    Core schema for a user's biometric identity profile.
    Utilizes Application-Level Envelope Encryption to store the 512-dimensional CNN facial embedding
    as an encrypted blob (LargeBinary), eliminating plain-text biometric exposure.
    Includes Wikimedia Commons legal attribution metadata for crawled targets.
    """
    __tablename__ = "identity_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, index=True, nullable=False)
    
    # Encrypted 512-D embedding payload (contains KMS-encrypted DEK + payload)
    encrypted_facial_embedding = Column(LargeBinary, nullable=False)
    
    # Store auxiliary data for Tier 2 and Tier 3 checks
    soft_biometrics_hash = Column(String(512), nullable=True) # E.g., geospatial mole map hash
    lbp_texture_hash = Column(String(512), nullable=True)

    # ── Wikimedia Legal Attribution Metadata ──
    image_url = Column(Text, nullable=True)
    thumbnail_url = Column(Text, nullable=True)
    file_page_url = Column(Text, nullable=True)
    creator = Column(Text, nullable=True)
    license_short_name = Column(String(255), nullable=True)
    license_url = Column(Text, nullable=True)
    credit = Column(Text, nullable=True)
    attribution_required = Column(Boolean, nullable=True, default=False)
    source = Column(Text, nullable=True)
    person_name = Column(Text, nullable=True)
    wikidata_id = Column(String(64), nullable=True, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# ---------------------------------------------------------
# IMMUTABLE FORENSIC AUDIT LEDGER
# ---------------------------------------------------------

class VerificationEvent(Base):
    """
    Immutable, append-only audit ledger for every biometric verification.
    Each row is a server-authoritative record of a pipeline transaction.
    fused_score_x100: Integer representation of percentage × 100
        (e.g. 99.50% → 9950) to eliminate floating-point drift in forensic records.
    """
    __tablename__ = "verification_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Chain of Custody — raw file hashes (SHA-256 of pre-decode byte stream)
    probe_hash = Column(String(64), nullable=False)
    gallery_hash = Column(String(64), nullable=True)

    # Match outcome
    matched_user_id = Column(String(255), nullable=True)
    fused_score_x100 = Column(Integer, nullable=False)  # 99.50% → 9950
    conclusion = Column(Text, nullable=False)

    # Pipeline provenance
    pipeline_version = Column(String(128), nullable=False)
    calibration_benchmark = Column(String(64), nullable=True)
    false_acceptance_rate = Column(String(128), nullable=True)
    veto_triggered = Column(Boolean, nullable=False, default=False)

    # Individual tier scores (×100 for consistency)
    structural_score_x100 = Column(Integer, nullable=True)
    geometric_score_x100 = Column(Integer, nullable=True)
    micro_topology_score_x100 = Column(Integer, nullable=True)

    # Neural Ensemble Audit Trail (Tier 1 v4.0)
    # Individual model scores before weighted fusion (×10000 for 4-decimal precision)
    arcface_score_x10000 = Column(Integer, nullable=True)       # Raw ArcFace cosine × 10000
    secondary_score_x10000 = Column(Integer, nullable=True)     # Raw secondary model cosine × 10000
    ensemble_model_secondary = Column(String(64), nullable=True) # Name of secondary model (e.g., 'Facenet512')

    # Tier 2: 3D Topographical Mapping Telemetry
    pose_corrected_3d = Column(Boolean, nullable=True)
    probe_pose_angles = Column(Text, nullable=True)     # JSON string of pitch, yaw, roll
    gallery_pose_angles = Column(Text, nullable=True)   # JSON string of pitch, yaw, roll
    occlusion_percentage = Column(Float, nullable=True)
    occluded_regions = Column(Text, nullable=True)      # JSON list of strings
    effective_geometric_ratios_used = Column(Integer, nullable=True)

    # Temporal & Spectral Telemetry (Tier 1/3)
    estimated_temporal_delta = Column(Float, nullable=True)
    cross_spectral_correction_applied = Column(Boolean, nullable=True, default=False)

    # Tier 4: Mark Correspondence (×100 for integer consistency)
    mark_correspondence_x100 = Column(Integer, nullable=True)

    # Bayesian Likelihood Ratio Audit Trail (Daubert v3.0)
    # Stored as Float for precision — these are scientific measurements, not financial values
    lr_arcface = Column(Float, nullable=True)
    lr_marks_product = Column(Float, nullable=True)
    lr_total = Column(Float, nullable=True)
    posterior_probability = Column(Float, nullable=True)

    # Phase 7: Synthetic Provenance Veto (Deepfake Detection)
    synthetic_anomaly_score = Column(Float, nullable=True)
    failed_provenance_veto = Column(Boolean, nullable=True, default=False)

    # Composite forensic receipt (GCS URI of stitched PNG)
    receipt_url = Column(Text, nullable=True)

# ---------------------------------------------------------
# DATABASE INITIALIZATION
# ---------------------------------------------------------

def init_db():
    # pgvector extension creation removed as we are now storing encrypted blobs.
    Base.metadata.create_all(bind=engine)
