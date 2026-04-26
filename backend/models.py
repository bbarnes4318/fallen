from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, Boolean, Text
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
# For Cloud Run, host is a unix socket: /cloudsql/hoppwhistle:us-central1:facial-db-instance
# For direct connection (crawler), set DATABASE_URL env var
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@127.0.0.1:5432/{DB_NAME}"
    if os.getenv("K_SERVICE"):  # Running in Cloud Run
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
# DATABASE INITIALIZATION
# ---------------------------------------------------------

def init_db():
    # pgvector extension creation removed as we are now storing encrypted blobs.
    Base.metadata.create_all(bind=engine)
