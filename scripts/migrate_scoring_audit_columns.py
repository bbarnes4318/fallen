import os
import sys
import logging
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_migration():
    """
    Idempotent migration to add Bayesian scoring audit columns to the verification_events table.
    """
    logger.info("Starting scoring audit column migration...")
    
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_NAME = os.getenv("DB_NAME")
    CLOUD_SQL_CONNECTION_NAME = os.getenv("CLOUD_SQL_CONNECTION_NAME")
    
    if not all([DB_USER, DB_PASS, DB_NAME, CLOUD_SQL_CONNECTION_NAME]):
        logger.error("CRITICAL: Missing database environment variables. Required: DB_USER, DB_PASS, DB_NAME, CLOUD_SQL_CONNECTION_NAME")
        sys.exit(1)
        
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@/{DB_NAME}?host=/cloudsql/{CLOUD_SQL_CONNECTION_NAME}"
    
    try:
        engine = create_engine(DATABASE_URL)
        # Test connection
        with engine.connect() as conn:
            logger.info("Successfully connected to database.")
            
            statements = [
                # Bayesian Scoring Audit (v1.0)
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS lr_arcface DOUBLE PRECISION;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS lr_marks_product DOUBLE PRECISION;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS lr_total DOUBLE PRECISION;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS posterior_probability DOUBLE PRECISION;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS bayesian_fused_score_x100 INTEGER;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS marks_matched INTEGER;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS calibration_status VARCHAR(32);",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS veto_reason VARCHAR(64);",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS veto_override_applied BOOLEAN DEFAULT FALSE;",
                # Mark Evidence Audit Trail (v2.0)
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS mark_match_status VARCHAR(32);",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS marks_detected_probe INTEGER;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS marks_detected_gallery INTEGER;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS mark_lrs_json TEXT;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS accepted_mark_correspondences_json TEXT;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS mark_detector_version VARCHAR(64);",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS mark_matcher_version VARCHAR(64);",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS mark_overlay_url TEXT;",
            ]
            
            for stmt in statements:
                logger.info(f"Executing: {stmt}")
                conn.execute(text(stmt))
                
            conn.commit()
            logger.info("Migration statements executed successfully.")
            
            # ── Post-migration verification ──
            required_columns = [
                "lr_arcface", "lr_marks_product", "lr_total", "posterior_probability",
                "bayesian_fused_score_x100", "marks_matched", "calibration_status",
                "veto_reason", "veto_override_applied",
                "mark_match_status", "marks_detected_probe", "marks_detected_gallery",
                "mark_lrs_json", "accepted_mark_correspondences_json",
                "mark_detector_version", "mark_matcher_version", "mark_overlay_url",
            ]
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'verification_events'"
            ))
            existing_columns = {row[0] for row in result}
            missing = [c for c in required_columns if c not in existing_columns]
            if missing:
                logger.error(f"CRITICAL: Post-migration verification FAILED. Missing columns: {missing}")
                sys.exit(1)
            logger.info(f"Post-migration verification passed. All {len(required_columns)} required columns present.")
            
    except Exception as e:
        logger.error(f"CRITICAL: Migration failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_migration()
