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
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS lr_arcface DOUBLE PRECISION;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS lr_marks_product DOUBLE PRECISION;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS lr_total DOUBLE PRECISION;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS posterior_probability DOUBLE PRECISION;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS bayesian_fused_score_x100 INTEGER;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS marks_matched INTEGER;",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS calibration_status VARCHAR(32);",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS veto_reason VARCHAR(64);",
                "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS veto_override_applied BOOLEAN DEFAULT FALSE;"
            ]
            
            for stmt in statements:
                logger.info(f"Executing: {stmt}")
                conn.execute(text(stmt))
                
            conn.commit()
            logger.info("Migration completed successfully.")
            
    except Exception as e:
        logger.error(f"CRITICAL: Migration failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_migration()
