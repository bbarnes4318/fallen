import os
import sys
from sqlalchemy import text

# Add the backend directory to sys.path so we can import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import engine

def migrate():
    print("Starting schema migration...")
    queries = [
        # Temporal/Spectral Engine
        "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS estimated_temporal_delta FLOAT;",
        "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS cross_spectral_correction_applied BOOLEAN DEFAULT FALSE;",
        
        # Phase 6 (Occlusion Recovery)
        "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS occlusion_percentage FLOAT;",
        "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS occluded_regions TEXT;",
        "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS effective_geometric_ratios_used INTEGER;",
        
        # Phase 7 (Anti-Spoofing)
        "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS synthetic_anomaly_score FLOAT;",
        "ALTER TABLE verification_events ADD COLUMN IF NOT EXISTS failed_provenance_veto BOOLEAN DEFAULT FALSE;"
    ]
    
    with engine.begin() as conn:
        for q in queries:
            print(f"Executing: {q}")
            conn.execute(text(q))
            
    print("Migration completed successfully.")

if __name__ == "__main__":
    migrate()
