import os
import sys
from sqlalchemy import inspect, text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import engine

def migrate():
    print("Starting Phase 5B Forensic Provenance Migration...")
    
    inspector = inspect(engine)
    if "verification_events" not in inspector.get_table_names():
        print("Table 'verification_events' does not exist. Please run init_db() first.")
        return

    columns = [col['name'] for col in inspector.get_columns("verification_events")]
    
    new_columns = {
        "probe_original_dimensions": "VARCHAR(32)",
        "gallery_original_dimensions": "VARCHAR(32)",
        "probe_decoded_dimensions": "VARCHAR(32)",
        "gallery_decoded_dimensions": "VARCHAR(32)",
        "probe_aligned_dimensions": "VARCHAR(32)",
        "gallery_aligned_dimensions": "VARCHAR(32)",
        "preprocessing_steps": "TEXT",
        "secondary_model_weight_hash": "VARCHAR(64)",
        "raw_arcface_similarity": "FLOAT",
        "raw_secondary_similarity": "FLOAT",
        "fused_face_model_similarity": "FLOAT",
        "lr_face_model": "FLOAT"
    }

    added_columns = []
    skipped_columns = []

    with engine.begin() as conn:
        for col_name, col_type in new_columns.items():
            if col_name not in columns:
                query = text(f"ALTER TABLE verification_events ADD COLUMN {col_name} {col_type};")
                conn.execute(query)
                added_columns.append(col_name)
            else:
                skipped_columns.append(col_name)

    print(f"Migration completed successfully.")
    print(f"Added columns ({len(added_columns)}): {', '.join(added_columns) if added_columns else 'None'}")
    print(f"Skipped columns ({len(skipped_columns)}): {', '.join(skipped_columns) if skipped_columns else 'None'}")

if __name__ == "__main__":
    migrate()
