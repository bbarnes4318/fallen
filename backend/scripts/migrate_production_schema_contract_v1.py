import os
import sys
import json
import argparse
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import ProgrammingError
from dotenv import load_dotenv

# Ensure safety guard is available
try:
    from migration_safety_guard import check_destructive_operation_allowed
except ImportError:
    print("CRITICAL: Safety guard module not found.")
    sys.exit(1)

# Try to load env
load_dotenv(dotenv_path='../.env.gcp')

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS") or os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DATABASE_URL = os.getenv("DATABASE_URL")
CLOUD_SQL_CONNECTION_NAME = os.getenv("CLOUD_SQL_CONNECTION_NAME") or os.getenv("SQL_CONNECTION_NAME")

if not DATABASE_URL:
    if not all([DB_USER, DB_PASS, DB_NAME]):
        print("CRITICAL: Missing required database credentials. Must supply DATABASE_URL or DB_USER, DB_PASS, DB_NAME.")
        sys.exit(1)
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@127.0.0.1:5433/{DB_NAME}"

engine = create_engine(DATABASE_URL)

CANONICAL_COLUMNS = {
    "arcface_score_x10000": "INTEGER",
    "secondary_score_x10000": "INTEGER",
    "ensemble_model_secondary": "VARCHAR(64)",
    "pose_corrected_3d": "BOOLEAN",
    "probe_pose_angles": "TEXT",
    "gallery_pose_angles": "TEXT",
    "occlusion_percentage": "DOUBLE PRECISION",
    "occluded_regions": "TEXT",
    "effective_geometric_ratios_used": "INTEGER",
    "estimated_temporal_delta": "DOUBLE PRECISION",
    "cross_spectral_correction_applied": "BOOLEAN",
    "mark_correspondence_x100": "INTEGER",
    "probe_aligned_crop_hash_pre_clahe": "VARCHAR(64)",
    "gallery_aligned_crop_hash_pre_clahe": "VARCHAR(64)",
    "probe_aligned_crop_hash_post_clahe": "VARCHAR(64)",
    "gallery_aligned_crop_hash_post_clahe": "VARCHAR(64)",
    "probe_original_dimensions": "VARCHAR(32)",
    "gallery_original_dimensions": "VARCHAR(32)",
    "probe_decoded_dimensions": "VARCHAR(32)",
    "gallery_decoded_dimensions": "VARCHAR(32)",
    "probe_aligned_dimensions": "VARCHAR(32)",
    "gallery_aligned_dimensions": "VARCHAR(32)",
    "preprocessing_steps": "TEXT",
    "secondary_model_weight_hash": "VARCHAR(64)",
    "raw_arcface_similarity": "DOUBLE PRECISION",
    "raw_secondary_similarity": "DOUBLE PRECISION",
    "fused_face_model_similarity": "DOUBLE PRECISION",
    "lr_face_model": "DOUBLE PRECISION",
    "synthetic_anomaly_score": "DOUBLE PRECISION",
    "failed_provenance_veto": "BOOLEAN",
    "receipt_url": "TEXT"
}

def migrate(dry_run=False, output_json=False):
    inspector = inspect(engine)
    if "verification_events" not in inspector.get_table_names():
        if output_json:
            print(json.dumps({"error": "Table 'verification_events' does not exist."}))
        else:
            print("CRITICAL: Table 'verification_events' does not exist.")
        sys.exit(1)

    existing_columns = {col["name"] for col in inspector.get_columns("verification_events")}
    
    results = {"added": [], "skipped": [], "errors": []}
    
    with engine.begin() as conn:
        for col_name, col_type in CANONICAL_COLUMNS.items():
            if col_name in existing_columns:
                results["skipped"].append(col_name)
                if not output_json:
                    print(f"SKIPPED: {col_name} (already exists)")
            else:
                results["added"].append(col_name)
                stmt = f"ALTER TABLE verification_events ADD COLUMN {col_name} {col_type};"
                
                # Check guard per statement
                try:
                    check_destructive_operation_allowed(DATABASE_URL, sql=stmt)
                except ValueError as e:
                    results["errors"].append({col_name: str(e)})
                    if not output_json:
                        print(f"ERROR: Safety guard blocked adding {col_name} - {str(e)}")
                    continue
                
                if not dry_run:
                    try:
                        conn.execute(text(stmt))
                        if not output_json:
                            print(f"ADDED: {col_name} ({col_type})")
                    except Exception as e:
                        results["errors"].append({col_name: str(e)})
                        if not output_json:
                            print(f"ERROR: Failed to add {col_name} - {str(e)}")
                else:
                    if not output_json:
                        print(f"DRY RUN (WOULD ADD): {col_name} ({col_type})")

    if output_json:
        print(json.dumps(results, indent=2))
    else:
        if dry_run:
            print(f"\nDRY RUN COMPLETE: Would add {len(results['added'])} columns, skipped {len(results['skipped'])}.")
        else:
            print(f"\nMIGRATION COMPLETE: Added {len(results['added'])} columns, skipped {len(results['skipped'])}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrate canonical schema contract.')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    args = parser.parse_args()
    migrate(dry_run=args.dry_run, output_json=args.json)
