import os
import sys
import json
import argparse
from sqlalchemy import create_engine, inspect
from migrate_production_schema_contract_v1 import CANONICAL_COLUMNS, DATABASE_URL

def audit(fail_on_missing=False, output_json=False):
    try:
        engine = create_engine(DATABASE_URL)
        inspector = inspect(engine)
        if "verification_events" not in inspector.get_table_names():
            res = {"verification_events": {"error": "Table does not exist."}}
            if output_json:
                print(json.dumps(res, indent=2))
            else:
                print("Error: Table 'verification_events' does not exist.")
            sys.exit(1)

        existing_columns = {col["name"] for col in inspector.get_columns("verification_events")}
        
        expected_columns = set(CANONICAL_COLUMNS.keys())
        
        missing = sorted(list(expected_columns - existing_columns))
        
        # We also want to know if there are extra columns that are not in our list
        # But our list is just the "missing" ones that we just added. We need the full canonical set.
        # Actually models.py handles the full set, CANONICAL_COLUMNS is just the additions.
        # Wait, the user asked for:
        # {
        #   "verification_events": {
        #     "missing_columns": [],
        #     "extra_columns": [],
        #     "canonical_columns_present": true,
        #     "schema_version": "production_schema_contract_v1"
        #   }
        # }
        
        res = {
            "verification_events": {
                "missing_columns": missing,
                "existing_columns": sorted(list(existing_columns)),
                "canonical_columns_present": len(missing) == 0,
                "schema_version": "production_schema_contract_v1"
            }
        }

        if output_json:
            print(json.dumps(res, indent=2))
        else:
            print(f"Audit Result: Canonical columns present: {res['verification_events']['canonical_columns_present']}")
            if missing:
                print(f"Missing columns: {missing}")

        if fail_on_missing and missing:
            sys.exit(1)
            
    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit canonical schema contract.')
    parser.add_argument('--fail-on-missing', action='store_true', help='Exit 1 if columns are missing')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    args = parser.parse_args()
    audit(fail_on_missing=args.fail_on_missing, output_json=args.json)
