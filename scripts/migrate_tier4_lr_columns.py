#!/usr/bin/env python3
"""
=============================================================================
  Fallen — SCHEMA MIGRATION: Bayesian LR Audit Columns
  Adds lr_arcface, lr_marks_product, lr_total, posterior_probability
  to the verification_events table (idempotent — safe to re-run).
=============================================================================
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.join(str(PROJECT_ROOT), "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text
from models import engine

COLUMNS_TO_ADD = [
    ("lr_arcface", "FLOAT"),
    ("lr_marks_product", "FLOAT"),
    ("lr_total", "FLOAT"),
    ("posterior_probability", "FLOAT"),
]

TABLE_NAME = "verification_events"


def main():
    print("═══ Fallen SCHEMA MIGRATION: Bayesian LR Columns ═══\n")

    with engine.connect() as conn:
        for col_name, col_type in COLUMNS_TO_ADD:
            # PostgreSQL: ADD COLUMN IF NOT EXISTS (PG 9.6+)
            sql = text(
                f"ALTER TABLE {TABLE_NAME} "
                f"ADD COLUMN IF NOT EXISTS {col_name} {col_type};"
            )
            try:
                conn.execute(sql)
                print(f"  ✓ {col_name} ({col_type}) — applied")
            except Exception as e:
                print(f"  ✗ {col_name} — ERROR: {e}")

        conn.commit()

    print(f"\n═══ MIGRATION COMPLETE ═══")


if __name__ == "__main__":
    main()
