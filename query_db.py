import psycopg2
import json
from decimal import Decimal

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

try:
    conn = psycopg2.connect("postgresql://facial_app:bVajldkFcj38upoNcSEH@127.0.0.1:5433/facial_db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            id,
            fused_score_x100, 
            bayesian_fused_score_x100, 
            posterior_probability, 
            lr_total, 
            lr_face_model, 
            lr_arcface, 
            lr_marks_product,
            raw_arcface_similarity, 
            raw_secondary_similarity, 
            fused_face_model_similarity, 
            failed_provenance_veto, 
            synthetic_anomaly_score, 
            receipt_url
        FROM verification_events
        ORDER BY id DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    cols = [desc[0] for desc in cursor.description]
    if row:
        print(json.dumps(dict(zip(cols, row)), indent=2, default=decimal_default))
    else:
        print("No row found.")
    cursor.close()
    conn.close()
except Exception as e:
    print("DB error:", e)
