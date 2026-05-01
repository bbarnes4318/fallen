import os
import sys
from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import engine, init_db

with engine.connect() as conn:
    res = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
    print("Tables:", [r[0] for r in res])

print("Running init_db() to ensure tables exist...")
init_db()
print("init_db() finished.")
