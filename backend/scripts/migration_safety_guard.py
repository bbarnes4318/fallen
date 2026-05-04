import os
import re

def is_destructive_sql(sql: str) -> bool:
    """Returns True if the SQL contains destructive operations."""
    sql_upper = sql.upper()
    destructive_patterns = [
        r'\bDROP\s+COLUMN\b',
        r'\bDROP\s+TABLE\b',
        r'\bTRUNCATE\b',
        r'\bDELETE\s+FROM\s+VERIFICATION_EVENTS\b',
        r'\bALTER\s+TABLE\s+.*?\s+DROP\b',
        r'\bALTER\s+TABLE\s+.*?\s+ALTER\s+COLUMN\b' # type changes
    ]
    for pattern in destructive_patterns:
        if re.search(pattern, sql_upper):
            return True
    return False

def check_destructive_operation_allowed(database_url: str, sql: str = None) -> None:
    """
    Checks if a destructive operation is allowed against the given database.
    Raises ValueError if the operation is blocked.
    """
    # If a specific SQL statement is provided and it's not destructive, allow it.
    if sql and not is_destructive_sql(sql):
        return

    # Check connection string for production indicators
    is_prod_db = "facial_db" in database_url.lower()
    is_cloud_sql = "hoppwhistle" in database_url.lower() or "facial-pg-instance" in database_url.lower()
    
    # 1. facial_db unconditionally rejects destructive operations
    if is_prod_db:
        raise ValueError("CRITICAL: Destructive operations are strictly prohibited against the production facial_db.")
    
    # 2. Cloud SQL instance unconditionally rejects destructive operations
    if is_cloud_sql:
        raise ValueError("CRITICAL: Destructive operations are strictly prohibited against Cloud SQL instances.")

    # 3. For any other DB, require explicit opt-in
    env = os.getenv("ENVIRONMENT", "").lower()
    allow_override = os.getenv("ALLOW_DESTRUCTIVE_SCHEMA_OPS", "").lower() == "true"
    
    if env != "development" or not allow_override:
        raise ValueError("CRITICAL: Destructive operations require ENVIRONMENT=development and ALLOW_DESTRUCTIVE_SCHEMA_OPS=true.")

    # If all checks pass, it's allowed (e.g. against a local test db with flags)
