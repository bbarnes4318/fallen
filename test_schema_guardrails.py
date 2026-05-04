import os
import re
import sys

def test_no_destructive_schema_operations_committed():
    """
    Scans the backend/scripts directory to ensure no destructive operations 
    are committed. The only exceptions are incident reports or approved test fixtures.
    """
    backend_scripts_dir = os.path.join(os.getcwd(), 'backend', 'scripts')
    
    destructive_patterns = [
        re.compile(r'\bDROP\s+COLUMN\b', re.IGNORECASE),
        re.compile(r'\bDROP\s+TABLE\b', re.IGNORECASE),
        re.compile(r'\bTRUNCATE\b', re.IGNORECASE),
        re.compile(r'\bDELETE\s+FROM\s+VERIFICATION_EVENTS\b', re.IGNORECASE),
        re.compile(r'rollback_migration', re.IGNORECASE)
    ]
    
    violating_files = []
    
    for root, dirs, files in os.walk(backend_scripts_dir):
        for file in files:
            if file.endswith('.py') and file != 'migration_safety_guard.py' and not file.startswith('test_'):
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for pattern in destructive_patterns:
                        if pattern.search(content):
                            violating_files.append((file_path, pattern.pattern))

    if len(violating_files) > 0:
        print(f"FAILED: Destructive operations found in committed scripts: {violating_files}")
        sys.exit(1)
    else:
        print("PASSED: test_no_destructive_schema_operations_committed")

def test_migration_safety_guard_exists():
    guard_path = os.path.join(os.getcwd(), 'backend', 'scripts', 'migration_safety_guard.py')
    if not os.path.exists(guard_path):
        print("FAILED: migration_safety_guard.py must exist to protect production DB.")
        sys.exit(1)
    print("PASSED: test_migration_safety_guard_exists")

def test_no_rollback_script_exists():
    rollback_path = os.path.join(os.getcwd(), 'backend', 'scripts', 'rollback_migration.py')
    if os.path.exists(rollback_path):
        print("FAILED: rollback_migration.py MUST NOT exist.")
        sys.exit(1)
    print("PASSED: test_no_rollback_script_exists")

def test_migration_safety_guard_logic():
    sys.path.insert(0, os.path.join(os.getcwd(), 'backend', 'scripts'))
    from migration_safety_guard import check_destructive_operation_allowed
    
    prod_url = "postgresql+psycopg2://facial_app:foo@127.0.0.1:5433/facial_db"
    
    # 1. Allowed: safe ADD COLUMN
    try:
        check_destructive_operation_allowed(prod_url, "ALTER TABLE verification_events ADD COLUMN test_col TEXT;")
    except ValueError:
        print("FAILED: Safety guard incorrectly blocked safe ADD COLUMN operation.")
        sys.exit(1)
        
    # 2. Blocked: DROP COLUMN
    try:
        check_destructive_operation_allowed(prod_url, "ALTER TABLE verification_events DROP COLUMN receipt_url;")
        print("FAILED: Safety guard allowed DROP COLUMN.")
        sys.exit(1)
    except ValueError:
        pass
        
    # 3. Blocked: DROP TABLE
    try:
        check_destructive_operation_allowed(prod_url, "DROP TABLE verification_events;")
        print("FAILED: Safety guard allowed DROP TABLE.")
        sys.exit(1)
    except ValueError:
        pass

    # 4. Blocked: TRUNCATE
    try:
        check_destructive_operation_allowed(prod_url, "TRUNCATE verification_events;")
        print("FAILED: Safety guard allowed TRUNCATE.")
        sys.exit(1)
    except ValueError:
        pass

    # 5. Blocked: DELETE FROM
    try:
        check_destructive_operation_allowed(prod_url, "DELETE FROM verification_events;")
        print("FAILED: Safety guard allowed DELETE.")
        sys.exit(1)
    except ValueError:
        pass
        
    print("PASSED: test_migration_safety_guard_logic")

if __name__ == '__main__':
    test_no_destructive_schema_operations_committed()
    test_migration_safety_guard_exists()
    test_no_rollback_script_exists()
    test_migration_safety_guard_logic()
    print("ALL TESTS PASSED")
