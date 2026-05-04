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

if __name__ == '__main__':
    test_no_destructive_schema_operations_committed()
    test_migration_safety_guard_exists()
    test_no_rollback_script_exists()
    print("ALL TESTS PASSED")
