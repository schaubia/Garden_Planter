"""
Diagnostic script for Garden Planner Streamlit deployment
Run this to check what's available in your environment
"""

import sys
import os

print("=" * 60)
print("GARDEN PLANNER - DIAGNOSTIC REPORT")
print("=" * 60)
print()

print(f"Python Version: {sys.version}")
print(f"Current Working Directory: {os.getcwd()}")
print()

print("=" * 60)
print("FILES IN CURRENT DIRECTORY")
print("=" * 60)
try:
    files = sorted(os.listdir('.'))
    for f in files:
        size = os.path.getsize(f) if os.path.isfile(f) else "DIR"
        print(f"  {'[D]' if os.path.isdir(f) else '[F]'} {f:40s} {size}")
except Exception as e:
    print(f"ERROR listing files: {e}")
print()

print("=" * 60)
print("CHECKING REQUIRED FILES")
print("=" * 60)

required_files = {
    'garden_planner_core.py': 'REQUIRED - Core functionality',
    'pfaf2.csv': 'REQUIRED - Plant database',
    'companion_plants.csv': 'OPTIONAL - Companion plant data',
    'streamlit_app.py': 'REQUIRED - Streamlit app',
    'requirements.txt': 'REQUIRED - Dependencies'
}

for filename, description in required_files.items():
    exists = os.path.exists(filename)
    status = "✓ FOUND" if exists else "✗ MISSING"
    print(f"  {status:12s} {filename:30s} - {description}")
print()

print("=" * 60)
print("TESTING IMPORTS")
print("=" * 60)

# Test standard imports
standard_packages = ['pandas', 'numpy', 'matplotlib', 'sklearn', 'requests', 'geopy', 'meteostat']
for pkg in standard_packages:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError as e:
        print(f"  ✗ {pkg} - {e}")
print()

# Test garden_planner_core import
print("=" * 60)
print("TESTING garden_planner_core.py IMPORT")
print("=" * 60)

if os.path.exists('garden_planner_core.py'):
    print("File exists, attempting import...")
    try:
        import garden_planner_core
        print("✓ Successfully imported garden_planner_core")
        print()
        print("Available classes/functions:")
        for item in dir(garden_planner_core):
            if not item.startswith('_'):
                print(f"  - {item}")
    except Exception as e:
        print(f"✗ Failed to import garden_planner_core")
        print(f"Error: {e}")
        print()
        import traceback
        print("Full traceback:")
        traceback.print_exc()
else:
    print("✗ garden_planner_core.py not found")
print()

print("=" * 60)
print("PYTHON PATH")
print("=" * 60)
for path in sys.path:
    print(f"  {path}")
print()

print("=" * 60)
print("END OF DIAGNOSTIC REPORT")
print("=" * 60)
