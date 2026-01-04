"""
Quick diagnostic to understand your code structure
Run this to see what's available in your files
"""

import os
import sys

print("="*60)
print("GARDEN PLANNER DIAGNOSTIC")
print("="*60)
print()

# Check files exist
print("1. CHECKING FILES:")
print("-" * 40)
files_to_check = [
    'garden_planner_main.py',
    'garden_planner_core.py', 
    'pfaf2.csv',
    'companion_plants.csv'
]

for f in files_to_check:
    exists = "✓" if os.path.exists(f) else "✗"
    print(f"  {exists} {f}")
print()

# Check garden_planner_core.py structure
print("2. GARDEN_PLANNER_CORE.PY CONTENTS:")
print("-" * 40)
try:
    import garden_planner_core as gpc
    
    functions = [item for item in dir(gpc) if callable(getattr(gpc, item)) and not item.startswith('_')]
    classes = [item for item in dir(gpc) if type(getattr(gpc, item)) == type and not item.startswith('_')]
    
    print(f"  Classes found: {len(classes)}")
    for cls in classes:
        print(f"    - {cls}")
    
    print(f"\n  Functions found: {len(functions)}")
    for func in functions:
        print(f"    - {func}")
    
    # Check for main
    if hasattr(gpc, 'main'):
        print("\n  ✓ Has main() function")
        import inspect
        sig = inspect.signature(gpc.main)
        print(f"    Signature: {sig}")
    else:
        print("\n  ✗ No main() function")
    
except Exception as e:
    print(f"  ✗ Error importing: {e}")
    import traceback
    traceback.print_exc()

print()

# Check garden_planner_main.py structure  
print("3. GARDEN_PLANNER_MAIN.PY STRUCTURE:")
print("-" * 40)
try:
    with open('garden_planner_main.py', 'r') as f:
        lines = f.readlines()
    
    print(f"  Total lines: {len(lines)}")
    print("\n  First 30 lines:")
    print("  " + "-" * 38)
    for i, line in enumerate(lines[:30], 1):
        print(f"  {i:3d}: {line.rstrip()}")
    
    # Look for key patterns
    print("\n  Key patterns found:")
    has_main = any('def main(' in line for line in lines)
    has_if_main = any("if __name__ == '__main__':" in line or 'if __name__ == "__main__":' in line for line in lines)
    has_argparse = any('argparse' in line for line in lines)
    has_input = any('input(' in line for line in lines)
    
    print(f"    - def main(): {has_main}")
    print(f"    - if __name__ == '__main__': {has_if_main}")
    print(f"    - Uses argparse: {has_argparse}")
    print(f"    - Uses input(): {has_input}")
    
except Exception as e:
    print(f"  ✗ Error reading: {e}")

print()
print("="*60)
print("RECOMMENDATIONS:")
print("="*60)

print("""
Based on the diagnostic above, the streamlit app needs to:

1. If garden_planner_core.py has a main() function:
   → Call it directly with parameters

2. If garden_planner_main.py uses argparse:
   → Set sys.argv before running

3. If garden_planner_main.py uses input():
   → Mock the input() function

4. If neither of above:
   → Share the code structure to create a custom integration

Please share this output!
""")
