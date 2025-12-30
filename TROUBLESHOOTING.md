# 🔧 Troubleshooting: "Setup Required" Error

## Problem
You're seeing this error even though the files are in your repository:
```
Setup Required
Please ensure the following files are present:
* garden_planner_core.py
* pfaf2.csv (plant database)
* companion_plants.csv (optional)
```

## Quick Fixes

### Fix 1: Use the Debug App (Recommended)

1. **Temporarily change your main file** in Streamlit Cloud:
   - Go to your app settings
   - Change "Main file path" from `streamlit_app.py` to `streamlit_debug.py`
   - Click "Save" and let it redeploy
   
2. **Check the debug output** to see exactly what's wrong

3. **Once fixed, switch back** to `streamlit_app.py`

### Fix 2: Verify File Structure

Your repository should look like this:
```
Garden_Planter/
├── streamlit_app.py          ← Main app
├── streamlit_debug.py         ← Debug app
├── garden_planner_core.py    ← MUST be in root
├── garden_planner_main.py    ← Original CLI app
├── pfaf2.csv                 ← MUST be in root
├── companion_plants.csv      ← Optional
├── requirements.txt          ← MUST be in root
├── .streamlit/
│   └── config.toml
└── jupyter_src/              ← Your notebooks
    └── ...
```

**Key Point:** All Python files and data files must be in the ROOT of your repository, not in subdirectories.

### Fix 3: Check for Common Issues

Run through this checklist:

#### ✅ Files Committed to Git?
```bash
git status
# Should show no uncommitted files
```

If you see uncommitted files:
```bash
git add garden_planner_core.py pfaf2.csv companion_plants.csv
git commit -m "Add missing files"
git push origin main
```

#### ✅ Files on GitHub?
- Go to https://github.com/schaubia/Garden_Planter
- You should see `garden_planner_core.py` in the file list
- Click it to verify the content is there

#### ✅ Correct File Names?
Check for typos:
- `garden_planner_core.py` (not `Garden_Planner_Core.py`)
- No extra spaces
- Correct extension (.py not .txt)

#### ✅ File Sizes?
```bash
ls -lh *.py *.csv
```
Make sure files aren't empty (0 bytes)

### Fix 4: Check Dependencies

Your `garden_planner_core.py` might import packages that aren't in `requirements.txt`.

**Check what it imports:**
```bash
grep "^import\|^from" garden_planner_core.py
```

**Common missing packages:**
- sqlite3 (built-in, don't add)
- json (built-in, don't add)
- datetime (built-in, don't add)
- pandas, numpy, sklearn, requests, geopy, meteostat (ADD to requirements.txt)

### Fix 5: Test Locally First

Before deploying, test on your computer:

```bash
# Install requirements
pip install -r requirements.txt

# Test import directly
python -c "from garden_planner_core import Config"

# If that works, run the app
streamlit run streamlit_app.py
```

If it fails locally, fix those errors first.

### Fix 6: Check Streamlit Cloud Logs

1. Go to your app on Streamlit Cloud
2. Click the hamburger menu (≡) in the top-right
3. Click "Manage app"
4. Check the logs for the actual error message
5. Look for lines with "ERROR" or "ImportError"

## Detailed Diagnostics

### Step 1: Run the Diagnostic Script

Add this to your repository:

```python
# diagnostic.py
import os
import sys

print("Current directory:", os.getcwd())
print("\nFiles present:")
for f in sorted(os.listdir('.')):
    print(f"  {f}")

print("\nTrying to import garden_planner_core...")
try:
    import garden_planner_core
    print("SUCCESS!")
    print("Available:", dir(garden_planner_core))
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
```

Then run it:
```bash
python diagnostic.py
```

### Step 2: Check Import Structure

Your `garden_planner_core.py` might have circular imports or missing dependencies.

**Look for these patterns:**
```python
# At the top of garden_planner_core.py

# These should be there:
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import requests
from geopy.geocoders import Nominatim
from meteostat import Point, Daily

# If any are missing, add them
```

### Step 3: Syntax Check

```bash
python -m py_compile garden_planner_core.py
```

If this shows syntax errors, fix them first.

## Common Root Causes

### Cause 1: Files in Wrong Directory

**Problem:** Files are in a subdirectory like `src/` or `code/`

**Solution:** Move them to the root:
```bash
mv src/garden_planner_core.py .
mv src/pfaf2.csv .
git add garden_planner_core.py pfaf2.csv
git commit -m "Move files to root"
git push
```

### Cause 2: Git LFS Issues

**Problem:** Large files (like pfaf2.csv) tracked by Git LFS aren't being loaded

**Check:**
```bash
git lfs ls-files
```

**Solution:** Either:
1. Push LFS files: `git lfs push origin main`
2. Or disable LFS for the file and commit normally

### Cause 3: Missing Dependencies

**Problem:** `garden_planner_core.py` imports packages not in `requirements.txt`

**Solution:** Add ALL imported packages to requirements.txt:
```txt
streamlit>=1.28.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
requests>=2.26.0
geopy>=2.2.0
meteostat>=1.6.0
xlsxwriter>=3.0.0
openpyxl>=3.0.0
Pillow>=9.0.0
```

### Cause 4: Python Version Mismatch

**Problem:** Code uses Python 3.10+ features but Streamlit uses 3.9

**Check your code for:**
- Pattern matching (match/case) - requires 3.10+
- Union operator (X | Y) - requires 3.10+
- Better error messages - may differ

**Solution:** Either:
1. Make code compatible with 3.9
2. Or specify Python version in Streamlit Cloud settings

## Still Not Working?

### Option A: Simplify First

Create a minimal test version:

```python
# streamlit_simple.py
import streamlit as st
import os

st.title("Simple Test")

st.write("Files here:", os.listdir('.'))

try:
    import garden_planner_core
    st.success("Import successful!")
    st.write("Classes available:", dir(garden_planner_core))
except Exception as e:
    st.error(f"Import failed: {e}")
```

Deploy this as your main file to confirm the import works.

### Option B: Show Me the Code

If you're still stuck:

1. Run `streamlit_debug.py` 
2. Take screenshots of the output
3. Check what the debug page shows
4. Share the specific error message

### Option C: Check Permissions

Ensure your repository is:
- Public (for free Streamlit Cloud)
- Or you have Streamlit Teams (for private repos)

## Prevention

To avoid this in the future:

1. **Always test locally first**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Use the debug app** whenever you have import issues

3. **Keep files in root directory**

4. **Commit and push all changes** before deploying

5. **Check GitHub** to verify files are there

6. **Use version control properly**
   ```bash
   git status        # Check what's staged
   git diff          # See changes
   git log --oneline # Verify commits
   ```

---

## Emergency Contact

If absolutely nothing works:

1. Clone a fresh copy of your repo
2. Test imports locally
3. Check if `garden_planner_core.py` is actually there
4. Run the diagnostic script
5. Share the full error output for more specific help

Remember: The files being "in the repo" means they're committed, pushed, and visible on GitHub.com - not just saved on your local computer!
