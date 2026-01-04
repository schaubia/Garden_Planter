import streamlit as st
import os
import sys

st.set_page_config(page_title="🔍 Garden Planner Debug", page_icon="🔍")

st.title("🔍 Garden Planner - Debug Mode")

st.info("This debug page will help identify why the main app isn't working.")

# Step 1: Check Python environment
st.header("1️⃣ Python Environment")
col1, col2 = st.columns(2)
with col1:
    st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
with col2:
    st.metric("Working Directory", "See below")

st.code(os.getcwd())

# Step 2: List files
st.header("2️⃣ Files in Directory")

try:
    files = sorted(os.listdir('.'))
    
    required_files = ['garden_planner_core.py', 'pfaf2.csv', 'streamlit_app.py', 'requirements.txt']
    optional_files = ['companion_plants.csv', 'garden_planner.db']
    
    st.subheader("Required Files")
    for f in required_files:
        if f in files:
            size = os.path.getsize(f) if os.path.isfile(f) else "N/A"
            st.success(f"✓ {f} ({size} bytes)")
        else:
            st.error(f"✗ {f} - MISSING")
    
    st.subheader("Optional Files")
    for f in optional_files:
        if f in files:
            size = os.path.getsize(f) if os.path.isfile(f) else "N/A"
            st.info(f"✓ {f} ({size} bytes)")
        else:
            st.warning(f"○ {f} - Not present (OK)")
    
    st.subheader("All Files")
    with st.expander("Show all files"):
        for f in files:
            is_dir = os.path.isdir(f)
            icon = "📁" if is_dir else "📄"
            st.text(f"{icon} {f}")
            
except Exception as e:
    st.error(f"Error listing files: {e}")

# Step 3: Test imports
st.header("3️⃣ Package Imports")

packages_to_test = {
    'Standard Library': ['os', 'sys', 'sqlite3', 'json'],
    'Data Science': ['pandas', 'numpy', 'matplotlib', 'sklearn'],
    'Web/APIs': ['requests', 'geopy'],
    'Specialized': ['meteostat', 'xlsxwriter', 'openpyxl'],
    'Streamlit': ['streamlit']
}

for category, packages in packages_to_test.items():
    st.subheader(category)
    cols = st.columns(len(packages))
    for idx, pkg in enumerate(packages):
        with cols[idx]:
            try:
                __import__(pkg)
                st.success(f"✓ {pkg}")
            except ImportError:
                st.error(f"✗ {pkg}")

# Step 4: Test garden_planner_core import
st.header("4️⃣ Garden Planner Core Import")

if os.path.exists('garden_planner_core.py'):
    st.success("✓ garden_planner_core.py file exists")
    
    # Read first few lines
    st.subheader("File Preview (first 20 lines)")
    try:
        with open('garden_planner_core.py', 'r') as f:
            lines = f.readlines()[:20]
        st.code(''.join(lines), language='python')
    except Exception as e:
        st.error(f"Could not read file: {e}")
    
    st.subheader("Import Test")
    try:
        import garden_planner_core
        st.success("✓ Successfully imported garden_planner_core")
        
        # List available items
        items = [item for item in dir(garden_planner_core) if not item.startswith('_')]
        st.write("**Available classes/functions:**")
        st.write(items)
        
        # Test specific imports
        st.subheader("Testing Specific Class Imports")
        required_classes = ['Config', 'LocationAnalyzer', 'PlantDatabase', 
                          'PlantSuitabilityScorer', 'ClusteringEngine', 
                          'CompanionPlantAnalyzer', 'ResultsExporter']
        
        for cls_name in required_classes:
            if hasattr(garden_planner_core, cls_name):
                st.success(f"✓ {cls_name}")
            else:
                st.error(f"✗ {cls_name} not found")
                
    except Exception as e:
        st.error(f"✗ Failed to import garden_planner_core")
        st.exception(e)
        
        # Show detailed error
        import traceback
        st.code(traceback.format_exc())
else:
    st.error("✗ garden_planner_core.py NOT FOUND")
    st.warning("The file is missing from your deployment. Please ensure it's committed to your repository.")

# Step 5: Python path
st.header("5️⃣ Python Path")
st.write("Python searches for modules in these locations:")
for idx, path in enumerate(sys.path, 1):
    st.text(f"{idx}. {path}")

# Step 6: Recommendations
st.header("6️⃣ Recommendations")

if os.path.exists('garden_planner_core.py'):
    st.info("""
    **Next Steps:**
    
    1. Check if the import test above passed
    2. If it failed, look at the error message carefully
    3. Common issues:
       - Missing dependencies in requirements.txt
       - Syntax errors in garden_planner_core.py
       - Circular imports
       - Missing imports within garden_planner_core.py
    
    4. Try running the diagnostic.py script locally
    5. Check Streamlit Cloud logs for more details
    """)
else:
    st.error("""
    **CRITICAL: garden_planner_core.py is missing!**
    
    **Fix this by:**
    
    1. Ensure garden_planner_core.py is in your repository root
    2. Commit the file: `git add garden_planner_core.py`
    3. Push to GitHub: `git push origin main`
    4. Redeploy the Streamlit app
    """)

# Footer
st.markdown("---")
st.caption("Debug mode • Use this to troubleshoot deployment issues")
