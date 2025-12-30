import streamlit as st
import pandas as pd
import os
import sys
from io import StringIO
import contextlib
import builtins

# Page configuration
st.set_page_config(
    page_title="🌱 Garden Planner",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('<h1 style="text-align: center; color: #2E7D32;">🌱 Garden Planner</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📍 Configuration")
garden_name = st.sidebar.text_input("Garden Name", value="MyGarden")
latitude = st.sidebar.number_input("Latitude", value=42.6977, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=23.3219, format="%.4f")
num_rec = st.sidebar.slider("Plants", 10, 100, 30, 10)
min_score = st.sidebar.slider("Min Score", 0.0, 1.0, 0.5, 0.05)
max_cluster = st.sidebar.slider("Cluster Size", 3, 10, 5, 1)
generate = st.sidebar.button("🌿 Generate", type="primary", use_container_width=True)

# Check files
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Files")
for f in ['garden_planner_main.py', 'garden_planner_core.py', 'pfaf2.csv']:
    exists = os.path.exists(f)
    st.sidebar.write(f"{'✅' if exists else '❌'} {f}")

if not os.path.exists('garden_planner_main.py'):
    st.error("Missing garden_planner_main.py")
    st.stop()

if 'log' not in st.session_state:
    st.session_state.log = []
    st.session_state.results = None

if generate:
    st.session_state.log = []
    st.session_state.results = None
    
    def log(msg):
        st.session_state.log.append(msg)
    
    log("🚀 Starting generation...")
    log(f"Garden: {garden_name}")
    log(f"Location: {latitude}, {longitude}")
    log(f"Settings: {num_rec} plants, score>={min_score}, cluster<={max_cluster}")
    log("")
    
    # Clean old files
    log("🧹 Cleaning old files...")
    for f in os.listdir('.'):
        if '_recommendations.csv' in f or '_results.xlsx' in f:
            try:
                os.remove(f)
                log(f"  Removed: {f}")
            except:
                pass
    log("")
    
    # Prepare inputs
    inputs = [garden_name, str(latitude), str(longitude), str(num_rec), str(min_score), str(max_cluster)]
    log(f"📝 Prepared {len(inputs)} inputs: {inputs}")
    log("")
    
    input_idx = [0]
    original_input = builtins.input
    
    def mock_input(prompt=""):
        if input_idx[0] < len(inputs):
            val = inputs[input_idx[0]]
            input_idx[0] += 1
            log(f"INPUT: {prompt}{val}")
            return val
        log(f"INPUT (empty): {prompt}")
        return ""
    
    log("🔧 Patching input()...")
    builtins.input = mock_input
    
    # Capture output
    captured = StringIO()
    
    log("📜 Executing garden_planner_main.py...")
    try:
        # Clear cache
        if 'garden_planner_main' in sys.modules:
            del sys.modules['garden_planner_main']
        if 'garden_planner_core' in sys.modules:
            del sys.modules['garden_planner_core']
        
        # Read and execute the file
        log("  Reading file...")
        with open('garden_planner_main.py', 'r') as f:
            code = f.read()
        
        log(f"  File size: {len(code)} characters")
        log("  Executing code...")
        
        # Create a namespace for execution
        namespace = {
            '__name__': '__main__',  # This is KEY - makes if __name__ == '__main__' work!
            '__file__': 'garden_planner_main.py',
            '__builtins__': builtins
        }
        
        with contextlib.redirect_stdout(captured):
            with contextlib.redirect_stderr(captured):
                try:
                    exec(code, namespace)
                    log("  ✅ Execution complete")
                except SystemExit as e:
                    log(f"  Script called sys.exit({e.code})")
                except Exception as e:
                    log(f"  ❌ Error during execution: {str(e)}")
                    import traceback
                    log("")
                    log("Traceback:")
                    for line in traceback.format_exc().split('\n'):
                        log(f"  {line}")
        
        output = captured.getvalue()
        if output:
            log("")
            log("📋 Script output:")
            log("─" * 40)
            for line in output.split('\n'):
                if line.strip():
                    log(f"  {line}")
        else:
            log("")
            log("⚠️ Script produced no output")
        
    except Exception as e:
        log(f"❌ Fatal error: {str(e)}")
        import traceback
        for line in traceback.format_exc().split('\n'):
            log(f"  {line}")
    finally:
        builtins.input = original_input
        log("")
        log("🔧 Restored input()")
    
    # Look for files
    log("")
    log("🔍 Searching for output files...")
    
    csv_files = [f for f in os.listdir('.') if f.endswith('_recommendations.csv')]
    xlsx_files = [f for f in os.listdir('.') if f.endswith('_results.xlsx')]
    png_files = [f for f in os.listdir('.') if 'plant_cluster' in f and f.endswith('.png')]
    
    log(f"  CSV files: {csv_files if csv_files else 'None'}")
    log(f"  Excel files: {xlsx_files if xlsx_files else 'None'}")
    log(f"  PNG files: {png_files if png_files else 'None'}")
    
    if csv_files:
        log("")
        log(f"✅ SUCCESS! Found {csv_files[0]}")
        try:
            df = pd.read_csv(csv_files[0])
            log(f"  Loaded {len(df)} plant recommendations")
            st.session_state.results = {
                'df': df,
                'csv': csv_files[0],
                'xlsx': xlsx_files[0] if xlsx_files else None,
                'png': png_files
            }
        except Exception as e:
            log(f"❌ Error loading CSV: {e}")
    else:
        log("")
        log("❌ No results generated")
        log("")
        log("💡 Debug info:")
        log(f"  Inputs provided: {input_idx[0]} out of {len(inputs)}")
        log(f"  Script output length: {len(captured.getvalue())} chars")
    
    st.rerun()

# Show log
if st.session_state.log:
    with st.expander("📋 Execution Log", expanded=True):
        st.code('\n'.join(st.session_state.log), language='text')

# Show results
if st.session_state.results:
    st.success("✅ Generation complete!")
    
    df = st.session_state.results['df']
    
    st.markdown("### 🌿 Top Recommendations")
    
    # Find score column
    score_col = None
    for col in ['suitability_score', 'score', 'Score']:
        if col in df.columns:
            score_col = col
            break
    
    if score_col:
        top = df.nlargest(10, score_col)
        for _, row in top.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                name = row.get('common_name', row.get('name', 'Unknown'))
                st.write(f"**{name}**")
            with col2:
                st.metric("Score", f"{row[score_col]:.2f}")
    else:
        st.dataframe(df.head(10))
    
    # Downloads
    st.markdown("### 📥 Downloads")
    cols = st.columns(3)
    
    if st.session_state.results['csv']:
        with cols[0]:
            with open(st.session_state.results['csv'], 'rb') as f:
                st.download_button("📄 CSV", f, st.session_state.results['csv'], use_container_width=True)
    
    if st.session_state.results['xlsx']:
        with cols[1]:
            with open(st.session_state.results['xlsx'], 'rb') as f:
                st.download_button("📊 Excel", f, st.session_state.results['xlsx'], use_container_width=True)
    
    if st.session_state.results['png']:
        with cols[2]:
            png = st.session_state.results['png'][0]
            with open(png, 'rb') as f:
                st.download_button("🖼️ Image", f, png, use_container_width=True)
    
    # Show images
    if st.session_state.results['png']:
        st.markdown("### 📊 Visualization")
        for png in st.session_state.results['png']:
            st.image(png, use_column_width=True)
    
    # Full data
    with st.expander("📋 All Data"):
        st.dataframe(df)

else:
    if not st.session_state.log:
        st.info("""
        ### 👋 Welcome!
        
        Configure settings in sidebar and click "Generate" to get plant recommendations.
        
        **Example coordinates:**
        - Sofia: 42.6977, 23.3219
        - London: 51.5074, -0.1278
        - New York: 40.7128, -74.0060
        """)
