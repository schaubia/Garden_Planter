import streamlit as st
import pandas as pd
import os
import sys
from io import StringIO
import contextlib
import builtins
import importlib

# Page configuration
st.set_page_config(
    page_title="🌱 Garden Planner",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E7D32;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌱 Garden Planner</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">An intelligent garden planning system based on real environmental data</p>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("📍 Garden Configuration")

garden_name = st.sidebar.text_input("Garden Name", value="My_Garden", help="Give your garden a memorable name (no spaces)")

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Location")

col1, col2 = st.sidebar.columns(2)
latitude = col1.number_input("Latitude", value=42.6977, format="%.4f", help="Enter your location's latitude")
longitude = col2.number_input("Longitude", value=23.3219, format="%.4f", help="Enter your location's longitude")

st.sidebar.markdown("""
<div class="info-box" style="font-size: 0.9rem;">
💡 <strong>Find coordinates:</strong><br>
1. Google Maps<br>
2. Right-click location<br>
3. Copy coordinates
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")

num_recommendations = st.sidebar.slider("Recommendations", 10, 200, 50, 10)
min_suitability = st.sidebar.slider("Min Score", 0.0, 1.0, 0.5, 0.05)
max_cluster_size = st.sidebar.slider("Max Cluster Size", 3, 10, 5, 1)

generate_button = st.sidebar.button("🌿 Generate Plan", type="primary", use_container_width=True)

# Check for required files
if not os.path.exists('garden_planner_main.py'):
    st.error("⚠️ garden_planner_main.py not found")
    st.stop()

# Initialize session state
if 'results_generated' not in st.session_state:
    st.session_state.results_generated = False

if generate_button:
    st.session_state.results_generated = False
    
    # Clean up old files first
    for f in os.listdir('.'):
        if f.endswith('_recommendations.csv') or f.endswith('_results.xlsx') or f.startswith('plant_clusters_max'):
            try:
                os.remove(f)
            except:
                pass
    
    with st.spinner("🌱 Generating your garden plan..."):
        # Prepare inputs
        input_queue = [
            garden_name,
            str(latitude),
            str(longitude),
            str(num_recommendations),
            str(min_suitability),
            str(max_cluster_size)
        ]
        
        input_index = [0]
        original_input = builtins.input
        
        def mock_input(prompt=""):
            if input_index[0] < len(input_queue):
                value = input_queue[input_index[0]]
                input_index[0] += 1
                print(f"{prompt}{value}")
                return value
            return ""
        
        # Capture output
        captured_output = StringIO()
        success = False
        error_msg = None
        
        try:
            # Replace input
            builtins.input = mock_input
            
            # Clear any cached imports
            if 'garden_planner_main' in sys.modules:
                del sys.modules['garden_planner_main']
            if 'garden_planner_core' in sys.modules:
                del sys.modules['garden_planner_core']
            
            with contextlib.redirect_stdout(captured_output):
                with contextlib.redirect_stderr(captured_output):
                    try:
                        # Import and execute
                        import garden_planner_main
                        success = True
                    except SystemExit:
                        # Script may call sys.exit() - that's OK
                        success = True
                    except Exception as e:
                        error_msg = str(e)
                        import traceback
                        print("\n" + "="*50)
                        print("ERROR OCCURRED:")
                        print("="*50)
                        traceback.print_exc()
            
            # Restore input
            builtins.input = original_input
            
            # Get output
            output = captured_output.getvalue()
            
            # Always show the log
            with st.expander("📋 Execution Log", expanded=not success):
                st.code(output, language="text")
            
            if error_msg:
                st.error(f"❌ Error during execution: {error_msg}")
            
            # Look for generated files
            st.write("🔍 Looking for generated files...")
            
            csv_files = [f for f in os.listdir('.') if f.endswith('_recommendations.csv')]
            xlsx_files = [f for f in os.listdir('.') if f.endswith('_results.xlsx')]
            png_files = [f for f in os.listdir('.') if f.startswith('plant_clusters_max') and f.endswith('.png')]
            
            st.write(f"Found {len(csv_files)} CSV files: {csv_files}")
            st.write(f"Found {len(xlsx_files)} Excel files: {xlsx_files}")
            st.write(f"Found {len(png_files)} PNG files: {png_files}")
            
            if csv_files:
                csv_file = csv_files[0]
                st.success(f"✅ Found results: {csv_file}")
                
                try:
                    df = pd.read_csv(csv_file)
                    st.write(f"📊 Loaded {len(df)} plant recommendations")
                    
                    st.session_state.results_df = df
                    st.session_state.csv_file = csv_file
                    st.session_state.xlsx_file = xlsx_files[0] if xlsx_files else None
                    st.session_state.png_files = png_files
                    st.session_state.results_generated = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {e}")
            else:
                st.warning("⚠️ No results CSV file was generated.")
                st.info("""
                **Possible issues:**
                1. The script may need different inputs
                2. Check the execution log above for errors
                3. The script may not be running to completion
                
                **Try:**
                - Use a simpler garden name (no spaces or special characters)
                - Reduce the number of recommendations to 20
                - Check if pfaf2.csv exists and is readable
                """)
                
        except Exception as e:
            builtins.input = original_input
            st.error(f"❌ Fatal error: {str(e)}")
            st.exception(e)
            
            output = captured_output.getvalue()
            if output:
                with st.expander("📋 Output"):
                    st.code(output)

# Display results
if st.session_state.results_generated:
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🌿 Your Garden Plan</h2>', unsafe_allow_html=True)
    
    df = st.session_state.results_df
    
    # Show summary
    st.write(f"**Total plants recommended:** {len(df)}")
    
    # Display top recommendations
    st.subheader("Top 10 Recommendations")
    
    # Find score column
    score_col = None
    for col in ['suitability_score', 'Suitability Score', 'score', 'Score']:
        if col in df.columns:
            score_col = col
            break
    
    if score_col:
        top_10 = df.nlargest(10, score_col)
    else:
        top_10 = df.head(10)
    
    for idx, row in top_10.iterrows():
        col1, col2 = st.columns([4, 1])
        with col1:
            name = row.get('common_name', row.get('Common Name', row.get('name', 'Unknown')))
            latin = row.get('latin_name', row.get('Latin Name', row.get('scientific_name', '')))
            st.markdown(f"**{name}**" + (f" *{latin}*" if latin else ""))
            
            family = row.get('family', row.get('Family', ''))
            growth = row.get('growth_rate', row.get('Growth Rate', ''))
            if family or growth:
                parts = []
                if family:
                    parts.append(f"Family: {family}")
                if growth:
                    parts.append(f"Growth: {growth}")
                st.caption(" | ".join(parts))
        with col2:
            if score_col:
                score = row[score_col]
                color = "#2E7D32" if score >= 0.8 else "#558B2F" if score >= 0.6 else "#FFA000"
                st.markdown(f"<h3 style='color: {color}; text-align: right;'>{score:.2f}</h3>", unsafe_allow_html=True)
        st.markdown("---")
    
    # Downloads
    st.markdown('<h2 class="sub-header">📥 Downloads</h2>', unsafe_allow_html=True)
    
    cols = st.columns(3)
    
    with cols[0]:
        if st.session_state.csv_file and os.path.exists(st.session_state.csv_file):
            with open(st.session_state.csv_file, 'rb') as f:
                st.download_button(
                    "📄 CSV File",
                    f.read(),
                    st.session_state.csv_file,
                    "text/csv",
                    use_container_width=True
                )
    
    with cols[1]:
        if st.session_state.xlsx_file and os.path.exists(st.session_state.xlsx_file):
            with open(st.session_state.xlsx_file, 'rb') as f:
                st.download_button(
                    "📊 Excel Report",
                    f.read(),
                    st.session_state.xlsx_file,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    with cols[2]:
        if st.session_state.png_files:
            png = st.session_state.png_files[0]
            if os.path.exists(png):
                with open(png, 'rb') as f:
                    st.download_button(
                        "🖼️ Visualization",
                        f.read(),
                        png,
                        "image/png",
                        use_container_width=True
                    )
    
    # Visualizations
    if st.session_state.png_files:
        st.markdown('<h2 class="sub-header">📊 Cluster Visualization</h2>', unsafe_allow_html=True)
        for png in st.session_state.png_files:
            if os.path.exists(png):
                st.image(png, use_column_width=True)
    
    # Full table
    with st.expander("📋 All Recommendations"):
        st.dataframe(df, use_container_width=True)

else:
    # Welcome
    st.markdown("""
    ### 👋 Welcome to Garden Planner!
    
    Get personalized plant recommendations based on your location's real environmental data.
    
    #### Quick Start:
    
    1. 📍 **Enter location** - Use your coordinates (Google Maps → right-click → copy)
    2. ⚙️ **Adjust settings** - Number of plants, score threshold, cluster size
    3. 🌿 **Generate** - Click the button in sidebar
    4. 📥 **Download** - Get your CSV and Excel reports
    
    ---
    
    **Ready?** Configure settings in the sidebar and click "Generate Plan"!
    """)
    
    with st.expander("🗺️ Example Coordinates"):
        st.markdown("""
        - **Sofia, Bulgaria**: 42.6977, 23.3219
        - **London, UK**: 51.5074, -0.1278
        - **New York, USA**: 40.7128, -74.0060
        - **Los Angeles, USA**: 34.0522, -118.2437
        - **Paris, France**: 48.8566, 2.3522
        """)

# Footer
st.markdown("---")
st.caption("🌱 Garden Planner • Data-driven recommendations based on climate, soil, and plant compatibility")
