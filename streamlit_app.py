import streamlit as st
import pandas as pd
import os
import sys

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
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌱 Garden Planner</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">An intelligent garden planning system based on real environmental data</p>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("📍 Garden Configuration")
garden_name = st.sidebar.text_input("Garden Name", value="My_Garden")
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Location")

col1, col2 = st.sidebar.columns(2)
latitude = col1.number_input("Latitude", value=42.6977, format="%.4f")
longitude = col2.number_input("Longitude", value=23.3219, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")

num_recommendations = st.sidebar.slider("Recommendations", 10, 200, 50, 10)
min_suitability = st.sidebar.slider("Min Score", 0.0, 1.0, 0.5, 0.05)
max_cluster_size = st.sidebar.slider("Max Cluster Size", 3, 10, 5, 1)

generate_button = st.sidebar.button("🌿 Generate Plan", type="primary", use_container_width=True)

# Check for required files
required_files = ['garden_planner_core.py', 'pfaf2.csv']
missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error(f"⚠️ Missing required files: {', '.join(missing)}")
    st.stop()

# Try to understand the structure of garden_planner_core.py
if 'structure_checked' not in st.session_state:
    st.session_state.structure_checked = False
    st.session_state.available_items = []
    
    try:
        import garden_planner_core as gpc
        st.session_state.available_items = [item for item in dir(gpc) if not item.startswith('_')]
        st.session_state.structure_checked = True
    except Exception as e:
        st.error(f"Error importing garden_planner_core: {e}")

# Show what's available
with st.expander("🔍 Debug: Available Functions/Classes"):
    if st.session_state.available_items:
        st.write("Found in garden_planner_core.py:")
        st.write(st.session_state.available_items)
    else:
        st.write("Unable to import garden_planner_core.py")

# Initialize session state
if 'results_generated' not in st.session_state:
    st.session_state.results_generated = False

if generate_button:
    st.session_state.results_generated = False
    
    with st.spinner("🌱 Generating your garden plan..."):
        try:
            # Try method 1: Look for a main function
            import garden_planner_core as gpc
            
            st.write("**Attempting to run garden planner...**")
            
            # Check if there's a main function
            if hasattr(gpc, 'main'):
                st.write("✓ Found main() function")
                try:
                    gpc.main(
                        garden_name=garden_name,
                        latitude=latitude,
                        longitude=longitude,
                        num_recommendations=num_recommendations,
                        min_suitability=min_suitability,
                        max_cluster_size=max_cluster_size
                    )
                except TypeError:
                    # Try without arguments
                    st.write("Trying main() without arguments...")
                    gpc.main()
            
            # Try method 2: Run the main script with sys.argv
            elif os.path.exists('garden_planner_main.py'):
                st.write("✓ Found garden_planner_main.py")
                st.write("Setting up sys.argv...")
                
                # Save original argv
                original_argv = sys.argv.copy()
                
                # Set up fake command line arguments
                sys.argv = [
                    'garden_planner_main.py',
                    '--garden-name', garden_name,
                    '--latitude', str(latitude),
                    '--longitude', str(longitude),
                    '--recommendations', str(num_recommendations),
                    '--min-score', str(min_suitability),
                    '--max-cluster', str(max_cluster_size)
                ]
                
                try:
                    # Try to run it
                    exec(open('garden_planner_main.py').read(), {'__name__': '__main__'})
                except SystemExit:
                    pass  # Script called exit(), that's OK
                finally:
                    sys.argv = original_argv
            
            else:
                st.error("❌ Don't know how to run this code structure")
                st.info("""
                Your code needs one of:
                1. A main() function in garden_planner_core.py
                2. Command-line argument support in garden_planner_main.py
                3. Or share the structure of your code so I can adapt
                """)
                st.stop()
            
            # Look for results
            st.write("🔍 Looking for generated files...")
            
            csv_files = [f for f in os.listdir('.') if f.endswith('_recommendations.csv')]
            xlsx_files = [f for f in os.listdir('.') if f.endswith('_results.xlsx')]
            png_files = [f for f in os.listdir('.') if f.startswith('plant_clusters_max') and f.endswith('.png')]
            
            st.write(f"CSV files: {csv_files}")
            st.write(f"Excel files: {xlsx_files}")
            st.write(f"PNG files: {png_files}")
            
            if csv_files:
                csv_file = csv_files[0]
                df = pd.read_csv(csv_file)
                
                st.session_state.results_df = df
                st.session_state.csv_file = csv_file
                st.session_state.xlsx_file = xlsx_files[0] if xlsx_files else None
                st.session_state.png_files = png_files
                st.session_state.results_generated = True
                st.success(f"✅ Generated {len(df)} recommendations!")
                st.rerun()
            else:
                st.warning("⚠️ No results generated")
                st.info("""
                **Next step:** Please share:
                1. The first 50 lines of your garden_planner_main.py
                2. Or the structure of your garden_planner_core.py
                
                This will help me create the right integration!
                """)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Display results
if st.session_state.results_generated:
    st.markdown("---")
    st.markdown("## 🌿 Your Garden Plan")
    
    df = st.session_state.results_df
    st.write(f"**{len(df)} plants recommended**")
    
    # Find score column
    score_col = None
    for col in ['suitability_score', 'Suitability Score', 'score', 'Score']:
        if col in df.columns:
            score_col = col
            break
    
    if score_col:
        top_10 = df.nlargest(10, score_col)
        
        st.subheader("Top 10 Plants")
        for idx, row in top_10.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                name = row.get('common_name', row.get('Common Name', 'Unknown'))
                latin = row.get('latin_name', row.get('Latin Name', ''))
                st.markdown(f"**{name}**" + (f" *{latin}*" if latin else ""))
            with col2:
                score = row[score_col]
                color = "#2E7D32" if score >= 0.8 else "#558B2F" if score >= 0.6 else "#FFA000"
                st.markdown(f"<h3 style='color: {color}; text-align: right;'>{score:.2f}</h3>", unsafe_allow_html=True)
    else:
        st.dataframe(df.head(10))
    
    # Downloads
    st.markdown("---")
    st.subheader("📥 Downloads")
    
    col1, col2, col3 = st.columns(3)
    
    if st.session_state.csv_file and os.path.exists(st.session_state.csv_file):
        with col1:
            with open(st.session_state.csv_file, 'rb') as f:
                st.download_button("📄 CSV", f, st.session_state.csv_file, "text/csv", use_container_width=True)
    
    if st.session_state.xlsx_file and os.path.exists(st.session_state.xlsx_file):
        with col2:
            with open(st.session_state.xlsx_file, 'rb') as f:
                st.download_button("📊 Excel", f, st.session_state.xlsx_file, use_container_width=True)
    
    if st.session_state.png_files:
        with col3:
            png = st.session_state.png_files[0]
            if os.path.exists(png):
                with open(png, 'rb') as f:
                    st.download_button("🖼️ Image", f, png, "image/png", use_container_width=True)
    
    # Show images
    if st.session_state.png_files:
        st.markdown("---")
        st.subheader("📊 Visualizations")
        for png in st.session_state.png_files:
            if os.path.exists(png):
                st.image(png, use_column_width=True)
    
    # Full table
    with st.expander("📋 All Data"):
        st.dataframe(df, use_container_width=True)

else:
    st.markdown("""
    ### 👋 Welcome!
    
    Get plant recommendations based on your location's environmental data.
    
    1. 📍 Enter your coordinates in the sidebar
    2. ⚙️ Adjust settings as needed
    3. 🌿 Click "Generate Plan"
    4. 📥 Download your results
    """)
    
    with st.expander("🗺️ Example Locations"):
        st.markdown("""
        - Sofia: 42.6977, 23.3219
        - London: 51.5074, -0.1278
        - New York: 40.7128, -74.0060
        """)

st.markdown("---")
st.caption("🌱 Garden Planner • Climate-based plant recommendations")
