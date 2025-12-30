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

garden_name = st.sidebar.text_input("Garden Name", value="My Garden", help="Give your garden a memorable name")

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Location")

col1, col2 = st.sidebar.columns(2)
latitude = col1.number_input("Latitude", value=42.6977, format="%.4f", help="Enter your location's latitude (e.g., 42.6977)")
longitude = col2.number_input("Longitude", value=23.3219, format="%.4f", help="Enter your location's longitude (e.g., 23.3219)")

st.sidebar.markdown("""
<div class="info-box" style="font-size: 0.9rem;">
💡 <strong>Find your coordinates:</strong><br>
1. Open Google Maps<br>
2. Right-click your location<br>
3. Click the coordinates to copy
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Recommendation Settings")

num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=10, max_value=200, value=100, step=10, help="How many plants to recommend")
min_suitability = st.sidebar.slider("Minimum Suitability Score", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Threshold for plant recommendations (0-1)")
max_cluster_size = st.sidebar.slider("Max Plants per Cluster", min_value=3, max_value=10, value=5, step=1, help="Maximum number of plants grouped together")

generate_button = st.sidebar.button("🌿 Generate Garden Plan", type="primary", use_container_width=True)

# Check if your main file exists
if not os.path.exists('garden_planner_main.py'):
    st.error("⚠️ garden_planner_main.py not found. Please ensure it's in your repository.")
    st.info("Your repository should contain: garden_planner_main.py, garden_planner_core.py, pfaf2.csv")
    st.stop()

# Initialize session state
if 'results_generated' not in st.session_state:
    st.session_state.results_generated = False

if generate_button:
    st.session_state.results_generated = False
    
    with st.spinner("🌱 Generating your garden plan..."):
        # Create a queue of inputs to replace input() calls
        input_queue = [
            garden_name,
            str(latitude),
            str(longitude),
            str(num_recommendations),
            str(min_suitability),
            str(max_cluster_size)
        ]
        
        input_index = [0]  # Use list to make it mutable in nested function
        
        # Monkey-patch input() - FIXED for Python 3.13
        original_input = builtins.input
        
        def mock_input(prompt=""):
            if input_index[0] < len(input_queue):
                value = input_queue[input_index[0]]
                input_index[0] += 1
                print(f"{prompt}{value}")  # Show what was "entered"
                return value
            return ""
        
        # Capture stdout
        captured_output = StringIO()
        
        try:
            # Replace input function
            builtins.input = mock_input
            
            # Capture print statements
            with contextlib.redirect_stdout(captured_output):
                # Import and run the main script
                import importlib
                import garden_planner_main
                
                # Reload to ensure fresh execution
                importlib.reload(garden_planner_main)
            
            # Restore original input
            builtins.input = original_input
            
            # Show output
            output = captured_output.getvalue()
            if output:
                with st.expander("📋 Execution Log"):
                    st.text(output)
            
            # Look for generated files
            csv_file = None
            xlsx_file = None
            png_files = []
            
            # Find actual files
            for f in os.listdir('.'):
                if f.endswith('_recommendations.csv'):
                    csv_file = f
                elif f.endswith('_results.xlsx'):
                    xlsx_file = f
                elif f.startswith('plant_clusters_max') and f.endswith('.png'):
                    png_files.append(f)
            
            if csv_file and os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                st.session_state.results_df = df
                st.session_state.csv_file = csv_file
                st.session_state.xlsx_file = xlsx_file
                st.session_state.png_files = png_files
                st.session_state.results_generated = True
                st.success("✅ Garden plan generated successfully!")
                st.rerun()
            else:
                st.warning("⚠️ Results file not found. Check the execution log above.")
                
        except Exception as e:
            builtins.input = original_input  # Restore on error
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
            
            # Show output anyway
            output = captured_output.getvalue()
            if output:
                st.text_area("Output:", output, height=300)

# Display results
if st.session_state.results_generated:
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🌿 Your Garden Plan</h2>', unsafe_allow_html=True)
    
    df = st.session_state.results_df
    
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
            # Try different column name variations
            name = row.get('common_name', row.get('Common Name', row.get('name', 'Unknown')))
            latin = row.get('latin_name', row.get('Latin Name', row.get('scientific_name', '')))
            st.markdown(f"**{name}**" + (f" *{latin}*" if latin else ""))
            
            # Add more details if available
            family = row.get('family', row.get('Family', ''))
            growth_rate = row.get('growth_rate', row.get('Growth Rate', ''))
            if family or growth_rate:
                details = []
                if family:
                    details.append(f"Family: {family}")
                if growth_rate:
                    details.append(f"Growth Rate: {growth_rate}")
                st.caption(" | ".join(details))
        with col2:
            if score_col:
                score = row[score_col]
                color = "#2E7D32" if score >= 0.8 else "#558B2F" if score >= 0.6 else "#FFA000"
                st.markdown(f"<h3 style='color: {color}; text-align: right;'>{score:.2f}</h3>", unsafe_allow_html=True)
        st.markdown("---")
    
    # Download section
    st.markdown('<h2 class="sub-header">📥 Downloads</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.csv_file and os.path.exists(st.session_state.csv_file):
            with open(st.session_state.csv_file, 'rb') as f:
                st.download_button(
                    "📄 CSV Recommendations",
                    f.read(),
                    st.session_state.csv_file,
                    "text/csv",
                    use_container_width=True
                )
    
    with col2:
        if st.session_state.xlsx_file and os.path.exists(st.session_state.xlsx_file):
            with open(st.session_state.xlsx_file, 'rb') as f:
                st.download_button(
                    "📊 Excel Report",
                    f.read(),
                    st.session_state.xlsx_file,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    with col3:
        if st.session_state.png_files:
            png_file = st.session_state.png_files[0]
            if os.path.exists(png_file):
                with open(png_file, 'rb') as f:
                    st.download_button(
                        "🖼️ Cluster Visualization",
                        f.read(),
                        png_file,
                        "image/png",
                        use_container_width=True
                    )
    
    # Show visualizations
    if st.session_state.png_files:
        st.markdown('<h2 class="sub-header">📊 Visualizations</h2>', unsafe_allow_html=True)
        for png_file in st.session_state.png_files:
            if os.path.exists(png_file):
                st.image(png_file, caption=png_file, use_column_width=True)
    
    # Full data table
    with st.expander("📋 View All Recommendations"):
        st.dataframe(df, use_container_width=True)

else:
    # Welcome message
    st.markdown("""
    ### 👋 Welcome to Garden Planner!
    
    This intelligent system recommends suitable plants based on **real environmental data** from your location.
    
    #### How it works:
    
    1. **📍 Enter your location** - Provide accurate latitude and longitude coordinates
    2. **⚙️ Configure preferences** - Set the number of recommendations and other parameters
    3. **🌿 Generate plan** - Click the button to analyze and get recommendations
    4. **📊 Review results** - Explore plant recommendations and clusters
    5. **📥 Export** - Download your personalized garden plan
    
    #### Features:
    
    - 🌍 **Real climate data** - Uses actual weather and soil data from your location
    - 🎯 **Smart scoring** - Evaluates plants based on hardiness, soil, shade, moisture, and more
    - 🔄 **Intelligent clustering** - Groups compatible plants together
    - 🤝 **Companion analysis** - Identifies beneficial plant relationships
    - 📊 **Professional reports** - Export results to Excel with visualizations
    
    ---
    
    **Ready to start?** Configure your garden in the sidebar and click "Generate Garden Plan"!
    """)
    
    # Example locations
    with st.expander("🗺️ Example Locations"):
        st.markdown("""
        Try these example coordinates:
        
        - **Sofia, Bulgaria**: 42.6977, 23.3219
        - **London, UK**: 51.5074, -0.1278
        - **New York, USA**: 40.7128, -74.0060
        - **Tokyo, Japan**: 35.6762, 139.6503
        - **Sydney, Australia**: -33.8688, 151.2093
        - **Paris, France**: 48.8566, 2.3522
        - **Vancouver, Canada**: 49.2827, -123.1207
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🌱 Garden Planner - Making gardening easier with data-driven recommendations</p>
    <p style="font-size: 0.9rem;">Based on real climate data, PFAF plant database, and companion planting research</p>
</div>
""", unsafe_allow_html=True)
