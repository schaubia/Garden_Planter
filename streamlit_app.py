import streamlit as st
import pandas as pd
import os
import sys
from io import StringIO
import contextlib

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

garden_name = st.sidebar.text_input("Garden Name", value="My Garden")
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Location")

col1, col2 = st.sidebar.columns(2)
latitude = col1.number_input("Latitude", value=42.6977, format="%.4f")
longitude = col2.number_input("Longitude", value=23.3219, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")

num_recommendations = st.sidebar.slider("Number of Recommendations", 10, 200, 100, 10)
min_suitability = st.sidebar.slider("Minimum Suitability Score", 0.0, 1.0, 0.5, 0.05)
max_cluster_size = st.sidebar.slider("Max Plants per Cluster", 3, 10, 5, 1)

generate_button = st.sidebar.button("🌿 Generate Garden Plan", type="primary", use_container_width=True)

# Check if your main file exists
if not os.path.exists('garden_planner_main.py'):
    st.error("⚠️ garden_planner_main.py not found")
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
        
        # Monkey-patch input()
        original_input = __builtins__.input
        
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
            __builtins__.input = mock_input
            
            # Capture print statements
            with contextlib.redirect_stdout(captured_output):
                # Import and run the main script
                import importlib
                import garden_planner_main
                
                # Reload to ensure fresh execution
                importlib.reload(garden_planner_main)
            
            # Restore original input
            __builtins__.input = original_input
            
            # Show output
            output = captured_output.getvalue()
            if output:
                with st.expander("📋 Execution Log"):
                    st.text(output)
            
            # Look for generated files
            csv_pattern = f"{garden_name.replace(' ', '_')}_recommendations.csv"
            xlsx_pattern = f"{garden_name.replace(' ', '_')}_results.xlsx"
            png_pattern = "plant_clusters_max*.png"
            
            # Find actual files
            csv_file = None
            xlsx_file = None
            png_files = []
            
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
            __builtins__.input = original_input  # Restore on error
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
            
            # Show output anyway
            output = captured_output.getvalue()
            if output:
                st.text_area("Output:", output, height=300)

# Display results
if st.session_state.results_generated:
    st.markdown("---")
    st.markdown("## 🌿 Your Garden Plan")
    
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
        with col2:
            if score_col:
                score = row[score_col]
                color = "#2E7D32" if score >= 0.8 else "#558B2F" if score >= 0.6 else "#FFA000"
                st.markdown(f"<h3 style='color: {color}; text-align: right;'>{score:.2f}</h3>", unsafe_allow_html=True)
        st.markdown("---")
    
    # Download section
    st.markdown("## 📥 Downloads")
    
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
        st.markdown("## 📊 Visualizations")
        for png_file in st.session_state.png_files:
            if os.path.exists(png_file):
                st.image(png_file, caption=png_file)
    
    # Full data table
    with st.expander("📋 View All Recommendations"):
        st.dataframe(df, use_container_width=True)

else:
    # Welcome message
    st.markdown("""
    ### 👋 Welcome!
    
    Configure your garden settings in the sidebar and click **"Generate Garden Plan"** to get started.
    
    #### What you'll get:
    - 🌿 Plant recommendations based on your location
    - 📊 Suitability scores for each plant  
    - 🔄 Optimal plant clusters
    - 🤝 Companion plant relationships
    - 📥 Downloadable reports
    
    #### Find your coordinates:
    1. Open Google Maps
    2. Right-click your location
    3. Copy the coordinates
    """)

# Footer
st.markdown("---")
st.caption("🌱 Garden Planner • Data-driven gardening recommendations")
