import streamlit as st
import pandas as pd
import os
import sys
from io import StringIO
import contextlib
import builtins
from PIL import Image, ImageDraw, ImageFont
import io

# Page configuration
st.set_page_config(
    page_title="🌱 Garden Planner",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #558B2F;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .plant-card {
        background-color: #F1F8F4;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E7D32;
        margin: 0.5rem 0;
    }
    .metric-container {
        background-color: #E8F5E9;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .info-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .download-section {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌱 Garden Planner</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent plant recommendations based on your location\'s environmental data</p>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Garden Configuration")

with st.sidebar:
    st.markdown("### 📍 Location")
    garden_name = st.text_input("Garden Name", value="My Garden", help="Give your garden a memorable name")
    
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=42.6977, format="%.4f")
    with col2:
        longitude = st.number_input("Longitude", value=23.3219, format="%.4f")
    
    st.info("💡 **Tip:** Right-click on Google Maps and copy coordinates")
    
    st.markdown("---")
    st.markdown("### 🎛️ Preferences")
    
    num_rec = st.slider("Number of Plants", 10, 100, 30, 10, 
                        help="How many plant recommendations to generate")
    min_score = st.slider("Minimum Suitability", 0.0, 1.0, 0.5, 0.05,
                         help="Only show plants with score above this threshold")
    max_cluster = st.slider("Plants per Cluster", 3, 10, 5, 1,
                           help="Maximum plants grouped together for companion planting")
    
    st.markdown("---")
    generate = st.button("🌿 Generate Garden Plan", type="primary", use_container_width=True)
    
    # Quick examples
    with st.expander("📍 Example Locations"):
        st.markdown("""
        **Popular cities:**
        - Sofia: `42.6977, 23.3219`
        - London: `51.5074, -0.1278`
        - New York: `40.7128, -74.0060`
        - Paris: `48.8566, 2.3522`
        - Tokyo: `35.6762, 139.6503`
        """)

# Field abbreviation explanations
FIELD_EXPLANATIONS = {
    'Shade': {
        'F': 'Full Sun - Needs direct sunlight most of the day',
        'S': 'Semi-Shade - Tolerates partial shade',
        'N': 'Full Shade - Thrives in shaded conditions'
    },
    'Moisture': {
        'D': 'Dry - Prefers well-drained, dry soil',
        'M': 'Moist - Needs consistently moist soil',
        'We': 'Wet - Tolerates waterlogged conditions',
        'Wa': 'Water - Aquatic plant, grows in water'
    },
    'Soil': {
        'L': 'Light - Sandy, well-draining soil',
        'M': 'Medium - Loamy soil',
        'H': 'Heavy - Clay soil',
        'acid': 'Acidic - pH < 6.5',
        'neutral': 'Neutral - pH 6.5-7.5',
        'alkaline': 'Alkaline - pH > 7.5'
    }
}

def add_legend_to_image(image_path):
    """Add a legend explaining clusters to the visualization"""
    try:
        img = Image.open(image_path)
        
        # Create a larger canvas
        legend_height = 150
        new_height = img.height + legend_height
        new_img = Image.new('RGB', (img.width, new_height), 'white')
        new_img.paste(img, (0, 0))
        
        draw = ImageDraw.Draw(new_img)
        
        # Try to use a nice font, fall back to default if not available
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Add legend title
        legend_y = img.height + 20
        draw.text((20, legend_y), "Plant Clustering Visualization", font=title_font, fill='#2E7D32')
        
        # Add explanation
        legend_y += 30
        explanation_text = (
            "• Each color represents a cluster of plants with similar growing requirements\n"
            "• Plants in the same cluster grow well together (companion planting)\n"
            "• Distance between points shows how similar plants are"
        )
        draw.text((20, legend_y), explanation_text, font=text_font, fill='#333333')
        
        # Save to bytes
        img_bytes = io.BytesIO()
        new_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
    except Exception as e:
        # If anything fails, return original
        with open(image_path, 'rb') as f:
            return io.BytesIO(f.read())

# Initialize session state
if 'log' not in st.session_state:
    st.session_state.log = []
    st.session_state.results = None
    st.session_state.show_log = False

if generate:
    st.session_state.log = []
    st.session_state.results = None
    st.session_state.show_log = False
    
    def log(msg):
        st.session_state.log.append(msg)
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🧹 Preparing environment...")
    progress_bar.progress(10)
    
    # Clean old files
    for f in os.listdir('.'):
        if '_recommendations.csv' in f or '_results.xlsx' in f:
            try:
                os.remove(f)
            except:
                pass
    
    # Prepare inputs
    inputs = [garden_name.replace(' ', '_'), str(latitude), str(longitude), 
              str(num_rec), str(min_score), str(max_cluster)]
    
    log("🚀 Garden Planner Execution")
    log(f"Garden: {garden_name}")
    log(f"Location: {latitude}, {longitude}")
    log(f"Settings: {num_rec} plants, score>={min_score}, cluster<={max_cluster}")
    log("")
    
    input_idx = [0]
    original_input = builtins.input
    
    def mock_input(prompt=""):
        if input_idx[0] < len(inputs):
            val = inputs[input_idx[0]]
            input_idx[0] += 1
            log(f"Input: {prompt}{val}")
            return val
        return ""
    
    builtins.input = mock_input
    captured = StringIO()
    
    status_text.text("🌱 Analyzing location and generating recommendations...")
    progress_bar.progress(30)
    
    try:
        # Clear cache
        if 'garden_planner_main' in sys.modules:
            del sys.modules['garden_planner_main']
        if 'garden_planner_core' in sys.modules:
            del sys.modules['garden_planner_core']
        
        # Execute the script
        with open('garden_planner_main.py', 'r') as f:
            code = f.read()
        
        namespace = {
            '__name__': '__main__',
            '__file__': 'garden_planner_main.py',
            '__builtins__': builtins
        }
        
        with contextlib.redirect_stdout(captured):
            with contextlib.redirect_stderr(captured):
                try:
                    exec(code, namespace)
                    log("\n✅ Execution completed successfully")
                except SystemExit:
                    log("\n✅ Script completed")
                except Exception as e:
                    log(f"\n❌ Error: {str(e)}")
        
        output = captured.getvalue()
        if output:
            log("\n--- Script Output ---")
            log(output)
        
    except Exception as e:
        log(f"\n❌ Fatal error: {str(e)}")
    finally:
        builtins.input = original_input
    
    status_text.text("🔍 Collecting results...")
    progress_bar.progress(80)
    
    # Look for files
    csv_files = [f for f in os.listdir('.') if f.endswith('_recommendations.csv')]
    xlsx_files = [f for f in os.listdir('.') if f.endswith('_results.xlsx')]
    png_files = [f for f in os.listdir('.') if 'plant_cluster' in f and f.endswith('.png')]
    
    if csv_files:
        try:
            df = pd.read_csv(csv_files[0])
            st.session_state.results = {
                'df': df,
                'csv': csv_files[0],
                'xlsx': xlsx_files[0] if xlsx_files else None,
                'png': png_files,
                'garden_name': garden_name,
                'location': f"{latitude}, {longitude}"
            }
            log(f"\n✅ Successfully loaded {len(df)} plant recommendations")
        except Exception as e:
            log(f"\n❌ Error loading results: {e}")
    else:
        log("\n❌ No results generated")
    
    status_text.text("✅ Complete!")
    progress_bar.progress(100)
    
    import time
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    st.rerun()

# Display Results
if st.session_state.results:
    df = st.session_state.results['df']
    
    # Success message
    st.success(f"✅ Successfully generated **{len(df)} plant recommendations** for {st.session_state.results['garden_name']}")
    
    # Summary metrics
    st.markdown("### 📊 Garden Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Plants", len(df))
    with col2:
        score_col = next((col for col in ['suitability_score', 'score'] if col in df.columns), None)
        if score_col:
            avg_score = df[score_col].mean()
            st.metric("Avg Suitability", f"{avg_score:.2f}")
    with col3:
        st.metric("Location", st.session_state.results['location'])
    with col4:
        families = df['family'].nunique() if 'family' in df.columns else 0
        st.metric("Plant Families", families)
    
    st.markdown("---")
    
    # Top Recommendations
    st.markdown("### 🌿 Top Plant Recommendations")
    
    # Find score column
    score_col = next((col for col in ['suitability_score', 'Suitability Score', 'score', 'Score'] if col in df.columns), None)
    
    if score_col:
        top_plants = df.nlargest(10, score_col)
        
        for idx, (_, row) in enumerate(top_plants.iterrows(), 1):
            with st.container():
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    # Plant name
                    common_name = row.get('common_name', row.get('Common Name', row.get('name', 'Unknown')))
                    latin_name = row.get('latin_name', row.get('Latin Name', row.get('scientific_name', '')))
                    
                    st.markdown(f"**{idx}. {common_name}**")
                    if latin_name:
                        st.caption(f"*{latin_name}*")
                    
                    # Details in columns
                    detail_cols = st.columns(4)
                    
                    with detail_cols[0]:
                        family = row.get('family', row.get('Family', 'Unknown'))
                        st.caption(f"🌿 Family: {family}")
                    
                    with detail_cols[1]:
                        shade = row.get('shade', row.get('Shade', ''))
                        if shade and shade in FIELD_EXPLANATIONS['Shade']:
                            st.caption(f"☀️ {FIELD_EXPLANATIONS['Shade'][shade]}")
                    
                    with detail_cols[2]:
                        moisture = row.get('moisture', row.get('Moisture', ''))
                        if moisture and moisture in FIELD_EXPLANATIONS['Moisture']:
                            st.caption(f"💧 {FIELD_EXPLANATIONS['Moisture'][moisture]}")
                    
                    with detail_cols[3]:
                        growth = row.get('growth_rate', row.get('Growth Rate', ''))
                        if growth:
                            st.caption(f"📈 Growth: {growth}")
                
                with col2:
                    score = row[score_col]
                    if score >= 0.8:
                        color = "#2E7D32"
                        label = "Excellent"
                    elif score >= 0.6:
                        color = "#558B2F"
                        label = "Good"
                    else:
                        color = "#FFA000"
                        label = "Fair"
                    
                    st.markdown(f"<div class='metric-container'><div style='font-size: 2rem; color: {color}; font-weight: bold;'>{score:.2f}</div><div style='font-size: 0.8rem; color: {color};'>{label}</div></div>", unsafe_allow_html=True)
                
                st.markdown("---")
    else:
        st.dataframe(df.head(10), use_container_width=True)
    
    # Cluster Visualization
    if st.session_state.results['png']:
        st.markdown("### 📊 Plant Cluster Visualization")
        
        st.info("""
        **Understanding the Visualization:**
        - 🎨 **Colors** represent different plant clusters
        - 🤝 **Same cluster** = plants grow well together (companion planting)
        - 📏 **Distance** = similarity in growing requirements
        - 🔄 **Clusters** help you organize your garden layout
        """)
        
        for png_file in st.session_state.results['png']:
            if os.path.exists(png_file):
                # Add legend to image
                img_with_legend = add_legend_to_image(png_file)
                st.image(img_with_legend, use_column_width=True)
    
    # Field Explanations
    with st.expander("📖 Understanding Plant Requirements"):
        st.markdown("### 🌤️ Shade Requirements")
        for code, desc in FIELD_EXPLANATIONS['Shade'].items():
            st.markdown(f"- **{code}**: {desc}")
        
        st.markdown("### 💧 Moisture Requirements")
        for code, desc in FIELD_EXPLANATIONS['Moisture'].items():
            st.markdown(f"- **{code}**: {desc}")
        
        st.markdown("### 🌱 Soil Type")
        for code, desc in FIELD_EXPLANATIONS['Soil'].items():
            st.markdown(f"- **{code}**: {desc}")
    
    # Downloads Section
    st.markdown("---")
    st.markdown('<div class="download-section">', unsafe_allow_html=True)
    st.markdown("### 📥 Download Your Garden Plan")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.results['csv'] and os.path.exists(st.session_state.results['csv']):
            with open(st.session_state.results['csv'], 'rb') as f:
                st.download_button(
                    label="📄 Download Plant List (CSV)",
                    data=f,
                    file_name=st.session_state.results['csv'],
                    mime="text/csv",
                    use_container_width=True
                )
            st.caption("Spreadsheet format for easy filtering and sorting")
    
    with col2:
        if st.session_state.results['xlsx'] and os.path.exists(st.session_state.results['xlsx']):
            with open(st.session_state.results['xlsx'], 'rb') as f:
                st.download_button(
                    label="📊 Download Full Report (Excel)",
                    data=f,
                    file_name=st.session_state.results['xlsx'],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            st.caption("Complete report with clusters and companion info")
    
    with col3:
        if st.session_state.results['png']:
            png_file = st.session_state.results['png'][0]
            if os.path.exists(png_file):
                img_with_legend = add_legend_to_image(png_file)
                st.download_button(
                    label="🖼️ Download Visualization (PNG)",
                    data=img_with_legend,
                    file_name=f"{garden_name.replace(' ', '_')}_clusters.png",
                    mime="image/png",
                    use_container_width=True
                )
                st.caption("Cluster diagram for garden planning")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Full data table
    with st.expander("📋 View Complete Plant Database"):
        st.dataframe(df, use_container_width=True, height=400)
    
    # Optional: Execution log (collapsed by default)
    if st.session_state.log:
        with st.expander("🔧 Technical Log (for debugging)"):
            log_text = '\n'.join(st.session_state.log)
            st.text_area("Execution Log", log_text, height=300)
            
            # Download log
            st.download_button(
                label="📥 Download Log",
                data=log_text,
                file_name="garden_planner_log.txt",
                mime="text/plain"
            )

else:
    # Welcome screen
    if not st.session_state.log:
        st.markdown("""
        ### 👋 Welcome to Garden Planner!
        
        Get personalized plant recommendations based on your location's real environmental data including:
        
        - 🌡️ **Climate Analysis** - Temperature, rainfall, and hardiness zones
        - 🌍 **Soil Assessment** - pH levels and soil composition
        - 🗺️ **Geographic Data** - Altitude and regional characteristics
        - 🤝 **Companion Planting** - Plants that grow well together
        
        #### How to Get Started:
        
        1. **📍 Enter Your Location** - Use your garden's coordinates (find them on Google Maps)
        2. **⚙️ Adjust Preferences** - Set the number of plants and suitability threshold
        3. **🌿 Generate** - Click the button to create your personalized garden plan
        4. **📥 Download** - Save your results as CSV, Excel, or visualization
        
        ---
        
        Ready to start planning your dream garden? Configure your settings in the sidebar! 🌱
        """)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🎯 Smart Scoring
            Plants are scored based on:
            - Hardiness match
            - Soil compatibility
            - Climate suitability
            - Water requirements
            """)
        
        with col2:
            st.markdown("""
            #### 🔄 Companion Planting
            Intelligent clustering:
            - Groups compatible plants
            - Identifies beneficial pairs
            - Optimizes garden layout
            - Maximizes yields
            """)
        
        with col3:
            st.markdown("""
            #### 📊 Detailed Reports
            Comprehensive outputs:
            - Suitability scores
            - Growing requirements
            - Visual cluster maps
            - Excel spreadsheets
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌱 <strong>Garden Planner</strong> • Powered by real environmental data and botanical science</p>
    <p style="font-size: 0.85rem;">Data sources: Climate records, PFAF plant database, companion planting research</p>
</div>
""", unsafe_allow_html=True)
