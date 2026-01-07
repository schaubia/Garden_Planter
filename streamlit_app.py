import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import traceback

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the core module directly
try:
    from garden_planner_core import (
        GardenPlanner, 
        PlantClusteringModule, 
        Config
    )
    from climate_projection import get_climate_projection_for_location
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")
    st.error("Please ensure garden_planner_core.py and climate_projection.py are in the same directory.")
    st.stop()

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
    .climate-warning {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .climate-info {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌱 Garden Planner</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent plant recommendations with climate change projections</p>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Garden Configuration")

with st.sidebar:
    st.markdown("### 📍 Location")
    garden_name = st.text_input("Garden Name", value="My Garden", help="Give your garden a memorable name")
    
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=42.6977, format="%.4f", min_value=-90.0, max_value=90.0)
    with col2:
        longitude = st.number_input("Longitude", value=23.3219, format="%.4f", min_value=-180.0, max_value=180.0)
    
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
    
    col1, col2 = st.columns(2)
    with col1:
        generate = st.button("🌿 Generate", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
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
if 'results' not in st.session_state:
    st.session_state.results = None

if 'climate_projection' not in st.session_state:
    st.session_state.climate_projection = None

if generate:
    st.session_state.results = None
    st.session_state.climate_projection = None
    
    # Check for required files
    plant_db = "pfaf2.csv"
    companion_db = "companion_plants.csv"
    
    if not Path(plant_db).exists():
        st.error(f"❌ Error: Plant database '{plant_db}' not found! Please ensure it's in the deployment directory.")
        st.stop()
    
    companion_available = Path(companion_db).exists()
    if not companion_available:
        st.warning(f"⚠️ Warning: Companion plants database '{companion_db}' not found. Companion analysis will be skipped.")
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🌍 Fetching location data...")
        progress_bar.progress(10)
        
        planner = GardenPlanner(plant_db)
        location_id = planner.setup_location(latitude, longitude, garden_name)
        
        status_text.text("🌡️ Analyzing climate data...")
        progress_bar.progress(30)
        
        # Generate climate scenarios
        planner.generate_climate_scenarios(location_id)
        
        status_text.text("🌱 Calculating plant suitability...")
        progress_bar.progress(50)
        
        # Get recommendations
        recommendations = planner.get_recommendations(
            location_id, 
            top_n=num_rec, 
            min_score=min_score
        )
        
        if recommendations.empty:
            st.error("❌ No suitable plants found for your criteria. Try lowering the minimum suitability score.")
            progress_bar.empty()
            status_text.empty()
            st.stop()
        
        status_text.text("🔄 Creating plant clusters...")
        progress_bar.progress(70)
        
        # Create clusters
        csv_file, xlsx_file, png_file, num_clusters = planner.create_clusters(
            recommendations, 
            max_cluster, 
            companion_available, 
            garden_name
        )
        
        # Get climate data for projection
        status_text.text("🌍 Generating climate projections...")
        progress_bar.progress(85)
        
        with planner.db.get_connection() as conn:
            climate_data = pd.read_sql(
                f"SELECT * FROM climate_data WHERE location_id = {location_id} AND scenario = 'current' LIMIT 1", 
                conn
            )
            
            if not climate_data.empty:
                climate_row = climate_data.iloc[0]
                projection, summary = get_climate_projection_for_location(
                    latitude,
                    longitude,
                    float(climate_row['avg_temp']),
                    float(climate_row['precip']),
                    int(climate_row['frost_days']),
                    garden_name
                )
                st.session_state.climate_projection = {
                    'projection': projection,
                    'summary': summary
                }
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Store results
        st.session_state.results = {
            'csv': csv_file,
            'xlsx': xlsx_file,
            'png': [png_file] if png_file else [],
            'num_clusters': num_clusters,
            'location': f"{latitude:.4f}, {longitude:.4f}"
        }
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Generated {len(recommendations)} plant recommendations with {num_clusters} clusters!")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error during generation: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()

# Display results
if st.session_state.results:
    df = pd.read_csv(st.session_state.results['csv'])
    
    # Climate Projection Section (New!)
    if st.session_state.climate_projection:
        st.markdown("---")
        st.markdown("### 🌍 Climate Change Projection (10 Years)")
        
        proj = st.session_state.climate_projection['projection']
        summary = st.session_state.climate_projection['summary']
        
        # Impact level indicator
        impact_colors = {
            'low': '#4CAF50',
            'moderate': '#FF9800',
            'high': '#F44336',
            'severe': '#D32F2F'
        }
        impact_color = impact_colors.get(proj.impact_level, '#FF9800')
        
        st.markdown(f"""
        <div style='background-color: {impact_color}20; border-left: 4px solid {impact_color}; padding: 1rem; border-radius: 5px;'>
            <h4 style='color: {impact_color}; margin: 0;'>
                Impact Level: {proj.impact_level.upper()} 
                <span style='font-size: 0.8em; color: #666;'>(Confidence: {proj.confidence})</span>
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Temperature Change", 
                f"+{proj.temp_change}°C",
                f"{proj.temp_change_min} to {proj.temp_change_max}°C",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "Precipitation Change", 
                f"{proj.precip_change:+.1f}%",
                f"{proj.precip_change_min:+.1f} to {proj.precip_change_max:+.1f}%"
            )
        
        with col3:
            st.metric(
                "Growing Season", 
                f"{proj.growing_season_change:+d} days",
                "Longer season" if proj.growing_season_change > 0 else "Shorter"
            )
        
        with col4:
            st.metric(
                "Hardiness Zone Shift", 
                f"+{proj.hardiness_zone_shift:.1f} zones",
                "Warmer zones"
            )
        
        # Detailed impacts
        with st.expander("📊 Detailed Climate Impacts", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🌡️ Temperature:**")
                st.info(summary['temperature'])
                
                st.markdown("**💧 Precipitation:**")
                st.info(summary['precipitation'])
                
                st.markdown("**📅 Growing Season:**")
                st.info(summary['growing_season'])
            
            with col2:
                st.markdown("**⚠️ Extreme Events:**")
                st.warning(summary['extreme_events'])
                
                st.markdown("**🌱 Gardening Implications:**")
                st.success(summary['gardening_implications'])
        
        # What this means for your garden
        st.markdown("#### 🌿 What This Means for Your Garden")
        
        recommendations_text = []
        
        if proj.temp_change > 1.0:
            recommendations_text.append("🌡️ **Consider heat-tolerant varieties** - Your location will be warmer, so plants that can handle heat stress will perform better")
        
        if proj.precip_change < -3:
            recommendations_text.append("💧 **Focus on drought-resistant plants** - Reduced rainfall means you'll want plants that can handle dry conditions")
        elif proj.precip_change > 3:
            recommendations_text.append("💧 **Ensure good drainage** - Increased rainfall means you'll want to avoid plants prone to root rot")
        
        if proj.growing_season_change > 10:
            recommendations_text.append("📅 **Extended growing season** - You'll be able to grow longer-season crops and potentially get multiple harvests")
        
        if proj.hardiness_zone_shift >= 0.5:
            recommendations_text.append(f"🗺️ **Hardiness zone shift** - Your garden will effectively be {proj.hardiness_zone_shift:.1f} zones warmer, opening up new plant possibilities")
        
        if proj.heat_wave_increase > 20:
            recommendations_text.append("☀️ **Prepare for more heat waves** - Consider shade structures and mulching to protect plants")
        
        if recommendations_text:
            for rec in recommendations_text:
                st.markdown(rec)
        else:
            st.info("✅ Your climate is expected to remain relatively stable for gardening purposes")
    
    st.markdown("---")
    
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
        st.metric("Clusters", st.session_state.results['num_clusters'])
    with col4:
        st.metric("Location", st.session_state.results['location'])
    
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
                    detail_cols = st.columns(3)
                    
                    with detail_cols[0]:
                        shade = row.get('shade', row.get('Shade', ''))
                        if shade and shade in FIELD_EXPLANATIONS['Shade']:
                            st.caption(f"☀️ {FIELD_EXPLANATIONS['Shade'][shade]}")
                        else:
                            st.caption("☀️ Shade info not available")
                    
                    with detail_cols[1]:
                        moisture = row.get('moisture', row.get('Moisture', ''))
                        if moisture and moisture in FIELD_EXPLANATIONS['Moisture']:
                            st.caption(f"💧 {FIELD_EXPLANATIONS['Moisture'][moisture]}")
                        else:
                            st.caption("💧 Moisture info not available")
                    
                    with detail_cols[2]:
                        growth = row.get('growth_rate', row.get('Growth Rate', ''))
                        if growth:
                            st.caption(f"📈 Growth: {growth}")
                        else:
                            st.caption("📈 Growth info not available")
                
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

else:
    # Welcome screen
    st.markdown("""
    ### 👋 Welcome to Garden Planner!
    
    Get personalized plant recommendations based on your location's real environmental data including:
    
    - 🌡️ **Climate Analysis** - Temperature, rainfall, and hardiness zones
    - 🌍 **Climate Projections** - Expected changes in the next 10 years
    - 🗺️ **Soil Assessment** - pH levels and soil composition
    - 🏔️ **Geographic Data** - Altitude and regional characteristics
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
        #### 🌍 Climate Projections
        See how climate change affects:
        - Temperature trends
        - Rainfall patterns
        - Growing season length
        - Plant zone shifts
        """)
    
    with col3:
        st.markdown("""
        #### 🔄 Companion Planting
        Intelligent clustering:
        - Groups compatible plants
        - Identifies beneficial pairs
        - Optimizes garden layout
        - Maximizes yields
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌱 <strong>Garden Planner</strong> • Powered by real environmental data and climate science</p>
    <p style="font-size: 0.85rem;">Data sources: Climate records, PFAF plant database, IPCC projections</p>
</div>
""", unsafe_allow_html=True)
