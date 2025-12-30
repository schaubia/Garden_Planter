import streamlit as st
import pandas as pd
import os
from io import BytesIO
import base64

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

# Import the garden planner functions
# These would come from garden_planner_core.py
try:
    from garden_planner_core import (
        Config, LocationAnalyzer, PlantDatabase, 
        PlantSuitabilityScorer, ClusteringEngine, 
        CompanionPlantAnalyzer, ResultsExporter
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    st.error("⚠️ Core garden planner modules not found. Please ensure garden_planner_core.py is in the same directory.")

# Title
st.markdown('<h1 class="main-header">🌱 Garden Planner</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">An intelligent garden planning system based on real environmental data</p>', unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("📍 Garden Configuration")

# Garden details
garden_name = st.sidebar.text_input(
    "Garden Name",
    value="My Garden",
    help="Give your garden a memorable name"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Location")

# Location input
col1, col2 = st.sidebar.columns(2)
latitude = col1.number_input(
    "Latitude",
    value=42.6977,
    format="%.4f",
    help="Enter your location's latitude (e.g., 42.6977)"
)
longitude = col2.number_input(
    "Longitude",
    value=23.3219,
    format="%.4f",
    help="Enter your location's longitude (e.g., 23.3219)"
)

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

# Recommendation parameters
num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=10,
    max_value=200,
    value=100,
    step=10,
    help="How many plants to recommend"
)

min_suitability = st.sidebar.slider(
    "Minimum Suitability Score",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Threshold for plant recommendations (0-1)"
)

max_cluster_size = st.sidebar.slider(
    "Max Plants per Cluster",
    min_value=3,
    max_value=10,
    value=5,
    step=1,
    help="Maximum number of plants grouped together"
)

# Generate button
generate_button = st.sidebar.button("🌿 Generate Garden Plan", type="primary", use_container_width=True)

# Main content area
if not CORE_AVAILABLE:
    st.warning("""
    ### Setup Required
    
    Please ensure the following files are present:
    - `garden_planner_core.py`
    - `pfaf2.csv` (plant database)
    - `companion_plants.csv` (optional)
    
    Upload these files to your Streamlit deployment or place them in the same directory as this app.
    """)
else:
    # Initialize session state
    if 'results_generated' not in st.session_state:
        st.session_state.results_generated = False
    
    if generate_button:
        with st.spinner("🌱 Analyzing your location and generating recommendations..."):
            try:
                # Create config
                config = Config()
                config.MAX_CLUSTER_SIZE = max_cluster_size
                
                # Step 1: Location Analysis
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📍 Analyzing location...")
                location_analyzer = LocationAnalyzer(latitude, longitude)
                location_data = location_analyzer.get_location_data()
                progress_bar.progress(20)
                
                # Step 2: Load plant database
                status_text.text("🌿 Loading plant database...")
                db = PlantDatabase(config)
                db.load_pfaf_data("pfaf2.csv")
                progress_bar.progress(40)
                
                # Step 3: Score plants
                status_text.text("🎯 Scoring plant suitability...")
                scorer = PlantSuitabilityScorer(config)
                scored_plants = scorer.score_all_plants(
                    db.plants,
                    location_data,
                    min_score=min_suitability
                )
                progress_bar.progress(60)
                
                # Step 4: Cluster plants
                status_text.text("🔄 Creating plant clusters...")
                clustering_engine = ClusteringEngine(config)
                clusters = clustering_engine.create_clusters(
                    scored_plants.head(num_recommendations)
                )
                progress_bar.progress(80)
                
                # Step 5: Analyze companions
                status_text.text("🤝 Analyzing companion plants...")
                try:
                    companion_analyzer = CompanionPlantAnalyzer(config)
                    companion_analyzer.load_companion_data("companion_plants.csv")
                    companion_links = companion_analyzer.analyze_clusters(clusters)
                except:
                    companion_links = {}
                    st.warning("⚠️ Companion plant data not available")
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Store results in session state
                st.session_state.results_generated = True
                st.session_state.location_data = location_data
                st.session_state.scored_plants = scored_plants
                st.session_state.clusters = clusters
                st.session_state.companion_links = companion_links
                st.session_state.garden_name = garden_name
                
                st.success("🎉 Garden plan generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating plan: {str(e)}")
                st.exception(e)
    
    # Display results if available
    if st.session_state.results_generated:
        st.markdown("---")
        
        # Location information
        st.markdown('<h2 class="sub-header">📍 Location Analysis</h2>', unsafe_allow_html=True)
        
        location_data = st.session_state.location_data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Altitude", f"{location_data.get('altitude', 'N/A')} m")
        with col2:
            st.metric("Soil pH", f"{location_data.get('soil_ph', 'N/A')}")
        with col3:
            st.metric("Avg Temp", f"{location_data.get('avg_temp', 'N/A')}°C")
        with col4:
            st.metric("Hardiness Zone", location_data.get('hardiness_zone', 'N/A'))
        
        # Climate info
        with st.expander("🌡️ Climate Details"):
            st.json(location_data.get('climate_data', {}))
        
        st.markdown("---")
        
        # Top recommendations
        st.markdown('<h2 class="sub-header">🌿 Top Plant Recommendations</h2>', unsafe_allow_html=True)
        
        scored_plants = st.session_state.scored_plants
        
        # Display top 10 in a nice format
        top_10 = scored_plants.head(10)
        
        for idx, row in top_10.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{row['common_name']}** _{row['latin_name']}_")
                st.caption(f"Family: {row.get('family', 'Unknown')} | Growth Rate: {row.get('growth_rate', 'Unknown')}")
            with col2:
                score_color = "#2E7D32" if row['suitability_score'] >= 0.8 else "#558B2F" if row['suitability_score'] >= 0.6 else "#FFA000"
                st.markdown(f"<h3 style='color: {score_color}; text-align: right;'>{row['suitability_score']:.2f}</h3>", unsafe_allow_html=True)
            st.markdown("---")
        
        # Download full recommendations
        csv = scored_plants.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Recommendations (CSV)",
            data=csv,
            file_name=f"{garden_name.replace(' ', '_')}_recommendations.csv",
            mime="text/csv",
        )
        
        st.markdown("---")
        
        # Clusters
        st.markdown('<h2 class="sub-header">🔄 Plant Clusters</h2>', unsafe_allow_html=True)
        
        clusters = st.session_state.clusters
        companion_links = st.session_state.companion_links
        
        st.info(f"Your plants have been organized into {len(clusters)} clusters based on similar growing requirements.")
        
        # Display clusters in tabs
        cluster_tabs = st.tabs([f"Cluster {i+1}" for i in range(len(clusters))])
        
        for i, tab in enumerate(cluster_tabs):
            with tab:
                cluster = clusters[i]
                
                st.markdown(f"**Cluster {i+1}** - {len(cluster)} plants")
                
                # Display plants in cluster
                for _, plant in cluster.iterrows():
                    st.markdown(f"- **{plant['common_name']}** ({plant['latin_name']}) - Score: {plant['suitability_score']:.2f}")
                
                # Companion relationships
                if i in companion_links and len(companion_links[i]) > 0:
                    st.markdown("**🤝 Companion Plant Relationships:**")
                    for link in companion_links[i]:
                        relationship_color = "green" if "helps" in link['relationship'].lower() else "red" if "harms" in link['relationship'].lower() else "gray"
                        st.markdown(f":{relationship_color}[{link['plant1']} ↔ {link['plant2']}: {link['relationship']}]")
        
        # Export functionality
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Export Results</h2>', unsafe_allow_html=True)
        
        try:
            # Create exporter
            exporter = ResultsExporter(config)
            
            # Generate Excel file in memory
            output = BytesIO()
            exporter.export_to_excel(
                clusters,
                companion_links,
                location_data,
                st.session_state.garden_name,
                output
            )
            output.seek(0)
            
            st.download_button(
                label="📥 Download Complete Report (Excel)",
                data=output.getvalue(),
                file_name=f"{garden_name.replace(' ', '_')}_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.warning(f"⚠️ Could not generate Excel report: {str(e)}")
    
    else:
        # Initial state - show instructions
        st.markdown("""
        ### 👋 Welcome to Garden Planner!
        
        This intelligent system recommends suitable plants based on **real environmental data** from your location.
        
        #### How it works:
        
        1. **📍 Enter your location** - Provide accurate latitude and longitude coordinates
        2. **⚙️ Configure preferences** - Set the number of recommendations and other parameters
        3. **🌿 Generate plan** - Click the button to analyze and get recommendations
        4. **📊 Review results** - Explore plant clusters and companion relationships
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
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🌱 Garden Planner - Making gardening easier with data-driven recommendations</p>
    <p style="font-size: 0.9rem;">Based on real climate data, PFAF plant database, and companion planting research</p>
</div>
""", unsafe_allow_html=True)
