#!/usr/bin/env python3
"""
Garden Planner - Plant Recommendation System
Main executable script with user input
Usage: python garden_planner_main.py
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from garden_planner_core import *

# ============================================================================
# USER INPUT FUNCTIONS
# ============================================================================

def get_user_location():
    """Get garden location from user input"""
    print("\n" + "="*60)
    print("🌱 GARDEN PLANNER - LOCATION SETUP")
    print("="*60)
    
    garden_name = input("\n📝 Enter your garden name: ").strip()
    if not garden_name:
        garden_name = "My Garden"
    
    print("\n📍 Enter location coordinates:")
    print("   (You can find these on Google Maps by right-clicking)")
    
    while True:
        try:
            lat_input = input("   Latitude (e.g., 42.6977): ").strip()
            lat = float(lat_input)
            if not -90 <= lat <= 90:
                print("   ❌ Latitude must be between -90 and 90")
                continue
            break
        except ValueError:
            print("   ❌ Please enter a valid number")
    
    while True:
        try:
            lon_input = input("   Longitude (e.g., 23.3219): ").strip()
            lon = float(lon_input)
            if not -180 <= lon <= 180:
                print("   ❌ Longitude must be between -180 and 180")
                continue
            break
        except ValueError:
            print("   ❌ Please enter a valid number")
    
    return garden_name, lat, lon


def get_user_preferences():
    """Get user preferences for recommendations"""
    print("\n" + "="*60)
    print("⚙️  RECOMMENDATION PREFERENCES")
    print("="*60)
    
    while True:
        try:
            top_n = input("\n📊 How many plant recommendations? (default 100): ").strip()
            top_n = int(top_n) if top_n else 100
            if top_n < 1:
                print("   ❌ Please enter a positive number")
                continue
            break
        except ValueError:
            print("   ❌ Please enter a valid number")
    
    while True:
        try:
            min_score = input("🎯 Minimum suitability score (0-1, default 0.5): ").strip()
            min_score = float(min_score) if min_score else 0.5
            if not 0 <= min_score <= 1:
                print("   ❌ Score must be between 0 and 1")
                continue
            break
        except ValueError:
            print("   ❌ Please enter a valid number")
    
    while True:
        try:
            max_cluster = input("🌿 Max plants per cluster (default 5): ").strip()
            max_cluster = int(max_cluster) if max_cluster else 5
            if max_cluster < 1:
                print("   ❌ Please enter a positive number")
                continue
            break
        except ValueError:
            print("   ❌ Please enter a valid number")
    
    return top_n, min_score, max_cluster


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("🌱 WELCOME TO GARDEN PLANNER")
    print("="*60)
    print("This tool will help you:")
    print("  • Find suitable plants for your location")
    print("  • Analyze environmental conditions")
    print("  • Create plant clusters for companion planting")
    print("  • Generate detailed Excel reports")
    
    # Check if plant database exists
    plant_db = "pfaf2.csv"
    if not Path(plant_db).exists():
        print(f"\n❌ Error: Plant database '{plant_db}' not found!")
        print("   Please ensure the file is in the current directory.")
        return
    
    # Check if companion plants file exists
    companion_db = "companion_plants.csv"
    if not Path(companion_db).exists():
        print(f"\n⚠️  Warning: Companion plants database '{companion_db}' not found!")
        print("   Clustering will work, but companion analysis will be skipped.")
        companion_available = False
    else:
        companion_available = True
    
    try:
        # Get user inputs
        garden_name, lat, lon = get_user_location()
        top_n, min_score, max_cluster = get_user_preferences()
        
        # Update config
        Config.MAX_CLUSTER_SIZE = max_cluster
        
        # Initialize planner
        print("\n" + "="*60)
        print("🚀 INITIALIZING GARDEN PLANNER")
        print("="*60)
        
        planner = GardenPlanner(use_vectorized=True)
        planner.initialize(plant_db)
        
        # Add location
        print("\n" + "="*60)
        print("📍 FETCHING LOCATION DATA")
        print("="*60)
        
        location_id = planner.add_location(lat, lon, garden_name)
        
        # Get recommendations
        print("\n" + "="*60)
        print("🌱 CALCULATING PLANT RECOMMENDATIONS")
        print("="*60)
        
        recommendations = planner.get_recommendations(location_id, top_n, min_score)
        
        if recommendations.empty:
            print("\n❌ No suitable plants found with the given criteria.")
            print("   Try lowering the minimum suitability score.")
            return
        
        # Export to CSV
        csv_filename = f"{garden_name.replace(' ', '_')}_recommendations.csv"
        recommendations.to_csv(csv_filename, index=False)
        print(f"\n✅ Recommendations saved to: {csv_filename}")
        
        # Perform clustering
        print("\n" + "="*60)
        print("🔬 CLUSTERING PLANTS")
        print("="*60)
        
        clustered_df = PlantClusteringModule.cluster_plants(recommendations, max_cluster)
        
        # Visualize clusters
        fig = PlantClusteringModule.visualize_clusters(clustered_df, garden_name)
        
        # Find companion plant relationships
        if companion_available:
            print("\n" + "="*60)
            print("🤝 ANALYZING COMPANION RELATIONSHIPS")
            print("="*60)
            
            cluster_companions = PlantClusteringModule.find_companions(
                clustered_df, companion_db
            )
        else:
            cluster_companions = {}
        
        # Export to Excel
        print("\n" + "="*60)
        print("📊 GENERATING EXCEL REPORT")
        print("="*60)
        
        excel_filename = f"{garden_name.replace(' ', '_')}_results.xlsx"
        PlantClusteringModule.export_to_excel(
            clustered_df, cluster_companions, fig, garden_name, excel_filename
        )
        
        print(f"\n✅ Full report saved to: {excel_filename}")
        
        # Summary
        print("\n" + "="*60)
        print("✨ SUMMARY")
        print("="*60)
        print(f"Garden: {garden_name}")
        print(f"Location: ({lat:.4f}, {lon:.4f})")
        print(f"Plants recommended: {len(recommendations)}")
        print(f"Clusters created: {clustered_df['cluster'].nunique()}")
        print(f"Max plants per cluster: {max_cluster}")
        
        if companion_available:
            total_companions = sum(len(df) for df in cluster_companions.values())
            print(f"Companion relationships found: {total_companions}")
        
        print("\n📁 Output files:")
        print(f"  • {csv_filename}")
        print(f"  • {excel_filename}")
        print(f"  • plant_clusters_max{max_cluster}.png")
        
        print("\n" + "="*60)
        print("🎉 GARDEN PLANNER COMPLETE!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()