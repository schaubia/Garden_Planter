# 🌱 Garden Planner - Plant Recommendation System
NB: The pfaf scraped plant data is taken from https://github.com/saulshanabrook/pfaf-data.

An intelligent garden planning system that recommends suitable plants based on real environmental data and creates optimal plant clusters for companion planting.

## 📋 Features

- **Location-based recommendations**: Analyzes real climate, soil, and geographic data
- **Intelligent plant clustering**: Groups compatible plants (max 5 per cluster by default)
- **Companion plant analysis**: Identifies beneficial plant relationships
- **Climate projections**: Includes future climate scenarios
- **Comprehensive reports**: Generates Excel reports with visualizations

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

1. **Clone the repository**
```bash
git clone https://github.com/Schaubia/Garden_Planter.git
cd Garden_Planter
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```


## 🚀 Usage

### Running the Program


```bash
python garden_planner_main.py
```

The program will interactively ask you for:

1. **Garden name**: A name for your garden (e.g., "My Backyard Garden")
2. **Latitude**: Geographic latitude (e.g., 42.6977)
3. **Longitude**: Geographic longitude (e.g., 23.3219)
4. **Number of recommendations**: How many plants to recommend (default: 100)
5. **Minimum suitability score**: Threshold from 0-1 (default: 0.5)
6. **Max plants per cluster**: Maximum plants grouped together (default: 5)

### Example Run

```
🌱 GARDEN PLANNER - LOCATION SETUP
📝 Enter your garden name: Sofia Garden
📍 Enter location coordinates:
   Latitude (e.g., 42.6977): 42.6977
   Longitude (e.g., 23.3219): 23.3219

⚙️  RECOMMENDATION PREFERENCES
📊 How many plant recommendations? (default 100): 100
🎯 Minimum suitability score (0-1, default 0.5): 0.5
🌿 Max plants per cluster (default 5): 5
```

### Finding Your Coordinates

1. Go to [Google Maps](https://maps.google.com)
2. Right-click on your location
3. Click the coordinates to copy them
4. Format: First number is latitude, second is longitude

## 📊 Output Files

The program generates three output files:

1. **`[GardenName]_recommendations.csv`**
   - Complete list of recommended plants
   - Includes suitability scores and plant characteristics

2. **`[GardenName]_results.xlsx`**
   - Multi-sheet Excel workbook with:
     - Main clusters sheet
     - Companion relationships per cluster
     - Visualization of clusters

3. **`plant_clusters_max[N].png`**
   - Visual representation of plant clusters
   - PCA projection showing plant relationships


## 🎯 How It Works

### 1. **Location Analysis**
- Fetches real altitude data
- Determines soil pH
- Identifies geological characteristics
- Retrieves climate data (10-year history)

### 2. **Plant Suitability Scoring**
Evaluates plants based on:
- **Hardiness** (40%): Temperature tolerance
- **Soil compatibility** (25%): pH and type matching
- **Physical characteristics** (20%): Growth habit and rate
- **Shade tolerance** (15%): Light requirements
- **Moisture needs** (15%): Water requirements
- **Usefulness** (10%): Edibility and medicinal value

### 3. **Clustering**
- Groups similar plants together
- Ensures clusters don't exceed maximum size
- Uses K-means clustering with trait-based features

### 4. **Companion Analysis**
- Identifies beneficial plant relationships
- Shows which plants help or harm each other
- Provides relationship details per cluster

**Features:**
- ✅ **Synonym mapping** for 20+ plant categories
- ✅ **Smart name normalization** 
- ✅ **Flexible matching** (exact, substring, word boundary)
- ✅ **Handles plurals, commas, special cases**

## 🔍 Understanding the Results

### Suitability Scores
- **0.8 - 1.0**: Excellent match for your location
- **0.6 - 0.8**: Good match, should thrive
- **0.4 - 0.6**: Moderate match, may need extra care
- **Below 0.4**: Poor match, not recommended

### Cluster Information
- Plants in the same cluster have similar requirements
- Smaller clusters (3-5 plants) are easier to manage
- Check companion sheets for plant interactions

### Companion Plant Links
- **"helps"**: Beneficial relationship
- **"harms"**: Negative relationship
- **"neutral"**: No significant interaction

## ⚙️ Configuration

You can modify settings in `garden_planner_core.py`:

```python
class Config:
    MAX_CLUSTER_SIZE = 5  # Change maximum plants per cluster
    
    WEIGHTS = {
        'hardiness': 0.4,    # Adjust scoring weights
        'shade': 0.15,
        'moisture': 0.15,
        'soil': 0.25,
        'physical': 0.20,
        'usefulness': 0.10,
    }
```

## 🐛 Troubleshooting

### "Plant database not found"
- Ensure `pfaf2.csv` is in the same directory as the script

### "No suitable plants found"
- Try lowering the minimum suitability score
- Increase the number of recommendations

### API Timeouts
- Some external API calls may fail
- The system will use default values and continue

### Installation Issues
- Make sure you have Python 3.8+: `python --version`
- Try upgrading pip: `pip install --upgrade pip`
- Install packages one by one if batch install fails

## 📝 Notes

- **Internet required**: First run fetches real environmental data
- **Processing time**: Depends on number of plants in database
- **Database**: SQLite database stores all data for future queries
- **Climate scenarios**: Includes current, 2050, and 2100 projections

## 🤝 Tips for Best Results

1. **Accurate coordinates**: Use precise lat/lon for your exact location
2. **Appropriate thresholds**: Start with 0.5 minimum score
3. **Cluster size**: Keep at 5 or less for easier management
4. **Review companions**: Check the companion sheets in Excel
5. **Multiple locations**: Run for different garden areas

## 📧 Support

For issues or questions:
- Check the troubleshooting section
- Review the output CSV for detailed plant information
- Verify your input coordinates are correct

## 🌟 Quick Start Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the program
python garden_planner_main.py

# 3. Enter your information when prompted

# 4. Check the generated files:
#    - [YourGarden]_recommendations.csv
#    - [YourGarden]_results.xlsx
#    - plant_clusters_max5.png
```

---

**Happy Gardening! 🌱🌻🌿**
