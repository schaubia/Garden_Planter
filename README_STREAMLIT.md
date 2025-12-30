# 🌱 Garden Planner - Streamlit Deployment Guide

This guide will help you deploy your Garden Planner app on Streamlit Cloud.

## 📋 Prerequisites

1. A GitHub account
2. Your garden planner files in a GitHub repository
3. A Streamlit Cloud account (free at https://streamlit.io/cloud)

## 🚀 Quick Deployment Steps

### 1. Prepare Your Repository

Make sure your GitHub repository contains these files:

```
Garden_Planter/
├── streamlit_app.py              # Main Streamlit application
├── garden_planner_core.py        # Core functionality (your existing file)
├── requirements.txt              # Python dependencies
├── pfaf2.csv                     # Plant database (required)
├── companion_plants.csv          # Companion plants (optional)
└── README_STREAMLIT.md           # This file
```

### 2. Update Your Repository

If you don't have `streamlit_app.py` yet:

1. Copy the `streamlit_app.py` file to your repository root
2. Update `requirements.txt` to include Streamlit and other dependencies
3. Commit and push all changes to GitHub:

```bash
git add streamlit_app.py requirements.txt
git commit -m "Add Streamlit web interface"
git push origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Sign in with your GitHub account
3. Click "New app"
4. Configure your app:
   - **Repository**: Select `schaubia/Garden_Planter`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. Click "Deploy!"

Your app will be live at: `https://[your-app-name].streamlit.app`

## 🔧 Configuration Options

### Environment Variables (Optional)

If you need to configure API keys or settings, add them in Streamlit Cloud:

1. Go to your app settings
2. Click "Secrets"
3. Add your secrets in TOML format:

```toml
[api]
weather_api_key = "your-api-key-here"
```

### Resource Limits

Streamlit Cloud free tier provides:
- 1 GB of RAM
- Share apps publicly
- Community support

If your app needs more resources, consider upgrading to a paid plan.

## 📁 File Requirements

### Required Files

1. **pfaf2.csv** - Plant database
   - This file MUST be present in your repository
   - Contains plant characteristics and growing requirements
   - Size: Make sure it's under GitHub's 100MB file limit

2. **garden_planner_core.py** - Core logic
   - Contains all the classes: LocationAnalyzer, PlantDatabase, etc.
   - This is your existing core Python file

### Optional Files

1. **companion_plants.csv** - Companion plant relationships
   - Adds companion planting analysis
   - App will work without it but with reduced functionality

## 🐛 Troubleshooting

### "Module not found" errors

**Solution**: Make sure all required packages are in `requirements.txt`

```bash
# Check your requirements.txt includes:
streamlit>=1.28.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
requests>=2.26.0
geopy>=2.2.0
meteostat>=1.6.0
xlsxwriter>=3.0.0
```

### "File not found" errors

**Solution**: Ensure data files are in the repository root or update paths in code:

```python
# In streamlit_app.py, update paths if needed:
db.load_pfaf_data("pfaf2.csv")  # or "data/pfaf2.csv"
```

### App is slow

**Solutions**:
1. Reduce the default number of recommendations
2. Add caching to expensive operations:

```python
@st.cache_data
def load_plant_database():
    # Your database loading code
    pass
```

3. Consider upgrading to Streamlit Cloud paid tier for more resources

### Data files too large

If `pfaf2.csv` exceeds GitHub limits:

**Option 1**: Use Git LFS (Large File Storage)
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes
git add pfaf2.csv
git commit -m "Add large files with LFS"
```

**Option 2**: Host data externally
- Upload to cloud storage (S3, Google Drive, etc.)
- Download in app on first run
- Cache the data

## 🎨 Customization

### Modify the UI

Edit `streamlit_app.py` to customize:

```python
# Change page title and icon
st.set_page_config(
    page_title="🌿 My Garden App",
    page_icon="🌿",
)

# Update colors
st.markdown("""
    <style>
    .main-header {
        color: #YOUR_COLOR;
    }
    </style>
""", unsafe_allow_html=True)
```

### Add Authentication (Optional)

For private deployments, add password protection:

```python
import streamlit as st

def check_password():
    def password_entered():
        if st.session_state["password"] == "your-password":
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", 
                     on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", 
                     on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if check_password():
    # Your main app code here
    pass
```

## 📊 Performance Tips

1. **Cache expensive operations**:
```python
@st.cache_data
def load_data():
    return pd.read_csv("pfaf2.csv")

@st.cache_resource
def create_database():
    return PlantDatabase(config)
```

2. **Show progress for long operations**:
```python
progress_bar = st.progress(0)
for i, step in enumerate(steps):
    # Do work
    progress_bar.progress((i + 1) / len(steps))
```

3. **Use session state to avoid recomputation**:
```python
if 'results' not in st.session_state:
    st.session_state.results = compute_results()
```

## 🔗 Useful Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Community Forum](https://discuss.streamlit.io)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [GitHub Large File Storage (LFS)](https://git-lfs.github.com)

## 📝 Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🆘 Support

If you encounter issues:

1. Check the Streamlit Cloud logs in your app dashboard
2. Search the [Streamlit Community Forum](https://discuss.streamlit.io)
3. Review [Streamlit debugging guide](https://docs.streamlit.io/knowledge-base/deploy)

## 🎉 Success!

Once deployed, your app will:
- ✅ Be accessible via a public URL
- ✅ Auto-update when you push to GitHub
- ✅ Handle multiple concurrent users
- ✅ Provide a beautiful web interface for your garden planner

Share your app URL with friends and fellow gardeners! 🌱

---

**Happy Gardening & Happy Coding!** 🌻
