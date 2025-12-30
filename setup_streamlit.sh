#!/bin/bash

# Garden Planner - Streamlit Deployment Setup Script
# This script helps you prepare your repository for Streamlit deployment

echo "🌱 Garden Planner - Streamlit Deployment Setup"
echo "=============================================="
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository!"
    echo "Please run this script from your Garden_Planter repository root."
    exit 1
fi

echo "✓ Git repository detected"
echo ""

# Check for required files
echo "📋 Checking required files..."
echo ""

required_files=("garden_planner_core.py" "pfaf2.csv")
missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file found"
    else
        echo "❌ $file NOT FOUND (REQUIRED)"
        missing_files+=("$file")
    fi
done

optional_files=("companion_plants.csv" "garden_planner.db")

for file in "${optional_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file found (optional)"
    else
        echo "⚠ $file not found (optional, app will work without it)"
    fi
done

echo ""

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ Missing required files. Please add them before deploying."
    exit 1
fi

# Check file sizes
echo "📊 Checking file sizes..."
echo ""

pfaf_size=$(du -h pfaf2.csv | cut -f1)
echo "pfaf2.csv size: $pfaf_size"

pfaf_bytes=$(stat -f%z pfaf2.csv 2>/dev/null || stat -c%s pfaf2.csv 2>/dev/null)
if [ $pfaf_bytes -gt 104857600 ]; then
    echo "⚠ WARNING: pfaf2.csv is larger than 100MB"
    echo "   GitHub has a 100MB file size limit."
    echo "   Consider using Git LFS or hosting data externally."
fi

echo ""

# Create Streamlit files if they don't exist
echo "📝 Setting up Streamlit files..."
echo ""

files_to_copy=("streamlit_app.py" "requirements.txt" "README_STREAMLIT.md" ".streamlit/config.toml")

for file in "${files_to_copy[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠ $file not found - you'll need to add this file"
    else
        echo "✓ $file ready"
    fi
done

echo ""

# Check if streamlit is installed
if command -v streamlit &> /dev/null; then
    echo "✓ Streamlit is installed"
    echo ""
    echo "🚀 You can test locally with: streamlit run streamlit_app.py"
else
    echo "⚠ Streamlit not installed locally"
    echo "   Install with: pip install streamlit"
fi

echo ""
echo "=============================================="
echo "📋 Next Steps:"
echo "=============================================="
echo ""
echo "1. Add the new files to git:"
echo "   git add streamlit_app.py requirements.txt README_STREAMLIT.md .streamlit/"
echo ""
echo "2. Commit the changes:"
echo "   git commit -m 'Add Streamlit web interface'"
echo ""
echo "3. Push to GitHub:"
echo "   git push origin main"
echo ""
echo "4. Deploy on Streamlit Cloud:"
echo "   - Go to https://streamlit.io/cloud"
echo "   - Sign in with GitHub"
echo "   - Click 'New app'"
echo "   - Select your repository: schaubia/Garden_Planter"
echo "   - Set main file: streamlit_app.py"
echo "   - Click 'Deploy!'"
echo ""
echo "=============================================="
echo "✨ Setup complete! Ready for deployment!"
echo "=============================================="
