#!/bin/bash
# Deployment Preparation Script
# Run this before deploying to Streamlit Cloud

echo "🚀 Preparing Nepal Real Estate App for Deployment..."
echo ""

# Step 1: Create .streamlit directory
echo "📁 Creating .streamlit directory..."
mkdir -p .streamlit

# Step 2: Create config.toml
echo "⚙️  Creating Streamlit config..."
cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
EOF

echo "✅ Config created!"
echo ""

# Step 3: Check model files
echo "🔍 Checking model files..."
models=(
    "xgboost_housing_final.pkl"
    "catboost_land_model_final.pkl"
    "catboost_lalpurja_house_v2_final.pkl"
    "catboost_lalpurja_model_final.pkl"
    "scaler_lalpurja_house_v2.pkl"
)

missing_models=()
for model in "${models[@]}"; do
    if [ -f "$model" ]; then
        size=$(du -h "$model" | cut -f1)
        echo "  ✅ $model ($size)"
    else
        echo "  ❌ $model (MISSING)"
        missing_models+=("$model")
    fi
done

if [ ${#missing_models[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  WARNING: Missing model files!"
    echo "Please ensure all model files are present before deploying."
    echo ""
fi

# Step 4: Check requirements.txt
echo ""
echo "📦 Checking requirements.txt..."
if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt found"
    echo "   Packages: $(wc -l < requirements.txt | tr -d ' ') lines"
else
    echo "❌ requirements.txt NOT FOUND"
    echo "   Creating basic requirements.txt..."
    cat > requirements.txt << 'EOF'
streamlit>=1.32.0
pandas>=2.0.0
numpy>=2.1.0
plotly>=5.18.0
scikit-learn>=1.4.0
xgboost>=2.0.0
catboost>=1.2.0
joblib>=1.3.0
langchain-core>=0.1.0
langchain-community>=0.0.20
langchain-openai>=0.1.0
langchain-huggingface>=0.0.3
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
openai>=1.0.0
python-dotenv>=1.0.0
EOF
    echo "✅ Created requirements.txt"
fi

# Step 5: Check .gitignore
echo ""
echo "🔒 Checking .gitignore..."
if [ -f ".gitignore" ]; then
    if grep -q ".env" .gitignore; then
        echo "✅ .env is in .gitignore (good!)"
    else
        echo "⚠️  Adding .env to .gitignore..."
        echo ".env" >> .gitignore
    fi
    if grep -q "venv/" .gitignore; then
        echo "✅ venv/ is in .gitignore"
    else
        echo "⚠️  Adding venv/ to .gitignore..."
        echo "venv/" >> .gitignore
    fi
else
    echo "⚠️  Creating .gitignore..."
    cat > .gitignore << 'EOF'
.env
venv/
__pycache__/
*.pyc
.DS_Store
.ipynb_checkpoints/
*.log
EOF
    echo "✅ Created .gitignore"
fi

# Step 6: Check if .env exists (should NOT be committed)
echo ""
echo "🔐 Checking secrets..."
if [ -f ".env" ]; then
    echo "✅ .env file found (for local development)"
    echo "⚠️  IMPORTANT: Make sure .env is in .gitignore!"
    echo "   You'll need to add secrets manually in Streamlit Cloud"
else
    echo "⚠️  .env file not found"
    echo "   You'll need to add secrets in Streamlit Cloud dashboard"
fi

# Step 7: Check app_final.py
echo ""
echo "📱 Checking main app file..."
if [ -f "app_final.py" ]; then
    lines=$(wc -l < app_final.py | tr -d ' ')
    echo "✅ app_final.py found ($lines lines)"
else
    echo "❌ app_final.py NOT FOUND"
    echo "   This is your main application file!"
fi

# Step 8: Git status
echo ""
echo "📊 Git status..."
if [ -d ".git" ]; then
    echo "✅ Git repository initialized"
    
    # Check if there are uncommitted changes
    if [[ -n $(git status -s) ]]; then
        echo "⚠️  You have uncommitted changes:"
        git status -s | head -5
        echo ""
        echo "Run these commands to commit:"
        echo "  git add ."
        echo "  git commit -m 'Prepare for deployment'"
        echo "  git push origin main"
    else
        echo "✅ No uncommitted changes"
    fi
else
    echo "⚠️  Git not initialized"
    echo "Run: git init"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 DEPLOYMENT READINESS SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ready=true

if [ ${#missing_models[@]} -eq 0 ]; then
    echo "✅ All model files present"
else
    echo "❌ Missing model files"
    ready=false
fi

if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt ready"
else
    echo "❌ requirements.txt missing"
    ready=false
fi

if [ -f ".streamlit/config.toml" ]; then
    echo "✅ Streamlit config ready"
else
    echo "❌ Streamlit config missing"
    ready=false
fi

if [ -f "app_final.py" ]; then
    echo "✅ Main app file ready"
else
    echo "❌ Main app file missing"
    ready=false
fi

if [ -f ".gitignore" ] && grep -q ".env" .gitignore; then
    echo "✅ .gitignore configured"
else
    echo "⚠️  .gitignore needs attention"
fi

echo ""
if [ "$ready" = true ]; then
    echo "🎉 READY FOR DEPLOYMENT!"
    echo ""
    echo "Next steps:"
    echo "1. Push to GitHub:"
    echo "   git add ."
    echo "   git commit -m 'Ready for deployment'"
    echo "   git push origin main"
    echo ""
    echo "2. Deploy on Streamlit Cloud:"
    echo "   → Go to: https://share.streamlit.io"
    echo "   → Click 'New app'"
    echo "   → Select your repo and app_final.py"
    echo "   → Add secrets (GITHUB_TOKEN, HUGGINGFACEHUB_API_TOKEN)"
    echo "   → Click 'Deploy!'"
    echo ""
    echo "3. Follow DEPLOYMENT_CHECKLIST.md for detailed steps"
else
    echo "⚠️  NOT READY - Please fix the issues above"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
