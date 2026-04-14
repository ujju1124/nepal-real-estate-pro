# FREE Deployment Guide for Students
## Nepal Land & House Price Prediction System

Complete guide for deploying your project **100% FREE** as a student.

---

## 🎯 Quick Recommendation

**BEST OPTION: Streamlit Community Cloud** ⭐
- ✅ 100% Free forever
- ✅ 1GB RAM (enough for your 4MB models)
- ✅ Easy deployment (3 clicks)
- ✅ Auto-updates from GitHub
- ✅ Custom domain support
- ✅ Perfect for student projects

**Your models total: ~4MB** → Will work perfectly on free tier!

---

## Option 1: Streamlit Community Cloud (RECOMMENDED) ⭐⭐⭐⭐⭐

### Why This is Best for You:
- **Cost:** $0 forever
- **RAM:** 1GB (your app needs ~450MB)
- **Storage:** Unlimited
- **Deployment Time:** 5 minutes
- **Maintenance:** Zero (auto-updates)
- **Perfect for:** Student projects, demos, portfolio

### Step-by-Step Deployment:

#### Step 1: Prepare Your Repository

**1.1 Create `.streamlit/config.toml` file:**

```bash
mkdir .streamlit
```

Create file `.streamlit/config.toml`:
```toml
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
```

**1.2 Optimize `requirements.txt` for Streamlit Cloud:**

Create `requirements_streamlit.txt`:
```txt
# Core (Required)
streamlit==1.32.0
pandas==2.0.3
numpy==1.24.3
plotly==5.18.0

# ML (Required)
scikit-learn==1.4.0
xgboost==2.0.3
catboost==1.2.2
joblib==1.3.2

# RAG Chatbot (Required)
langchain-core==0.1.52
langchain-community==0.0.38
langchain-openai==0.1.7
langchain-huggingface==0.0.3
faiss-cpu==1.7.4
sentence-transformers==2.6.1
openai==1.30.1

# Utilities
python-dotenv==1.0.0
```

**Why optimized?** Removed unnecessary packages (selenium, beautifulsoup - only needed for scraping)

**1.3 Push to GitHub:**

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Prepare for Streamlit Cloud deployment"

# Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/Nepal-Land-Price-Prediction.git
git branch -M main
git push -u origin main
```

#### Step 2: Deploy on Streamlit Cloud

**2.1 Sign Up (FREE):**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign up with GitHub"
3. Authorize Streamlit

**2.2 Deploy App:**
1. Click "New app" button
2. Fill in:
   - **Repository:** `YOUR_USERNAME/Nepal-Land-Price-Prediction`
   - **Branch:** `main`
   - **Main file path:** `app_final.py`
   - **App URL:** `nepal-realestate` (or your choice)
3. Click "Deploy!"

**2.3 Add Secrets (API Keys):**
1. While app is deploying, click "Advanced settings" → "Secrets"
2. Add your API keys:

```toml
# Paste this in the Secrets section
GITHUB_TOKEN = "ghp_your_github_token_here"
HUGGINGFACEHUB_API_TOKEN = "hf_your_huggingface_token_here"
```

3. Click "Save"

#### Step 3: Wait for Deployment

- **First deployment:** 5-10 minutes
- **Status:** Watch the logs in real-time
- **Success:** You'll see "Your app is live!" 🎉

#### Step 4: Access Your App

**Your app URL:** `https://nepal-realestate.streamlit.app`

**Share with:**
- Professors
- Classmates
- Portfolio
- LinkedIn

### Troubleshooting Streamlit Cloud:

**Issue 1: "App exceeds resource limits"**

**Solution:** Optimize memory usage in `app_final.py`:

```python
# Add at the top of app_final.py
import streamlit as st

# Optimize caching
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_models():
    # Only load models when needed
    return {
        'general_housing': joblib.load('xgboost_housing_final.pkl'),
        # Load others on demand
    }

# Lazy loading - only load model user selects
model_type = st.selectbox("Select Model", [...])
if model_type == 'general_housing':
    model = load_models()['general_housing']
```

**Issue 2: "Module not found"**

**Solution:** Ensure `requirements_streamlit.txt` is named `requirements.txt` in your repo

**Issue 3: "Secrets not loading"**

**Solution:** Access secrets in code:
```python
import os
github_token = st.secrets.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")
```

### Updating Your Deployed App:

```bash
# Make changes locally
# Test: streamlit run app_final.py

# Push to GitHub
git add .
git commit -m "Update feature X"
git push

# Streamlit Cloud auto-deploys in 2-3 minutes!
```

---

## Option 2: Hugging Face Spaces (Alternative) ⭐⭐⭐⭐

### Why Consider This:
- **Cost:** $0 forever
- **RAM:** 16GB (way more than you need!)
- **Storage:** 50GB
- **GPU:** Free CPU (GPU costs money)
- **Community:** ML/AI focused

### Deployment Steps:

#### Step 1: Create Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in:
   - **Space name:** `nepal-realestate`
   - **License:** MIT
   - **SDK:** Streamlit
   - **Hardware:** CPU basic (FREE)
4. Click "Create Space"

#### Step 2: Upload Files

**Option A: Git (Recommended)**
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/nepal-realestate
cd nepal-realestate

# Copy your files
cp /path/to/your/project/* .

# Push
git add .
git commit -m "Initial deployment"
git push
```

**Option B: Web Interface**
- Click "Files" → "Add file" → Upload your files

#### Step 3: Configure

Create `README.md` in Space:
```markdown
---
title: Nepal Real Estate Price Prediction
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app_final.py
pinned: false
---

# Nepal Land & House Price Prediction System
ML-powered real estate price prediction for Kathmandu Valley.
```

#### Step 4: Add Secrets

1. Go to Space settings
2. Click "Repository secrets"
3. Add:
   - `GITHUB_TOKEN`: your token
   - `HUGGINGFACEHUB_API_TOKEN`: your token

#### Step 5: Access App

**URL:** `https://huggingface.co/spaces/YOUR_USERNAME/nepal-realestate`

### Pros vs Cons:

**Pros:**
- More RAM (16GB vs 1GB)
- ML community visibility
- Better for ML portfolios

**Cons:**
- Slower cold start (30s vs 5s)
- Less intuitive than Streamlit Cloud
- URL is longer

---

## Option 3: Render (Backup Option) ⭐⭐⭐

### Free Tier:
- **Cost:** $0
- **RAM:** 512MB (tight but possible)
- **Limitation:** Spins down after 15 min inactivity
- **Cold start:** 30-60 seconds

### Quick Deploy:

1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New" → "Web Service"
4. Select your repo
5. Configure:
   - **Name:** nepal-realestate
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app_final.py --server.port=$PORT --server.address=0.0.0.0`
6. Add environment variables (secrets)
7. Click "Create Web Service"

**Note:** Free tier sleeps after 15 min → First load takes 30-60s

---

## Option 4: Railway (Another Alternative) ⭐⭐⭐

### Free Tier:
- **Cost:** $5 credit/month (FREE)
- **RAM:** 8GB
- **Usage:** ~$5/month if running 24/7
- **Strategy:** Deploy only when needed

### Deploy:

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repo
5. Add environment variables
6. Deploy!

**Cost Management:**
- Only run when demoing
- Pause when not in use
- $5 credit = ~500 hours/month

---

## 📊 Comparison Table

| Platform | Cost | RAM | Storage | Cold Start | Best For |
|----------|------|-----|---------|------------|----------|
| **Streamlit Cloud** ⭐ | $0 | 1GB | Unlimited | 5s | **Student projects** |
| Hugging Face | $0 | 16GB | 50GB | 30s | ML portfolios |
| Render | $0 | 512MB | Limited | 60s | Backup option |
| Railway | $5 credit | 8GB | 100GB | 10s | Temporary demos |

---

## 🎯 My Recommendation for You

### Deploy on Streamlit Cloud because:

1. **Your models are small (4MB)** → 1GB RAM is plenty
2. **Zero cost forever** → No credit card needed
3. **Always online** → No cold starts
4. **Auto-updates** → Push to GitHub = auto-deploy
5. **Professional URL** → `your-app.streamlit.app`
6. **Perfect for demos** → Fast, reliable, easy to share

### Deployment Timeline:

```
Today (30 minutes):
├── Push to GitHub (5 min)
├── Sign up Streamlit Cloud (2 min)
├── Deploy app (3 min)
├── Add secrets (2 min)
└── Wait for deployment (10 min)

Result: Live app at https://nepal-realestate.streamlit.app
```

---

## 🚀 Step-by-Step: Deploy RIGHT NOW

### 1. Prepare Repository (5 minutes)

```bash
# Create Streamlit config
mkdir .streamlit
echo '[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
port = 8501' > .streamlit/config.toml

# Ensure requirements.txt is optimized
# (Use the requirements_streamlit.txt content from above)

# Push to GitHub
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main
```

### 2. Deploy on Streamlit Cloud (3 minutes)

1. Open [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign in with GitHub"
3. Click "New app"
4. Select:
   - Repo: `YOUR_USERNAME/Nepal-Land-Price-Prediction`
   - Branch: `main`
   - File: `app_final.py`
5. Click "Deploy"

### 3. Add Secrets (2 minutes)

While deploying, click "Advanced settings" → "Secrets":

```toml
GITHUB_TOKEN = "ghp_xxxxx"
HUGGINGFACEHUB_API_TOKEN = "hf_xxxxx"
```

### 4. Share Your App (1 minute)

Once deployed:
- Copy URL: `https://your-app.streamlit.app`
- Add to your resume/portfolio
- Share with professors
- Include in project report

---

## 💡 Pro Tips for Free Deployment

### Tip 1: Optimize for Free Tier

**Reduce memory usage:**
```python
# In app_final.py
import streamlit as st

# Only load model user needs
@st.cache_resource
def load_model(model_name):
    if model_name == 'general_housing':
        return joblib.load('xgboost_housing_final.pkl')
    # ... load others only when needed

# Don't load all 4 models at once
model_type = st.selectbox("Select Property Type", [...])
model = load_model(model_type)  # Load only selected model
```

### Tip 2: Use Lightweight Dependencies

**Replace heavy packages:**
- ❌ `scipy` (large) → ✅ Use only if needed
- ❌ `lightgbm` (not used) → ✅ Remove from requirements.txt
- ✅ Keep only what you actually use

### Tip 3: Add Loading Indicators

```python
with st.spinner('Loading model...'):
    model = load_model(model_type)
st.success('Model loaded!')
```

### Tip 4: Handle Errors Gracefully

```python
try:
    result = predict_price(user_input)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.info("Please check your inputs and try again")
```

---

## 📱 Bonus: Make It Mobile-Friendly

Add to `.streamlit/config.toml`:
```toml
[browser]
gatherUsageStats = false

[server]
enableCORS = false
enableXsrfProtection = false
```

Add responsive layout in `app_final.py`:
```python
# Use columns for mobile
col1, col2 = st.columns([1, 1])
with col1:
    district = st.selectbox("District", [...])
with col2:
    neighborhood = st.selectbox("Neighborhood", [...])
```

---

## 🎓 For Your Project Report

**Add this section to your report:**

### Deployment

The application is deployed on **Streamlit Community Cloud**, a free platform for hosting Streamlit applications. The deployment process involves:

1. **Version Control:** Code hosted on GitHub for version management
2. **Continuous Deployment:** Automatic updates when code is pushed to main branch
3. **Environment Management:** Secrets (API keys) stored securely in Streamlit Cloud
4. **Resource Optimization:** Application optimized to run within 1GB RAM limit

**Live Application:** [https://nepal-realestate.streamlit.app](https://your-actual-url.streamlit.app)

**Deployment Architecture:**
```
GitHub Repository → Streamlit Cloud → Live Application
     ↓                    ↓                  ↓
  Code Push         Auto-Deploy        Public Access
```

---

## ❓ FAQ

**Q: Will my app stay online forever?**
A: Yes! Streamlit Cloud free tier has no time limit.

**Q: Can I use a custom domain?**
A: Yes! Streamlit Cloud supports custom domains (e.g., nepal-realestate.com)

**Q: What if I exceed 1GB RAM?**
A: Optimize by loading only one model at a time (see Tip 1 above)

**Q: Can I deploy multiple apps?**
A: Yes! Unlimited apps on Streamlit Cloud free tier.

**Q: Is my data secure?**
A: Yes! Secrets are encrypted. Your data never leaves the app.

**Q: Can I add authentication?**
A: Yes! Use `streamlit-authenticator` package for login.

---

## 🎉 Next Steps

1. **Deploy now** using Streamlit Cloud (30 minutes)
2. **Test your app** - Try all features
3. **Share URL** - Add to resume, LinkedIn, project report
4. **Monitor usage** - Check Streamlit Cloud analytics
5. **Iterate** - Push updates as needed

---

## 📞 Need Help?

**Streamlit Community:**
- Forum: [discuss.streamlit.io](https://discuss.streamlit.io)
- Docs: [docs.streamlit.io](https://docs.streamlit.io)
- Discord: [streamlit.io/community](https://streamlit.io/community)

**Common Issues:**
- Deployment fails → Check requirements.txt
- App crashes → Check logs in Streamlit Cloud
- Secrets not working → Verify format in Secrets section

---

**Good luck with your deployment! 🚀**

Your app will be live in 30 minutes and you can share it with the world!
