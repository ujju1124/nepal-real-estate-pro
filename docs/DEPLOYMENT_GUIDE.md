# Deployment Guide
## Nepal Land & House Price Prediction System

Complete step-by-step guide for deploying the application locally and in production.

---

## Table of Contents

1. [Local Development Setup](#1-local-development-setup)
2. [Environment Configuration](#2-environment-configuration)
3. [Running the Application](#3-running-the-application)
4. [Testing](#4-testing)
5. [Production Deployment](#5-production-deployment)
6. [Troubleshooting](#6-troubleshooting)
7. [Maintenance](#7-maintenance)

---

## 1. Local Development Setup

### 1.1 Prerequisites

**System Requirements:**
- **Operating System:** Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python:** 3.8 or higher (3.11 recommended)
- **RAM:** Minimum 4GB, Recommended 8GB
- **Disk Space:** 5GB free space
- **Internet:** Required for initial setup and RAG chatbot

**Check Python Version:**
```bash
python --version
# Should show Python 3.8.x or higher
```

If Python not installed, download from [python.org](https://www.python.org/downloads/)

### 1.2 Clone Repository

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Nepal-Land-Price-Prediction.git
cd Nepal-Land-Price-Prediction

# Verify files
ls -la
# Should see: app_final.py, requirements.txt, .env.example, etc.
```

### 1.3 Create Virtual Environment

**Why Virtual Environment?**
- Isolates project dependencies
- Prevents conflicts with system Python packages
- Makes deployment reproducible

**Create and Activate:**

**On Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Verify activation (should show (venv) prefix)
```

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Verify activation
which python
# Should show: /path/to/project/venv/bin/python
```

### 1.4 Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list
# Should show: streamlit, pandas, scikit-learn, xgboost, catboost, etc.
```

**Installation Time:** 5-10 minutes (depends on internet speed)

**Common Issues:**
- **Error: "Microsoft Visual C++ required"** (Windows)
  - Solution: Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- **Error: "Command 'gcc' failed"** (Linux)
  - Solution: `sudo apt-get install build-essential python3-dev`

---

## 2. Environment Configuration

### 2.1 Create .env File

```bash
# Copy example file
cp .env.example .env

# Edit .env file
nano .env  # or use any text editor
```

### 2.2 Required Environment Variables

**File: `.env`**
```bash
# GitHub Models API (for RAG chatbot)
GITHUB_TOKEN=ghp_your_github_personal_access_token_here

# HuggingFace API (for embeddings)
HUGGINGFACEHUB_API_TOKEN=hf_your_huggingface_token_here
```

### 2.3 Obtain API Tokens

**GitHub Token (for GPT-4o-mini via GitHub Models):**

1. Go to [GitHub Settings → Developer Settings → Personal Access Tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Name: "Nepal Real Estate App"
4. Scopes: Select `repo` (if private repo) or leave unchecked (if public)
5. Click "Generate token"
6. Copy token (starts with `ghp_`)
7. Paste into `.env` file

**HuggingFace Token (for sentence-transformers embeddings):**

1. Go to [HuggingFace Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name: "Nepal Real Estate App"
4. Role: "Read"
5. Click "Generate"
6. Copy token (starts with `hf_`)
7. Paste into `.env` file

**Security Note:**
- **Never commit `.env` to Git** (already in `.gitignore`)
- Keep tokens private
- Rotate tokens if exposed

### 2.4 Verify Configuration

```bash
# Test environment variables
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('GitHub Token:', os.getenv('GITHUB_TOKEN')[:10] + '...'); print('HF Token:', os.getenv('HUGGINGFACEHUB_API_TOKEN')[:10] + '...')"

# Should output:
# GitHub Token: ghp_abc123...
# HF Token: hf_xyz789...
```

---

## 3. Running the Application

### 3.1 Start Streamlit Server

```bash
# Ensure virtual environment is activated
# (venv) should be visible in terminal

# Run application
streamlit run app_final.py

# Expected output:
#   You can now view your Streamlit app in your browser.
#   Local URL: http://localhost:8501
#   Network URL: http://192.168.x.x:8501
```

**First Run:** May take 30-60 seconds to load models

### 3.2 Access Application

1. Open browser
2. Navigate to `http://localhost:8501`
3. Application should load with sidebar navigation

**Sections:**
- 📊 Market Analytics
- 🧠 Inference Engine
- 🔍 Property Recommendations
- 💬 Property Assistant (RAG Chatbot)

### 3.3 Test Basic Functionality

**Test 1: Analytics**
1. Click "Market Analytics" in sidebar
2. Select "General Housing"
3. Verify price distribution chart loads

**Test 2: Prediction**
1. Click "Inference Engine"
2. Select "General Housing"
3. Fill form:
   - District: Kathmandu
   - Neighborhood: Baluwatar
   - Land: 5 aana
   - Built-up: 3500 sq.ft.
   - Bedrooms: 4, Bathrooms: 3
4. Click "Predict Price"
5. Verify prediction displays (should be ~8-10 crore)

**Test 3: Chatbot**
1. Click "Property Assistant"
2. Type: "What's the average price in Lalitpur?"
3. Verify response generates (may take 2-4 seconds)

### 3.4 Stop Application

```bash
# In terminal where Streamlit is running
Ctrl + C

# Deactivate virtual environment
deactivate
```

---

## 4. Testing

### 4.1 Unit Tests

**Create test file:** `tests/test_predictions.py`

```python
import pytest
import pandas as pd
import numpy as np
from app_final import (
    predict_general_housing,
    predict_general_land,
    predict_lalpurja_housing,
    predict_lalpurja_land,
    engineer_features,
    validate_input
)

def test_general_housing_prediction():
    """Test general housing prediction with valid input"""
    user_input = {
        'district': 'Kathmandu',
        'neighborhood': 'Baluwatar',
        'land_aana': 5,
        'built_sqft': 3500,
        'bedrooms': 4,
        'bathrooms': 3,
        'total_floors': 3,
        'road_width_feet': 20,
        'road_type': 'Commercial',
        'parking': 2,
        'furnishing': 1,
        'property_type': 'House',
        'house_age': 5
    }
    
    result = predict_general_housing(user_input)
    
    assert 'price' in result
    assert 'price_crore' in result
    assert result['price'] > 0
    assert result['price_crore'] > 0
    assert result['confidence_lower'] < result['price_crore'] < result['confidence_upper']

def test_input_validation():
    """Test input validation catches invalid data"""
    invalid_input = {
        'land_aana': -5,  # Invalid: negative
        'bedrooms': 0,    # Invalid: zero
    }
    
    is_valid, error = validate_input(invalid_input, 'general_housing')
    assert not is_valid
    assert error is not None

def test_feature_engineering():
    """Test feature engineering produces correct number of features"""
    user_input = {
        'district': 'Kathmandu',
        'land_aana': 5,
        'bedrooms': 3,
        # ... other required fields
    }
    
    X = engineer_features(user_input, 'lalpurja_housing')
    
    assert X.shape[0] == 1  # Single row
    assert X.shape[1] == 42  # 42 features for Lalpurja Housing
    assert not X.isnull().any().any()  # No missing values

def test_all_models_load():
    """Test all four models load successfully"""
    from app_final import load_models
    
    models = load_models()
    
    assert 'general_housing' in models
    assert 'general_land' in models
    assert 'lalpurja_housing' in models
    assert 'lalpurja_land' in models
    assert all(model is not None for model in models.values())

def test_prediction_consistency():
    """Test same input produces same prediction"""
    user_input = {
        'district': 'Lalitpur',
        'neighborhood': 'Imadol',
        'land_aana': 10,
        'road_width_feet': 15,
        'road_type': 'Gravel'
    }
    
    result1 = predict_general_land(user_input)
    result2 = predict_general_land(user_input)
    
    assert result1['price'] == result2['price']

def test_confidence_intervals():
    """Test confidence intervals are reasonable"""
    user_input = {
        'district': 'Kathmandu',
        'land_aana': 5,
        # ... other fields
    }
    
    result = predict_general_housing(user_input)
    
    # CI should be within ±50% of prediction
    lower_bound = result['price_crore'] * 0.5
    upper_bound = result['price_crore'] * 1.5
    
    assert lower_bound < result['confidence_lower'] < result['price_crore']
    assert result['price_crore'] < result['confidence_upper'] < upper_bound
```

**Run Tests:**
```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/test_predictions.py -v

# Expected output:
# test_predictions.py::test_general_housing_prediction PASSED
# test_predictions.py::test_input_validation PASSED
# test_predictions.py::test_feature_engineering PASSED
# test_predictions.py::test_all_models_load PASSED
# test_predictions.py::test_prediction_consistency PASSED
# test_predictions.py::test_confidence_intervals PASSED
```

### 4.2 Integration Tests

**Test RAG Chatbot:**
```python
def test_rag_chatbot():
    """Test RAG chatbot initialization and query"""
    from app_final import initialize_rag_system, query_chatbot, load_dataset
    
    df = load_dataset('general_housing')
    qa_chain = initialize_rag_system(df)
    
    response = query_chatbot(qa_chain, "What's the average price?")
    
    assert 'result' in response
    assert len(response['result']) > 0
    assert 'source_documents' in response
```

---

## 5. Production Deployment

### 5.1 Streamlit Cloud (Recommended for Quick Deployment)

**Steps:**

1. **Push to GitHub:**
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select repository: `YOUR_USERNAME/Nepal-Land-Price-Prediction`
   - Main file: `app_final.py`
   - Click "Deploy"

3. **Add Secrets:**
   - In Streamlit Cloud dashboard, go to "Settings" → "Secrets"
   - Add:
   ```toml
   GITHUB_TOKEN = "ghp_your_token_here"
   HUGGINGFACEHUB_API_TOKEN = "hf_your_token_here"
   ```

4. **Access App:**
   - URL: `https://your-app-name.streamlit.app`

**Limitations:**
- Free tier: 1GB RAM (may struggle with all 4 models)
- Solution: Deploy only 1-2 models or upgrade to paid tier

### 5.2 AWS EC2 Deployment

**Step 1: Launch EC2 Instance**

1. Go to AWS Console → EC2
2. Launch instance:
   - AMI: Ubuntu 22.04 LTS
   - Instance type: t3.medium (4GB RAM)
   - Storage: 20GB
   - Security group: Allow ports 22 (SSH), 8501 (Streamlit)

**Step 2: Connect and Setup**

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip -y

# Clone repository
git clone https://github.com/YOUR_USERNAME/Nepal-Land-Price-Prediction.git
cd Nepal-Land-Price-Prediction

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Step 3: Configure Environment**

```bash
# Create .env file
nano .env

# Add tokens (paste from local .env)
# Save: Ctrl+O, Enter, Ctrl+X
```

**Step 4: Run with systemd (Auto-restart)**

Create service file:
```bash
sudo nano /etc/systemd/system/nepal-realestate.service
```

Add:
```ini
[Unit]
Description=Nepal Real Estate Prediction App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/Nepal-Land-Price-Prediction
Environment="PATH=/home/ubuntu/Nepal-Land-Price-Prediction/venv/bin"
ExecStart=/home/ubuntu/Nepal-Land-Price-Prediction/venv/bin/streamlit run app_final.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable nepal-realestate
sudo systemctl start nepal-realestate

# Check status
sudo systemctl status nepal-realestate

# View logs
sudo journalctl -u nepal-realestate -f
```

**Step 5: Access Application**

- URL: `http://your-ec2-ip:8501`
- For custom domain, configure Route 53 + ALB

**Cost Estimate:**
- t3.medium: ~$30/month
- 20GB storage: ~$2/month
- Total: ~$32/month

### 5.3 Docker Deployment

**Create Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "app_final.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and Run:**
```bash
# Build image
docker build -t nepal-realestate .

# Run container
docker run -d \
  -p 8501:8501 \
  -e GITHUB_TOKEN=ghp_your_token \
  -e HUGGINGFACEHUB_API_TOKEN=hf_your_token \
  --name nepal-realestate-app \
  nepal-realestate

# Check logs
docker logs -f nepal-realestate-app

# Access: http://localhost:8501
```

---

## 6. Troubleshooting

### 6.1 Common Issues

**Issue 1: "ModuleNotFoundError: No module named 'streamlit'"**

**Solution:**
```bash
# Verify virtual environment is activated
which python  # Should show venv path

# Reinstall dependencies
pip install -r requirements.txt
```

**Issue 2: "FileNotFoundError: Model file not found"**

**Solution:**
```bash
# Verify model files exist
ls -lh *.pkl

# Should see:
# xgboost_housing_final.pkl
# catboost_land_model_final.pkl
# catboost_lalpurja_house_v2_final.pkl
# catboost_lalpurja_model_final.pkl

# If missing, re-run model training notebooks
```

**Issue 3: "RAG chatbot not responding"**

**Solution:**
```bash
# Check environment variables
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GITHUB_TOKEN'))"

# Should output token (not None)

# Test API connection
curl -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user
```

**Issue 4: "Memory Error / Application Crashes"**

**Solution:**
- Increase system RAM (minimum 4GB)
- Or deploy only 1-2 models instead of all 4
- Or use cloud deployment with more resources

**Issue 5: "Slow Predictions"**

**Solution:**
```python
# Enable caching in app_final.py
@st.cache_data
def predict_general_housing(user_input):
    # ... prediction code
```

### 6.2 Debugging Tips

**Enable Debug Mode:**
```bash
streamlit run app_final.py --logger.level=debug
```

**Check Logs:**
```bash
# Streamlit logs location
~/.streamlit/logs/

# View latest log
tail -f ~/.streamlit/logs/streamlit.log
```

**Test Individual Components:**
```python
# Test model loading
python -c "import joblib; model = joblib.load('xgboost_housing_final.pkl'); print('Model loaded successfully')"

# Test prediction
python -c "from app_final import predict_general_housing; print(predict_general_housing({'district': 'Kathmandu', 'land_aana': 5, ...}))"
```

---

## 7. Maintenance

### 7.1 Updating Models

**When to Update:**
- New data available (quarterly recommended)
- Model performance degrades
- Market conditions change significantly

**Steps:**
1. Scrape new data
2. Re-run cleaning notebooks
3. Re-run feature engineering notebooks
4. Re-run model building notebooks
5. Replace `.pkl` files
6. Restart application

### 7.2 Monitoring

**Metrics to Track:**
- Prediction latency (should be < 200ms)
- Memory usage (should be < 1GB)
- Error rate (should be < 1%)
- User queries (for chatbot improvement)

**Tools:**
- Streamlit built-in metrics
- AWS CloudWatch (if on EC2)
- Custom logging

### 7.3 Backup

**What to Backup:**
- Model files (`.pkl`)
- Cleaned datasets (`.csv`)
- Environment configuration (`.env` - securely)
- Application code (`app_final.py`)

**Backup Script:**
```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR="backups/$DATE"

mkdir -p $BACKUP_DIR
cp *.pkl $BACKUP_DIR/
cp *.csv $BACKUP_DIR/
cp app_final.py $BACKUP_DIR/
cp requirements.txt $BACKUP_DIR/

echo "Backup completed: $BACKUP_DIR"
```

### 7.4 Security Updates

**Regular Tasks:**
- Update Python packages: `pip install --upgrade -r requirements.txt`
- Rotate API tokens (every 6 months)
- Review access logs
- Update system packages: `sudo apt update && sudo apt upgrade`

---

## 8. Performance Optimization

### 8.1 Caching Strategy

```python
# Cache model loading
@st.cache_resource
def load_models():
    return joblib.load('model.pkl')

# Cache data loading
@st.cache_data
def load_dataset(name):
    return pd.read_csv(f'{name}.csv')

# Cache predictions (careful with this)
@st.cache_data
def predict_price(_model, features_tuple):
    return _model.predict(features_tuple)
```

### 8.2 Lazy Loading

```python
# Load models only when needed
if page == "Predict":
    models = load_models()  # Only load when user goes to prediction page
```

### 8.3 Compression

```python
# Compress model files
import joblib
joblib.dump(model, 'model.pkl', compress=3)  # Compression level 0-9
```

---

## 9. Scaling Considerations

**For High Traffic (>1000 users/day):**

1. **Load Balancer:** Use AWS ALB to distribute traffic
2. **Multiple Instances:** Run 3-5 EC2 instances
3. **Model Serving:** Use dedicated model serving (TensorFlow Serving, TorchServe)
4. **Caching Layer:** Redis for prediction caching
5. **CDN:** CloudFront for static assets

**Architecture:**
```
Users → CloudFront → ALB → [EC2-1, EC2-2, EC2-3] → Redis Cache
                                    ↓
                              Model Server (GPU)
```

---

**End of Deployment Guide**

*For technical details, see [METHODOLOGY.md](METHODOLOGY.md). For API reference, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).*
