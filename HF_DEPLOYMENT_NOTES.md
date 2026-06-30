# Hugging Face Spaces Deployment — Memory Assessment

## 📊 Resource Analysis

### Data Files (loaded at startup)
| File | Size | Rows | Purpose |
|------|------|------|---------|
| housing_model_ready_after_outlier_treatment.csv | 0.57 MB | 2,506 | General housing analytics |
| cleaned_land_merged_final_after_eda.csv | 0.78 MB | 4,063 | General land analytics |
| cleaned_lalpurja_house_v2_after_cleaning.csv | 0.44 MB | 2,187 | Lalpurja housing analytics |
| cleaned_lalpurja_land_final_after_eda.csv | 0.20 MB | 1,214 | Lalpurja land analytics |
| housing_features_ready_after_feature_engineering.csv | 0.34 MB | ~2,005 | Feature engineering mappings |
| lalpurja_house_v2_features_ready.csv | 0.89 MB | ~1,749 | Feature engineering mappings |
| lalpurja_dataset_ready_after_feature_engineering.csv | 0.33 MB | ~971 | Feature engineering mappings |
| land_features_final_modeled.csv | 0.44 MB | ~3,250 | Feature engineering mappings |
| **TOTAL DATA** | **3.99 MB** | **~10,000 rows** | |

### Model Files (loaded at startup)
| File | Size | Type | Purpose |
|------|------|------|---------|
| xgboost_housing_final.pkl | 0.33 MB | XGBoost | General housing predictions |
| catboost_land_model_final.pkl | 0.07 MB | CatBoost | General land predictions |
| catboost_lalpurja_house_v2_final.pkl | 1.07 MB | CatBoost | Lalpurja housing predictions |
| catboost_lalpurja_model_final.pkl | 0.48 MB | CatBoost | Lalpurja land predictions |
| scaler_lalpurja_house_v2.pkl | <0.01 MB | Scaler | Preprocessing |
| **TOTAL MODELS** | **1.95 MB** | | |

### Memory Footprint Estimate

**Disk Storage**: 5.94 MB (data + models on disk)

**RAM Usage Breakdown** (approximate):
1. **Python + Streamlit base**: ~200-300 MB
2. **Pandas DataFrames** (10k rows loaded):
   - On disk: 3.99 MB
   - In memory: ~40-60 MB (pandas overhead, dtype expansion)
3. **ML Models** (4 models + 1 scaler):
   - On disk: 1.95 MB
   - In memory: ~50-100 MB (model structures, internal arrays)
4. **Dependencies** (numpy, scipy, plotly, langchain, sentence-transformers):
   - ~500-800 MB (includes torch for transformers)
5. **Streamlit session state & caching**: ~100-200 MB

**ESTIMATED TOTAL RAM**: **1.0-1.5 GB per user session**

### Multi-User Scenario
- **HF Spaces Free Tier**: Shared 16 GB RAM
- **Realistic concurrent users**: 8-12 users before memory pressure
- **Cold start time**: 15-30 seconds (loading models + data)

---

## ⚠️ Memory Risk Assessment

### Current Loading Strategy
- **Status**: ❌ **ALL 4 models loaded upfront** (line 298: `MODELS = load_models()`)
- **Impact**: ~50-100 MB of models sit in memory even if user only uses 1 prediction type
- **Caching**: Models use `@st.cache_resource` — loaded once, shared across sessions (good!)

### Risk Level: 🟡 **LOW-MEDIUM**

**Why LOW risk:**
- Total memory footprint (~1-1.5 GB) is **well within** HF Spaces free tier limits
- Data is small (10k rows = ~40 MB in memory)
- Models are small (largest is 1.07 MB on disk, ~25 MB in memory)
- `@st.cache_resource` means models load once, not per user

**Why MEDIUM caution:**
- HF Spaces **terminates apps that exceed memory limits**
- Multiple concurrent users could push usage to 3-5 GB
- RAG chatbot (if enabled) loads sentence-transformers + FAISS (~500 MB extra)
- Torch backend for transformers can be memory-hungry

---

## 🛠️ Optimization Options

### Option 1: Keep Current Approach (Recommended)
**Pros:**
- Simplest — no code changes
- Fast predictions (models already loaded)
- Likely fine for 5-10 concurrent users

**Cons:**
- 50-100 MB "wasted" on unused models per session
- Could hit limits with 15+ concurrent users + RAG

**Verdict**: ✅ **Start here** — monitor HF Spaces logs for OOM errors

### Option 2: Lazy-Load Models (If memory issues occur)
Change `load_models()` to load only when needed:

```python
@st.cache_resource
def load_model(model_key):
    """Load a single model on-demand."""
    model_files = {
        "gen_house": "models/xgboost_housing_final.pkl",
        "gen_land":  "models/catboost_land_model_final.pkl",
        "lph_house": "models/catboost_lalpurja_house_v2_final.pkl",
        "lph_land":  "models/catboost_lalpurja_model_final.pkl",
    }
    with open(model_files[model_key], "rb") as f:
        return pickle.load(f)

# Then call: model = load_model("gen_house") only when that tab is selected
```

**Memory savings**: ~40-75 MB per session (75% reduction if user only uses 1 model type)

**Trade-off**: Slightly slower first prediction in each section (~1-2 sec delay)

### Option 3: Disable RAG Chatbot on HF Spaces
If memory becomes tight, add to Dockerfile:

```dockerfile
ENV DISABLE_RAG=true
```

And modify app to check this env var before loading langchain/sentence-transformers.

**Memory savings**: ~500-800 MB (RAG dependencies + model embeddings)

---

## 📋 Deployment Checklist

### Before Creating HF Space:
- [x] Dockerfile.hf created with port 7860, 0.0.0.0 binding
- [x] README.md frontmatter added (title, emoji, sdk, app_port)
- [x] System dependencies identified (libgomp1 for LightGBM, build-essential)
- [x] Memory assessment complete (~1-1.5 GB per session, LOW-MEDIUM risk)

### After Creating HF Space:
- [ ] Upload Dockerfile.hf as `Dockerfile` (HF Spaces expects this name)
- [ ] Push all code (app_final.py, data/, models/, requirements.txt)
- [ ] Add secrets (optional): GITHUB_TOKEN for RAG chatbot
- [ ] Monitor first deployment logs for OOM errors
- [ ] Test all 4 prediction types (gen_house, gen_land, lph_house, lph_land)
- [ ] Test with 2-3 concurrent browser tabs to simulate load

### If Memory Issues Arise:
1. Check HF Spaces logs for "OOM" or "killed" messages
2. Implement lazy-loading (Option 2 above)
3. Consider disabling RAG chatbot (Option 3 above)
4. Upgrade to HF Spaces paid tier (if justified by traffic)

---

## 🚀 Deployment Commands

**For HF Spaces** (after creating Space):
```bash
# Clone your new HF Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/nepal-real-estate-pro
cd nepal-real-estate-pro

# Copy files from this repo
cp ../nepal-real-estate-pro/Dockerfile.hf ./Dockerfile
cp ../nepal-real-estate-pro/app_final.py .
cp ../nepal-real-estate-pro/requirements.txt .
cp ../nepal-real-estate-pro/.env.example .env
cp -r ../nepal-real-estate-pro/data ./
cp -r ../nepal-real-estate-pro/models ./

# Commit and push
git add .
git commit -m "Initial deployment: Nepal Real Estate Pro"
git push
```

**For testing locally** (Docker):
```bash
docker build -f Dockerfile.hf -t nepal-real-estate-hf .
docker run -p 7860:7860 nepal-real-estate-hf
# Open http://localhost:7860
```

---

## 📊 Expected Performance

**Cold Start**: 15-30 seconds
- Base image download: 5-10s
- Pip install: 10-15s
- Model loading: 2-5s

**Warm Requests**: <1 second for predictions

**Memory Usage**:
- Idle: ~800 MB - 1 GB
- Active (1 user): ~1-1.5 GB
- Active (5 users): ~2-3 GB
- Active (10 users): ~4-6 GB

---

## ✅ Ready to Deploy

**What's included:**
1. ✅ `Dockerfile.hf` — HF Spaces Docker config (port 7860, system deps)
2. ✅ `README.md` — Updated with HF frontmatter
3. ✅ Memory assessment — 1-1.5 GB per session, LOW-MEDIUM risk
4. ✅ Optimization plan — Lazy-loading available if needed

**What you need to do:**
1. Create HF Space at https://huggingface.co/new-space
2. Choose "Docker" as SDK
3. Clone the Space repo
4. Copy files using commands above
5. Push to HF Space repo

**Streamlit Cloud stays untouched** — this is a parallel deployment. Your existing Streamlit Cloud app continues to run independently as a backup.
