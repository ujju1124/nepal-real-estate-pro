---
title: Nepal Real Estate Pro
emoji: 🏠
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🏠 Nepal Land & House Price Prediction System

**Originally developed as a final year university project; refined here with cleaner dependencies, documentation, and known limitations.**

Kathmandu Valley real estate price prediction using scraped listing data,
4 production ML models, interactive EDA dashboards, and a RAG-powered chatbot.

---

## 📊 Project Highlights
| Metric | Value |
|--------|-------|
| Raw listings scraped | 13,114 |
| Cleaned records | 11,706 |
| Districts covered | Kathmandu, Lalitpur, Bhaktapur |
| ML models deployed | 4 (CatBoost × 3, XGBoost × 1) |
| Best R² score | 0.777 (General Housing) |
| App sections | Analytics · Inference Engine · Recommendations · RAG Chatbot |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ujju1124/nepal-real-estate-pro
cd nepal-real-estate-pro

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your tokens

# 4. Run the app
streamlit run app_final.py
```

---

## 🔑 Environment Variables
Create a `.env` file (see `.env.example`):
```
GITHUB_TOKEN=ghp_your_github_pat_here         # For RAG chatbot (GitHub Models API)
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here   # For HuggingFace embeddings
```

---

## 🧠 Models
| Pipeline | Algorithm | R² | Avg Error | Train Rows | Features |
|----------|-----------|-----|-----------|------------|----------|
| General Housing | XGBoost | 0.777 | ±18.8% | 2,005 | 24 |
| Lalpurja Land | CatBoost | 0.744 | ±19.1% | 971 | 29 |
| General Land | CatBoost | 0.744 | ±27.4% | 3,250 | 16 |
| Lalpurja Housing | CatBoost | 0.648 | ±23.7% | 1,749 | 42 |

---

## 📁 Project Structure
```
nepal-real-estate-pro/
├── app_final.py                        # Main Streamlit app
├── requirements.txt                    # Python dependencies (pinned versions)
├── readme.md
├── Dockerfile                          # Docker configuration
├── docker-compose.yml
├── .env.example                        # Environment variable template
│
├── data/                               # Cleaned datasets (8 CSV files, ~4 MB)
│   ├── housing_model_ready_after_outlier_treatment.csv
│   ├── cleaned_land_merged_final_after_eda.csv
│   ├── cleaned_lalpurja_house_v2_after_cleaning.csv
│   ├── cleaned_lalpurja_land_final_after_eda.csv
│   ├── land_features_final_modeled.csv
│   ├── housing_features_ready_after_feature_engineering.csv
│   ├── lalpurja_house_v2_features_ready.csv
│   └── lalpurja_dataset_ready_after_feature_engineering.csv
│
├── models/                             # Trained ML models (5 PKL files, ~2 MB)
│   ├── xgboost_housing_final.pkl
│   ├── catboost_land_model_final.pkl
│   ├── catboost_lalpurja_house_v2_final.pkl
│   ├── catboost_lalpurja_model_final.pkl
│   └── scaler_lalpurja_house_v2.pkl
│
├── notebooks/                          # Jupyter notebooks (organized by phase)
│   ├── 01-data-cleaning/
│   ├── 02-eda/
│   ├── 03-feature-engineering/
│   └── 04-model-building/
│
├── archive/                            # Raw and intermediate data (excluded from git)
│   ├── raw-data/
│   └── intermediate-data/
│
└── utilities/                          # Old files (excluded from git)
```

**Note on Data Files**: The `data/` and `models/` directories (~6 MB total) are committed to git for easy deployment. All files are required at runtime and are under GitHub's 100 MB limit per file. The `archive/` directory (raw/intermediate data) is excluded via `.gitignore`.

---

## 🕷️ Web Scraping
- **Hamrobazar.com** — Selenium + BeautifulSoup, 3,869 land listings
- **lalpurjanepal.com.np** — Nuxt.js SPA, dual extraction (Nuxt state + DOM fallback)
  - Speed optimised from ~30s/listing → 4-6s/listing
  - 817 Kathmandu listings in ~90 minutes

---

## 📈 App Features
1. **📊 Market Analytics** — Price distributions, neighborhood comparisons, 
   road type premiums, district breakdowns, amenity correlations
2. **🧠 Inference Engine** — Price prediction with local perturbation analysis 
   (model-agnostic explainability — shows which features drive the price)
3. **🔍 Recommendations** — Filter properties by budget, size, amenities
4. **💬 Property Assistant** — RAG chatbot (LangChain + FAISS + GPT-4o-mini)

---

## ☁️ Streamlit Cloud Deployment

This app can be deployed to [Streamlit Cloud](https://streamlit.io/cloud) for free:

1. **Fork or push this repo** to your GitHub account
2. **Go to** [share.streamlit.io](https://share.streamlit.io/)
3. **Click "New app"** and select this repository
4. **Set the main file** to `app_final.py`
5. **Add secrets** (optional, for RAG chatbot):
   - Click "Advanced settings" → "Secrets"
   - Add your `GITHUB_TOKEN` for the chatbot feature
6. **Deploy** — Streamlit Cloud will install dependencies and start the app

**Note**: The pinned `requirements.txt` ensures consistent builds. If deployment fails, check that Streamlit Cloud is using Python 3.11+.

---

## 🐳 Docker Deployment (Optional)

For containerized deployment:

```bash
# Build Docker image
docker build -t nepal-realestate .

# Run container
docker run -p 8501:8501 --env-file .env nepal-realestate

# Or use docker-compose
docker-compose up -d
```

**Benefits:**
- Consistent environment across platforms
- Easy deployment to cloud services
- Isolated dependencies

---

## ⚠️ Known Limitations
- **No automated testing**: This project prioritized model training and deployment over test coverage
- **Standard ML algorithms**: Uses well-established XGBoost and CatBoost models rather than novel architectures
- **Data staleness**: Prices based on 2025 listing data — market conditions and neighborhoods evolve
- **Limited geographic scope**: Lalpurja datasets cover only 3 districts (Kathmandu, Lalitpur, Bhaktapur)
- **Model accuracy varies**: General Land model R² = 0.61 (land valuation is inherently harder than housing)
- **Scraped data quality**: Data extracted from listing sites may have inconsistencies or missing fields

---

## 👨‍💻 Author
**Ujwal** — [GitHub](https://github.com/ujju1124)