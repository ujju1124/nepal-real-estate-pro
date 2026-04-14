# Methodology Documentation
## Nepal Land & House Price Prediction System

This document provides detailed technical methodology for the complete machine learning pipeline.

---

## 1. Data Collection Methodology

### 1.1 Web Scraping Architecture

**Platform 1: Hamrobazar.com**
- **Technology Stack:** Selenium WebDriver + BeautifulSoup4
- **Target:** Land listings in Kathmandu Valley
- **Challenges:**
  - Dynamic JavaScript content loading
  - Pagination with infinite scroll
  - Anti-bot detection mechanisms

**Implementation:**
```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup

# Initialize headless Chrome
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

# Navigate and wait for dynamic content
driver.get(url)
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, "listing-card"))
)

# Parse with BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')
```

**Platform 2: lalpurjanepal.com.np**
- **Technology Stack:** Nuxt.js state extraction + DOM fallback
- **Target:** Government-registered (Lalpurja) properties
- **Optimization:** Dual extraction strategy
  1. **Primary:** Extract from Nuxt.js `__NUXT__` state object (fast)
  2. **Fallback:** Parse DOM if state unavailable (reliable)

**Performance Improvement:**
- **Before:** 30 seconds per listing (DOM-only parsing)
- **After:** 4-6 seconds per listing (state extraction)
- **Result:** 817 Kathmandu listings scraped in ~90 minutes

**Rate Limiting Strategy:**
```python
import time
import random

# Respectful scraping with random delays
for listing in listings:
    scrape_listing(listing)
    time.sleep(random.uniform(2, 5))  # 2-5 second delay
```

### 1.2 Data Extraction Schema

**Extracted Fields (28 total):**

| Category | Fields |
|----------|--------|
| **Identifiers** | listing_id, url, scrape_date |
| **Location** | district, municipality, ward_no, neighborhood, latitude, longitude |
| **Size** | land_size_aana, land_size_sqft, built_up_sqft, road_width_feet |
| **Structure** | bedrooms, bathrooms, kitchens, living_rooms, total_floors, parking |
| **Amenities** | property_type, property_face, road_type, furnishing |
| **Price** | total_price, price_per_aana, price_per_sqft |
| **Metadata** | listing_date, seller_type, description |

### 1.3 Data Quality Checks

**Validation Rules:**
1. **Price sanity:** 1 lakh < price < 100 crore
2. **Size sanity:** 1 aana < land < 100 aana (for typical residential)
3. **Location completeness:** District + municipality required
4. **Duplicate detection:** Same URL or (location + size + price) combination

**Rejection Criteria:**
- Missing price or location
- Duplicate listings
- Commercial properties (if filtering for residential)
- Outliers beyond 3 standard deviations (flagged for manual review)

---

## 2. Data Preprocessing Pipeline

### 2.1 Data Cleaning Steps

**Step 1: Duplicate Removal**
```python
# Remove exact duplicates
df = df.drop_duplicates(subset=['url'])

# Remove near-duplicates (same property, different listings)
df = df.drop_duplicates(
    subset=['district', 'municipality', 'neighborhood', 
            'land_size_aana', 'total_price'],
    keep='first'
)
```

**Step 2: Missing Value Treatment**

| Feature | Strategy | Justification |
|---------|----------|---------------|
| `bedrooms`, `bathrooms` | Median imputation | Central tendency for count data |
| `road_width_feet` | Mode imputation | Most common road width in area |
| `property_face` | "Unknown" category | Missing = not specified |
| `built_up_sqft` | Drop row | Critical for housing models |
| `neighborhood` | "Other" category | Rare neighborhoods grouped |

**Step 3: Outlier Treatment**

**Method:** IQR (Interquartile Range) with domain knowledge
```python
def remove_outliers(df, column, lower_percentile=1, upper_percentile=99):
    lower = df[column].quantile(lower_percentile / 100)
    upper = df[column].quantile(upper_percentile / 100)
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Apply to price and size features
df = remove_outliers(df, 'total_price', 1, 99)
df = remove_outliers(df, 'land_size_sqft', 1, 99)
```

**Domain-Specific Rules:**
- Land > 50 aana → Flag as commercial/agricultural (exclude if residential focus)
- Price per sq.ft. > 50,000 → Flag as luxury (keep but note)
- Road width > 40 feet → Likely highway (premium feature)

**Step 4: Unit Standardization**

```python
# Convert aana to square feet (1 aana = 342.25 sq.ft. in Kathmandu)
df['land_size_sqft'] = df['land_size_aana'] * 342.25

# Convert lakhs to crores for consistency
df['total_price_crore'] = df['total_price_lakh'] / 100

# Standardize road width to feet
df['road_width_feet'] = df['road_width_meters'] * 3.28084
```

**Step 5: Text Normalization**

```python
import re

def normalize_neighborhood(name):
    # Remove special characters
    name = re.sub(r'[^\w\s]', '', name)
    # Convert to title case
    name = name.strip().title()
    # Fix common misspellings
    replacements = {
        'Boudha': 'Boudhanath',
        'Ktm': 'Kathmandu',
        'Lalitpur': 'Lalitpur'
    }
    return replacements.get(name, name)

df['neighborhood'] = df['neighborhood'].apply(normalize_neighborhood)
```

### 2.2 Train-Test Split Strategy

**Split Ratio:** 80% train, 20% test

**Method:** Random split (not temporal)
- **Justification:** Data collected within similar time period (2024-2025)
- **Alternative considered:** Temporal split (train on older, test on newer)
  - **Rejected:** Insufficient temporal variation in dataset

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['district']
)
```

**Stratification:** By district to ensure balanced geographic representation

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Univariate Analysis

**Price Distribution:**
- **Observation:** Right-skewed (long tail of expensive properties)
- **Action:** Log transformation for modeling

```python
import numpy as np
y = np.log1p(df['total_price'])  # log1p handles zeros gracefully
```

**Size Distribution:**
- **Land:** Right-skewed (few very large plots)
- **Built-up:** More normal distribution
- **Action:** Log transformation for land, keep built-up as-is

**Categorical Features:**
- **District:** Kathmandu (45%), Lalitpur (35%), Bhaktapur (20%)
- **Property Type:** House (60%), Land (40%)
- **Road Type:** Gravel (50%), Pitched (35%), Commercial (15%)

### 3.2 Bivariate Analysis

**Price vs. Location:**
- **Highest:** Baluwatar, Lazimpat, Sanepa (Kathmandu/Lalitpur core)
- **Lowest:** Bhaktapur outskirts, Imadol, Lubhu
- **Insight:** 3x price difference between premium and budget neighborhoods

**Price vs. Size:**
- **Correlation:** 0.68 (strong positive)
- **Non-linearity:** Diminishing returns for very large plots
- **Insight:** Price per sq.ft. decreases for larger plots

**Price vs. Road Width:**
- **Correlation:** 0.42 (moderate positive)
- **Threshold effect:** 20+ feet roads command significant premium
- **Insight:** Road accessibility is critical valuation factor

### 3.3 Multivariate Analysis

**Correlation Heatmap Insights:**
- `total_price` ↔ `land_size_sqft`: 0.68
- `total_price` ↔ `built_up_sqft`: 0.71
- `total_price` ↔ `road_width_feet`: 0.42
- `total_price` ↔ `bedrooms`: 0.55
- `land_size_sqft` ↔ `built_up_sqft`: 0.62 (multicollinearity concern)

**Action:** Create `floor_area_ratio` = built_up / land to capture relationship

---

## 4. Feature Engineering

### 4.1 Location Features

**4.1.1 Target Encoding for Neighborhoods**

**Problem:** 50+ unique neighborhoods → high cardinality
**Solution:** Encode by median price in neighborhood

```python
neighborhood_medians = df.groupby('neighborhood')['total_price'].median()
df['neighborhood_encoded'] = df['neighborhood'].map(neighborhood_medians)

# Handle unseen neighborhoods in test set
df['neighborhood_encoded'].fillna(df['total_price'].median(), inplace=True)
```

**Importance:** 10.1% (3rd most important feature)

**4.1.2 Urban Centrality Score**

**Concept:** Distance-weighted score to major city centers

```python
from geopy.distance import geodesic

city_centers = {
    'Kathmandu': (27.7172, 85.3240),  # Ratna Park
    'Lalitpur':  (27.6683, 85.3206),  # Patan Durbar
    'Bhaktapur': (27.6710, 85.4298)   # Bhaktapur Durbar
}

def urban_centrality(row):
    property_loc = (row['latitude'], row['longitude'])
    center = city_centers[row['district']]
    distance_km = geodesic(property_loc, center).km
    # Inverse distance score (closer = higher score)
    return 1 / (1 + distance_km)

df['urban_centrality'] = df.apply(urban_centrality, axis=1)
```

**4.1.3 Interaction Terms**

```python
# Neighborhood × District (captures micro-location effects)
df['neighborhood_x_district'] = (
    df['neighborhood_encoded'] * df['district'].astype('category').cat.codes
)

# Municipality × Ward (captures administrative pricing patterns)
df['municipality_x_ward'] = (
    df['municipality_encoded'] * df['ward_no']
)
```

### 4.2 Size Features

**4.2.1 Log Transformations**

```python
# Reduce skewness
df['log_land'] = np.log1p(df['land_size_sqft'])
df['log_built'] = np.log1p(df['built_up_sqft'])
```

**4.2.2 Ratio Features**

```python
# Floor Area Ratio (development intensity)
df['floor_area_ratio'] = df['built_up_sqft'] / df['land_size_sqft']

# Space efficiency
df['sqft_per_room'] = df['built_up_sqft'] / df['rooms_total']

# Bathroom luxury indicator
df['bath_per_bed'] = df['bathrooms'] / df['bedrooms']
```

### 4.3 Proximity Features

**4.3.1 Distance Calculations**

**Landmarks:**
- Ring Road (major highway)
- Boudhanath (religious/tourist site)
- Bhatbhateni (major supermarket chain)
- Airport
- Hospitals, Schools, Colleges
- Pharmacies, Police Stations
- Public Transport hubs

```python
from geopy.distance import geodesic

landmarks = {
    'ring_road':        (27.7172, 85.3240),
    'boudhanath':       (27.7215, 85.3621),
    'bhatbhateni':      (27.7089, 85.3247),
    'airport':          (27.6966, 85.3591),
    # ... more landmarks
}

for name, coords in landmarks.items():
    df[f'{name}_m'] = df.apply(
        lambda row: geodesic(
            (row['latitude'], row['longitude']), coords
        ).meters,
        axis=1
    )
```

**4.3.2 Amenity Access Score**

**Composite metric:** Weighted average of proximity to essential services

```python
# Normalize distances to 0-1 scale (closer = higher score)
def normalize_distance(dist, max_dist=5000):
    return 1 - min(dist / max_dist, 1)

amenity_weights = {
    'hospital_m': 0.25,
    'school_m': 0.20,
    'public_transport_m': 0.20,
    'pharmacy_m': 0.15,
    'bhatbhateni_m': 0.10,
    'police_station_m': 0.10
}

df['amenity_access_score'] = sum(
    normalize_distance(df[col]) * weight
    for col, weight in amenity_weights.items()
)
```

### 4.4 Structural Features

**4.4.1 Interaction Terms**

```python
# Most important feature (24.4% importance)
df['floors_x_land'] = df['total_floors'] * df['log_land']
```

**Interpretation:** Captures development intensity - tall buildings on large plots command premium

**4.4.2 Luxury Score**

**Composite metric:** Weighted sum of luxury indicators

```python
luxury_features = {
    'parking': 2,           # Parking availability
    'furnishing': 3,        # Furnished property
    'bathrooms': 1,         # More bathrooms = luxury
    'total_floors': 1,      # Multi-story = luxury
}

df['luxury_score'] = sum(
    df[feature] * weight
    for feature, weight in luxury_features.items()
)
```

**4.4.3 Age-Condition Score**

```python
# Newer properties command premium
df['age_condition_score'] = np.where(
    df['house_age'] < 5, 1.0,      # New
    np.where(df['house_age'] < 15, 0.7,  # Moderate
             0.4)                   # Old
)
```

### 4.5 Road Features

**4.5.1 Commercial Road Premium**

```python
# Binary indicator for commercial road access
df['comm_road_premium'] = (df['road_type'] == 'Commercial').astype(int)
```

**4.5.2 Parking Premium**

```python
# Parking availability indicator
df['parking_premium'] = (df['parking'] > 0).astype(int)
```

### 4.6 Final Feature Set

**Lalpurja Housing (42 features):**
- Location: 5 features (neighborhood_encoded, municipality_encoded, urban_centrality, etc.)
- Size: 6 features (log_land, log_built, floor_area_ratio, etc.)
- Proximity: 11 features (distances to landmarks)
- Structural: 12 features (bedrooms, bathrooms, floors_x_land, luxury_score, etc.)
- Road: 3 features (road_width_feet, road_type, comm_road_premium)
- Amenity: 1 feature (amenity_access_score)
- Administrative: 4 features (district, ward_no, municipality_x_ward, etc.)

---

## 5. Model Development

### 5.1 Algorithm Selection

**Candidates Evaluated (9 algorithms):**

| Category | Algorithms | Rationale |
|----------|-----------|-----------|
| **Linear** | Ridge, Lasso | Baseline, interpretable |
| **Tree Ensemble** | Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost | Handle non-linearity, feature interactions |
| **Instance-based** | K-Nearest Neighbors | Captures local patterns |
| **Kernel** | Support Vector Regression | Non-linear relationships |

### 5.2 Hyperparameter Tuning

**Method:** RandomizedSearchCV
- **Iterations:** 50 random combinations
- **Cross-validation:** 5-fold
- **Scoring:** R² (coefficient of determination)
- **Parallelization:** n_jobs=-1 (use all CPU cores)

**Search Spaces:**

**CatBoost:**
```python
{
    'iterations':    [500, 1000, 1500, 2000],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'depth':         [4, 6, 8],
    'l2_leaf_reg':   [1, 3, 5, 9]
}
```

**Best Parameters (Lalpurja Housing):**
- iterations: 1000
- learning_rate: 0.05
- depth: 6
- l2_leaf_reg: 5

**XGBoost:**
```python
{
    'n_estimators':     [300, 500, 800],
    'learning_rate':    [0.01, 0.05, 0.1],
    'max_depth':        [3, 5, 7],
    'subsample':        [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}
```

**Best Parameters (General Housing):**
- n_estimators: 500
- learning_rate: 0.05
- max_depth: 3
- subsample: 0.8
- colsample_bytree: 0.8
- min_child_weight: 5

### 5.3 Model Training Process

**Step 1: Baseline Models**
```python
# Train all 9 algorithms with default parameters
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: R² = {r2:.4f}")
```

**Step 2: Select Top 3**
- Based on R² score
- Typically: CatBoost, XGBoost, Gradient Boosting

**Step 3: Hyperparameter Tuning**
```python
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=50,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
```

**Step 4: Final Model Selection**
- Pick model with highest test R²
- Save model and scaler

```python
import joblib
joblib.dump(best_model, 'model_final.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

### 5.4 Feature Scaling

**For Linear Models Only:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Tree-based models:** No scaling required (scale-invariant)

---

## 6. Model Evaluation

### 6.1 Evaluation Metrics

**Primary Metric: R² (Coefficient of Determination)**
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
```

**Interpretation:**
- R² = 0.77 → Model explains 77% of price variance
- R² = 1.0 → Perfect predictions
- R² = 0.0 → Model no better than mean baseline

**Secondary Metrics:**

**MAE (Mean Absolute Error) in log-space:**
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
error_pct = (np.exp(mae) - 1) * 100  # Convert to percentage
```

**RMSE (Root Mean Squared Error):**
```python
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

**Penalizes large errors more than MAE**

### 6.2 Cross-Validation

**5-Fold Cross-Validation during tuning:**
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model, X_train, y_train, 
    cv=5, scoring='r2'
)
print(f"CV R² Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

**Purpose:** Ensure model generalizes across different data splits

### 6.3 Feature Importance Analysis

**CatBoost Built-in Importance:**
```python
feature_importance = model.get_feature_importance()
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)
```

**Top Features (Lalpurja Housing):**
1. floors_x_land: 24.4%
2. log_land: 10.3%
3. neighborhood_encoded: 10.1%

---

## 7. Application Development

### 7.1 Streamlit Architecture

**File:** `app_final.py` (1,743 lines)

**Structure:**
```python
import streamlit as st
import pandas as pd
import joblib

# Load models once (cached)
@st.cache_resource
def load_models():
    return {
        'general_housing': joblib.load('xgboost_housing_final.pkl'),
        'general_land': joblib.load('catboost_land_model_final.pkl'),
        'lalpurja_housing': joblib.load('catboost_lalpurja_house_v2_final.pkl'),
        'lalpurja_land': joblib.load('catboost_lalpurja_model_final.pkl')
    }

models = load_models()

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", 
    ["Analytics", "Predict", "Recommendations", "Chatbot"])

if page == "Analytics":
    show_analytics()
elif page == "Predict":
    show_prediction_engine()
# ... etc
```

### 7.2 Prediction Engine

**User Input Form:**
```python
def get_user_input():
    col1, col2 = st.columns(2)
    with col1:
        district = st.selectbox("District", ["Kathmandu", "Lalitpur", "Bhaktapur"])
        land_aana = st.number_input("Land Size (aana)", 1, 50, 5)
        bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    with col2:
        neighborhood = st.selectbox("Neighborhood", neighborhoods)
        built_sqft = st.number_input("Built-up (sq.ft.)", 500, 10000, 2000)
        bathrooms = st.number_input("Bathrooms", 1, 8, 2)
    return {
        'district': district,
        'land_aana': land_aana,
        # ... etc
    }
```

**Feature Engineering for Prediction:**
```python
def prepare_features(user_input):
    # Apply same transformations as training
    features = {}
    features['log_land'] = np.log1p(user_input['land_aana'] * 342.25)
    features['neighborhood_encoded'] = neighborhood_map[user_input['neighborhood']]
    features['floors_x_land'] = user_input['total_floors'] * features['log_land']
    # ... all 42 features
    return pd.DataFrame([features])

X_pred = prepare_features(user_input)
prediction = model.predict(X_pred)
price = np.expm1(prediction[0])  # Inverse log transform
```

**Confidence Intervals:**
```python
# Bootstrap confidence intervals
predictions = []
for _ in range(100):
    # Add small random noise to features
    X_noisy = X_pred + np.random.normal(0, 0.01, X_pred.shape)
    pred = model.predict(X_noisy)
    predictions.append(np.expm1(pred[0]))

lower = np.percentile(predictions, 5)
upper = np.percentile(predictions, 95)

st.write(f"Predicted Price: {price/1e7:.2f} crore")
st.write(f"90% Confidence Interval: {lower/1e7:.2f} - {upper/1e7:.2f} crore")
```

### 7.3 Perturbation Analysis (Explainability)

**Local Feature Importance:**
```python
def perturbation_analysis(model, X_base, feature_names):
    base_pred = model.predict(X_base)[0]
    importances = {}
    
    for i, feature in enumerate(feature_names):
        X_perturbed = X_base.copy()
        # Increase feature by 10%
        X_perturbed[0, i] *= 1.1
        perturbed_pred = model.predict(X_perturbed)[0]
        
        # Impact = change in prediction
        impact = (perturbed_pred - base_pred) / base_pred * 100
        importances[feature] = impact
    
    return sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)

# Display top 10 features
top_features = perturbation_analysis(model, X_pred, feature_names)[:10]
for feature, impact in top_features:
    st.write(f"{feature}: {impact:+.2f}% price change")
```

### 7.4 RAG Chatbot

**Architecture:**
```
User Query → Embedding → FAISS Search → Context Retrieval → LLM → Response
```

**Implementation:**
```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Load embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store from property listings
texts = df.apply(lambda row: 
    f"Property in {row['neighborhood']}, {row['district']}. "
    f"Land: {row['land_aana']} aana, Price: {row['total_price']/1e7:.2f} crore. "
    f"{row['bedrooms']} bed, {row['bathrooms']} bath.",
    axis=1
).tolist()

vectorstore = FAISS.from_texts(texts, embeddings)

# Create QA chain
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("GITHUB_TOKEN"))
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Query
response = qa_chain({"query": "Show me houses under 1 crore in Lalitpur"})
st.write(response['result'])
```

---

## 8. Deployment Considerations

### 8.1 System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB disk space

**Recommended:**
- Python 3.11+
- 8GB RAM
- 5GB disk space (for all models + data)

### 8.2 Dependencies

See `requirements.txt` for complete list. Key dependencies:
- streamlit >= 1.32.0
- scikit-learn >= 1.4.0
- xgboost >= 2.0.0
- catboost >= 1.2.0
- langchain-core >= 0.1.0
- faiss-cpu >= 1.7.4

### 8.3 Environment Variables

Required in `.env` file:
```
GITHUB_TOKEN=ghp_xxxxx          # For RAG chatbot (GitHub Models API)
HUGGINGFACEHUB_API_TOKEN=hf_xxx # For embeddings
```

### 8.4 Performance Optimization

**Model Loading:**
```python
@st.cache_resource
def load_models():
    # Load once, cache in memory
    return joblib.load('model.pkl')
```

**Data Loading:**
```python
@st.cache_data
def load_data():
    # Cache dataframes
    return pd.read_csv('data.csv')
```

**Prediction Caching:**
```python
@st.cache_data
def predict_price(_model, features):
    # Cache predictions for same inputs
    return _model.predict(features)
```

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

**Data:**
- Listing prices (not actual transaction prices)
- Limited temporal coverage (2024-2025 only)
- Geographic scope (Kathmandu Valley only)

**Models:**
- Static predictions (no market trend forecasting)
- Limited handling of extreme luxury properties
- No incorporation of macroeconomic factors

**System:**
- Local deployment only (not web-accessible)
- Manual retraining required for updates
- Limited Nepali language support

### 9.2 Proposed Improvements

**Short-term:**
1. Add cross-validation results to notebooks
2. Create error analysis visualizations
3. Implement automated testing
4. Add deployment documentation

**Medium-term:**
1. Collect time-series data for trend analysis
2. Incorporate property images (computer vision)
3. Expand to other cities (Pokhara, Chitwan)
4. Deploy as web service (AWS/Azure)

**Long-term:**
1. Partner with agencies for transaction data
2. Implement reinforcement learning for pricing optimization
3. Build mobile application
4. Add Nepali language interface

---

**End of Methodology Document**

*This document provides complete technical details for reproducing the Nepal Land & House Price Prediction System. For high-level overview, see [PROJECT_REPORT.md](PROJECT_REPORT.md).*
