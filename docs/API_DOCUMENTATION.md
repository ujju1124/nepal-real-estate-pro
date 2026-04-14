# API Documentation
## Nepal Land & House Price Prediction System

This document provides detailed documentation for all functions in `app_final.py`.

---

## Table of Contents

1. [Model Loading Functions](#1-model-loading-functions)
2. [Data Loading Functions](#2-data-loading-functions)
3. [Feature Engineering Functions](#3-feature-engineering-functions)
4. [Prediction Functions](#4-prediction-functions)
5. [Analytics Functions](#5-analytics-functions)
6. [Utility Functions](#6-utility-functions)
7. [RAG Chatbot Functions](#7-rag-chatbot-functions)

---

## 1. Model Loading Functions

### `load_models()`

**Purpose:** Load all four trained ML models into memory (cached).

**Returns:**
- `dict`: Dictionary containing four model objects
  - `'general_housing'`: XGBoost model for general housing
  - `'general_land'`: CatBoost model for general land
  - `'lalpurja_housing'`: CatBoost model for Lalpurja housing
  - `'lalpurja_land'`: CatBoost model for Lalpurja land

**Caching:** Uses `@st.cache_resource` to load models once per session

**Example:**
```python
models = load_models()
prediction = models['general_housing'].predict(X)
```

**File Paths:**
- `xgboost_housing_final.pkl`
- `catboost_land_model_final.pkl`
- `catboost_lalpurja_house_v2_final.pkl`
- `catboost_lalpurja_model_final.pkl`

---

### `load_scaler(model_type)`

**Purpose:** Load StandardScaler for models that require feature scaling.

**Parameters:**
- `model_type` (str): One of `['general_housing', 'general_land', 'lalpurja_housing', 'lalpurja_land']`

**Returns:**
- `StandardScaler` object or `None` if model doesn't require scaling

**Example:**
```python
scaler = load_scaler('lalpurja_housing')
X_scaled = scaler.transform(X)
```

**Note:** Only Lalpurja Housing model uses scaler (`scaler_lalpurja_house_v2.pkl`)

---

## 2. Data Loading Functions

### `load_dataset(dataset_name)`

**Purpose:** Load cleaned property datasets for analytics and recommendations.

**Parameters:**
- `dataset_name` (str): One of:
  - `'general_housing'`: General housing dataset
  - `'general_land'`: General land dataset
  - `'lalpurja_housing'`: Lalpurja housing dataset
  - `'lalpurja_land'`: Lalpurja land dataset

**Returns:**
- `pd.DataFrame`: Cleaned dataset with all features

**Caching:** Uses `@st.cache_data` for performance

**Example:**
```python
df = load_dataset('general_housing')
print(f"Loaded {len(df)} properties")
```

**File Paths:**
- `housing_model_ready_after_outlier_treatment.csv`
- `cleaned_land_merged_final_after_eda.csv`
- `cleaned_lalpurja_house_v2_after_cleaning.csv`
- `cleaned_lalpurja_land_final_after_eda.csv`

---

## 3. Feature Engineering Functions

### `engineer_features(user_input, model_type)`

**Purpose:** Transform raw user input into model-ready features.

**Parameters:**
- `user_input` (dict): Dictionary containing property characteristics
  - Required keys depend on `model_type`
  - Example: `{'district': 'Kathmandu', 'land_aana': 5, 'bedrooms': 3, ...}`
- `model_type` (str): Target model type

**Returns:**
- `pd.DataFrame`: Single-row dataframe with engineered features (42 columns for Lalpurja Housing)

**Feature Engineering Steps:**
1. Log transformations (`log_land`, `log_built`)
2. Target encoding (`neighborhood_encoded`, `municipality_encoded`)
3. Interaction terms (`floors_x_land`, `neighborhood_x_district`)
4. Proximity calculations (distances to landmarks)
5. Composite scores (`luxury_score`, `amenity_access_score`)

**Example:**
```python
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

X = engineer_features(user_input, 'lalpurja_housing')
# Returns DataFrame with 42 engineered features
```

**Engineered Features (Lalpurja Housing - 42 total):**

| Category | Features | Count |
|----------|----------|-------|
| Location | neighborhood_encoded, municipality_encoded, urban_centrality, neighborhood_x_district, municipality_x_ward | 5 |
| Size | log_land, log_built, floor_area_ratio, sqft_per_room, house_size_score | 5 |
| Proximity | ring_road_m, boudhanath_m, bhatbhateni_m, airport_m, hospital_m, school_m, college_m, pharmacy_m, police_station_m, public_transport_m | 10 |
| Structural | bedrooms, bathrooms, kitchens, living_rooms, total_floors, parking, rooms_total, bath_per_bed, floors_x_land, luxury_score, age_condition_score | 11 |
| Road | road_width_feet, road_type, comm_road_premium, parking_premium | 4 |
| Amenity | amenity_access_score | 1 |
| Administrative | district, ward_no, municipality, property_face, property_type, furnishing | 6 |

---

### `calculate_proximity(lat, lon, landmark)`

**Purpose:** Calculate distance from property to landmark using Haversine formula.

**Parameters:**
- `lat` (float): Property latitude
- `lon` (float): Property longitude
- `landmark` (str): Landmark name (e.g., 'ring_road', 'boudhanath')

**Returns:**
- `float`: Distance in meters

**Example:**
```python
distance = calculate_proximity(27.7172, 85.3240, 'ring_road')
print(f"Distance to Ring Road: {distance:.0f} meters")
```

**Available Landmarks:**
- `ring_road`: Major highway
- `boudhanath`: Religious/tourist site
- `bhatbhateni`: Supermarket chain
- `airport`: Tribhuvan International Airport
- `hospital`: Nearest major hospital
- `school`: Nearest school
- `college`: Nearest college
- `pharmacy`: Nearest pharmacy
- `police_station`: Nearest police station
- `public_transport`: Nearest bus stop/station

---

### `encode_neighborhood(neighborhood, district, model_type)`

**Purpose:** Target encode neighborhood based on median price.

**Parameters:**
- `neighborhood` (str): Neighborhood name
- `district` (str): District name
- `model_type` (str): Model type for lookup table

**Returns:**
- `float`: Encoded value (median price in neighborhood)

**Example:**
```python
encoded = encode_neighborhood('Baluwatar', 'Kathmandu', 'general_housing')
# Returns median price for Baluwatar properties
```

**Fallback:** If neighborhood unseen, returns district median

---

## 4. Prediction Functions

### `predict_general_housing(user_input)`

**Purpose:** Predict price for general housing properties.

**Parameters:**
- `user_input` (dict): Property characteristics (see `engineer_features` for required keys)

**Returns:**
- `dict`: Prediction results
  - `'price'` (float): Predicted price in NPR
  - `'price_crore'` (float): Predicted price in crores
  - `'confidence_lower'` (float): Lower bound of 90% confidence interval
  - `'confidence_upper'` (float): Upper bound of 90% confidence interval
  - `'model'` (str): Model name ('XGBoost')
  - `'r2'` (float): Model R² score (0.777)

**Example:**
```python
result = predict_general_housing(user_input)
print(f"Predicted Price: {result['price_crore']:.2f} crore")
print(f"90% CI: {result['confidence_lower']:.2f} - {result['confidence_upper']:.2f} crore")
```

**Model:** XGBoost (R² = 0.777, Error = ±18.8%)

---

### `predict_general_land(user_input)`

**Purpose:** Predict price for general land plots.

**Parameters:**
- `user_input` (dict): Land characteristics
  - Required: `district`, `neighborhood`, `land_aana`, `road_width_feet`, `road_type`

**Returns:**
- `dict`: Prediction results (same structure as `predict_general_housing`)

**Example:**
```python
land_input = {
    'district': 'Lalitpur',
    'neighborhood': 'Imadol',
    'land_aana': 10,
    'road_width_feet': 15,
    'road_type': 'Gravel'
}
result = predict_general_land(land_input)
```

**Model:** CatBoost (R² = 0.612, Error = ±27.4%)

---

### `predict_lalpurja_housing(user_input)`

**Purpose:** Predict price for Lalpurja-registered housing properties.

**Parameters:**
- `user_input` (dict): Property characteristics (42 features required)

**Returns:**
- `dict`: Prediction results

**Example:**
```python
result = predict_lalpurja_housing(user_input)
```

**Model:** CatBoost (R² = 0.648, Error = ±23.7%)

**Note:** Requires scaler transformation before prediction

---

### `predict_lalpurja_land(user_input)`

**Purpose:** Predict price for Lalpurja-registered land plots.

**Parameters:**
- `user_input` (dict): Land characteristics

**Returns:**
- `dict`: Prediction results

**Example:**
```python
result = predict_lalpurja_land(user_input)
```

**Model:** CatBoost (R² = 0.744, Error = ±19.1%)

---

### `calculate_confidence_interval(model, X, n_iterations=100)`

**Purpose:** Calculate prediction confidence interval using bootstrap method.

**Parameters:**
- `model`: Trained ML model
- `X` (pd.DataFrame): Feature matrix (single row)
- `n_iterations` (int): Number of bootstrap samples (default: 100)

**Returns:**
- `tuple`: (lower_bound, upper_bound) for 90% confidence interval

**Method:**
1. Add small random noise to features (Gaussian, σ=0.01)
2. Generate 100 predictions
3. Calculate 5th and 95th percentiles

**Example:**
```python
lower, upper = calculate_confidence_interval(model, X)
print(f"90% CI: {lower:.2f} - {upper:.2f} crore")
```

---

## 5. Analytics Functions

### `plot_price_distribution(df, property_type)`

**Purpose:** Create histogram of price distribution.

**Parameters:**
- `df` (pd.DataFrame): Property dataset
- `property_type` (str): Property type for title

**Returns:**
- `plotly.graph_objects.Figure`: Interactive histogram

**Example:**
```python
fig = plot_price_distribution(df, 'General Housing')
st.plotly_chart(fig)
```

**Features:**
- Bins: 50
- Color: Blue gradient
- Hover: Count and price range
- X-axis: Price in crores
- Y-axis: Frequency

---

### `plot_neighborhood_comparison(df, district=None)`

**Purpose:** Create box plot comparing prices across neighborhoods.

**Parameters:**
- `df` (pd.DataFrame): Property dataset
- `district` (str, optional): Filter by district

**Returns:**
- `plotly.graph_objects.Figure`: Interactive box plot

**Example:**
```python
fig = plot_neighborhood_comparison(df, district='Kathmandu')
st.plotly_chart(fig)
```

**Features:**
- Shows median, quartiles, outliers
- Sorted by median price (descending)
- Top 15 neighborhoods displayed
- Hover: Min, Q1, Median, Q3, Max

---

### `plot_road_premium(df)`

**Purpose:** Analyze price premium by road type.

**Parameters:**
- `df` (pd.DataFrame): Property dataset

**Returns:**
- `plotly.graph_objects.Figure`: Bar chart

**Example:**
```python
fig = plot_road_premium(df)
st.plotly_chart(fig)
```

**Calculation:**
- Baseline: Gravel road (100%)
- Premium: (Median price / Gravel median - 1) × 100%

---

### `plot_district_breakdown(df)`

**Purpose:** Create pie chart of property distribution by district.

**Parameters:**
- `df` (pd.DataFrame): Property dataset

**Returns:**
- `plotly.graph_objects.Figure`: Pie chart

**Example:**
```python
fig = plot_district_breakdown(df)
st.plotly_chart(fig)
```

---

### `plot_amenity_correlation(df)`

**Purpose:** Create heatmap of feature correlations with price.

**Parameters:**
- `df` (pd.DataFrame): Property dataset

**Returns:**
- `plotly.graph_objects.Figure`: Correlation heatmap

**Example:**
```python
fig = plot_amenity_correlation(df)
st.plotly_chart(fig)
```

**Features:**
- Top 15 correlated features
- Color scale: Red (negative) to Green (positive)
- Hover: Correlation coefficient

---

## 6. Utility Functions

### `perturbation_analysis(model, X_base, feature_names, top_n=10)`

**Purpose:** Perform local feature importance analysis using perturbation method.

**Parameters:**
- `model`: Trained ML model
- `X_base` (np.array): Base feature vector (single prediction)
- `feature_names` (list): List of feature names
- `top_n` (int): Number of top features to return (default: 10)

**Returns:**
- `list`: List of tuples `(feature_name, impact_percentage)`

**Method:**
1. Get baseline prediction
2. For each feature:
   - Increase by 10%
   - Calculate new prediction
   - Compute impact = (new - baseline) / baseline × 100%
3. Sort by absolute impact

**Example:**
```python
impacts = perturbation_analysis(model, X.values[0], X.columns)
for feature, impact in impacts:
    st.write(f"{feature}: {impact:+.2f}% price change")
```

**Output Example:**
```
floors_x_land: +8.5% price change
log_land: +6.2% price change
neighborhood_encoded: +4.1% price change
```

---

### `format_price(price)`

**Purpose:** Format price in Nepali currency notation.

**Parameters:**
- `price` (float): Price in NPR

**Returns:**
- `str`: Formatted price string

**Example:**
```python
formatted = format_price(85000000)
# Returns: "8.50 crore"

formatted = format_price(2500000)
# Returns: "25.00 lakh"
```

**Rules:**
- >= 1 crore: Display in crores (1 crore = 10 million)
- < 1 crore: Display in lakhs (1 lakh = 100,000)

---

### `validate_input(user_input, model_type)`

**Purpose:** Validate user input before prediction.

**Parameters:**
- `user_input` (dict): User-provided property characteristics
- `model_type` (str): Target model type

**Returns:**
- `tuple`: (is_valid, error_message)
  - `is_valid` (bool): True if input valid
  - `error_message` (str): Error description if invalid, else None

**Validation Rules:**
- Land size: 1-50 aana
- Built-up area: 500-10,000 sq.ft.
- Bedrooms: 1-10
- Bathrooms: 1-8
- Road width: 5-40 feet
- House age: 0-100 years

**Example:**
```python
is_valid, error = validate_input(user_input, 'general_housing')
if not is_valid:
    st.error(error)
else:
    result = predict_general_housing(user_input)
```

---

### `get_similar_properties(df, user_input, n=5)`

**Purpose:** Find similar properties in dataset based on user input.

**Parameters:**
- `df` (pd.DataFrame): Property dataset
- `user_input` (dict): Target property characteristics
- `n` (int): Number of similar properties to return (default: 5)

**Returns:**
- `pd.DataFrame`: Top n similar properties

**Similarity Metric:**
- Euclidean distance in normalized feature space
- Features: land_size, built_up, bedrooms, bathrooms, road_width
- Weights: Equal (can be customized)

**Example:**
```python
similar = get_similar_properties(df, user_input, n=5)
st.dataframe(similar[['neighborhood', 'total_price', 'land_aana', 'bedrooms']])
```

---

## 7. RAG Chatbot Functions

### `initialize_rag_system(df)`

**Purpose:** Initialize RAG (Retrieval-Augmented Generation) chatbot system.

**Parameters:**
- `df` (pd.DataFrame): Property dataset to index

**Returns:**
- `RetrievalQA`: LangChain QA chain object

**Components:**
1. **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
2. **Vector Store:** FAISS (Facebook AI Similarity Search)
3. **LLM:** GPT-4o-mini via GitHub Models API
4. **Retriever:** Top-5 similar documents

**Example:**
```python
qa_chain = initialize_rag_system(df)
response = qa_chain({"query": "Show houses under 1 crore in Lalitpur"})
st.write(response['result'])
```

**Document Format:**
```
Property in {neighborhood}, {district}.
Land: {land_aana} aana, Price: {price} crore.
{bedrooms} bed, {bathrooms} bath, {total_floors} floors.
Road: {road_width} feet, {road_type}.
```

---

### `query_chatbot(qa_chain, user_query)`

**Purpose:** Query RAG chatbot with natural language question.

**Parameters:**
- `qa_chain` (RetrievalQA): Initialized QA chain
- `user_query` (str): User's natural language question

**Returns:**
- `dict`: Response dictionary
  - `'result'` (str): Generated answer
  - `'source_documents'` (list): Retrieved context documents

**Example:**
```python
response = query_chatbot(qa_chain, "What's the average price in Baluwatar?")
st.write(response['result'])

# Show sources
st.write("Sources:")
for doc in response['source_documents']:
    st.write(f"- {doc.page_content}")
```

**Supported Query Types:**
1. **Factual:** "Average price in Lalitpur?"
2. **Comparison:** "Compare Kathmandu vs Bhaktapur prices"
3. **Recommendation:** "Show houses under 1 crore with 3 bedrooms"
4. **Explanation:** "Why are Baluwatar properties expensive?"

---

## Error Handling

All functions include error handling:

```python
try:
    result = predict_general_housing(user_input)
except ValueError as e:
    st.error(f"Invalid input: {e}")
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
except Exception as e:
    st.error(f"Prediction error: {e}")
```

---

## Performance Considerations

**Caching:**
- Models: `@st.cache_resource` (load once per session)
- Data: `@st.cache_data` (cache dataframes)
- Predictions: Cache for identical inputs

**Memory Usage:**
- 4 models: ~200MB
- Datasets: ~50MB
- FAISS index: ~100MB
- Total: ~450MB

**Latency:**
- Prediction: < 200ms
- Analytics: 1-2s
- Chatbot: 2-4s (depends on LLM API)

---

**End of API Documentation**

*For usage examples, see `app_final.py`. For methodology, see [METHODOLOGY.md](METHODOLOGY.md).*
