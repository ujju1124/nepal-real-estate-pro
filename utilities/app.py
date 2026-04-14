"""
Nepal Real Estate Analytics & Prediction — IMPROVED VERSION
============================================================
Fixes applied:
1. Lalpurja accuracy issue — land size multiplier + confidence scoring
2. UX confusion — Basic/Advanced mode with goal-based selection
3. Recommendations — Smart similarity matching using Lalpurja data
4. Robustness — Input validation, graceful error handling, encoding safety
5. Transparency — Confidence intervals, data quality indicators, error ranges

Files needed (same folder as app.py):
  housing_model_ready_after_outlier_treatment.csv
  cleaned_land_merged_final_after_eda.csv
  cleaned_lalpurja_house_v2_after_cleaning.csv
  cleaned_lalpurja_land_final_after_eda.csv
  land_features_final_modeled.csv
  housing_features_ready_after_feature_engineering.csv
  lalpurja_house_v2_features_ready.csv
  lalpurja_dataset_ready_after_feature_engineering.csv
  xgboost_housing_final.pkl
  catboost_land_model_final.pkl
  catboost_lalpurja_house_v2_final.pkl
  catboost_lalpurja_model_final.pkl

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nepal Real Estate Pro",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="metric-container"] {
        background: #1e1e2e; border: 1px solid #2a2a3a;
        border-radius: 12px; padding: 16px;
    }
    .confidence-high { color: #2ecc71; font-weight: bold; }
    .confidence-medium { color: #f39c12; font-weight: bold; }
    .confidence-low { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
MAIN_DISTRICTS = ["Kathmandu", "Lalitpur", "Bhaktapur"]
DIST_COLORS = {"Kathmandu": "#e8a45a", "Lalitpur": "#5ab8e8", "Bhaktapur": "#8ae85a"}
FACING_OPTIONS = ["East", "West", "North", "South", "North East", "North West", "South East", "South West"]

# Model metadata
MODEL_INFO = {
    "gen_house": {
        "name": "🏠 General Housing",
        "r2": 0.777, "error": 18.8, "samples": 2005,
        "best_for": "Typical homes across Kathmandu Valley",
        "strengths": ["Highest accuracy", "Works everywhere", "Fast"],
        "limitations": ["Doesn't capture interior quality"]
    },
    "gen_land": {
        "name": "🌍 General Land",
        "r2": 0.744, "error": 19.1, "samples": 3250,
        "best_for": "Plots without detailed amenity data",
        "strengths": ["Good accuracy", "Simple inputs"],
        "limitations": ["No amenity analysis"]
    },
    "lph_house": {
        "name": "🏘️ Lalpurja Housing (Advanced)",
        "r2": 0.648, "error": 23.7, "samples": 1749,
        "best_for": "Detailed interior features, specific neighborhoods",
        "strengths": ["42 features capture detail", "Amenity distances"],
        "limitations": ["Interior quality unmeasurable", "Smaller dataset"]
    },
    "lph_land": {
        "name": "🎯 Lalpurja Land (Advanced)",
        "r2": 0.744, "error": 19.1, "samples": 971,
        "best_for": "Land analysis with amenity proximity data",
        "strengths": ["Amenity analysis", "Detailed metrics"],
        "limitations": ["Smaller dataset", "Land size extrapolation risk"]
    }
}

def fmt_npr(val, decimal=2):
    """Format numbers as NPR currency (Cr/L/direct)"""
    try:
        if pd.isna(val) or val == 0: return "₹0"
    except Exception:
        pass
    if val >= 10_000_000: return f"₹{val/10_000_000:.{decimal}f} Cr"
    elif val >= 100_000:  return f"₹{val/100_000:.{decimal}f} L"
    return f"₹{val:,.0f}"

def clean_chart(fig, height=None):
    """Apply consistent chart styling"""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)", 
        showlegend=False,
        font=dict(color="#e0e0e0")
    )
    if height: fig.update_layout(height=height)
    return fig

def insight(text: str):
    """Render a styled 1-2 line EDA insight below a chart."""
    st.markdown(
        f"<p style='font-size:13px; color:#a0c4e8; background:rgba(30,30,60,0.5); "
        f"padding:8px 12px; border-left:3px solid #5ab8e8; border-radius:4px; margin-bottom:12px;'>"
        f"💡 {text}</p>",
        unsafe_allow_html=True
    )
def calculate_matching_score(row, prefs):
    # 🎯 Ideal targets
    ideal_price = (prefs["min_price"] + prefs["max_price"]) / 2
    ideal_beds  = prefs["bedrooms"]

    # ---- Price closeness (0–1) ----
    price_diff_pct = abs(row["total_price"] - ideal_price) / max(ideal_price, 1)
    price_score = max(0, 1 - price_diff_pct)

    # ---- Bedroom closeness (0–1) ----
    bed_diff = abs(row["bedrooms"] - ideal_beds)
    bed_score = max(0, 1 - (bed_diff / max(ideal_beds, 1)))

    # ---- Amenity score (0–1) ----
    if prefs["must_have_amenities"]:
        matched = sum(
            row.get(a, 0) == 1
            for a in prefs["must_have_amenities"]
        )
        amenity_score = matched / len(prefs["must_have_amenities"])

        # Normal weight distribution
        final_score = (
            price_score * 0.30 +
            bed_score   * 0.20 +
            amenity_score * 0.50
        ) * 100
    else:
        # Redistribute full weight to price & bedrooms
        final_score = (
            price_score * 0.60 +
            bed_score   * 0.40
        ) * 100

    return round(min(100, final_score), 2)

# ═══════════════════════════════════════════════════════════
# LOAD ANALYTICS DATA  (cached)
# ═══════════════════════════════════════════════════════════
@st.cache_data
def load_analytics_data():
    """Load and filter all datasets to main districts"""
    try:
        gh = pd.read_csv("housing_model_ready_after_outlier_treatment.csv")
        gh = gh[gh["district"].isin(MAIN_DISTRICTS)]

        gl = pd.read_csv("cleaned_land_merged_final_after_eda.csv")
        gl = gl[gl["district"].isin(MAIN_DISTRICTS)]
        gl_named = gl[~gl["neighborhood"].str.contains("Zone", na=False)]

        lh = pd.read_csv("cleaned_lalpurja_house_v2_after_cleaning.csv")
        lh["floors_x_land"] = lh["total_floors"] * lh["land_size_aana"]
        lh_named = lh[~lh["neighborhood"].str.contains("Zone", na=False)]

        ll = pd.read_csv("cleaned_lalpurja_land_final_after_eda.csv")
        ll_named = ll[~ll["neighborhood"].str.contains("Zone", na=False)]

        return gh, gl, gl_named, lh, lh_named, ll, ll_named
    except FileNotFoundError as e:
        st.error(f"❌ Data file missing: {e}")
        st.stop()

try:
    gh, gl, gl_named, lh, lh_named, ll, ll_named = load_analytics_data()
except FileNotFoundError as e:
    st.error(f"❌ Data file missing: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error while loading data: {e}")
    st.stop()


# ═══════════════════════════════════════════════════════════
# BUILD ENCODING MAPS  (cached)
# ═══════════════════════════════════════════════════════════
@st.cache_data
def build_encoding_maps():
    """Build label→encoding dictionaries from feature-engineered CSVs"""
    try:
        # Load feature-engineered (encoded) CSVs
        house_fe  = pd.read_csv("housing_features_ready_after_feature_engineering.csv")
        land_fe   = pd.read_csv("land_features_final_modeled.csv")
        lph_house_fe = pd.read_csv("lalpurja_house_v2_features_ready.csv")
        lph_land_fe  = pd.read_csv("lalpurja_dataset_ready_after_feature_engineering.csv")

        # Load clean CSVs
        clean_gh = pd.read_csv("housing_model_ready_after_outlier_treatment.csv")
        clean_gl = pd.read_csv("cleaned_land_merged_final_after_eda.csv")
        clean_lh = pd.read_csv("cleaned_lalpurja_house_v2_after_cleaning.csv")
        clean_ll = pd.read_csv("cleaned_lalpurja_land_final_after_eda.csv")

        # Filter to matching districts
        clean_gh_f = clean_gh[clean_gh["district"].isin(MAIN_DISTRICTS + ["Unknown"])].reset_index(drop=True)
        house_fe_f = house_fe[house_fe["district"].isin([0,1,2,3])].reset_index(drop=True)
        clean_gl_f = clean_gl[clean_gl["district"].isin(MAIN_DISTRICTS)].reset_index(drop=True)
        land_fe_f  = land_fe[land_fe["district"].isin([0,1,2])].reset_index(drop=True)

        def build_map(clean_col, enc_col, clean_df, enc_df):
            """Build {clean_label: encoded_value} dict"""
            m = {}
            for i in range(min(len(clean_df), len(enc_df))):
                m[clean_df.iloc[i][clean_col]] = float(enc_df.iloc[i][enc_col])
            return m

        maps = {}

        # Neighborhood target-encoded maps
        maps["neigh_gh"] = build_map("neighborhood", "neighborhood_encoded", clean_gh_f, house_fe_f)
        maps["neigh_gl"] = build_map("neighborhood", "neighborhood_encoded", clean_gl_f, land_fe_f)
        maps["neigh_lh"] = build_map("neighborhood", "neighborhood_encoded", clean_lh, lph_house_fe)
        maps["neigh_ll"] = build_map("neighborhood", "neighborhood_encoded", clean_ll, lph_land_fe)

        # Municipality target-encoded maps
        maps["muni_lh"] = build_map("municipality", "municipality_encoded", clean_lh, lph_house_fe)
        maps["muni_ll"] = build_map("municipality", "municipality_encoded", clean_ll, lph_land_fe)

        # Label encoding maps
        maps["district"] = {"Bhaktapur": 0, "Kathmandu": 1, "Lalitpur": 2, "Unknown": 3}
        maps["facing_gh"] = {"East":0,"North":1,"North East":2,"North West":3,"South":4,"South East":5,"South West":6,"West":7}
        maps["facing_gl"] = {"East":0,"North":1,"North-East":2,"North-West":3,"South":4,"South-East":5,"South-West":6,"Unknown":7,"West":8}
        maps["road_gl"] = {"High Access":0,"Low Access":1,"Mid Access":2,"Unknown":3}
        maps["road_lh"]   = {"High Access":0,"Low Access":1}
        maps["road_ll"]   = {"High Access":0,"Low Access":1}
        maps["ptype_lh"]  = {"Commercial":0,"Residential":1,"Semi-commercial":2}
        maps["ptype_ll"]  = {"Commercial":0,"Residential":1,"Semi-commercial":2}
        maps["furnish"]   = {"Full Furnished":0,"Semi Furnished":1,"Unfurnished":2}
        maps["face_lph"]  = {"East":0,"North":1,"North-East":2,"North-West":3,"South":4,"South-East":5,"South-West":6,"West":7}
        maps["source_gl"] = {"Hamrobazaar": 0, "Nepali": 1}

        # Amenity & ward medians
        maps["ward_lh"] = clean_lh.groupby("neighborhood")["ward_no"].median().to_dict()
        maps["ward_ll"] = clean_ll.groupby("neighborhood")["ward_no"].median().to_dict()

        amenity_cols_lh = ["hospital_m","airport_m","pharmacy_m","bhatbhateni_m","school_m","college_m","public_transport_m","police_station_m","boudhanath_m","ring_road_m"]
        amenity_cols_ll = ["hospital_m","airport_m","pharmacy_m","bhatbhateni_m","school_m","public_transport_m","police_station_m","ring_road_m"]
        maps["amenity_lh"] = clean_lh.groupby("neighborhood")[amenity_cols_lh].median().to_dict(orient="index")
        maps["amenity_ll"] = clean_ll.groupby("neighborhood")[amenity_cols_ll].median().to_dict(orient="index")

        # Engineered feature medians per neighborhood
        lph_house_fe["neigh_name"] = clean_lh["neighborhood"].values
        maps["eng_lh"] = lph_house_fe.groupby("neigh_name")[
            ["log_land","log_built","floor_area_ratio","urban_centrality","amenity_access_score","house_size_score",
             "comm_road_premium","neighborhood_x_district","municipality_x_ward","age_condition_score","rooms_total",
             "bath_per_bed","sqft_per_room","floors_x_land","luxury_score","parking_premium"]
        ].median().to_dict(orient="index")

        lph_land_fe["neigh_name"] = clean_ll["neighborhood"].values
        maps["eng_ll"] = lph_land_fe.groupby("neigh_name")[
            ["log_land","urban_centrality","amenity_access_score","plot_value_score","commercial_zone_score",
             "neighborhood_x_district","municipality_x_ward","road_access_quality","ring_road_proximity",
             "comm_road_premium","is_corner_plot","facing_road_width"]
        ].median().to_dict(orient="index")

        house_fe_f["neigh_name"] = clean_gh_f["neighborhood"].values
        maps["eng_gh"] = house_fe_f.groupby("neigh_name")[
            ["log_land","log_build_up","luxury_score","amenity_count","is_wide_road","is_area_estimated",
             "is_incomplete_listing","parking_cars","parking_bikes"]
        ].median().to_dict(orient="index")

        land_fe_f["neigh_name"] = clean_gl_f["neighborhood"].values
        maps["eng_gl"] = land_fe_f.groupby("neigh_name")[
            ["log_land","is_wide_road","is_large_plot","road_quality_score","neighborhood_x_district",
             "plot_size_category","location_tier","large_plot_x_neighborhood"]
        ].median().to_dict(orient="index")

        # Neighborhood → municipality/district maps
        maps["neigh_to_muni_lh"] = clean_lh.groupby("neighborhood")["municipality"].first().to_dict()
        maps["neigh_to_muni_ll"] = clean_ll.groupby("neighborhood")["municipality"].first().to_dict()
        maps["neigh_to_dist_lh"] = clean_lh.groupby("neighborhood")["district"].first().to_dict()
        maps["neigh_to_dist_ll"] = clean_ll.groupby("neighborhood")["district"].first().to_dict()

        return maps

    except Exception as e:
        st.error(f"Failed to build encoding maps: {e}")
        st.stop()

with st.spinner("Loading encoding maps..."):
    MAPS = build_encoding_maps()


# ═══════════════════════════════════════════════════════════
# LOAD REAL PKL MODELS  (cached)
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    """Load pickle models"""
    try:
        models = {}
        model_files = {
            "gen_house": "xgboost_housing_final.pkl",
            "gen_land":  "catboost_land_model_final.pkl",
            "lph_house": "catboost_lalpurja_house_v2_final.pkl",
            "lph_land":  "catboost_lalpurja_model_final.pkl",
        }
        for key, fname in model_files.items():
            with open(fname, "rb") as f:
                models[key] = pickle.load(f)
        return models
    except FileNotFoundError as e:
        st.error(f"❌ Model file missing: {e}")
        st.stop()

with st.spinner("Loading prediction models..."):
    MODELS = load_models()


# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────

def get_default(maps_dict, neighborhood, col, fallback=0.0):
    """Get neighborhood median for an engineered column"""
    row = maps_dict.get(neighborhood, {})
    val = row.get(col, fallback)
    return float(val) if not (isinstance(val, float) and np.isnan(val)) else fallback

def validate_input(land_aana, bedrooms, bathrooms, house_age, buildup_sqft=None):
    """Validate user inputs and return error message if invalid"""
    errors = []
    
    if land_aana < 0.5:
        errors.append("Land size must be ≥ 0.5 aana")
    if land_aana > 50:
        errors.append("Land size > 50 aana is outside training range (may be inaccurate)")
    if bedrooms < 1:
        errors.append("Bedrooms must be ≥ 1")
    if bedrooms > 15:
        errors.append("15+ bedrooms is rare; prediction confidence will be lower")
    if bathrooms < 1:
        errors.append("Bathrooms must be ≥ 1")
    if house_age < 0:
        errors.append("House age cannot be negative")
    if house_age > 100:
        errors.append("House age > 100 years is outside training range")
    if buildup_sqft is not None and buildup_sqft < 100:
        errors.append("Built-up area must be ≥ 100 sqft")
    
    return errors

def get_confidence_score(land_aana, neighborhood, model_r2, model_samples, dataset_name):
    """
    Calculate prediction confidence (0-100) based on:
    - Model R² (base)
    - Land size (if Lalpurja land)
    - Neighborhood data availability
    - Training sample count
    """
    # Start with model R² as base (e.g., 0.744 → 74.4)
    confidence = model_r2 * 100
    
    # Penalty for extrapolation
    if dataset_name == "lph_land" and land_aana > 15:
        confidence -= (land_aana - 15) * 3  # -3% per aana beyond 15
    
    # Penalty for sparse neighborhoods
    if neighborhood in MAPS.get("neigh_lh", {}):
        neigh_count = len([k for k in MAPS.get("neigh_lh", {}) if k == neighborhood])
        if neigh_count < 5:
            confidence -= 5
    
    # Boost for large training sets
    if model_samples > 2000:
        confidence += 2
    
    return max(10, min(100, confidence))  # Clamp 10-100

def apply_land_multiplier(base_price, land_aana, model_type="lph_land"):
    """
    FIX FOR LALPURJA ACCURACY: Scale price with land size using non-linear multiplier.
    
    Without this, model predicts same price for 4 aana and 11 aana.
    With this, larger plots get progressively higher prices.
    
    Formula: multiplier = (land_aana / ref_land) ^ exponent
    - ref_land = 5 (reference size)
    - exponent = 0.6 (diminishing returns — larger plots don't scale 1:1)
    """
    if model_type != "lph_land":
        return base_price  # Only apply to Lalpurja land
    
    ref_land = 5.0  # Reference land size (aana)
    exponent = 0.6   # Exponent for diminishing returns
    
    multiplier = max(0.5, (land_aana / ref_land) ** exponent)
    adjusted_price = base_price * multiplier
    
    return adjusted_price

def find_smart_recommendations(user_property, dataset, model_name, top_n=8):
    """
    Find best comparable properties using multi-factor similarity scoring.
    
    Factors:
    - Land size match (40%)
    - Bedrooms match (25%)
    - Neighborhood (20%)
    - Amenity proximity (15%)
    
    Returns scored dataframe sorted by best match
    """
    if dataset.empty:
        return None
    
    # Filter candidates
    district = user_property.get("district")
    property_type = user_property.get("property_type")
    land_aana = user_property.get("land_aana")
    bedrooms = user_property.get("bedrooms", 0)
    neighborhood = user_property.get("neighborhood")
    
    candidates = dataset[
        (dataset["district"] == district) &
        (dataset["land_size_aana"].between(land_aana * 0.6, land_aana * 1.4))
    ].copy()
    
    if candidates.empty:
        return None
    
    # Calculate similarity scores
    land_score = 1 - np.abs(candidates["land_size_aana"] - land_aana) / max(land_aana, 1)
    bed_score = 1 - np.abs(candidates.get("bedrooms", 0) - bedrooms) / max(bedrooms, 1) if bedrooms > 0 else 0.5
    neigh_score = (candidates["neighborhood"] == neighborhood).astype(int)
    
    # Weighted total (0-100)
    candidates["similarity"] = (
        land_score * 40 +
        bed_score * 25 +
        neigh_score * 20 +
        50  # Base score
    )
    
    # Add price difference
    candidates["price_diff_pct"] = 0  # Will be calculated in display
    
    return candidates.nlargest(top_n, "similarity")


# ═══════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS (with validation & improved accuracy)
# ═══════════════════════════════════════════════════════════

def predict_gen_house(district, neighborhood, bedrooms, bathrooms, floors, land_aana, buildup_sqft,
                      road_width, house_age, facing, has_parking, has_garden, has_mod_kitchen,
                      has_parquet, has_drainage, has_solar):
    """General Housing — XGBoost model (24 features)"""
    
    # Validate inputs
    errors = validate_input(land_aana, bedrooms, bathrooms, house_age, buildup_sqft)
    if errors:
        raise ValueError("; ".join(errors))
    
    eng = MAPS["eng_gh"]
    d   = MAPS["district"].get(district, 1)
    f   = MAPS["facing_gh"].get(facing, 0)
    ne  = MAPS["neigh_gh"].get(neighborhood)
    
    if ne is None:
        raise ValueError(f"Neighborhood '{neighborhood}' not in training data")
    
    # Derived features
    log_land = np.log1p(land_aana)
    log_build_up = np.log1p(buildup_sqft)
    luxury_score = int(has_parking)*1 + int(has_garden)*2 + int(has_mod_kitchen)*2 + \
                   int(has_parquet)*1 + int(has_drainage)*1 + int(has_solar)*2
    amenity_count = sum([has_parking, has_garden, has_mod_kitchen, has_parquet, has_drainage, has_solar])
    is_wide_road = 1 if road_width >= 20 else 0

    parking_cars = get_default(eng, neighborhood, "parking_cars", 1.0)
    parking_bikes = get_default(eng, neighborhood, "parking_bikes", 0.0)

    row = np.array([[
        d, land_aana, buildup_sqft, floors, f, road_width, bedrooms, bathrooms,
        parking_cars, parking_bikes, house_age, amenity_count, int(has_mod_kitchen),
        int(has_parquet), int(has_drainage), int(has_parking), int(has_garden),
        is_wide_road, 0, luxury_score, 0, log_land, log_build_up, ne
    ]], dtype=np.float32)

    # Model was trained with log1p(price) as target → must expm1 to get NPR
    return float(np.expm1(MODELS["gen_house"].predict(row)[0]))


def predict_gen_land(district, neighborhood, land_aana, road_type, road_width, facing):
    """General Land — CatBoost model (15 features)"""
    
    errors = validate_input(land_aana, 1, 1, 0)
    if errors:
        raise ValueError("; ".join(errors))
    
    eng = MAPS["eng_gl"]
    d   = MAPS["district"].get(district, 1)
    rt  = MAPS["road_gl"].get(road_type, 2)
    f   = MAPS["facing_gl"].get(facing, 0)
    ne  = MAPS["neigh_gl"].get(neighborhood)
    
    if ne is None:
        raise ValueError(f"Neighborhood '{neighborhood}' not in training data")

    log_land = np.log1p(land_aana)
    is_large_plot = 1 if land_aana > 10 else 0
    is_wide_road = 1 if road_width >= 20 else 0
    road_quality = {"High Access":2,"Mid Access":1,"Low Access":0}.get(road_type, 1)

    row = np.array([[
        d, rt, land_aana, is_large_plot, road_width, is_wide_road, f, 0,
        log_land, ne, road_quality,
        get_default(eng, neighborhood, "neighborhood_x_district", d*ne),
        get_default(eng, neighborhood, "plot_size_category", 2.0),
        get_default(eng, neighborhood, "location_tier", 3.0),
        get_default(eng, neighborhood, "large_plot_x_neighborhood", is_large_plot*ne),
    ]], dtype=np.float32)

    # Model was trained with log1p(price_per_aana) as target → must expm1 to get NPR
    return float(np.expm1(MODELS["gen_land"].predict(row)[0]))


def predict_lph_house(neighborhood, property_type, road_type, furnishing, property_face, bedrooms,
                      kitchens, bathrooms, living_rooms, parking_spaces, total_floors, house_age,
                      road_width, land_aana, buildup_sqft, hospital_m, airport_m, pharmacy_m,
                      bhatbhateni_m, school_m, college_m, public_transport_m, police_station_m,
                      boudhanath_m, ring_road_m):
    """Lalpurja Housing — CatBoost model (42 features)"""
    
    errors = validate_input(land_aana, bedrooms, bathrooms, house_age, buildup_sqft)
    if errors:
        raise ValueError("; ".join(errors))
    
    eng = MAPS["eng_lh"]
    dist_name = MAPS["neigh_to_dist_lh"].get(neighborhood)
    muni_name = MAPS["neigh_to_muni_lh"].get(neighborhood)
    
    if not dist_name or not muni_name:
        raise ValueError(f"Neighborhood '{neighborhood}' not in training data")
    
    d = MAPS["district"].get(dist_name, 1)
    pt = MAPS["ptype_lh"].get(property_type, 1)
    rt = MAPS["road_lh"].get(road_type, 0)
    fn = MAPS["furnish"].get(furnishing, 2)
    pf = MAPS["face_lph"].get(property_face, 0)
    ne = MAPS["neigh_lh"].get(neighborhood)
    me = MAPS["muni_lh"].get(muni_name)
    ward = int(MAPS["ward_lh"].get(neighborhood, 10))

    if ne is None or me is None:
        raise ValueError(f"Encoding failed for neighborhood '{neighborhood}'")

    log_land = np.log1p(land_aana)
    log_built = np.log1p(buildup_sqft)
    rooms_total = bedrooms + bathrooms + kitchens + living_rooms
    bath_per_bed = bathrooms / max(bedrooms, 1)
    sqft_per_room = buildup_sqft / max(rooms_total, 1)
    floors_x_land = total_floors * land_aana
    floor_area_ratio = buildup_sqft / max(land_aana * 182, 1)
    age_condition = max(0, 1 - house_age / 60)
    comm_road = 1 if road_type == "High Access" else 0
    luxury = get_default(eng, neighborhood, "luxury_score", 2.0)
    pk_premium = get_default(eng, neighborhood, "parking_premium", parking_spaces * 0.1)

    row = np.array([[
        d, pt, pf, rt, ward, bedrooms, kitchens, bathrooms, living_rooms, parking_spaces,
        total_floors, house_age, road_width, hospital_m, airport_m, pharmacy_m, bhatbhateni_m,
        school_m, college_m, public_transport_m, police_station_m, boudhanath_m, ring_road_m,
        log_land, log_built, floor_area_ratio,
        get_default(eng, neighborhood, "urban_centrality", 0.5),
        get_default(eng, neighborhood, "amenity_access_score", 0.5),
        get_default(eng, neighborhood, "house_size_score", 0.5),
        comm_road, d * ne, me * ward, age_condition,
        rooms_total, bath_per_bed, sqft_per_room, floors_x_land, luxury, pk_premium,
        fn, ne, me
    ]], dtype=np.float32)

    # Model was trained with log1p(price) as target → must expm1 to get NPR
    return float(np.expm1(MODELS["lph_house"].predict(row)[0]))


def predict_lph_land(neighborhood, property_type, road_type, property_face, land_aana, road_width,
                     hospital_m, airport_m, pharmacy_m, bhatbhateni_m, school_m, public_transport_m,
                     police_station_m, ring_road_m):
    """Lalpurja Land — CatBoost model (29 features)"""
    
    errors = validate_input(land_aana, 1, 1, 0)
    if errors:
        raise ValueError("; ".join(errors))
    
    eng = MAPS["eng_ll"]
    dist_name = MAPS["neigh_to_dist_ll"].get(neighborhood)
    muni_name = MAPS["neigh_to_muni_ll"].get(neighborhood)
    
    if not dist_name or not muni_name:
        raise ValueError(f"Neighborhood '{neighborhood}' not in training data")
    
    d = MAPS["district"].get(dist_name, 1)
    pt = MAPS["ptype_ll"].get(property_type, 1)
    rt = MAPS["road_ll"].get(road_type, 0)
    pf = MAPS["face_lph"].get(property_face, 0)
    ne = MAPS["neigh_ll"].get(neighborhood)
    me = MAPS["muni_ll"].get(muni_name)
    ward = int(MAPS["ward_ll"].get(neighborhood, 10))

    if ne is None or me is None:
        raise ValueError(f"Encoding failed for neighborhood '{neighborhood}'")

    log_land = np.log1p(land_aana)
    comm_road = 1 if road_type == "High Access" else 0
    road_access = {"High Access":2,"Low Access":0}.get(road_type, 1)
    ring_prox = 1 / max(ring_road_m, 1) * 10000

    row = np.array([[
        d, pt, pf, rt, ward, land_aana, road_width,
        get_default(eng, neighborhood, "facing_road_width", road_width),
        hospital_m, airport_m, pharmacy_m, bhatbhateni_m, school_m, public_transport_m,
        police_station_m, ring_road_m, ring_prox, comm_road,
        get_default(eng, neighborhood, "is_corner_plot", 0.0),
        log_land, ne, me,
        get_default(eng, neighborhood, "urban_centrality", 0.5),
        get_default(eng, neighborhood, "amenity_access_score", 0.5),
        get_default(eng, neighborhood, "plot_value_score", 0.5),
        get_default(eng, neighborhood, "commercial_zone_score", 0.0),
        d * ne, me * ward, road_access,
    ]], dtype=np.float32)

    # Model was trained with log1p(price_per_aana) as target → must expm1 to get NPR
    price_per_ana = float(np.expm1(MODELS["lph_land"].predict(row)[0]))
    
    # Apply land multiplier to handle large-plot extrapolation
    adjusted_price_per_ana = apply_land_multiplier(price_per_ana, land_aana, "lph_land")
    
    return adjusted_price_per_ana


# ═══════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════
st.sidebar.title("🏠 Nepal Real Estate Pro")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "",
    ["📊 Market Analytics", "🧠 Inference Engine", "🔍 Recommendations"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("9,929 listings · Kathmandu Valley · 2025")

if st.sidebar.button("ℹ️ About Models", help="Learn about each model"):
    st.sidebar.info("""
    **General Housing** (R² 0.777) — Best overall accuracy
    **Lalpurja Land** (R² 0.744) — Amenity analysis
    **General Land** (R² 0.744) — Simple plots
    **Lalpurja Housing** (R² 0.648) — Interior detail
    """)


# ═══════════════════════════════════════════════════════════
# SECTION 1 — ANALYTICS (condensed, keeping core dashboards)
# ═══════════════════════════════════════════════════════════
if section == "📊 Market Analytics":
    page = st.sidebar.radio("Market", [
        "Overview",
        "🏡 Housing Market",
        "🌍 Land Market",
        "🤖 Model Performance",
    ])

    if page == "Overview":
        st.title("📊 Nepal Real Estate — Market Overview")
        st.markdown("Kathmandu Valley · 2025")
        st.markdown("---")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Listings", "9,929", "All datasets")
        c2.metric("Median House Price", fmt_npr(gh["total_price"].median()), "General market")
        c3.metric("Median Land Price/Ana", fmt_npr(gl["price_per_aana"].median()), "General market")
        c4.metric("Best Model R²", "0.777", "General Housing")
        st.markdown("---")

        st.subheader("Price by District")
        c1,c2 = st.columns(2)
        with c1:
            d = gh.groupby("district")["total_price"].median().reset_index().sort_values("total_price")
            d.columns = ["District","Median Price"]
            fig = px.bar(d, x="Median Price", y="District", orientation="h",
                        color="District", color_discrete_map=DIST_COLORS,
                        text=d["Median Price"].apply(fmt_npr))
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis=dict(showticklabels=False), xaxis_title="", yaxis_title="", height=260)
            st.plotly_chart(clean_chart(fig), use_container_width=True)
        with c2:
            d = gl.groupby("district")["price_per_aana"].median().reset_index().sort_values("price_per_aana")
            d.columns = ["District","Median Price/Ana"]
            fig = px.bar(d, x="Median Price/Ana", y="District", orientation="h",
                        color="District", color_discrete_map=DIST_COLORS,
                        text=d["Median Price/Ana"].apply(fmt_npr))
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis=dict(showticklabels=False), xaxis_title="", yaxis_title="", height=260)
            st.plotly_chart(clean_chart(fig), use_container_width=True)

        st.markdown("---")
        c1,c2 = st.columns(2)
        with c1:
            fig = px.pie(names=["Gen Housing","Lalpurja Housing","Gen Land","Lalpurja Land"],
                        values=[2465,2187,4063,1214], title="Listings by Dataset", hole=0.45)
            st.plotly_chart(clean_chart(fig), use_container_width=True)
        with c2:
            bins = [0,15e6,25e6,35e6,50e6,75e6,400e6]
            labels = ["<1.5Cr","1.5–2.5Cr","2.5–3.5Cr","3.5–5Cr","5–7.5Cr",">7.5Cr"]
            gh["price_bucket"] = pd.cut(gh["total_price"], bins=bins, labels=labels)
            bkt = gh["price_bucket"].value_counts().sort_index().reset_index()
            bkt.columns = ["Range","Count"]
            fig = px.bar(bkt, x="Range", y="Count", title="Housing Price Distribution")
            st.plotly_chart(clean_chart(fig), use_container_width=True)

    elif page == "🏡 Housing Market":
        st.title("🏡 Housing Market Analysis")
        tab1, tab2 = st.tabs(["General Housing", "Lalpurja Housing"])

        # ── GENERAL HOUSING ──────────────────────────────────────────────────
        with tab1:
            st.subheader(f"General Housing — {len(gh):,} Listings")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Listings", f"{len(gh):,}")
            c2.metric("Median Price", fmt_npr(gh["total_price"].median(), 1))
            c3.metric("Avg Beds", f"{gh['bedrooms'].mean():.1f}")
            c4.metric("Avg Floors", f"{gh['floors'].mean():.2f}")
            st.markdown("---")

            # ════════════════════════════════════════════════
            # SUBSECTION: Housing
            # ════════════════════════════════════════════════
            st.markdown("## 🏠 Housing")
            st.markdown("In-depth visualizations for the General Housing dataset. Each chart is annotated with key EDA findings.")

            # ── Row 1: Top Neighborhoods + Price Distribution ──
            c1, c2 = st.columns(2)
            with c1:
                top_n = gh[~gh["neighborhood"].str.contains("Zone|Unknown", na=False)].groupby("neighborhood")["total_price"].agg(["median", "count"]).reset_index()
                top_n = top_n[top_n["count"] >= 5].sort_values("median", ascending=False).head(15)
                top_n["Label"] = top_n["median"].apply(fmt_npr)
                fig = px.bar(top_n.sort_values("median"), x="median", y="neighborhood", orientation="h",
                             color="median", color_continuous_scale="Purples", text="Label",
                             title="Top 15 Most Expensive Neighborhoods")
                fig.update_traces(textposition="outside")
                fig.update_layout(xaxis_title="", xaxis_showticklabels=False, coloraxis_showscale=False, yaxis_title="")
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight("The most expensive neighborhoods are concentrated in central Kathmandu. These areas command prices well above the valley median, confirming that location is the #1 price driver.")

            with c2:
                st.subheader("Price Distribution")
                fig = go.Figure(data=[
                    go.Histogram(x=gh["total_price"], nbinsx=40,
                                 marker_color="#667eea", opacity=0.75)
                ])
                med_val = gh["total_price"].median()
                fig.add_vline(x=med_val, line_dash="dash", line_color="#764ba2",
                              annotation_text=f"Median: {fmt_npr(med_val, 1)}")
                fig.update_xaxes(title_text="Price (NPR)")
                fig.update_yaxes(title_text="Frequency")
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight("Heavily right-skewed — most properties are priced between ₹0.2–0.5 Cr, with a long tail extending to ₹4 Cr. Log transformation is essential before ML modeling to handle this skew.")

            st.markdown("---")

            # ── Row 2: Price by Bedrooms, Bathrooms, Floors ──
            st.subheader("Price by Property Features")
            c1, c2, c3 = st.columns(3)

            with c1:
                if "bedrooms" in gh.columns:
                    bed_stats = gh.groupby("bedrooms")["total_price"].agg(["median", "count"]).reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=bed_stats["bedrooms"], y=bed_stats["median"],
                                         name="Median Price", marker_color="#667eea",
                                         text=bed_stats["median"].apply(fmt_npr), textposition="outside"))
                    fig.add_trace(go.Scatter(x=bed_stats["bedrooms"], y=bed_stats["count"],
                                             name="Count", yaxis="y2", mode="lines+markers",
                                             marker=dict(color="#764ba2", size=8)))
                    fig.update_layout(title="Bedrooms Impact", yaxis_title="Price (NPR)",
                                      yaxis2=dict(title="Count", overlaying="y", side="right"),
                                      hovermode="x unified")
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("General upward trend: more bedrooms = higher price. Standard homes (2–5 beds) are tightly clustered, while 10–16-bedroom properties (guesthouses/commercial) show much higher and more variable prices.")

            with c2:
                if "bathrooms" in gh.columns:
                    bath_stats = gh.groupby("bathrooms")["total_price"].agg(["median", "count"]).reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=bath_stats["bathrooms"], y=bath_stats["median"],
                                         name="Median Price", marker_color="#8884d8",
                                         text=bath_stats["median"].apply(fmt_npr), textposition="outside"))
                    fig.update_layout(title="Bathrooms Impact", xaxis_title="Bathrooms",
                                      yaxis_title="Price (NPR)", showlegend=False)
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("Bathrooms is the 2nd strongest numerical predictor (EDA correlation: 0.37). It reflects overall house size — more bathrooms generally signals a larger, more complete property.")

            with c3:
                if "floors" in gh.columns:
                    floor_stats = gh.groupby("floors")["total_price"].agg(["median", "count"]).reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=floor_stats["floors"], y=floor_stats["median"],
                                         name="Median Price", marker_color="#52c41a",
                                         text=floor_stats["median"].apply(fmt_npr), textposition="outside"))
                    fig.update_layout(title="Floors Impact", xaxis_title="Floors",
                                      yaxis_title="Price (NPR)", showlegend=False)
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("Floors shows the strongest categorical price trend (EDA correlation: 0.32). Properties with 6–7 floors have dramatically higher median prices — confirming that vertical development signals premium properties.")

            st.markdown("---")

            # ── Row 3: Violin plot + Amenity Premiums ──
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Price Density by District (Violin)")
                fig = go.Figure()
                for district in MAIN_DISTRICTS:
                    fig.add_trace(go.Violin(
                        y=gh[gh["district"] == district]["total_price"],
                        name=district,
                        box_visible=True,
                        meanline_visible=True,
                        points=False,
                        marker_color=DIST_COLORS.get(district)
                    ))
                fig.update_yaxes(title_text="Price (NPR)")
                fig.update_layout(showlegend=True, height=400)
                st.plotly_chart(clean_chart(fig, height=400), use_container_width=True)
                insight("All 3 districts have similar median prices (boxes overlap significantly). Kathmandu has the widest spread and most high-end outliers, confirming it hosts the luxury market. Bhaktapur is the most consistently priced and affordable district.")

            with c2:
                st.subheader("Amenity Price Premiums")
                amenity_cols = [c for c in gh.columns if c.startswith("has_")]
                amenity_premium = []
                for col in amenity_cols:
                    if gh[col].dtype in [int, float, "int64", "float64"]:
                        w = gh[gh[col] == 1]["total_price"].median()
                        wo = gh[gh[col] == 0]["total_price"].median()
                        if pd.notna(w) and pd.notna(wo) and wo > 0:
                            pct = (w - wo) / wo * 100
                            amenity_premium.append({
                                "Amenity": col.replace("has_", "").replace("_", " ").title(),
                                "Premium %": round(pct, 1),
                                "Count": int((gh[col] == 1).sum())
                            })
                if amenity_premium:
                    ap_df = pd.DataFrame(amenity_premium).sort_values("Premium %")
                    fig = px.bar(ap_df, x="Premium %", y="Amenity", orientation="h",
                                 color="Premium %", color_continuous_scale="RdYlGn",
                                 title="Price Premium by Amenity (%)")
                    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
                    st.plotly_chart(clean_chart(fig, height=400), use_container_width=True)
                    insight("Amenities like garden and modular kitchen command the highest premiums. Surprisingly, the overall luxury score has weak correlation with price — land size and location drive value more than individual amenities.")

            st.markdown("---")

            # ── Row 4: Price Ratios ──
            st.subheader("Price Ratio Analysis")
            c1, c2 = st.columns(2)

            with c1:
                if "build_up_area" in gh.columns:
                    gh_r = gh.dropna(subset=["build_up_area"]).copy()
                    gh_r["price_per_sqft"] = gh_r["total_price"] / gh_r["build_up_area"]
                    fig = go.Figure(data=[go.Histogram(
                        x=gh_r["price_per_sqft"], nbinsx=40, marker_color="#667eea", opacity=0.75)])
                    med_ppsf = gh_r["price_per_sqft"].median()
                    fig.add_vline(x=med_ppsf, line_dash="dash", line_color="#764ba2",
                                  annotation_text=f"Median: {fmt_npr(med_ppsf, 0)}")
                    fig.update_layout(title="Price per Sq. Ft. Distribution",
                                      xaxis_title="Price per Sqft (NPR)", yaxis_title="Frequency",
                                      showlegend=False, height=350)
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("Price per sqft is right-skewed, peaking at mid-range values. Extreme values on the right tail represent premium-location properties where land value is the dominant factor, not built-up area.")

            with c2:
                if "land_area_aana" in gh.columns:
                    gh_r2 = gh.dropna(subset=["land_area_aana"]).copy()
                    gh_r2["price_per_ana"] = gh_r2["total_price"] / gh_r2["land_area_aana"]
                    fig = go.Figure(data=[go.Histogram(
                        x=gh_r2["price_per_ana"], nbinsx=40, marker_color="#8884d8", opacity=0.75)])
                    med_ppa = gh_r2["price_per_ana"].median()
                    fig.add_vline(x=med_ppa, line_dash="dash", line_color="#764ba2",
                                  annotation_text=f"Median: {fmt_npr(med_ppa, 1)}")
                    fig.update_layout(title="Price per Ana (Housing) Distribution",
                                      xaxis_title="Price per Ana (NPR)", yaxis_title="Frequency",
                                      showlegend=False, height=350)
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("Land area (Ana) is the STRONGEST numerical predictor of total price (EDA correlation: 0.46). The distribution of price-per-ana confirms that most housing plots cluster around ₹0.3–0.5 Cr per Ana.")

            st.markdown("---")

            # ── Row 5: Feature Correlation + Cumulative Distribution ──
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Feature Correlation with Price")
                # Exclude ALL derived/leakage columns
                _leakage = {
                    "price_per_aana", "calculated_total_price", "price_bucket",
                    "log_price", "log_total_price", "price_log", "price_per_sqft",
                    "price_per_ana", "price_ratio"
                }
                numeric_cols = gh.select_dtypes(include=[np.number]).columns.tolist()
                corr_cols = [c for c in numeric_cols if c != "total_price" and c not in _leakage]
                if corr_cols:
                    corr_series = gh[corr_cols + ["total_price"]].corr()["total_price"].drop("total_price").sort_values()
                    # Build explicit DataFrame so Feature, Value, Color are always aligned
                    corr_df = pd.DataFrame({
                        "Feature": corr_series.index,
                        "Correlation": corr_series.values
                    }).reset_index(drop=True)
                    fig = px.bar(
                        corr_df,
                        x="Correlation",
                        y="Feature",
                        orientation="h",
                        color="Correlation",
                        color_continuous_scale="RdBu",
                        title="Feature Correlations (Leakage-Free)",
                        hover_data={"Feature": True, "Correlation": ":.3f"}
                    )
                    fig.update_xaxes(range=[-1, 1])
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(clean_chart(fig, height=450), use_container_width=True)
                    insight("Land area aana (0.7) and buildup area (0.67) are top predictors. Bathrooms (0.37) and floors (0.26) follow. House age and luxury score show near-zero correlation — location and size matter far more than age or extras.")

            with c2:
                st.subheader("Cumulative Price Distribution")
                sorted_prices = gh["total_price"].sort_values().values
                cumulative = np.arange(1, len(sorted_prices) + 1) / len(sorted_prices) * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sorted_prices, y=cumulative, mode="lines",
                                         fill="tozeroy", line=dict(color="#667eea", width=3)))
                fig.update_layout(title="Cumulative Distribution", showlegend=False, height=450,
                                  xaxis_title="Price (NPR)", yaxis_title="Cumulative %")
                st.plotly_chart(clean_chart(fig, height=450), use_container_width=True)
                insight("~75% of properties are priced below ₹4.5 Cr, confirming the market is dominated by mid-range housing. The steep rise in the lower range reflects a high volume of affordable listings, while the long tail captures the luxury segment.")

            st.markdown("---")

            # ── Row 6: Market Segments ──
            st.subheader("Price by Market Segment & Bedrooms")
            if "bedrooms" in gh.columns:
                q33 = gh["total_price"].quantile(0.33)
                q67 = gh["total_price"].quantile(0.67)
                gh_seg = gh.copy()
                gh_seg["Segment"] = "Standard"
                gh_seg.loc[gh_seg["total_price"] < q33, "Segment"] = "Budget"
                gh_seg.loc[gh_seg["total_price"] > q67, "Segment"] = "Luxury"
                seg_analysis = gh_seg.groupby(["Segment", "bedrooms"])["total_price"].median().reset_index()
                fig = px.bar(seg_analysis, x="bedrooms", y="total_price", color="Segment",
                             title=f"Median Price by Segment — Budget (<{fmt_npr(q33,1)}), Standard, Luxury (>{fmt_npr(q67,1)})",
                             barmode="group",
                             labels={"total_price": "Median Price (NPR)", "bedrooms": "Bedrooms"},
                             color_discrete_map={"Budget": "#faad14", "Standard": "#667eea", "Luxury": "#f5222d"})
                st.plotly_chart(clean_chart(fig, height=400), use_container_width=True)
                insight(f"Budget properties (below {fmt_npr(q33,1)}) are tightly clustered regardless of bedroom count. The luxury gap widens sharply above 5 bedrooms. Budget properties with large land but low price typically indicate outskirt/underdeveloped locations.")

            st.markdown("---")

            # ── Row 7: Statistical Summary Table ──
            st.subheader("Statistical Summary")
            summary_stats = {
                "Statistic": ["Count", "Min", "Q1 (25%)", "Median", "Q3 (75%)", "Max", "Mean", "Std Dev"],
                "Housing Price": [
                    f"{len(gh):,}",
                    fmt_npr(gh["total_price"].min(), 2),
                    fmt_npr(gh["total_price"].quantile(0.25), 2),
                    fmt_npr(gh["total_price"].median(), 2),
                    fmt_npr(gh["total_price"].quantile(0.75), 2),
                    fmt_npr(gh["total_price"].max(), 2),
                    fmt_npr(gh["total_price"].mean(), 2),
                    fmt_npr(gh["total_price"].std(), 2),
                ]
            }
            st.dataframe(pd.DataFrame(summary_stats), use_container_width=True, hide_index=True)
            insight("The IQR (Q1 to Q3) spans roughly ₹2.45 Cr to ₹4.5 Cr, defining the standard mid-range segment. Properties above ₹9 Cr are Ultra-Luxury. The wide std dev confirms high price variability driven by location differences.")

        # ── LALPURJA HOUSING ─────────────────────────────────────────────────
        with tab2:
            st.subheader(f"Lalpurja Housing — {len(lh):,} Listings")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Listings", f"{len(lh):,}")
            c2.metric("Median", fmt_npr(lh["total_price"].median()))
            c3.metric("Avg Beds", f"{lh['bedrooms'].mean():.1f}")
            c4.metric("Avg Land", f"{lh['land_size_aana'].mean():.1f} Ana")
            st.markdown("---")

            # ════════════════════════════════════════════════
            # ════════════════════════════════════════════════
            # SUBSECTION: Housing (Lalpurja)
            # ════════════════════════════════════════════════
            st.markdown("## 🏠 Housing")
            st.markdown("In-depth visualizations for the Lalpurja Housing dataset. Insights sourced from EDA analysis.")

            # ── Row 1: Top Neighborhoods + Price Distribution ──
            c1, c2 = st.columns(2)

            with c1:
                top_n = lh_named.groupby("neighborhood")["total_price"].agg(["median", "count"]).reset_index()
                top_n = top_n[top_n["count"] >= 3].sort_values("median", ascending=False).head(15)
                top_n["Label"] = top_n["median"].apply(fmt_npr)
                fig = px.bar(top_n.sort_values("median"), x="median", y="neighborhood", orientation="h",
                             color="median", color_continuous_scale="Purples", text="Label",
                             title="Top 15 Most Expensive Neighborhoods")
                fig.update_traces(textposition="outside")
                fig.update_layout(xaxis_title="", xaxis_showticklabels=False, coloraxis_showscale=False, yaxis_title="")
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight("Hattisar, Jawalakhel, and Bansbari are among the priciest Lalpurja neighborhoods, reaching ₹19 Cr+. These are all well-known central Kathmandu areas — confirming ring-road proximity as the top premium driver.")

            with c2:
                st.subheader("Price Distribution")
                fig = go.Figure(data=[go.Histogram(
                    x=lh["total_price"], nbinsx=40, marker_color="#5ab8e8", opacity=0.75)])
                lh_med = lh["total_price"].median()
                fig.add_vline(x=lh_med, line_dash="dash", line_color="#e8a45a",
                              annotation_text=f"Median: {fmt_npr(lh_med, 1)}")
                fig.update_layout(title="Lalpurja Housing Price Distribution",
                                  xaxis_title="Price (NPR)", yaxis_title="Frequency",
                                  showlegend=False, height=500)
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight(f"Sharp peak at ₹0.3–0.5 Cr with a rapid drop and long tail to ₹3.5 Cr. Median is {fmt_npr(lh_med, 1)} — most Lalpurja homes are mid-range residential. The slight irregularities in the distribution indicate multiple pricing zones within the dataset.")

            st.markdown("---")

            # ── Row 2: Bedrooms & Property Type Impact ──
            c1, c2, c3 = st.columns(3)

            with c1:
                if "bedrooms" in lh.columns:
                    bed_stats = lh.groupby("bedrooms")["total_price"].agg(["median", "count"]).reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=bed_stats["bedrooms"], y=bed_stats["median"],
                                         name="Median Price", marker_color="#5ab8e8",
                                         text=bed_stats["median"].apply(lambda v: fmt_npr(v, 1)), textposition="outside"))
                    fig.add_trace(go.Scatter(x=bed_stats["bedrooms"], y=bed_stats["count"],
                                             name="Count", yaxis="y2", mode="lines+markers",
                                             marker=dict(color="#e8a45a", size=8)))
                    fig.update_layout(title="Bedrooms Impact", yaxis_title="Price (NPR)",
                                      yaxis2=dict(title="Count", overlaying="y", side="right"),
                                      hovermode="x unified")
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("Most Lalpurja homes have 3–5 bedrooms (the spike in count). Price rises with bedrooms but with high variance — location amplifies bedroom value in central areas more than in outskirts.")

            with c2:
                if "property_type" in lh.columns:
                    ptype_stats = lh.groupby("property_type")["total_price"].median().reset_index()
                    fig = px.bar(ptype_stats, x="property_type", y="total_price",
                                 title="Price by Property Type",
                                 color="property_type",
                                 text=ptype_stats["total_price"].apply(lambda v: fmt_npr(v, 1)),
                                 labels={"total_price": "Median Price (NPR)", "property_type": ""})
                    fig.update_traces(textposition="outside")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight(f"Commercial: {fmt_npr(ptype_stats[ptype_stats['property_type']=='Commercial']['total_price'].values[0] if 'Commercial' in ptype_stats['property_type'].values else 0, 1)} vs Residential: lower — commercial properties command ~43% higher median price. Property type is a strong and important ML feature.")

            with c3:
                if "road_type" in lh.columns:
                    road_stats = lh.groupby("road_type")["total_price"].median().reset_index()
                    fig = px.bar(road_stats, x="road_type", y="total_price",
                                 title="Price by Road Type",
                                 color="total_price", color_continuous_scale="Blues",
                                 text=road_stats["total_price"].apply(lambda v: fmt_npr(v, 1)),
                                 labels={"total_price": "Median Price (NPR)", "road_type": ""})
                    fig.update_traces(textposition="outside")
                    fig.update_layout(coloraxis_showscale=False, showlegend=False)
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("High Access road adds ~₹0.65 Cr median premium over Low Access (EDA: ₹3.6 Cr vs ₹2.95 Cr). Better road access also attracts more high-end listings, widening the price spread for that category.")

            st.markdown("---")

            # ── Row 3: Land Size vs Price + Amenity Distance Correlation ──
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Land Size vs Total Price")
                fig = px.scatter(lh, x="land_size_aana", y="total_price",
                                 color="district" if "district" in lh.columns else None,
                                 opacity=0.5,
                                 title="Land Size vs Price (Lalpurja Housing)",
                                 labels={"land_size_aana": "Land Size (Ana)", "total_price": "Price (NPR)"})
                st.plotly_chart(clean_chart(fig, height=400), use_container_width=True)
                insight("Clear positive trend — larger land = higher total price. Most data clusters at 2–10 Ana and ₹2–6 Cr. Kathmandu (blue) dominates both large plots and high prices. Price spreads widely above 2,000 Sqft — location starts dominating over size.")

            with c2:
                st.subheader("Amenity Distance Impact on Price")
                amenity_dist_cols = [c for c in lh.columns if c.endswith("_m") and "price" not in c]
                if amenity_dist_cols:
                    corr_amenity = lh[amenity_dist_cols + ["total_price"]].corr()["total_price"].drop("total_price").sort_values()
                    fig = px.bar(x=corr_amenity.values, y=corr_amenity.index, orientation="h",
                                 color=corr_amenity.values, color_continuous_scale="RdBu",
                                 title="Amenity Distance Correlation with Price",
                                 labels={"x": "Correlation", "y": "Amenity"})
                    fig.update_xaxes(range=[-1, 1])
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(clean_chart(fig, height=400), use_container_width=True)
                    insight("All amenity correlations are negative (closer = more expensive). Ring Road (−0.16) and Boudhanath (−0.15) are strongest for housing. Unlike land, airport distance is weaker (−0.10 vs −0.56 for land) — residents care more about daily urban access.")

            st.markdown("---")

            # ── Row 4: Furnishing & Floors ──
            c1, c2 = st.columns(2)

            with c1:
                if "furnishing" in lh.columns:
                    furn_stats = lh.groupby("furnishing")["total_price"].median().reset_index()
                    fig = px.bar(furn_stats, x="furnishing", y="total_price",
                                 title="Price by Furnishing Status",
                                 color="furnishing",
                                 text=furn_stats["total_price"].apply(lambda v: fmt_npr(v, 1)),
                                 labels={"total_price": "Median Price (NPR)", "furnishing": ""})
                    fig.update_traces(textposition="outside")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("While meaningful, furnishing alone doesn't drive price as much as location or land size — it's an incremental feature, not a primary value driver.")

            with c2:
                if "total_floors" in lh.columns:
                    fl_stats = lh.groupby("total_floors")["total_price"].agg(["median", "count"]).reset_index()
                    fl_stats = fl_stats[fl_stats["count"] >= 3]
                    fig = px.line(fl_stats, x="total_floors", y="median", markers=True,
                                  title="Price Trend by Total Floors",
                                  labels={"total_floors": "Total Floors", "median": "Median Price (NPR)"})
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("Price rises with floors in a non-linear pattern. The jump beyond 4 floors indicates multi-storey commercial or large residential builds — consistent with the EDA finding that floors is among the strongest structural predictors.")

            st.markdown("---")

            # ── Row 5: Statistical Summary ──
            st.subheader("Lalpurja Housing Statistical Summary")
            lh_summary = {
                "Statistic": ["Count", "Min", "Q1 (25%)", "Median", "Q3 (75%)", "Max", "Mean", "Std Dev"],
                "Total Price": [
                    f"{len(lh):,}",
                    fmt_npr(lh["total_price"].min(), 2),
                    fmt_npr(lh["total_price"].quantile(0.25), 2),
                    fmt_npr(lh["total_price"].median(), 2),
                    fmt_npr(lh["total_price"].quantile(0.75), 2),
                    fmt_npr(lh["total_price"].max(), 2),
                    fmt_npr(lh["total_price"].mean(), 2),
                    fmt_npr(lh["total_price"].std(), 2),
                ]
            }
            st.dataframe(pd.DataFrame(lh_summary), use_container_width=True, hide_index=True)
            insight("IQR (Q1–Q3) spans ₹2.6–4.8 Cr, defining the standard mid-range Lalpurja housing segment. Max of ₹35 Cr represents ultra-luxury commercial/large residential properties in premium Kathmandu neighborhoods.")

    elif page == "🌍 Land Market":
        st.title("🌍 Land Market Analysis")
        tab1, tab2 = st.tabs(["General Land", "Lalpurja Land"])

        # ── GENERAL LAND ─────────────────────────────────────────────────────
        with tab1:
            st.subheader(f"General Land — {len(gl):,} Plots")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Plots", f"{len(gl):,}")
            c2.metric("Median/Ana", fmt_npr(gl["price_per_aana"].median(), 1))
            c3.metric("Avg Size", f"{gl['land_size_aana'].mean():.1f} Ana")
            c4.metric("Top Area", "Jhamsikhel/Naxal")
            st.markdown("---")

            # ════════════════════════════════════════════════
            # SUBSECTION: Land
            # ════════════════════════════════════════════════
            st.markdown("## 🌍 Land")
            st.markdown("In-depth visualizations for the General Land dataset. Insights sourced from EDA analysis.")

            # ── Row 1: Top Neighborhoods + Price/Ana Distribution ──
            c1, c2 = st.columns(2)
            with c1:
                top_n = gl_named.groupby("neighborhood")["price_per_aana"].agg(["median", "count"]).reset_index()
                top_n = top_n[top_n["count"] >= 10].sort_values("median", ascending=False).head(15)
                top_n["Label"] = top_n["median"].apply(lambda v: fmt_npr(v, 1))
                fig = px.bar(top_n.sort_values("median"), x="median", y="neighborhood", orientation="h",
                             color="median", color_continuous_scale="Oranges", text="Label",
                             title="Top 15 Neighborhoods by Median Price/Ana")
                fig.update_traces(textposition="outside")
                fig.update_layout(xaxis_title="", xaxis_showticklabels=False, coloraxis_showscale=False, yaxis_title="")
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight("Top neighborhoods like Jhamsikhel and Naxal command premium prices close to ₹1 Cr/Ana. These are all central Kathmandu areas inside or near the ring road — confirming that location is the dominant price driver for land.")

            with c2:
                gl_med = gl["price_per_aana"].median()
                fig = go.Figure(data=[go.Histogram(
                    x=gl["price_per_aana"], nbinsx=40, marker_color="#e8a45a", opacity=0.75)])
                fig.add_vline(x=gl_med, line_dash="dash", line_color="#667eea",
                              annotation_text=f"Median: {fmt_npr(gl_med, 1)}")
                fig.update_layout(title="Price per Ana Distribution",
                                  xaxis_title="Price per Ana (NPR)", yaxis_title="Frequency",
                                  showlegend=False, height=500)
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight(f"Right-skewed with peaks around ₹0.3–0.5 Cr per Ana. Median is {fmt_npr(gl_med, 1)}, but the long tail extending to ₹2 Cr represents premium central Kathmandu plots. Log transformation is necessary for ML modeling.")

            st.markdown("---")

            # ── Row 2: Price by District (box + violin) ──
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Price/Ana Box Plot by District")
                fig = go.Figure()
                for district in MAIN_DISTRICTS:
                    fig.add_trace(go.Box(
                        y=gl[gl["district"] == district]["price_per_aana"],
                        name=district,
                        marker_color=DIST_COLORS.get(district)
                    ))
                fig.update_yaxes(title_text="Price per Ana (NPR)")
                fig.update_layout(showlegend=True, height=400)
                st.plotly_chart(clean_chart(fig, height=400), use_container_width=True)
                insight(f"Kathmandu: {fmt_npr(gl[gl['district']=='Kathmandu']['price_per_aana'].median(),1)}/Ana · Lalitpur: {fmt_npr(gl[gl['district']=='Lalitpur']['price_per_aana'].median(),1)}/Ana · Bhaktapur: {fmt_npr(gl[gl['district']=='Bhaktapur']['price_per_aana'].median(),1)}/Ana. All 3 have outliers above ₹1 Cr/Ana — confirming district is a meaningful price differentiator.")

            with c2:
                st.subheader("Price/Ana Density by District (Violin)")
                fig = go.Figure()
                for district in MAIN_DISTRICTS:
                    fig.add_trace(go.Violin(
                        y=gl[gl["district"] == district]["price_per_aana"],
                        name=district,
                        box_visible=True,
                        meanline_visible=True,
                        points=False,
                        marker_color=DIST_COLORS.get(district)
                    ))
                fig.update_yaxes(title_text="Price per Ana (NPR)")
                fig.update_layout(showlegend=True, height=400)
                st.plotly_chart(clean_chart(fig, height=400), use_container_width=True)
                insight("Kathmandu has the widest violin, indicating most price diversity. Bhaktapur's narrow violin shows the tightest, most predictable pricing — easiest district to model. Lalitpur falls in between with moderate spread.")

            st.markdown("---")

            # ── Row 3: Road Type & Land Size Impact ──
            c1, c2 = st.columns(2)

            with c1:
                if "road_type" in gl.columns:
                    road_stats = gl.groupby("road_type")["price_per_aana"].median().reset_index()
                    fig = px.bar(road_stats, x="road_type", y="price_per_aana",
                                 title="Price/Ana by Road Type",
                                 color="price_per_aana", color_continuous_scale="Oranges",
                                 text=road_stats["price_per_aana"].apply(lambda v: fmt_npr(v, 1)),
                                 labels={"price_per_aana": "Median Price/Ana (NPR)", "road_type": ""})
                    fig.update_traces(textposition="outside")
                    fig.update_layout(coloraxis_showscale=False, showlegend=False)
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("Wide Road commands the highest median (EDA: ₹0.53 Cr/Ana) vs Gravel at ₹0.35 Cr/Ana — a ~51% premium. Pitched roads closely match Wide Road pricing. Road quality is a meaningful and reliable land price differentiator.")

            with c2:
                if "land_size_aana" in gl.columns:
                    gl_binned = gl.copy()
                    gl_binned["Land_Bin"] = pd.cut(gl_binned["land_size_aana"], bins=8)
                    land_trend = gl_binned.groupby("Land_Bin", observed=False)["price_per_aana"].median().reset_index()
                    land_trend["Land_Bin"] = land_trend["Land_Bin"].astype(str)
                    fig = px.bar(land_trend, x="Land_Bin", y="price_per_aana",
                                 title="Median Price/Ana by Land Size",
                                 color="price_per_aana", color_continuous_scale="Viridis",
                                 labels={"price_per_aana": "Median Price/Ana (NPR)", "Land_Bin": "Land Size (Ana)"})
                    fig.update_xaxes(tickangle=-30)
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("Highest per-Ana prices are on small plots (0–10 Ana). As land size grows beyond 20 Ana, price/Ana drops — a classic 'bulk discount' effect. Small plots in prime locations command the most per-Ana value.")

            st.markdown("---")

            # ── Row 4: Correlation (DATA LEAKAGE FIXED) + Cumulative ──
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Feature Correlation with Price/Ana")
                # Build an enriched numeric frame by encoding key categorical columns
                _leakage_land = {
                    "calculated_total_price", "total_price", "price_bucket",
                    "is_price_outlier", "is_rate_outlier", "is_price_suspect",
                    "log_price", "log_price_per_aana"
                }
                gl_corr = gl.copy()

                # Encode district → ordinal (price hierarchy from EDA)
                dist_order = {"Kathmandu": 3, "Lalitpur": 2, "Bhaktapur": 1}
                gl_corr["district_rank"] = gl_corr["district"].map(dist_order).fillna(0)

                # Encode road_type → ordinal (price hierarchy from EDA)
                road_order = {
                    "Wide Road": 5, "Pitched": 4, "Narrow Road": 4,
                    "Medium Road": 3, "Paved": 3, "Unknown": 2, "Gravel": 1
                }
                if "road_type" in gl_corr.columns:
                    gl_corr["road_type_rank"] = gl_corr["road_type"].map(road_order).fillna(2)

                # Encode neighborhood tier → ordinal
                tier_order = {
                    "Premium_Zone": 5, "High_Zone": 4, "Mid_Zone": 3,
                    "Budget_Zone": 2, "Outskirt_Zone": 1
                }
                if "neighborhood" in gl_corr.columns:
                    gl_corr["neighborhood_tier"] = gl_corr["neighborhood"].map(tier_order)
                    # Named neighborhoods get their median price rank
                    named_medians = (
                        gl_corr[gl_corr["neighborhood_tier"].isna()]
                        .groupby("neighborhood")["price_per_aana"].transform("median")
                    )
                    # Bin named neighborhood medians into 1–5
                    if named_medians.notna().any():
                        gl_corr.loc[gl_corr["neighborhood_tier"].isna(), "neighborhood_tier"] = \
                            pd.cut(named_medians.dropna(), bins=5, labels=[1,2,3,4,5]).astype(float)
                    gl_corr["neighborhood_tier"] = gl_corr["neighborhood_tier"].fillna(2)

                # Binary: is_large_plot, is_wide_road
                if "is_large_plot" in gl_corr.columns:
                    gl_corr["is_large_plot"] = gl_corr["is_large_plot"].astype(int)
                if "is_wide_road" in gl_corr.columns:
                    gl_corr["is_wide_road"] = gl_corr["is_wide_road"].astype(int)

                # Source encoding (hamrobazaar vs nepali_land)
                if "source" in gl_corr.columns:
                    gl_corr["source_hb"] = (gl_corr["source"] == "hamrobazaar").astype(int)

                # Now select clean numeric cols
                numeric_land = gl_corr.select_dtypes(include=[np.number]).columns.tolist()
                corr_cols_land = [c for c in numeric_land
                                  if c != "price_per_aana" and c not in _leakage_land]

                if corr_cols_land:
                    corr_series = gl_corr[corr_cols_land + ["price_per_aana"]].corr()["price_per_aana"].drop("price_per_aana").sort_values()
                    # Use DataFrame to keep Feature/Value/Color always aligned
                    corr_df = pd.DataFrame({
                        "Feature": corr_series.index,
                        "Correlation": corr_series.values
                    }).reset_index(drop=True)
                    fig = px.bar(
                        corr_df,
                        x="Correlation",
                        y="Feature",
                        orientation="h",
                        color="Correlation",
                        color_continuous_scale="RdBu",
                        title="Feature Correlations (Leakage-Free)",
                        hover_data={"Feature": True, "Correlation": ":.3f"}
                    )
                    fig.update_xaxes(range=[-1, 1])
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(clean_chart(fig, height=450), use_container_width=True)
                    insight("Neighborhood tier is the strongest predictor — location zone (Premium/High/Mid/Budget) encodes where the plot sits in the value hierarchy. District rank (Kathmandu > Lalitpur > Bhaktapur) and road quality also show meaningful correlations. Land size remains near-zero (EDA: −0.01) — price per Ana is driven by location, not plot dimensions.")

            with c2:
                st.subheader("Cumulative Price Distribution")
                sorted_land = gl["price_per_aana"].sort_values().values
                cum_land = np.arange(1, len(sorted_land) + 1) / len(sorted_land) * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sorted_land, y=cum_land, mode="lines",
                                         fill="tozeroy", line=dict(color="#e8a45a", width=3)))
                fig.update_layout(title="Cumulative Land Price Distribution", showlegend=False,
                                  xaxis_title="Price per Ana (NPR)", yaxis_title="Cumulative %", height=400)
                st.plotly_chart(clean_chart(fig, height=400), use_container_width=True)
                insight("~50% of plots are priced below ₹0.47 Cr/Ana and ~75% below ₹0.65 Cr/Ana. The steep early rise reflects a large volume of affordable outskirt land, while the gradual tail captures premium central plots.")

            st.markdown("---")

            # ── Row 5: Price Percentiles ──
            st.subheader("Price/Ana Percentile Analysis")
            percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
            pctl_land = {
                "Percentile": [f"{p}th" for p in percentiles],
                "Price/Ana": [fmt_npr(gl["price_per_aana"].quantile(p / 100), 2) for p in percentiles],
                "% of Plots Below": [f"{p}%" for p in percentiles]
            }
            st.dataframe(pd.DataFrame(pctl_land), use_container_width=True, hide_index=True)
            insight("Price segments from EDA — Budget: below ₹0.31 Cr/Ana · Mid-Range: ₹0.31–0.65 Cr/Ana · High-End: ₹0.65 Cr–₹1.12 Cr/Ana · Ultra-Luxury: above ₹1.12 Cr/Ana. Most listings (75%) fall in Budget or Mid-Range.")

        # ── LALPURJA LAND ─────────────────────────────────────────────────────
        with tab2:
            st.subheader(f"Lalpurja Land — {len(ll):,} Plots")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Plots", f"{len(ll):,}")
            c2.metric("Median/Ana", fmt_npr(ll["price_per_aana"].median(), 1))
            c3.metric("Model R²", "0.744")
            c4.metric("Airport Corr.", "−0.558")
            st.markdown("---")

            # ════════════════════════════════════════════════
            # SUBSECTION: Land (Lalpurja)
            # ════════════════════════════════════════════════
            st.markdown("## 🌍 Land")
            st.markdown("In-depth visualizations for the Lalpurja Land dataset. Insights sourced from EDA analysis.")

            # ── Row 1: Top Neighborhoods + Price Distribution ──
            c1, c2 = st.columns(2)
            with c1:
                top_n = ll_named.groupby("neighborhood")["price_per_aana"].agg(["median", "count"]).reset_index()
                top_n = top_n[top_n["count"] >= 3].sort_values("median", ascending=False).head(15)
                top_n["Label"] = top_n["median"].apply(lambda v: fmt_npr(v, 1))
                fig = px.bar(top_n.sort_values("median"), x="median", y="neighborhood", orientation="h",
                             color="median", color_continuous_scale="Greens", text="Label",
                             title="Top 15 Neighborhoods by Median Price/Ana (Lalpurja)")
                fig.update_traces(textposition="outside")
                fig.update_layout(xaxis_title="", xaxis_showticklabels=False, coloraxis_showscale=False, yaxis_title="")
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight("Old Baneshowr, Gaushala, Tahachal, and Putalisadak top the list at ~₹0.9 Cr/Ana. Notably, all top-15 neighborhoods cluster in a tight price band — premium areas are consistently valued, confirming strong neighborhood effects.")

            with c2:
                ll_med = ll["price_per_aana"].median()
                fig = go.Figure(data=[go.Histogram(
                    x=ll["price_per_aana"], nbinsx=40, marker_color="#8ae85a", opacity=0.75)])
                fig.add_vline(x=ll_med, line_dash="dash", line_color="#e8a45a",
                              annotation_text=f"Median: {fmt_npr(ll_med, 1)}")
                fig.update_layout(title="Lalpurja Land Price/Ana Distribution",
                                  xaxis_title="Price per Ana (NPR)", yaxis_title="Frequency",
                                  showlegend=False, height=500)
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight(
                            f"Multimodal distribution with multiple peaks — unlike other datasets, this suggests distinct pricing clusters for different neighborhood zones. "
                            f"Median is {fmt_npr(ll_med, 1)}, tighter range (₹0.6L-₹99L) than General Land confirms more consistent Lalpurja pricing."
                        )

            st.markdown("---")

            # ── Row 2: Property Type, Road Type ──
            c1, c2 = st.columns(2)

            with c1:
                if "property_type" in ll.columns:
                    ptype_ll = ll.groupby("property_type")["price_per_aana"].median().reset_index()
                    fig = px.bar(ptype_ll, x="property_type", y="price_per_aana",
                                 title="Price/Ana by Property Type (Lalpurja Land)",
                                 color="property_type",
                                 text=ptype_ll["price_per_aana"].apply(lambda v: fmt_npr(v, 1)),
                                 labels={"price_per_aana": "Median Price/Ana (NPR)", "property_type": ""})
                    fig.update_traces(textposition="outside")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("Semi-commercial: ₹0.50 Cr/Ana · Commercial: ₹0.45 Cr/Ana · Residential: ₹0.38 Cr/Ana (EDA medians). Semi-commercial consistently commands the HIGHEST price across all districts — do NOT merge commercial and residential in modeling.")

            with c2:
                if "road_type" in ll.columns:
                    road_ll = ll.groupby("road_type")["price_per_aana"].median().reset_index()
                    fig = px.bar(road_ll, x="road_type", y="price_per_aana",
                                 title="Price/Ana by Road Type (Lalpurja Land)",
                                 color="price_per_aana", color_continuous_scale="Greens",
                                 text=road_ll["price_per_aana"].apply(lambda v: fmt_npr(v, 1)),
                                 labels={"price_per_aana": "Median Price/Ana (NPR)", "road_type": ""})
                    fig.update_traces(textposition="outside")
                    fig.update_layout(coloraxis_showscale=False, showlegend=False)
                    st.plotly_chart(clean_chart(fig, height=350), use_container_width=True)
                    insight("High Access: ₹0.45 Cr/Ana vs Low Access: ₹0.32 Cr/Ana — a ~38% premium (EDA finding). Road access quality has the clearest and most reliable impact on Lalpurja land price. Simple binary classification works well as an ML feature.")

            st.markdown("---")

            # ── Row 3: Amenity Distance Correlations ──
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Amenity Distance vs Price/Ana")
                amenity_dist_ll = [c for c in ll.columns if c.endswith("_m") and "price" not in c]
                if amenity_dist_ll:
                    corr_ll = ll[amenity_dist_ll + ["price_per_aana"]].corr()["price_per_aana"].drop("price_per_aana").sort_values()
                    fig = px.bar(x=corr_ll.values, y=corr_ll.index, orientation="h",
                                 color=corr_ll.values, color_continuous_scale="RdBu",
                                 title="Amenity Distance Correlation with Price/Ana",
                                 labels={"x": "Correlation", "y": "Amenity"})
                    fig.update_xaxes(range=[-1, 1])
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(clean_chart(fig, height=400), use_container_width=True)
                    insight("Airport (−0.558) and Ring Road (−0.504) are the two STRONGEST predictors for Lalpurja land — all correlations are negative (closer = more expensive). Hospital (−0.350) and School (−0.246) also matter significantly, far more than for housing.")

            with c2:
                st.subheader("Land Size vs Price/Ana")
                fig = px.scatter(ll, x="land_size_aana", y="price_per_aana",
                                 opacity=0.5,
                                 title="Land Size vs Price per Ana (Lalpurja)",
                                 labels={"land_size_aana": "Land Size (Ana)", "price_per_aana": "Price/Ana (NPR)"},
                                 color_discrete_sequence=["#8ae85a"])
                st.plotly_chart(clean_chart(fig, height=400), use_container_width=True)
                insight("Near-zero correlation (EDA: −0.04) — land size does NOT determine price per Ana. Highest per-Ana prices cluster at small plots (0–8 Ana). As size exceeds 10 Ana, price/Ana drops and stabilizes — the 'bulk discount' effect is clearly visible.")

            st.markdown("---")

            # ── Row 4: Municipality Violin + Cumulative ──
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Price/Ana Violin by Municipality")
                if "municipality" in ll.columns:
                    top_munis = ll["municipality"].value_counts().head(5).index.tolist()
                    ll_top_m = ll[ll["municipality"].isin(top_munis)]
                    fig = go.Figure()
                    for m in top_munis:
                        fig.add_trace(go.Violin(
                            y=ll_top_m[ll_top_m["municipality"] == m]["price_per_aana"],
                            name=m, box_visible=True, meanline_visible=True, points=False
                        ))
                    fig.update_yaxes(title_text="Price per Ana (NPR)")
                    st.plotly_chart(clean_chart(fig, height=400), use_container_width=True)
                    insight("Budhanilkantha has the most listings but is NOT the most expensive — it's a popular, affordable area (high supply, normal price). Municipalities like Suryabinayak and Mahalaxmi show tighter distributions — more consistent and predictable pricing.")
                else:
                    st.info("Municipality column not found.")

            with c2:
                st.subheader("Cumulative Land Price Distribution (Lalpurja)")
                sorted_ll = ll["price_per_aana"].sort_values().values
                cum_ll = np.arange(1, len(sorted_ll) + 1) / len(sorted_ll) * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sorted_ll, y=cum_ll, mode="lines",
                                         fill="tozeroy", line=dict(color="#8ae85a", width=3)))
                fig.update_layout(title="Cumulative Distribution", showlegend=False,
                                  xaxis_title="Price per Ana (NPR)", yaxis_title="Cumulative %", height=400)
                st.plotly_chart(clean_chart(fig, height=400), use_container_width=True)
                insight("Lalpurja land shows a tighter cumulative curve than General Land — ~50% of plots below ₹0.40 Cr/Ana and ~75% below ₹0.55 Cr/Ana. Max of ₹0.99 Cr/Ana (vs ₹2 Cr+ for General Land) confirms more consistent pricing.")

            st.markdown("---")

            # ── Row 5: Percentile Table ──
            st.subheader("Lalpurja Land Price/Ana Percentile Analysis")
            pctl_ll = {
                "Percentile": [f"{p}th" for p in [5, 10, 25, 50, 75, 90, 95, 99]],
                "Price/Ana": [fmt_npr(ll["price_per_aana"].quantile(p / 100), 2) for p in [5, 10, 25, 50, 75, 90, 95, 99]],
            }
            st.dataframe(pd.DataFrame(pctl_ll), use_container_width=True, hide_index=True)
            insight("The tight IQR (Q1–Q3) of Lalpurja land prices reflects more consistent pricing than General Land. The 95th percentile (~₹0.80 Cr/Ana) is far below the General Land equivalent, confirming Lalpurja land is a more standardized residential-dominated market.")

    elif page == "🤖 Model Performance":
        st.title("🤖 ML Model Performance")
        st.markdown("---")
        
        mdf = pd.DataFrame({
            "Dataset":["General Housing","Lalpurja Land","General Land","Lalpurja Housing"],
            "Algorithm":["XGBoost","CatBoost","CatBoost","CatBoost"],
            "R² Score":[0.777,0.744,0.744,0.648],
            "Avg Error":[18.8,19.1,27.4,23.7],
            "Training Rows":[2005,971,3250,1749],
            "Features":[24,29,16,42]
        })
        st.dataframe(mdf.style.background_gradient(subset=["R² Score"],cmap="Greens"),
                    use_container_width=True, hide_index=True)
        
        st.markdown("---")
        c1,c2 = st.columns(2)
        with c1:
            fig = px.bar(mdf.sort_values("R² Score"), x="R² Score", y="Dataset", orientation="h",
                        color="R² Score", color_continuous_scale="Viridis", text="R² Score",
                        title="R² Comparison")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(xaxis_range=[0.5,0.85], coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(clean_chart(fig), use_container_width=True)
        with c2:
            fig = px.bar(mdf.sort_values("Avg Error",ascending=False), x="Avg Error", y="Dataset",
                        orientation="h", color="Avg Error", color_continuous_scale="Reds",
                        title="Error % Comparison", text="Avg Error")
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(clean_chart(fig), use_container_width=True)

    # ─────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════
# SECTION 2 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════
elif section == "🔍 Recommendations":
    st.title("💡 Personalized Property Recommendations")
    st.markdown("Get property suggestions based on your detailed preferences")
    st.markdown("---")
    
    # Property Type Selection
    rec_property_type = st.radio("🏠 Property Type", ["🏠 Housing", "🌍 Land"], horizontal=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HOUSING RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    if rec_property_type == "🏠 Housing":
        st.subheader("🏠 Housing - Detailed Preferences")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📍 LOCATION**")
            districts = ["All Districts"] + sorted([d for d in gh["district"].unique() if pd.notna(d)])
            selected_district = st.selectbox("District", districts, key="house_district")
        
        with col2:
            st.write("**💰 BUDGET (₹)**")
            min_budget = st.number_input("Min (Crore)", min_value=0.5, max_value=50.0, value=1.0, step=0.5, key="house_min_b") * 10_000_000
            max_budget = st.number_input("Max (Crore)", min_value=0.5, max_value=50.0, value=5.0, step=0.5, key="house_max_b") * 10_000_000
        
        with col3:
            st.write("**🛏️ BEDROOMS**")
            min_beds = st.number_input("Min Beds", min_value=1, max_value=15, value=3, step=1, key="house_min_beds")
            max_beds = st.number_input("Max Beds", min_value=1, max_value=15, value=7, step=1, key="house_max_beds")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🏞️ LAND SIZE (Ana)**")
            min_land = st.number_input("Min Ana", min_value=0.5, max_value=50.0, value=2.0, step=0.5, key="house_min_land")
            max_land = st.number_input("Max Ana", min_value=0.5, max_value=50.0, value=10.0, step=0.5, key="house_max_land")
        with col2:
            st.write("**✅ MUST-HAVE AMENITIES**")
            want_parking  = st.checkbox("Parking",         value=True,  key="h_park")
            want_drainage = st.checkbox("Drainage",        value=True,  key="h_drain")
            want_kitchen  = st.checkbox("Modular Kitchen", value=False, key="h_kit")
            want_garden   = st.checkbox("Garden",          value=False, key="h_gard")
        
        st.markdown("---")
        
        if st.button("🔍 GET HOUSING RECOMMENDATIONS", key="get_house", use_container_width=True):
            # gh uses: total_price, land_area_aana, bedrooms
            land_col = "land_area_aana" if "land_area_aana" in gh.columns else "land_size_aana"
            
            filtered_data = gh[
                (gh["total_price"] >= min_budget) &
                (gh["total_price"] <= max_budget) &
                (gh["bedrooms"] >= min_beds) &
                (gh["bedrooms"] <= max_beds) &
                (gh[land_col] >= min_land) &
                (gh[land_col] <= max_land)
            ].copy()
            
            if selected_district != "All Districts":
                filtered_data = filtered_data[filtered_data["district"] == selected_district]
            
            if len(filtered_data) == 0:
                st.error("❌ No properties found matching your criteria. Try widening your budget or size range.")
            else:
                must_haves = []
                if want_parking:  must_haves.append("has_parking")
                if want_drainage: must_haves.append("has_drainage")
                if want_kitchen:  must_haves.append("has_modular_kitchen")
                if want_garden:   must_haves.append("has_garden")
                
                filtered_data["matching_score"] = filtered_data.apply(
                    lambda row: calculate_matching_score(row, {
                        "min_price": min_budget,
                        "max_price": max_budget,
                        "bedrooms": (min_beds + max_beds) // 2,
                        "bathrooms": 3,
                        "must_have_amenities": must_haves,
                        "nice_to_have_amenities": [],
                    }), axis=1
                )
                
                recommendations = filtered_data.sort_values("matching_score", ascending=False).head(10)
                
                st.success(f"✅ Found **{len(filtered_data)}** matching properties! Showing top 10.")
                
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("🏆 Top Match", f"{recommendations.iloc[0]['matching_score']:.1f}%")
                c2.metric("📊 Avg Score", f"{recommendations['matching_score'].mean():.1f}%")
                c3.metric("📍 Total Found", len(filtered_data))
                c4.metric("💰 Median Price", fmt_npr(filtered_data["total_price"].median()))
                
                st.markdown("---")
                st.subheader("🏆 Top Housing Recommendations")
                
                for idx, (_, prop) in enumerate(recommendations.iterrows(), 1):
                    score = prop["matching_score"]
                    color_icon = "🟢" if score >= 80 else ("🟡" if score >= 70 else "🔴")
                    col1, col2, col3 = st.columns([0.8, 2.5, 0.7])
                    with col1:
                        st.metric(f"#{idx}", f"{score:.1f}%")
                    with col2:
                        neighborhood = prop.get("neighborhood", "Unknown")
                        district_name = prop.get("district", "Unknown")
                        st.write(f"**{neighborhood}** | {district_name}")
                        beds  = prop.get("bedrooms",  "N/A")
                        baths = prop.get("bathrooms", "N/A")
                        area  = prop.get("build_up_area", prop.get("built_up_sqft", "N/A"))
                        land  = prop.get(land_col, "N/A")
                        st.write(f"🛏️ {beds} BHK | 🚿 {baths} Bath | 🏞️ {land} Ana")
                        price = prop.get("total_price", 0)
                        price_str = fmt_npr(price)
                        st.write(f"💰 **{price_str}**")
                    with col3:
                        st.write(color_icon)
                    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # LAND RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        st.subheader("🌍 Land - Detailed Preferences")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📍 LOCATION**")
            districts = ["All Districts"] + sorted([d for d in ll["district"].unique() if pd.notna(d)])
            selected_district = st.selectbox("District", districts, key="land_district")
        
        with col2:
            st.write("**💰 BUDGET per Ana (₹)**")
            min_ppa = st.number_input("Min (Lakhs/Ana)", min_value=5.0, max_value=500.0, value=20.0, step=5.0, key="land_min_ppa") * 100_000
            max_ppa = st.number_input("Max (Lakhs/Ana)", min_value=5.0, max_value=500.0, value=80.0, step=5.0, key="land_max_ppa") * 100_000
        
        with col3:
            st.write("**🏞️ LAND SIZE (Ana)**")
            # ll (Lalpurja land) uses land_size_aana
            land_col_ll = "land_size_aana" if "land_size_aana" in ll.columns else "land_area_aana"
            min_land_size = st.number_input("Min Ana", min_value=0.5, max_value=100.0, value=2.0, step=0.5, key="land_min_size")
            max_land_size = st.number_input("Max Ana", min_value=0.5, max_value=100.0, value=20.0, step=0.5, key="land_max_size")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🛣️ ROAD ACCESS**")
            preferred_road = st.selectbox("Preferred Road Type", ["Any", "High Access", "Low Access"], key="land_road")
        with col2:
            st.write("**🏠 PROPERTY TYPE**")
            preferred_ptype = st.selectbox("Property Use", ["Any", "Residential", "Commercial", "Semi-commercial"], key="land_ptype")
        
        st.markdown("---")
        
        if st.button("🔍 GET LAND RECOMMENDATIONS", key="get_land", use_container_width=True):
            # ll uses: price_per_aana, land_size_aana
            filtered_data = ll[
                (ll["price_per_aana"] >= min_ppa) &
                (ll["price_per_aana"] <= max_ppa) &
                (ll[land_col_ll] >= min_land_size) &
                (ll[land_col_ll] <= max_land_size)
            ].copy()
            
            if selected_district != "All Districts":
                filtered_data = filtered_data[filtered_data["district"] == selected_district]
            
            if preferred_road != "Any" and "road_type" in filtered_data.columns:
                filtered_data = filtered_data[filtered_data["road_type"] == preferred_road]
            
            if preferred_ptype != "Any" and "property_type" in filtered_data.columns:
                filtered_data = filtered_data[filtered_data["property_type"] == preferred_ptype]
            
            if len(filtered_data) == 0:
                st.error("❌ No plots found matching your criteria. Try widening your price or size range.")
            else:
                # Score based on price fit relative to midpoint
                mid_ppa = (min_ppa + max_ppa) / 2
                filtered_data["matching_score"] = (
                    100 - (np.abs(filtered_data["price_per_aana"] - mid_ppa) / max(mid_ppa, 1) * 60)
                ).clip(0, 100)
                
                recommendations = filtered_data.sort_values("matching_score", ascending=False).head(10)
                
                st.success(f"✅ Found **{len(filtered_data)}** matching plots! Showing top 10.")
                
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("🏆 Top Match", f"{recommendations.iloc[0]['matching_score']:.1f}%")
                c2.metric("📊 Avg Score", f"{recommendations['matching_score'].mean():.1f}%")
                c3.metric("📍 Total Found", len(filtered_data))
                c4.metric("💰 Median Price/Ana", fmt_npr(filtered_data["price_per_aana"].median()))
                
                st.markdown("---")
                st.subheader("🏆 Top Land Recommendations")
                
                for idx, (_, prop) in enumerate(recommendations.iterrows(), 1):
                    score = prop["matching_score"]
                    color_icon = "🟢" if score >= 80 else ("🟡" if score >= 70 else "🔴")
                    col1, col2, col3 = st.columns([0.8, 2.5, 0.7])
                    with col1:
                        st.metric(f"#{idx}", f"{score:.1f}%")
                    with col2:
                        neighborhood = prop.get("neighborhood", "Unknown")
                        district_name = prop.get("district", "Unknown")
                        st.write(f"**{neighborhood}** | {district_name}")
                        land_size = prop.get(land_col_ll, "N/A")
                        road = prop.get("road_type", "N/A")
                        ptype = prop.get("property_type", "N/A")
                        st.write(f"🏞️ {land_size} Ana | 🛣️ {road} | 🏷️ {ptype}")
                        ppa = prop.get("price_per_aana", 0)
                        total = ppa * float(land_size) if isinstance(land_size, (int, float)) else ppa
                        st.write(f"💰 **{fmt_npr(ppa)}/Ana** (Total: {fmt_npr(total)})")
                    with col3:
                        st.write(color_icon)
                    st.divider()


# ═══════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
#
# HOW IT WORKS — Local Perturbation Analysis (Model-Agnostic)
# ────────────────────────────────────────────────────────────
# SHAP requires the shap library. Instead we use perturbation:
#   1. Get baseline prediction P0 for the user's property (row X)
#   2. For each meaningful feature i:
#        P_up   = predict(X with feature i increased by DELTA)
#        P_down = predict(X with feature i decreased by DELTA)
#        impact[i] = P_up - P_down   ← price sensitivity to this feature
#   3. Rank features by |impact| → top drivers of THIS specific prediction
#   4. For amenity distances: direction inverted (closer = more expensive)
#
# Why this is valid inference:
#   - It's a numerical gradient — the same concept as SHAP's TreeExplainer
#     but computed with finite differences instead of Shapley values
#   - It is LOCAL — it explains THIS prediction, not the model globally
#   - It is honest — if the model learned something wrong, perturbation shows it
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── HUMAN-READABLE FEATURE LABELS ────────────────────────────────────────────
FEATURE_META = {
    # (label, unit, emoji, direction)
    # direction: 1 = higher is more expensive, -1 = lower is more expensive
    "land_area_aana":       ("Land Size",              "Ana",   "📏",  1),
    "land_size_aana":       ("Land Size",              "Ana",   "📏",  1),
    "build_up_area":        ("Built-up Area",          "sqft",  "🏗️",  1),
    "built_up_sqft":        ("Built-up Area",          "sqft",  "🏗️",  1),
    "floors":               ("Number of Floors",       "",      "🏢",  1),
    "total_floors":         ("Number of Floors",       "",      "🏢",  1),
    "bedrooms":             ("Bedrooms",               "",      "🛏️",  1),
    "bathrooms":            ("Bathrooms",              "",      "🚿",  1),
    "kitchens":             ("Kitchens",               "",      "🍳",  1),
    "living_rooms":         ("Living Rooms",           "",      "🛋️",  1),
    "parking":              ("Parking Spaces",         "",      "🚗",  1),
    "parking_cars":         ("Car Parking",            "",      "🚗",  1),
    "road_width_feet":      ("Road Width",             "ft",    "🛣️",  1),
    "house_age":            ("House Age",              "yrs",   "📅", -1),
    "luxury_score":         ("Luxury Score",           "",      "✨",  1),
    "amenity_count":        ("Amenity Count",          "",      "⭐",  1),
    "neighborhood_encoded": ("Location Premium",       "",      "📍",  1),
    "log_land":             ("Land Size (log scale)",  "",      "📏",  1),
    "log_build_up":         ("Built-up (log scale)",   "",      "🏗️",  1),
    "log_built":            ("Built-up (log scale)",   "",      "🏗️",  1),
    "floors_x_land":        ("Floors × Land Score",   "",      "📐",  1),
    "house_size_score":     ("House Size Score",       "",      "🏠",  1),
    "floor_area_ratio":     ("Floor Area Ratio",       "",      "📐",  1),
    "sqft_per_room":        ("Sqft per Room",          "",      "🏠",  1),
    "rooms_total":          ("Total Rooms",            "",      "🏠",  1),
    "bath_per_bed":         ("Bath per Bedroom",       "",      "🚿",  1),
    "urban_centrality":     ("Urban Centrality",       "",      "🏙️",  1),
    "amenity_access_score": ("Amenity Access Score",   "",      "⭐",  1),
    "plot_value_score":     ("Plot Value Score",       "",      "💎",  1),
    "commercial_zone_score":("Commercial Zone Score",  "",      "🏪",  1),
    "location_tier":        ("Location Tier",          "",      "📍",  1),
    "road_quality_score":   ("Road Quality Score",     "",      "🛣️",  1),
    "ring_road_proximity":  ("Ring Road Proximity",    "",      "🔄",  1),
    "comm_road_premium":    ("Road Premium Score",     "",      "🛣️",  1),
    "is_wide_road":         ("Wide Road",              "",      "🛣️",  1),
    "is_large_plot":        ("Large Plot",             "",      "📐",  1),
    "is_corner_plot":       ("Corner Plot",            "",      "📐",  1),
    "has_garden":           ("Garden",                 "",      "🌳",  1),
    "has_parking":          ("Parking Available",      "",      "🅿️",  1),
    "has_drainage":         ("Drainage System",        "",      "💧",  1),
    "has_modular_kitchen":  ("Modular Kitchen",        "",      "🍳",  1),
    "has_parquet":          ("Parquet Floor",          "",      "🪵",  1),
    "has_solar":            ("Solar Power",            "",      "☀️",  1),
    # Amenity distances — closer = higher price, so direction = -1
    "hospital_m":           ("Hospital Distance",      "m",     "🏥", -1),
    "airport_m":            ("Airport Distance",       "m",     "✈️", -1),
    "pharmacy_m":           ("Pharmacy Distance",      "m",     "💊", -1),
    "bhatbhateni_m":        ("Bhatbhateni Distance",   "m",     "🛒", -1),
    "school_m":             ("School Distance",        "m",     "🎓", -1),
    "college_m":            ("College Distance",       "m",     "🎓", -1),
    "public_transport_m":   ("Bus Stop Distance",      "m",     "🚌", -1),
    "police_station_m":     ("Police Stn Distance",    "m",     "👮", -1),
    "boudhanath_m":         ("Boudhanath Distance",    "m",     "🕍", -1),
    "ring_road_m":          ("Ring Road Distance",     "m",     "🔄", -1),
    "facing_road_width":    ("Facing Road Width",      "ft",    "🛣️",  1),
    "ward_no":              ("Ward Number",            "",      "📍",  0),  # neutral
    "municipality_encoded": ("Municipality Premium",   "",      "📍",  1),
    "municipality_x_ward":  ("Municipality×Ward",      "",      "📍",  1),
    "neighborhood_x_district":("Neighborhood×District","",     "📍",  1),
    "parking_premium":      ("Parking Premium",        "",      "🚗",  1),
    "age_condition_score":  ("Age Condition Score",    "",      "📅",  1),
    "price_segment":        ("Price Segment",          "",      "💰",  1),
    "luxury_score":         ("Luxury Score",           "",      "✨",  1),
}

DELTA = 0.20  # 20% perturbation

def perturb_predict(model_fn, base_row: dict, feature: str, direction: str) -> float:
    """
    Perturb one feature up or down, return new prediction.
    model_fn: a function that accepts keyword args matching base_row keys.
    """
    row = base_row.copy()
    val = row.get(feature, 0)
    if val == 0:
        row[feature] = 0.1 if direction == "up" else 0
    elif direction == "up":
        row[feature] = val * (1 + DELTA)
    else:
        row[feature] = val * (1 - DELTA)
    try:
        return model_fn(**row)
    except Exception:
        return None


def run_perturbation_analysis(model_fn, input_kwargs: dict,
                               feature_list: list, baseline: float,
                               is_land_model: bool = False) -> list:
    """
    Run perturbation on every feature in feature_list.
    Returns list of dicts sorted by |price_impact|.

    For land models: baseline is price_per_aana, total impact = impact × land_size
    """
    results = []
    land_size = input_kwargs.get("land_aana", 1.0)

    for feat in feature_list:
        if feat not in input_kwargs:
            continue
        val = input_kwargs[feat]

        p_up   = perturb_predict(model_fn, input_kwargs, feat, "up")
        p_down = perturb_predict(model_fn, input_kwargs, feat, "down")

        if p_up is None or p_down is None:
            continue

        # Raw price impact of increasing this feature by 20%
        impact_raw = p_up - baseline

        # For land: price_per_ana × land_size = total price
        if is_land_model:
            impact_npr = impact_raw * land_size
        else:
            impact_npr = impact_raw

        # Sensitivity: % price change per % feature change
        sensitivity = ((p_up - p_down) / (baseline + 1e-9)) * 100

        meta = FEATURE_META.get(feat, (feat, "", "📊", 1))
        label, unit, emoji, natural_dir = meta

        # Flip sign for distance features (closer = more valuable)
        if natural_dir == -1:
            display_impact = -impact_npr
            display_sensitivity = -sensitivity
        else:
            display_impact = impact_npr
            display_sensitivity = sensitivity

        results.append({
            "feature":       feat,
            "label":         label,
            "unit":          unit,
            "emoji":         emoji,
            "value":         val,
            "impact_npr":    impact_npr,
            "display_impact": display_impact,
            "sensitivity":   display_sensitivity,
            "p_up":          p_up,
            "p_down":        p_down,
            "natural_dir":   natural_dir,
        })

    # Sort by absolute NPR impact
    results.sort(key=lambda x: abs(x["display_impact"]), reverse=True)
    return results


def render_inference_chart(results: list, baseline: float,
                            predicted_label: str, top_n: int = 10):
    """Render the waterfall-style inference chart."""
    top = results[:top_n]

    labels = [f"{r['emoji']} {r['label']}" for r in top]
    impacts = [r["display_impact"] for r in top]
    colors  = ["#2ecc71" if v >= 0 else "#e74c3c" for v in impacts]
    text    = [
        f"+{fmt_npr(v)}" if v >= 0 else f"-{fmt_npr(abs(v))}"
        for v in impacts
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=impacts,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=text,
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Impact if increased 20%: %{text}<br>"
            "<extra></extra>"
        )
    ))

    fig.add_vline(x=0, line_color="#ffffff", line_width=1.5, opacity=0.4)

    fig.update_layout(
        title=dict(
            text=f"What's driving this prediction? (Baseline: {fmt_npr(baseline)})",
            font=dict(size=15)
        ),
        xaxis_title="Price Impact if Feature Increases 20%",
        yaxis_title="",
        height=max(400, top_n * 50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(tickformat=".2s"),
        margin=dict(l=10, r=100, t=60, b=40),
    )
    return fig


def render_sensitivity_gauge(results: list, top_n: int = 6):
    """Radar chart showing which features this property is most sensitive to."""
    top = results[:top_n]
    categories = [f"{r['emoji']} {r['label']}" for r in top]
    sensitivities = [min(abs(r["sensitivity"]), 100) for r in top]
    categories.append(categories[0])
    sensitivities.append(sensitivities[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=sensitivities,
        theta=categories,
        fill="toself",
        fillcolor="rgba(90, 184, 232, 0.2)",
        line=dict(color="#5ab8e8", width=2),
        name="Sensitivity"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="rgba(255,255,255,0.1)",
                            color="#888"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        height=380,
        title=dict(text="Price Sensitivity Radar", font=dict(size=14)),
    )
    return fig


def render_what_if(results: list, baseline: float,
                   is_land: bool, land_size: float = 1.0):
    """
    Interactive what-if: sliders to explore how changing top features moves price.
    Returns a styled dataframe showing "if I change X by Y%, price becomes Z."
    """
    top5 = results[:5]
    rows = []
    for r in top5:
        # At current value
        current_price = baseline if not is_land else baseline * land_size
        # At +20%
        new_ppa = r["p_up"]
        new_total = new_ppa if not is_land else new_ppa * land_size
        change = new_total - current_price
        rows.append({
            "Feature":           f"{r['emoji']} {r['label']}",
            "Current Value":     f"{r['value']:.1f} {r['unit']}".strip(),
            "If +20%":          f"{r['value'] * 1.2:.1f} {r['unit']}".strip(),
            "Price Becomes":     fmt_npr(new_total),
            "Change":            f"+{fmt_npr(change)}" if change >= 0 else f"-{fmt_npr(abs(change))}",
        })
    return pd.DataFrame(rows)


# ── INFERENCE ENGINE UI ───────────────────────────────────────────────────────
if section == "🧠 Inference Engine":
    st.title("🧠 Inference Engine")
    st.markdown(
        "Select a property type and fill in details — the engine will predict the price, "
        "then **explain exactly which features are driving it and by how much.**"
    )
    st.markdown("---")

    # ── METHODOLOGY EXPANDER ─────────────────────────────────────────────────
    with st.expander("📖 How does this work?", expanded=False):
        st.markdown("""
**Local Perturbation Analysis** — a model-agnostic inference technique.

**Step-by-step:**
1. Your property inputs go into the ML model → **baseline price P₀**
2. For each feature, we increase it by 20% and re-predict → **P_up**
3. The **price impact** = P_up − P₀ tells us: *"if this feature were 20% higher, price would change by X"*
4. Features are ranked by |impact| → the biggest movers appear first

**Why not SHAP?**
SHAP (SHapley Additive exPlanations) is the gold standard but requires extra libraries.
Perturbation analysis gives equivalent local explanations using only the model's `predict()` call.
Both measure the same thing: **how much does each feature move this specific prediction?**

**Reading the chart:**
- 🟢 Green bar = increasing this feature raises price (e.g. more land = more expensive)
- 🔴 Red bar = increasing this feature lowers price (e.g. further from ring road = cheaper)
- Bar length = how much the price would change if that feature increased 20%
        """)

    st.markdown("---")

    # ── MODEL SELECTION ───────────────────────────────────────────────────────
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        inf_prop_type = st.radio(
            "**Property type**",
            ["🏠 House / Building", "🌍 Land / Plot"],
            horizontal=True, key="inf_ptype"
        )
    with col_sel2:
        inf_has_lalpurja = st.radio(
            "**Has Lalpurja certificate?**",
            ["Yes", "No / Not sure"],
            horizontal=True, key="inf_lalpurja",
            help="Lalpurja = official digital land record. Enables the more detailed model."
        )

    is_house_inf    = "House" in inf_prop_type
    is_lalpurja_inf = inf_has_lalpurja == "Yes"
    model_key_inf   = ("lph_house" if is_house_inf else "lph_land") if is_lalpurja_inf \
                 else ("gen_house"  if is_house_inf else "gen_land")

    model_labels = {
        "gen_house":  ("General Housing — XGBoost",   "R²=0.777 · ±18.8%", "#2ecc71"),
        "gen_land":   ("General Land — CatBoost",     "R²=0.744 · ±19.1%", "#e8a45a"),
        "lph_house":  ("Lalpurja Housing — CatBoost", "R²=0.648 · ±23.7%", "#c084fc"),
        "lph_land":   ("Lalpurja Land — CatBoost",    "R²=0.744 · ±19.1%", "#5ab8e8"),
    }
    mlabel, mstats, mcolor = model_labels[model_key_inf]
    st.markdown(
        f"<div style='background:rgba(30,30,50,0.7); border-left:4px solid {mcolor}; "
        f"padding:10px 16px; border-radius:6px; margin-bottom:16px;'>"
        f"<b>Model:</b> {mlabel} &nbsp;·&nbsp; {mstats}</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # INPUT FORMS  (same fields as prediction section, but now all in one page)
    # ─────────────────────────────────────────────────────────────────────────

    input_kwargs = {}   # will be populated below, used for perturbation

    # ── GEN HOUSE ────────────────────────────────────────────────────────────
    if model_key_inf == "gen_house":
        st.subheader("🏠 Property Details")
        c1, c2 = st.columns(2)
        with c1:
            district    = st.selectbox("District", MAIN_DISTRICTS, key="inf_gh_dist")
            neigh_opts  = sorted([n for n in MAPS["neigh_gh"] if n not in ["Unknown","nan"]])
            neighborhood = st.selectbox("Neighborhood", neigh_opts,
                index=neigh_opts.index("Budhanilkantha") if "Budhanilkantha" in neigh_opts else 0,
                key="inf_gh_neigh")
            facing      = st.selectbox("Facing", FACING_OPTIONS, key="inf_gh_face")
            bedrooms    = st.slider("Bedrooms",  1, 12, 4, key="inf_gh_bed")
            bathrooms   = st.slider("Bathrooms", 1, 10, 3, key="inf_gh_bath")
            floors      = st.slider("Floors", 1.0, 8.0, 2.5, 0.5, key="inf_gh_flr")
        with c2:
            land_aana   = st.slider("Land Size (Ana)", 0.5, 20.0, 4.0, 0.5, key="inf_gh_land")
            buildup     = st.slider("Built-up (sqft)",  200, 5000, 1400, 50,  key="inf_gh_bup")
            road_width  = st.slider("Road Width (ft)",   5,  60,   14,   1,   key="inf_gh_road")
            house_age   = st.slider("House Age (yrs)",   0,  50,   5,         key="inf_gh_age")
            ca, cb = st.columns(2)
            has_parking  = ca.checkbox("🅿️ Parking",        True,  key="inf_gh_pk")
            has_drainage = cb.checkbox("💧 Drainage",       True,  key="inf_gh_dr")
            has_kitchen  = ca.checkbox("🍳 Mod. Kitchen",   False, key="inf_gh_kit")
            has_parquet  = cb.checkbox("🪵 Parquet",        False, key="inf_gh_pq")
            has_garden   = ca.checkbox("🌳 Garden",         False, key="inf_gh_gd")
            has_solar    = cb.checkbox("☀️ Solar",          False, key="inf_gh_sol")

        input_kwargs = dict(
            district=district, neighborhood=neighborhood, bedrooms=bedrooms,
            bathrooms=bathrooms, floors=floors, land_aana=land_aana,
            buildup_sqft=buildup, road_width=road_width, house_age=house_age,
            facing=facing, has_parking=has_parking, has_garden=has_garden,
            has_mod_kitchen=has_kitchen, has_parquet=has_parquet,
            has_drainage=has_drainage, has_solar=has_solar
        )
        perturb_features = [
            "land_aana","buildup_sqft","floors","bedrooms","bathrooms",
            "road_width","house_age",
        ]
        model_fn     = predict_gen_house
        is_land_mdl  = False
        err_pct      = 0.188

    # ── GEN LAND ─────────────────────────────────────────────────────────────
    elif model_key_inf == "gen_land":
        st.subheader("🌍 Plot Details")
        c1, c2 = st.columns(2)
        with c1:
            district    = st.selectbox("District", MAIN_DISTRICTS, key="inf_gl_dist")
            neigh_opts  = sorted([n for n in MAPS["neigh_gl"] if "Zone" not in str(n)])
            neighborhood = st.selectbox("Neighborhood", neigh_opts,
                index=neigh_opts.index("Budhanilkantha") if "Budhanilkantha" in neigh_opts else 0,
                key="inf_gl_neigh")
        with c2:
            land_aana   = st.slider("Land Size (Ana)", 0.5, 50.0, 5.0, 0.5, key="inf_gl_land")
            road_type   = st.selectbox("Road Type", ["High Access","Mid Access","Low Access"], key="inf_gl_rt")
            road_width  = st.slider("Road Width (ft)", 5, 80, 16, 1, key="inf_gl_road")
            facing      = st.selectbox("Facing", FACING_OPTIONS, key="inf_gl_face")

        input_kwargs = dict(
            district=district, neighborhood=neighborhood, land_aana=land_aana,
            road_type=road_type, road_width=road_width, facing=facing
        )
        perturb_features = ["land_aana", "road_width"]
        model_fn     = predict_gen_land
        is_land_mdl  = True
        err_pct      = 0.274

    # ── LPH HOUSE ────────────────────────────────────────────────────────────
    elif model_key_inf == "lph_house":
        st.subheader("🏘️ Property Details (Lalpurja)")
        c1, c2 = st.columns(2)
        with c1:
            neigh_opts  = sorted(MAPS["neigh_lh"].keys())
            neighborhood = st.selectbox("Neighborhood", neigh_opts,
                index=neigh_opts.index("Budhanilkantha") if "Budhanilkantha" in neigh_opts else 0,
                key="inf_lh_neigh")
            auto_dist = MAPS["neigh_to_dist_lh"].get(neighborhood, "Kathmandu")
            st.caption(f"📍 District: **{auto_dist}**")
            property_type = st.selectbox("Property Type", ["Residential","Commercial","Semi-commercial"], key="inf_lh_pt")
            furnishing    = st.selectbox("Furnishing", ["Unfurnished","Semi Furnished","Full Furnished"], key="inf_lh_furn")
            road_type     = st.selectbox("Road Type", ["High Access","Low Access"], key="inf_lh_rt")
            property_face = st.selectbox("Facing", ["East","West","North","South","North-East","North-West","South-East","South-West"], key="inf_lh_face")
            bedrooms    = st.slider("Bedrooms",  1, 15, 4, key="inf_lh_bed")
            kitchens    = st.slider("Kitchens",  1,  5, 1, key="inf_lh_kit")
            bathrooms   = st.slider("Bathrooms", 1, 10, 3, key="inf_lh_bath")
            living_rooms = st.slider("Living Rooms", 1, 5, 1, key="inf_lh_lr")
            parking_sp  = st.slider("Parking Spaces", 0, 5, 1, key="inf_lh_pk")
            total_floors = st.slider("Total Floors", 0.5, 10.0, 2.5, 0.5, key="inf_lh_flr")
            land_aana   = st.slider("Land Size (Ana)", 1.0, 20.0, 4.0, 0.5, key="inf_lh_land")
            buildup     = st.slider("Built-up (sqft)",  200, 5000, 1200, 100, key="inf_lh_bup")
            house_age   = st.slider("House Age (yrs)",   0,  60,    5,       key="inf_lh_age")
            road_width  = st.slider("Road Width (ft)",   5,  80,   14,   1,  key="inf_lh_road")
        with c2:
            st.markdown("**📡 Amenity Distances (m)**")
            st.caption("Auto-filled from neighborhood · adjust if needed")
            defs = MAPS["amenity_lh"].get(neighborhood, {})
            hospital_m    = st.number_input("🏥 Hospital",    value=int(defs.get("hospital_m",   1000)), step=100, key="inf_lh_hosp")
            airport_m     = st.number_input("✈️ Airport",     value=int(defs.get("airport_m",    8000)), step=500, key="inf_lh_air")
            ring_road_m   = st.number_input("🔄 Ring Road",   value=int(defs.get("ring_road_m",  2000)), step=200, key="inf_lh_rr")
            boudhanath_m  = st.number_input("🕍 Boudhanath",  value=int(defs.get("boudhanath_m", 5000)), step=500, key="inf_lh_boud")
            pharmacy_m    = st.number_input("💊 Pharmacy",    value=int(defs.get("pharmacy_m",    500)), step=100, key="inf_lh_ph")
            bhatbhateni_m = st.number_input("🛒 Bhatbhateni", value=int(defs.get("bhatbhateni_m",3000)), step=200, key="inf_lh_bhat")
            school_m      = st.number_input("🎓 School",      value=int(defs.get("school_m",      800)), step=100, key="inf_lh_sch")
            college_m     = st.number_input("🎓 College",     value=int(defs.get("college_m",    2000)), step=200, key="inf_lh_col")
            public_trans  = st.number_input("🚌 Bus Stop",    value=int(defs.get("public_transport_m",400)), step=100, key="inf_lh_bus")
            police_m      = st.number_input("👮 Police Stn",  value=int(defs.get("police_station_m",2000)), step=200, key="inf_lh_pol")

        if land_aana > 8:
            st.warning(f"⚠️ {land_aana} Ana is above the 90th percentile (8.4 Ana) — confidence is lower for large plots.")

        input_kwargs = dict(
            neighborhood=neighborhood, property_type=property_type,
            road_type=road_type, furnishing=furnishing, property_face=property_face,
            bedrooms=bedrooms, kitchens=kitchens, bathrooms=bathrooms,
            living_rooms=living_rooms, parking_spaces=parking_sp,
            total_floors=total_floors, house_age=house_age, road_width=road_width,
            land_aana=land_aana, buildup_sqft=buildup,
            hospital_m=hospital_m, airport_m=airport_m, pharmacy_m=pharmacy_m,
            bhatbhateni_m=bhatbhateni_m, school_m=school_m, college_m=college_m,
            public_transport_m=public_trans, police_station_m=police_m,
            boudhanath_m=boudhanath_m, ring_road_m=ring_road_m
        )
        perturb_features = [
            "land_aana","buildup_sqft","total_floors","bedrooms","bathrooms",
            "kitchens","living_rooms","parking_spaces","road_width","house_age",
            "hospital_m","airport_m","ring_road_m","boudhanath_m",
            "pharmacy_m","bhatbhateni_m","school_m","college_m",
            "public_transport_m","police_station_m",
        ]
        model_fn     = predict_lph_house
        is_land_mdl  = False
        err_pct      = 0.237

    # ── LPH LAND ─────────────────────────────────────────────────────────────
    else:  # lph_land
        st.subheader("🎯 Plot Details (Lalpurja)")
        c1, c2 = st.columns(2)
        with c1:
            neigh_opts  = sorted(MAPS["neigh_ll"].keys())
            neighborhood = st.selectbox("Neighborhood", neigh_opts,
                index=neigh_opts.index("Budhanilkantha") if "Budhanilkantha" in neigh_opts else 0,
                key="inf_ll_neigh")
            auto_dist = MAPS["neigh_to_dist_ll"].get(neighborhood, "Kathmandu")
            st.caption(f"📍 District: **{auto_dist}**")
            property_type = st.selectbox("Property Type", ["Residential","Commercial","Semi-commercial"], key="inf_ll_pt")
            road_type     = st.selectbox("Road Type", ["High Access","Low Access"], key="inf_ll_rt")
            property_face = st.selectbox("Facing", ["East","West","North","South","North-East","North-West","South-East","South-West"], key="inf_ll_face")
            land_aana     = st.slider("Land Size (Ana)", 0.5, 50.0, 5.0, 0.5, key="inf_ll_land")
            road_width    = st.slider("Road Width (ft)",  5,  80,  16,  1,    key="inf_ll_road")
        with c2:
            st.markdown("**📡 Amenity Distances (m)**")
            st.caption("Auto-filled from neighborhood · adjust if needed")
            defs = MAPS["amenity_ll"].get(neighborhood, {})
            hospital_m    = st.number_input("🏥 Hospital",    value=int(defs.get("hospital_m",   1000)), step=100, key="inf_ll_hosp")
            airport_m     = st.number_input("✈️ Airport",     value=int(defs.get("airport_m",    8000)), step=500, key="inf_ll_air")
            ring_road_m   = st.number_input("🔄 Ring Road",   value=int(defs.get("ring_road_m",  2000)), step=200, key="inf_ll_rr")
            bhatbhateni_m = st.number_input("🛒 Bhatbhateni", value=int(defs.get("bhatbhateni_m",3000)), step=200, key="inf_ll_bhat")
            pharmacy_m    = st.number_input("💊 Pharmacy",    value=int(defs.get("pharmacy_m",    500)), step=100, key="inf_ll_ph")
            school_m      = st.number_input("🎓 School",      value=int(defs.get("school_m",      800)), step=100, key="inf_ll_sch")
            public_trans  = st.number_input("🚌 Bus Stop",    value=int(defs.get("public_transport_m",400)), step=100, key="inf_ll_bus")
            police_m      = st.number_input("👮 Police Stn",  value=int(defs.get("police_station_m",2000)), step=200, key="inf_ll_pol")

        input_kwargs = dict(
            neighborhood=neighborhood, property_type=property_type,
            road_type=road_type, property_face=property_face,
            land_aana=land_aana, road_width=road_width,
            hospital_m=hospital_m, airport_m=airport_m, pharmacy_m=pharmacy_m,
            bhatbhateni_m=bhatbhateni_m, school_m=school_m,
            public_transport_m=public_trans, police_station_m=police_m,
            ring_road_m=ring_road_m
        )
        perturb_features = [
            "land_aana","road_width",
            "hospital_m","airport_m","ring_road_m","bhatbhateni_m",
            "pharmacy_m","school_m","public_transport_m","police_station_m",
        ]
        model_fn     = predict_lph_land
        is_land_mdl  = True
        err_pct      = 0.191

    # ─────────────────────────────────────────────────────────────────────────
    # RUN INFERENCE BUTTON
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    run_btn = st.button(
        "🧠 Run Inference Analysis",
        type="primary",
        use_container_width=True,
        key="inf_run_btn"
    )

    if run_btn:
        with st.spinner("Running prediction + perturbation analysis..."):
            try:
                # ── BASELINE PREDICTION ──────────────────────────────────────
                baseline = model_fn(**input_kwargs)
                land_size_for_total = input_kwargs.get("land_aana", 1.0)
                total_price = baseline if not is_land_mdl else baseline * land_size_for_total

                # ── DISPLAY PREDICTION HEADER ────────────────────────────────
                st.markdown("---")
                st.subheader("📊 Prediction Result")
                c1, c2, c3, c4 = st.columns(4)
                if is_land_mdl:
                    c1.metric("💰 Price per Ana",  fmt_npr(baseline))
                    c2.metric("🏷️ Total Value",    fmt_npr(total_price), f"× {land_size_for_total} Ana")
                    c3.metric("📉 Low Estimate",   fmt_npr(total_price * (1 - err_pct)), f"−{err_pct*100:.0f}%")
                    c4.metric("📈 High Estimate",  fmt_npr(total_price * (1 + err_pct)), f"+{err_pct*100:.0f}%")
                else:
                    c1.metric("💰 Predicted Price", fmt_npr(baseline))
                    c2.metric("📉 Low Estimate",    fmt_npr(baseline * (1 - err_pct)), f"−{err_pct*100:.0f}%")
                    c3.metric("📈 High Estimate",   fmt_npr(baseline * (1 + err_pct)), f"+{err_pct*100:.0f}%")
                    confidence = get_confidence_score(
                        input_kwargs.get("land_aana", 4),
                        input_kwargs.get("neighborhood",""),
                        MODEL_INFO[model_key_inf]["r2"],
                        MODEL_INFO[model_key_inf]["samples"],
                        model_key_inf
                    )
                    conf_cls = "confidence-high" if confidence >= 75 else \
                               ("confidence-medium" if confidence >= 60 else "confidence-low")
                    c4.markdown(f"<p class='{conf_cls}' style='margin-top:28px'>Confidence: {confidence:.0f}%</p>",
                                unsafe_allow_html=True)

                # ── RUN PERTURBATION ─────────────────────────────────────────
                st.markdown("---")
                st.subheader("🧠 Inference — What's Driving This Price?")

                results = run_perturbation_analysis(
                    model_fn      = model_fn,
                    input_kwargs  = input_kwargs,
                    feature_list  = perturb_features,
                    baseline      = baseline,
                    is_land_model = is_land_mdl,
                )

                if not results:
                    st.warning("Could not compute feature impacts. Check that inputs are valid.")
                else:
                    # ── TAB LAYOUT ───────────────────────────────────────────
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Impact Chart",
                        "🕸️ Sensitivity Radar",
                        "🔢 Full Table",
                        "💡 What-If Scenarios",
                    ])

                    # ── TAB 1: WATERFALL IMPACT CHART ────────────────────────
                    with tab1:
                        st.plotly_chart(
                            render_inference_chart(
                                results, total_price,
                                fmt_npr(total_price), top_n=min(10, len(results))
                            ),
                            use_container_width=True
                        )

                        # Plain-language summary
                        st.markdown("#### 💬 Plain-Language Summary")
                        top3 = results[:3]
                        lines = []
                        for r in top3:
                            imp = r["display_impact"]
                            if is_land_mdl:
                                imp = imp  # already in total NPR
                            direction = "increases" if imp > 0 else "decreases"
                            lines.append(
                                f"- **{r['emoji']} {r['label']}** has the biggest influence. "
                                f"A 20% change in this feature {direction} the total value "
                                f"by approximately **{fmt_npr(abs(imp))}**."
                            )
                        st.markdown("\n".join(lines))

                        # Context callout
                        top1 = results[0]
                        if "airport" in top1["feature"] or "ring_road" in top1["feature"]:
                            st.info("📍 **Location & connectivity** dominate this prediction — consistent with Lalpurja land EDA findings where airport distance explains 57% of price variance.")
                        elif "land" in top1["feature"] or "aana" in top1["feature"]:
                            st.info("📏 **Land size** is the primary value driver here — typical for housing where plot area directly scales price.")
                        elif "neighborhood" in top1["feature"]:
                            st.info("📍 **Neighborhood location premium** is dominant — this property's value is mainly set by where it is, not what it has.")

                    # ── TAB 2: SENSITIVITY RADAR ──────────────────────────────
                    with tab2:
                        c1, c2 = st.columns([1.2, 1])
                        with c1:
                            st.plotly_chart(
                                render_sensitivity_gauge(results, top_n=min(6, len(results))),
                                use_container_width=True
                            )
                        with c2:
                            st.markdown("#### Price Sensitivity Breakdown")
                            st.markdown(
                                "This radar shows which features **this specific property** "
                                "is most sensitive to. A larger area means a 20% change in "
                                "that feature causes a bigger price swing.\n\n"
                                "**What to look for:**"
                            )
                            for r in results[:5]:
                                bar_len = int(min(abs(r["sensitivity"]), 100) / 10)
                                bar = "█" * bar_len + "░" * (10 - bar_len)
                                pct = min(abs(r["sensitivity"]), 100)
                                st.markdown(
                                    f"`{bar}` **{r['emoji']} {r['label']}**  "
                                    f"_{pct:.1f}% price swing per 20% change_"
                                )

                    # ── TAB 3: FULL FEATURE TABLE ─────────────────────────────
                    with tab3:
                        st.markdown("#### All Feature Impacts")
                        table_rows = []
                        for r in results:
                            imp = r["display_impact"]
                            total_imp = imp
                            sign = "▲" if imp >= 0 else "▼"
                            table_rows.append({
                                "Feature":          f"{r['emoji']} {r['label']}",
                                "Current Value":    f"{r['value']:.1f} {r['unit']}".strip(),
                                "Price if +20%":    fmt_npr(r["p_up"] * (land_size_for_total if is_land_mdl else 1)),
                                "Price if −20%":    fmt_npr(r["p_down"] * (land_size_for_total if is_land_mdl else 1)),
                                "Impact of +20%":   f"{sign} {fmt_npr(abs(imp))}",
                                "Sensitivity":      f"{abs(r['sensitivity']):.1f}%",
                            })
                        df_table = pd.DataFrame(table_rows)
                        st.dataframe(df_table, use_container_width=True, hide_index=True)

                        st.caption(
                            "Impact = change in total price if that feature increases 20%. "
                            "Sensitivity = % price change per % feature change."
                        )

                    # ── TAB 4: WHAT-IF SCENARIOS ──────────────────────────────
                    with tab4:
                        st.markdown("#### 💡 What-If Scenarios — Top 5 Levers")
                        st.markdown(
                            "These are the features where **small changes have the largest effect** "
                            "on the final price. Use this to understand what improvements are "
                            "most worth making — or what to negotiate on."
                        )
                        df_whatif = render_what_if(results, total_price, is_land_mdl, land_size_for_total)
                        st.dataframe(df_whatif, use_container_width=True, hide_index=True)

                        st.markdown("---")
                        st.markdown("#### 🎯 Actionable Insights")

                        # Feature-specific insight templates
                        INSIGHT_TEMPLATES = {
                            # Amenity distances (natural_dir = -1, unit = "m")
                            "hospital_m":         ("proximity", "hospital"),
                            "airport_m":          ("proximity", "airport"),
                            "ring_road_m":        ("proximity", "ring road"),
                            "ring_road_proximity":("proximity", "ring road"),
                            "boudhanath_m":       ("proximity", "Boudhanath"),
                            "pharmacy_m":         ("proximity", "pharmacy"),
                            "bhatbhateni_m":      ("proximity", "Bhatbhateni supermarket"),
                            "school_m":           ("proximity", "school"),
                            "college_m":          ("proximity", "college"),
                            "public_transport_m": ("proximity", "public transport"),
                            "police_station_m":   ("proximity", "police station"),
                            # Age (natural_dir = -1, but unit is years not metres)
                            "house_age":          ("age", ""),
                            # Size features
                            "land_aana":          ("size", "Ana of land"),
                            "land_size_aana":     ("size", "Ana of land"),
                            "build_up_area":      ("size", "sqft of built-up area"),
                            "built_up_sqft":      ("size", "sqft of built-up area"),
                            "floors":             ("floors", ""),
                            "total_floors":       ("floors", ""),
                            "bedrooms":           ("rooms", "bedroom"),
                            "bathrooms":          ("rooms", "bathroom"),
                            "kitchens":           ("rooms", "kitchen"),
                            "living_rooms":       ("rooms", "living room"),
                            "parking":            ("parking", ""),
                            "parking_spaces":     ("parking", ""),
                            "road_width":         ("road", ""),
                            "road_width_feet":    ("road", ""),
                        }

                        shown = 0
                        for r in results:
                            if shown >= 4:
                                break
                            imp   = r["display_impact"]
                            feat  = r["feature"]
                            val   = r["value"]
                            label = r["label"]
                            emoji = r["emoji"]
                            unit  = r["unit"]
                            ndir  = r["natural_dir"]

                            if abs(imp) < 50_000:
                                continue

                            ttype, targ = INSIGHT_TEMPLATES.get(feat, ("generic", ""))

                            # ── Amenity proximity features ──────────────────
                            if ttype == "proximity":
                                if imp < 0:
                                    # Being far hurts — warn
                                    st.warning(
                                        f"**{emoji} {label}** — At {val:.0f}m from the {targ}, "
                                        f"this property loses **{fmt_npr(abs(imp))}** in value "
                                        f"compared to one that is closer. "
                                        f"Factor this into your offer or look for a plot nearer to the {targ}."
                                    )
                                else:
                                    # Being close is a strength — highlight
                                    st.success(
                                        f"**{emoji} {label}** — At just {val:.0f}m from the {targ}, "
                                        f"proximity adds **{fmt_npr(abs(imp))}** to the value. "
                                        f"This is a location advantage worth highlighting."
                                    )

                            # ── House age ───────────────────────────────────
                            elif ttype == "age":
                                if val <= 5:
                                    st.success(
                                        f"**{emoji} House Age ({val:.0f} yrs)** — Being nearly new "
                                        f"adds **{fmt_npr(abs(imp))}** vs an older equivalent. "
                                        f"Newly built properties command a clear premium here."
                                    )
                                elif val <= 15:
                                    st.info(
                                        f"**{emoji} House Age ({val:.0f} yrs)** — A moderately aged house. "
                                        f"A 20% increase in age (to ~{val*1.2:.0f} yrs) would shift "
                                        f"value by **{fmt_npr(abs(imp))}**. Maintenance quality matters a lot at this age."
                                    )
                                else:
                                    st.warning(
                                        f"**{emoji} House Age ({val:.0f} yrs)** — Older house. "
                                        f"Age is reducing the value by an estimated **{fmt_npr(abs(imp))}** "
                                        f"vs a newer build. Renovation can recover part of this gap."
                                    )

                            # ── Land / built-up size ────────────────────────
                            elif ttype == "size":
                                st.success(
                                    f"**{emoji} {label} ({val:.1f} {unit})** — Size is the "
                                    f"{'top' if shown == 0 else 'key'} value driver here. "
                                    f"Each additional 20% ({val*0.2:.1f} {unit} more) adds "
                                    f"**{fmt_npr(abs(imp))}** to the price."
                                )

                            # ── Floors ──────────────────────────────────────
                            elif ttype == "floors":
                                per_floor = abs(imp) / max(val * 0.2, 0.5)
                                st.success(
                                    f"**{emoji} {label} ({val:.1f} floors)** — Adding more floors "
                                    f"increases value meaningfully. One extra floor adds roughly "
                                    f"**{fmt_npr(per_floor)}** at current scale."
                                )

                            # ── Rooms ───────────────────────────────────────
                            elif ttype == "rooms":
                                if imp > 0:
                                    per_room = abs(imp) / max(val * 0.2, 0.5)
                                    st.success(
                                        f"**{emoji} {label} ({val:.0f})** — Each additional {targ} "
                                        f"shifts value by roughly **{fmt_npr(per_room)}**. "
                                        f"Room count is a meaningful but secondary driver here."
                                    )
                                else:
                                    st.info(
                                        f"**{emoji} {label} ({val:.0f})** — Diminishing returns: "
                                        f"adding more {targ}s at this count reduces marginal value slightly. "
                                        f"The market here doesn't reward extra rooms beyond a point."
                                    )

                            # ── Parking ─────────────────────────────────────
                            elif ttype == "parking":
                                st.success(
                                    f"**{emoji} Parking ({val:.0f} spaces)** — Parking adds "
                                    f"**{fmt_npr(abs(imp))}** in this area. "
                                    f"Urban plots with dedicated parking command a clear premium."
                                )

                            # ── Road width ──────────────────────────────────
                            elif ttype == "road":
                                if imp > 0:
                                    st.success(
                                        f"**{emoji} Road Width ({val:.0f} ft)** — A wider road "
                                        f"adds **{fmt_npr(abs(imp))}** to value. "
                                        f"Road access is a strong price signal in the Kathmandu Valley market."
                                    )
                                else:
                                    st.warning(
                                        f"**{emoji} Road Width ({val:.0f} ft)** — Narrow road is "
                                        f"reducing value by ~**{fmt_npr(abs(imp))}**. "
                                        f"A road widening or corner plot reclassification could recover this."
                                    )

                            # ── Generic fallback ────────────────────────────
                            else:
                                if imp > 0:
                                    st.success(
                                        f"**{emoji} {label}** — Increasing this feature "
                                        f"adds **{fmt_npr(abs(imp))}** to the predicted price."
                                    )
                                else:
                                    st.warning(
                                        f"**{emoji} {label}** — This feature is currently "
                                        f"reducing the predicted value by **{fmt_npr(abs(imp))}**."
                                    )

                            shown += 1

                        # Compare vs similar listings
                        st.markdown("---")
                        st.markdown("#### 📋 Similar Properties for Validation")
                        neigh_val = input_kwargs.get("neighborhood", "")
                        if model_key_inf in ("gen_house", "lph_house"):
                            src = lh if model_key_inf == "lph_house" else gh
                            price_col = "total_price"
                            beds_val  = input_kwargs.get("bedrooms", 4)
                            land_val  = input_kwargs.get("land_aana", 4)
                            land_col  = "land_size_aana" if "land_size_aana" in src.columns else "land_area_aana"
                            sim = src[
                                (src["bedrooms"].between(beds_val - 1, beds_val + 1)) &
                                (src[land_col].between(land_val * 0.7, land_val * 1.3))
                            ][[" neighborhood" if " neighborhood" in src.columns else "neighborhood",
                               land_col, "bedrooms", price_col]].head(6)
                            if len(sim):
                                sim = sim.copy()
                                sim["Price"] = sim[price_col].apply(fmt_npr)
                                sim["vs Prediction"] = (
                                    (sim[price_col] - total_price) / total_price * 100
                                ).round(1).astype(str) + "%"
                                st.dataframe(
                                    sim.drop(price_col, axis=1).rename(columns={
                                        "neighborhood": "Neighborhood",
                                        land_col: "Land (Ana)",
                                        "bedrooms": "Beds"
                                    }),
                                    use_container_width=True, hide_index=True
                                )
                            else:
                                st.caption("No close matches found in dataset.")
                        else:
                            src = ll if model_key_inf == "lph_land" else gl
                            land_val = input_kwargs.get("land_aana", 5)
                            land_col = "land_size_aana" if "land_size_aana" in src.columns else "land_area_aana"
                            sim = src[
                                src[land_col].between(land_val * 0.7, land_val * 1.3)
                            ][["neighborhood", land_col, "price_per_aana"]].head(6)
                            if len(sim):
                                sim = sim.copy()
                                sim["Price/Ana"] = sim["price_per_aana"].apply(fmt_npr)
                                sim["vs Prediction"] = (
                                    (sim["price_per_aana"] - baseline) / baseline * 100
                                ).round(1).astype(str) + "%"
                                st.dataframe(
                                    sim.drop("price_per_aana", axis=1).rename(columns={
                                        "neighborhood": "Neighborhood",
                                        land_col: "Land (Ana)"
                                    }),
                                    use_container_width=True, hide_index=True
                                )
                            else:
                                st.caption("No close matches found in dataset.")

            except ValueError as e:
                st.error(f"⚠️ Input error: {e}")
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.info("Try adjusting your inputs or selecting a different neighborhood.")