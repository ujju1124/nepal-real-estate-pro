"""
Nepal Real Estate Analytics & Prediction — FINAL VERSION
=========================================================
Sections:
  1. 📊 Market Analytics  — EDA dashboards
  2. 🧠 Inference Engine  — Price prediction + perturbation analysis
  3. 🔍 Recommendations   — Personalised property search
  4. 💬 Property Assistant — RAG chatbot (LangChain + FAISS + HuggingFace + OpenAI)

Files needed (same folder as app_final.py):
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

Optional (for RAG chatbot):
  Set GITHUB_TOKEN in Streamlit secrets or .env

Run:  streamlit run app_final.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# ── RAG Chatbot imports ────────────────────────────────────────────────────────
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    
from dotenv import load_dotenv

# Load .env file
load_dotenv()
GITHUB_API_KEY = os.getenv("GITHUB_TOKEN")  # reads from .env

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
    .chat-user   { background: #2a2a3a; border-radius: 10px; padding: 10px 14px; margin: 6px 0; }
    .chat-bot    { background: #1a3a2a; border-radius: 10px; padding: 10px 14px; margin: 6px 0;
                   border-left: 3px solid #2ecc71; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
MAIN_DISTRICTS = ["Kathmandu", "Lalitpur", "Bhaktapur"]
DIST_COLORS = {"Kathmandu": "#e8a45a", "Lalitpur": "#5ab8e8", "Bhaktapur": "#8ae85a"}
FACING_OPTIONS = ["East", "West", "North", "South", "North East", "North West", "South East", "South West"]

# ── FIX 1: Corrected gen_land R² from 0.744 → 0.6117 and error from 19.1 → 27.4 ──
MODEL_INFO = {
    "gen_house": {"name": "🏠 General Housing",          "r2": 0.777,  "error": 18.8, "samples": 2005,"note": "Neighborhood target-encoded on full train set — R² may be ~0.01 optimistic"},
    "gen_land":  {"name": "🌍 General Land",              "r2": 0.6117, "error": 27.4, "samples": 3250},
    "lph_house": {"name": "🏘️ Lalpurja Housing (Adv.)",  "r2": 0.648,  "error": 23.7, "samples": 1749},
    "lph_land":  {"name": "🎯 Lalpurja Land (Adv.)",     "r2": 0.744,  "error": 19.1, "samples": 971},
}

# ─────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────

def fmt_npr(val, decimal=2):
    try:
        if pd.isna(val) or val == 0: return "₹0"
    except Exception:
        pass
    if val >= 10_000_000: return f"₹{val/10_000_000:.{decimal}f} Cr"
    elif val >= 100_000:  return f"₹{val/100_000:.{decimal}f} L"
    return f"₹{val:,.0f}"

def clean_chart(fig, height=None):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(color="#e0e0e0")
    )
    if height: fig.update_layout(height=height)
    return fig

def insight(text: str):
    st.markdown(
        f"<p style='font-size:13px; color:#a0c4e8; background:rgba(30,30,60,0.5); "
        f"padding:8px 12px; border-left:3px solid #5ab8e8; border-radius:4px; margin-bottom:12px;'>"
        f"💡 {text}</p>",
        unsafe_allow_html=True
    )

def calculate_matching_score(row, prefs):
    ideal_price = (prefs["min_price"] + prefs["max_price"]) / 2
    ideal_beds  = prefs["bedrooms"]

    price_diff_pct = abs(row["total_price"] - ideal_price) / max(ideal_price, 1)
    price_score = max(0, 1 - price_diff_pct)

    bed_diff = abs(row["bedrooms"] - ideal_beds)
    bed_score = max(0, 1 - (bed_diff / max(ideal_beds, 1)))

    if prefs["must_have_amenities"]:
        matched = sum(row.get(a, 0) == 1 for a in prefs["must_have_amenities"])
        amenity_score = matched / len(prefs["must_have_amenities"])
        final_score = (price_score * 0.30 + bed_score * 0.20 + amenity_score * 0.50) * 100
    else:
        final_score = (price_score * 0.60 + bed_score * 0.40) * 100

    return round(min(100, final_score), 2)

# ─────────────────────────────────────────────────────────
# LOAD ANALYTICS DATA
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_analytics_data():
    try:
        gh = pd.read_csv("data/housing_model_ready_after_outlier_treatment.csv")
        gh = gh[gh["district"].isin(MAIN_DISTRICTS)]

        gl = pd.read_csv("data/cleaned_land_merged_final_after_eda.csv")
        gl = gl[gl["district"].isin(MAIN_DISTRICTS)]
        gl_named = gl[~gl["neighborhood"].str.contains("Zone", na=False)]

        lh = pd.read_csv("data/cleaned_lalpurja_house_v2_after_cleaning.csv")
        lh["floors_x_land"] = lh["total_floors"] * lh["land_size_aana"]
        lh_named = lh[~lh["neighborhood"].str.contains("Zone", na=False)]

        ll = pd.read_csv("data/cleaned_lalpurja_land_final_after_eda.csv")
        ll_named = ll[~ll["neighborhood"].str.contains("Zone", na=False)]

        return gh, gl, gl_named, lh, lh_named, ll, ll_named
    except FileNotFoundError as e:
        st.error(f"❌ Data file missing: {e}")
        st.stop()

try:
    gh, gl, gl_named, lh, lh_named, ll, ll_named = load_analytics_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────
# BUILD ENCODING MAPS
# ─────────────────────────────────────────────────────────
@st.cache_data
def build_encoding_maps():
    try:
        house_fe     = pd.read_csv("data/housing_features_ready_after_feature_engineering.csv")
        land_fe      = pd.read_csv("data/land_features_final_modeled.csv")
        lph_house_fe = pd.read_csv("data/lalpurja_house_v2_features_ready.csv")
        lph_land_fe  = pd.read_csv("data/lalpurja_dataset_ready_after_feature_engineering.csv")

        clean_gh = pd.read_csv("data/housing_model_ready_after_outlier_treatment.csv")
        clean_gl = pd.read_csv("data/cleaned_land_merged_final_after_eda.csv")
        clean_lh = pd.read_csv("data/cleaned_lalpurja_house_v2_after_cleaning.csv")
        clean_ll = pd.read_csv("data/cleaned_lalpurja_land_final_after_eda.csv")

        clean_gh_f = clean_gh[clean_gh["district"].isin(MAIN_DISTRICTS + ["Unknown"])].reset_index(drop=True)
        house_fe_f = house_fe[house_fe["district"].isin([0,1,2,3])].reset_index(drop=True)
        clean_gl_f = clean_gl[clean_gl["district"].isin(MAIN_DISTRICTS)].reset_index(drop=True)
        land_fe_f  = land_fe[land_fe["district"].isin([0,1,2])].reset_index(drop=True)

        def build_map(clean_col, enc_col, clean_df, enc_df):
            m = {}
            for i in range(min(len(clean_df), len(enc_df))):
                m[clean_df.iloc[i][clean_col]] = float(enc_df.iloc[i][enc_col])
            return m

        maps = {}
        maps["neigh_gh"] = build_map("neighborhood", "neighborhood_encoded", clean_gh_f, house_fe_f)
        maps["neigh_gl"] = build_map("neighborhood", "neighborhood_encoded", clean_gl_f, land_fe_f)
        maps["neigh_lh"] = build_map("neighborhood", "neighborhood_encoded", clean_lh, lph_house_fe)
        maps["neigh_ll"] = build_map("neighborhood", "neighborhood_encoded", clean_ll, lph_land_fe)
        maps["muni_lh"]  = build_map("municipality", "municipality_encoded", clean_lh, lph_house_fe)
        maps["muni_ll"]  = build_map("municipality", "municipality_encoded", clean_ll, lph_land_fe)

        maps["district"]  = {"Bhaktapur": 0, "Kathmandu": 1, "Lalitpur": 2, "Unknown": 3}
        maps["facing_gh"] = {"East":0,"North":1,"North East":2,"North West":3,"South":4,"South East":5,"South West":6,"West":7}
        maps["facing_gl"] = {"East":0,"North":1,"North-East":2,"North-West":3,"South":4,"South-East":5,"South-West":6,"Unknown":7,"West":8}
        maps["road_gl"]   = {"High Access":0,"Low Access":1,"Mid Access":2,"Unknown":3}
        maps["road_lh"]   = {"High Access":0,"Low Access":1}
        maps["road_ll"]   = {"High Access":0,"Low Access":1}
        maps["ptype_lh"]  = {"Commercial":0,"Residential":1,"Semi-commercial":2}
        maps["ptype_ll"]  = {"Commercial":0,"Residential":1,"Semi-commercial":2}
        maps["furnish"]   = {"Full Furnished":0,"Semi Furnished":1,"Unfurnished":2}
        maps["face_lph"]  = {"East":0,"North":1,"North-East":2,"North-West":3,"South":4,"South-East":5,"South-West":6,"West":7}

        maps["ward_lh"] = clean_lh.groupby("neighborhood")["ward_no"].median().to_dict()
        maps["ward_ll"] = clean_ll.groupby("neighborhood")["ward_no"].median().to_dict()

        amenity_cols_lh = ["hospital_m","airport_m","pharmacy_m","bhatbhateni_m","school_m","college_m","public_transport_m","police_station_m","boudhanath_m","ring_road_m"]
        amenity_cols_ll = ["hospital_m","airport_m","pharmacy_m","bhatbhateni_m","school_m","public_transport_m","police_station_m","ring_road_m"]
        maps["amenity_lh"] = clean_lh.groupby("neighborhood")[amenity_cols_lh].median().to_dict(orient="index")
        maps["amenity_ll"] = clean_ll.groupby("neighborhood")[amenity_cols_ll].median().to_dict(orient="index")

        lph_house_fe["neigh_name"] = clean_lh["neighborhood"].values
        maps["eng_lh"] = lph_house_fe.groupby("neigh_name")[
            ["log_land","log_built","floor_area_ratio","urban_centrality","amenity_access_score",
             "house_size_score","comm_road_premium","neighborhood_x_district","municipality_x_ward",
             "age_condition_score","rooms_total","bath_per_bed","sqft_per_room","floors_x_land",
             "luxury_score","parking_premium"]
        ].median().to_dict(orient="index")

        lph_land_fe["neigh_name"] = clean_ll["neighborhood"].values
        maps["eng_ll"] = lph_land_fe.groupby("neigh_name")[
            ["log_land","urban_centrality","amenity_access_score","plot_value_score",
             "commercial_zone_score","neighborhood_x_district","municipality_x_ward",
             "road_access_quality","ring_road_proximity","comm_road_premium",
             "is_corner_plot","facing_road_width"]
        ].median().to_dict(orient="index")

        house_fe_f["neigh_name"] = clean_gh_f["neighborhood"].values
        maps["eng_gh"] = house_fe_f.groupby("neigh_name")[
            ["log_land","log_build_up","luxury_score","amenity_count","is_wide_road",
             "is_area_estimated","is_incomplete_listing","parking_cars","parking_bikes"]
        ].median().to_dict(orient="index")

        land_fe_f["neigh_name"] = clean_gl_f["neighborhood"].values
        maps["eng_gl"] = land_fe_f.groupby("neigh_name")[
            ["log_land","is_wide_road","is_large_plot","road_quality_score",
             "neighborhood_x_district","plot_size_category","location_tier",
             "large_plot_x_neighborhood"]
        ].median().to_dict(orient="index")

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

# ─────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        models = {}
        model_files = {
            "gen_house": "models/xgboost_housing_final.pkl",
            "gen_land":  "models/catboost_land_model_final.pkl",
            "lph_house": "models/catboost_lalpurja_house_v2_final.pkl",
            "lph_land":  "models/catboost_lalpurja_model_final.pkl",
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
    row = maps_dict.get(neighborhood, {})
    val = row.get(col, fallback)
    return float(val) if not (isinstance(val, float) and np.isnan(val)) else fallback

def validate_input(land_aana, bedrooms, bathrooms, house_age, buildup_sqft=None):
    errors = []
    if land_aana < 0.5:    errors.append("Land size must be ≥ 0.5 aana")
    if land_aana > 50:     errors.append("Land size > 50 aana is outside training range")
    if bedrooms < 1:       errors.append("Bedrooms must be ≥ 1")
    if bedrooms > 15:      errors.append("15+ bedrooms — confidence will be lower")
    if bathrooms < 1:      errors.append("Bathrooms must be ≥ 1")
    if house_age < 0:      errors.append("House age cannot be negative")
    if house_age > 100:    errors.append("House age > 100 years is outside training range")
    if buildup_sqft is not None and buildup_sqft < 100:
        errors.append("Built-up area must be ≥ 100 sqft")
    return errors

def get_confidence_score(land_aana, neighborhood, model_r2, model_samples, dataset_name):
    confidence = model_r2 * 100
    if dataset_name == "lph_land" and land_aana > 15:
        confidence -= (land_aana - 15) * 3
    if model_samples > 2000:
        confidence += 2
    return max(10, min(100, confidence))

def apply_land_multiplier(base_price, land_aana, model_type="lph_land"):
    if model_type != "lph_land":
        return base_price
    ref_land = 5.0
    exponent = 0.6
    multiplier = max(0.5, (land_aana / ref_land) ** exponent)
    return base_price * multiplier

# ─────────────────────────────────────────────────────────
# PREDICTION FUNCTIONS
# ─────────────────────────────────────────────────────────

def predict_gen_house(district, neighborhood, bedrooms, bathrooms, floors, land_aana, buildup_sqft,
                      road_width, house_age, facing, has_parking, has_garden, has_mod_kitchen,
                      has_parquet, has_drainage, has_solar):
    errors = validate_input(land_aana, bedrooms, bathrooms, house_age, buildup_sqft)
    if errors: raise ValueError("; ".join(errors))

    eng = MAPS["eng_gh"]
    d   = MAPS["district"].get(district, 1)
    f   = MAPS["facing_gh"].get(facing, 0)
    ne  = MAPS["neigh_gh"].get(neighborhood)
    if ne is None: raise ValueError(f"Neighborhood '{neighborhood}' not in training data")

    log_land     = np.log1p(land_aana)
    log_build_up = np.log1p(buildup_sqft)
    luxury_score = int(has_parking)*1 + int(has_garden)*2 + int(has_mod_kitchen)*2 + \
                   int(has_parquet)*1 + int(has_drainage)*1 + int(has_solar)*2
    amenity_count = sum([has_parking, has_garden, has_mod_kitchen, has_parquet, has_drainage, has_solar])
    is_wide_road  = 1 if road_width >= 20 else 0
    parking_cars  = get_default(eng, neighborhood, "parking_cars", 1.0)
    parking_bikes = get_default(eng, neighborhood, "parking_bikes", 0.0)

    row = np.array([[
        d, land_aana, buildup_sqft, floors, f, road_width, bedrooms, bathrooms,
        parking_cars, parking_bikes, house_age, amenity_count, int(has_mod_kitchen),
        int(has_parquet), int(has_drainage), int(has_parking), int(has_garden),
        is_wide_road, 0, luxury_score, 0, log_land, log_build_up, ne
    ]], dtype=np.float32)
    return float(np.expm1(MODELS["gen_house"].predict(row)[0]))


def predict_gen_land(district, neighborhood, land_aana, road_type, road_width, facing):
    errors = validate_input(land_aana, 1, 1, 0)
    if errors: raise ValueError("; ".join(errors))

    eng = MAPS["eng_gl"]
    d   = MAPS["district"].get(district, 1)
    rt  = MAPS["road_gl"].get(road_type, 2)
    f   = MAPS["facing_gl"].get(facing, 0)
    ne  = MAPS["neigh_gl"].get(neighborhood)
    if ne is None: raise ValueError(f"Neighborhood '{neighborhood}' not in training data")

    log_land      = np.log1p(land_aana)
    is_large_plot = 1 if land_aana > 10 else 0
    is_wide_road  = 1 if road_width >= 20 else 0
    road_quality  = {"High Access":2,"Mid Access":1,"Low Access":0}.get(road_type, 1)

    row = np.array([[
        d, rt, land_aana, is_large_plot, road_width, is_wide_road, f, 0,
        log_land, ne, road_quality,
        get_default(eng, neighborhood, "neighborhood_x_district", d*ne),
        get_default(eng, neighborhood, "plot_size_category", 2.0),
        get_default(eng, neighborhood, "location_tier", 3.0),
        get_default(eng, neighborhood, "large_plot_x_neighborhood", is_large_plot*ne),
    ]], dtype=np.float32)
    return float(np.expm1(MODELS["gen_land"].predict(row)[0]))


def predict_lph_house(neighborhood, property_type, road_type, furnishing, property_face, bedrooms,
                      kitchens, bathrooms, living_rooms, parking_spaces, total_floors, house_age,
                      road_width, land_aana, buildup_sqft, hospital_m, airport_m, pharmacy_m,
                      bhatbhateni_m, school_m, college_m, public_transport_m, police_station_m,
                      boudhanath_m, ring_road_m):
    errors = validate_input(land_aana, bedrooms, bathrooms, house_age, buildup_sqft)
    if errors: raise ValueError("; ".join(errors))

    eng       = MAPS["eng_lh"]
    dist_name = MAPS["neigh_to_dist_lh"].get(neighborhood)
    muni_name = MAPS["neigh_to_muni_lh"].get(neighborhood)
    if not dist_name or not muni_name:
        raise ValueError(f"Neighborhood '{neighborhood}' not in training data")

    d    = MAPS["district"].get(dist_name, 1)
    pt   = MAPS["ptype_lh"].get(property_type, 1)
    rt   = MAPS["road_lh"].get(road_type, 0)
    fn   = MAPS["furnish"].get(furnishing, 2)
    pf   = MAPS["face_lph"].get(property_face, 0)
    ne   = MAPS["neigh_lh"].get(neighborhood)
    me   = MAPS["muni_lh"].get(muni_name)
    ward = int(MAPS["ward_lh"].get(neighborhood, 10))
    if ne is None or me is None:
        raise ValueError(f"Encoding failed for neighborhood '{neighborhood}'")

    log_land        = np.log1p(land_aana)
    log_built       = np.log1p(buildup_sqft)
    rooms_total     = bedrooms + bathrooms + kitchens + living_rooms
    bath_per_bed    = bathrooms / max(bedrooms, 1)
    sqft_per_room   = buildup_sqft / max(rooms_total, 1)
    floors_x_land   = total_floors * land_aana
    floor_area_ratio= buildup_sqft / max(land_aana * 182, 1)
    age_condition   = max(0, 1 - house_age / 60)
    comm_road       = 1 if road_type == "High Access" else 0
    luxury          = get_default(eng, neighborhood, "luxury_score", 2.0)
    pk_premium      = get_default(eng, neighborhood, "parking_premium", parking_spaces * 0.1)

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
    return float(np.expm1(MODELS["lph_house"].predict(row)[0]))


def predict_lph_land(neighborhood, property_type, road_type, property_face, land_aana, road_width,
                     hospital_m, airport_m, pharmacy_m, bhatbhateni_m, school_m, public_transport_m,
                     police_station_m, ring_road_m):
    errors = validate_input(land_aana, 1, 1, 0)
    if errors: raise ValueError("; ".join(errors))

    eng       = MAPS["eng_ll"]
    dist_name = MAPS["neigh_to_dist_ll"].get(neighborhood)
    muni_name = MAPS["neigh_to_muni_ll"].get(neighborhood)
    if not dist_name or not muni_name:
        raise ValueError(f"Neighborhood '{neighborhood}' not in training data")

    d    = MAPS["district"].get(dist_name, 1)
    pt   = MAPS["ptype_ll"].get(property_type, 1)
    rt   = MAPS["road_ll"].get(road_type, 0)
    pf   = MAPS["face_lph"].get(property_face, 0)
    ne   = MAPS["neigh_ll"].get(neighborhood)
    me   = MAPS["muni_ll"].get(muni_name)
    ward = int(MAPS["ward_ll"].get(neighborhood, 10))
    if ne is None or me is None:
        raise ValueError(f"Encoding failed for neighborhood '{neighborhood}'")

    log_land    = np.log1p(land_aana)
    comm_road   = 1 if road_type == "High Access" else 0
    road_access = {"High Access":2,"Low Access":0}.get(road_type, 1)
    ring_prox   = 1 / max(ring_road_m, 1) * 10000

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

    price_per_ana = float(np.expm1(MODELS["lph_land"].predict(row)[0]))
    return apply_land_multiplier(price_per_ana, land_aana, "lph_land")

# ═══════════════════════════════════════════════════════════
# RAG KNOWLEDGE BASE BUILDER
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Building property knowledge base...")
def build_rag_knowledge_base():
    if not RAG_AVAILABLE:
        return None, None

    docs = []

    docs.append(f"""
Nepal Real Estate Market Overview — Kathmandu Valley 2025
Total listings analysed: 9,929 across four datasets.
Datasets: General Housing (2,465 listings), Lalpurja Housing (2,187 listings),
General Land (4,063 plots), Lalpurja Land (1,214 plots).
Districts covered: Kathmandu, Lalitpur, Bhaktapur.
Median house price (general market): {fmt_npr(gh['total_price'].median(), 1)}.
Median land price per Ana (general market): {fmt_npr(gl['price_per_aana'].median(), 1)}.
Best ML model R² score: 0.777 (General Housing — XGBoost).
Data source year: 2025. Currency: Nepalese Rupee (NPR).
1 Crore (Cr) = 10,000,000 NPR. 1 Lakh (L) = 100,000 NPR.
1 Ana = approximately 342.25 square feet of land area.
""")

    gh_stats = gh["total_price"].describe()
    docs.append(f"""
General Housing Dataset — Statistics
Total listings: {len(gh):,}
Minimum price: {fmt_npr(gh['total_price'].min(), 1)}
Maximum price: {fmt_npr(gh['total_price'].max(), 1)}
Mean price: {fmt_npr(gh['total_price'].mean(), 1)}
Median price: {fmt_npr(gh['total_price'].median(), 1)}
25th percentile: {fmt_npr(gh['total_price'].quantile(0.25), 1)}
75th percentile: {fmt_npr(gh['total_price'].quantile(0.75), 1)}
Average bedrooms: {gh['bedrooms'].mean():.1f}
Average bathrooms: {gh['bathrooms'].mean():.1f}
Average floors: {gh['floors'].mean():.2f}
District breakdown:
  Kathmandu median price: {fmt_npr(gh[gh['district']=='Kathmandu']['total_price'].median(), 1)}
  Lalitpur median price: {fmt_npr(gh[gh['district']=='Lalitpur']['total_price'].median(), 1)}
  Bhaktapur median price: {fmt_npr(gh[gh['district']=='Bhaktapur']['total_price'].median(), 1)}
Price distribution is right-skewed. Most properties are priced between 2 Cr and 4.5 Cr.
Top price drivers: land area (correlation 0.70), built-up area (0.67), bathrooms (0.37), floors (0.32).
""")

    docs.append(f"""
General Land Dataset — Statistics
Total plots: {len(gl):,}
Minimum price per Ana: {fmt_npr(gl['price_per_aana'].min(), 1)}
Maximum price per Ana: {fmt_npr(gl['price_per_aana'].max(), 1)}
Mean price per Ana: {fmt_npr(gl['price_per_aana'].mean(), 1)}
Median price per Ana: {fmt_npr(gl['price_per_aana'].median(), 1)}
Average land size: {gl['land_size_aana'].mean():.1f} Ana
District breakdown:
  Kathmandu median price/Ana: {fmt_npr(gl[gl['district']=='Kathmandu']['price_per_aana'].median(), 1)}
  Lalitpur median price/Ana: {fmt_npr(gl[gl['district']=='Lalitpur']['price_per_aana'].median(), 1)}
  Bhaktapur median price/Ana: {fmt_npr(gl[gl['district']=='Bhaktapur']['price_per_aana'].median(), 1)}
Price segments: Budget below 0.31 Cr/Ana, Mid-Range 0.31–0.65 Cr/Ana, High-End 0.65–1.12 Cr/Ana, Ultra-Luxury above 1.12 Cr/Ana.
Road type premium: Wide Road median 0.53 Cr/Ana vs Gravel 0.35 Cr/Ana (about 51% premium).
Land size does NOT significantly affect price per Ana (correlation near zero).
""")

    docs.append(f"""
Lalpurja Housing Dataset — Statistics
Total listings: {len(lh):,}
Lalpurja refers to official digital land records (land ownership certificate) in Nepal.
Minimum price: {fmt_npr(lh['total_price'].min(), 1)}
Maximum price: {fmt_npr(lh['total_price'].max(), 1)}
Median price: {fmt_npr(lh['total_price'].median(), 1)}
Average bedrooms: {lh['bedrooms'].mean():.1f}
Average land size: {lh['land_size_aana'].mean():.1f} Ana
Most expensive neighborhoods: Hattisar, Jawalakhel, Bansbari (prices reaching 19 Cr+).
Amenity correlations with price (all negative — closer = more expensive):
  Ring Road distance: -0.16
  Boudhanath distance: -0.15
  Airport distance: -0.10
Road type premium: High Access road adds about 0.65 Cr median premium.
Property type premium: Commercial properties command about 43% higher median price than Residential.
""")

    docs.append(f"""
Lalpurja Land Dataset — Statistics
Total plots: {len(ll):,}
Median price per Ana: {fmt_npr(ll['price_per_aana'].median(), 1)}
Top neighborhoods by price: Old Baneshowr, Gaushala, Tahachal, Putalisadak (around 0.9 Cr/Ana).
The strongest price predictors are amenity distances (all negative correlations):
  Airport distance correlation: -0.558 (STRONGEST predictor)
  Ring Road distance correlation: -0.504
  Hospital distance correlation: -0.350
  School distance correlation: -0.246
Road access premium: High Access 0.45 Cr/Ana vs Low Access 0.32 Cr/Ana (about 38% premium).
Property type: Semi-commercial commands highest price, then Commercial, then Residential.
Land size does NOT determine price per Ana (correlation: -0.04). Small plots in prime locations have highest per-Ana value.
""")

    docs.append("""
Machine Learning Models — Nepal Real Estate Price Prediction
Four models are deployed:

1. General Housing Model (XGBoost)
   R² score: 0.777 (best accuracy)
   Average prediction error: ±18.8%
   Training samples: 2,005
   Features: 24
   Best for: Typical homes across Kathmandu Valley

2. General Land Model (CatBoost)
   R² score: 0.6117
   Average prediction error: ±27.4%
   Training samples: 3,250
   Features: 16
   Best for: Plots without detailed amenity data

3. Lalpurja Housing Model (CatBoost)
   R² score: 0.648
   Average prediction error: ±23.7%
   Training samples: 1,749
   Features: 42 (most detailed model)
   Best for: Properties with Lalpurja certificate

4. Lalpurja Land Model (CatBoost)
   R² score: 0.744
   Average prediction error: ±19.1%
   Training samples: 971
   Features: 29
   Best for: Land plots with amenity proximity data
""")

    top_gh = gh[~gh["neighborhood"].str.contains("Zone|Unknown", na=False)]\
               .groupby("neighborhood")["total_price"].median()\
               .sort_values(ascending=False).head(10)
    neigh_text = "Top 10 most expensive housing neighborhoods (General dataset):\n"
    for n, p in top_gh.items():
        neigh_text += f"  {n}: {fmt_npr(p, 1)}\n"
    docs.append(neigh_text)

    top_gl = gl_named.groupby("neighborhood")["price_per_aana"].median()\
                     .sort_values(ascending=False).head(10)
    land_neigh_text = "Top 10 most expensive land neighborhoods (General dataset, price per Ana):\n"
    for n, p in top_gl.items():
        land_neigh_text += f"  {n}: {fmt_npr(p, 1)}\n"
    docs.append(land_neigh_text)

    docs.append("""
Nepal Real Estate Buying and Investment Guide — Kathmandu Valley

Location is the #1 price driver for both housing and land in Kathmandu Valley.
Properties closer to the Ring Road, airport, hospitals, and schools command significantly higher prices.

For Housing Buyers:
- Kathmandu has the widest price range and luxury options. Bhaktapur is most affordable and consistent.
- A typical mid-range home (3-5 bedrooms, 2-4 Ana land) costs between 2.45 Cr and 4.5 Cr.
- Land area (Ana) is the strongest predictor of total price.
- Amenities like garden (+2 luxury points) and modular kitchen (+2) add value.
- Newer houses (0-5 years old) command a clear premium over older ones.
- Wide roads (20+ feet) significantly boost property value.

For Land Investors:
- Small plots (0-10 Ana) in prime locations have the highest price per Ana.
- Airport proximity and Ring Road proximity are the two most powerful value drivers for land.
- Semi-commercial designated land commands higher prices than pure residential.

Price Ranges (2025 market):
- Budget housing: below 2.45 Cr
- Mid-range housing: 2.45 Cr to 4.5 Cr
- Luxury housing: 4.5 Cr to 9 Cr
- Ultra-luxury housing: above 9 Cr
- Budget land: below 0.31 Cr per Ana
- Mid-range land: 0.31 Cr to 0.65 Cr per Ana
- High-end land: 0.65 Cr to 1.12 Cr per Ana
- Ultra-luxury land: above 1.12 Cr per Ana
""")

    docs.append(f"""
District Comparison — Kathmandu Valley Real Estate

Kathmandu District:
  Housing median price: {fmt_npr(gh[gh['district']=='Kathmandu']['total_price'].median(), 1)}
  Land median price/Ana: {fmt_npr(gl[gl['district']=='Kathmandu']['price_per_aana'].median(), 1)}
  Characteristics: Widest price range, most luxury properties, highest demand, urban commercial mix.

Lalitpur District:
  Housing median price: {fmt_npr(gh[gh['district']=='Lalitpur']['total_price'].median(), 1)}
  Land median price/Ana: {fmt_npr(gl[gl['district']=='Lalitpur']['price_per_aana'].median(), 1)}
  Characteristics: Moderate spread, cultural heritage areas, good connectivity.

Bhaktapur District:
  Housing median price: {fmt_npr(gh[gh['district']=='Bhaktapur']['total_price'].median(), 1)}
  Land median price/Ana: {fmt_npr(gl[gl['district']=='Bhaktapur']['price_per_aana'].median(), 1)}
  Characteristics: Most affordable, tightest pricing, consistent and predictable market.
""")

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.create_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, embeddings


def build_rag_chain(vectorstore, github_api_key: str):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=github_api_key,
        base_url="https://models.inference.ai.azure.com",
        streaming=True,
    )

    prompt = PromptTemplate.from_template("""
You are a knowledgeable Nepal Real Estate Assistant specialising in the Kathmandu Valley property market.
Use ONLY the context provided below to answer the question. If the context does not contain enough information,
say so honestly and suggest the user explore the Analytics or Inference Engine sections of this app.

Always format prices in NPR (Crore/Lakh format), mention relevant districts when applicable,
and provide practical buying/selling advice where appropriate.

Context:
{context}

Question: {question}

Answer (be concise, helpful, and use bullet points where appropriate):
""")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ═══════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════
st.sidebar.title("🏠 Nepal Real Estate Pro")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "",
    ["📊 Market Analytics", "🧠 Inference Engine", "🔍 Recommendations", "💬 Property Assistant"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("9,929 listings · Kathmandu Valley · 2025")

if st.sidebar.button("ℹ️ About Models"):
    st.sidebar.info("""
    **General Housing** (R² 0.777) — Best overall accuracy
    **Lalpurja Land** (R² 0.744) — Amenity analysis
    **General Land** (R² 0.612) — Simple plots
    **Lalpurja Housing** (R² 0.648) — Interior detail
    """)


# ═══════════════════════════════════════════════════════════
# SECTION 1 — ANALYTICS
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

        with tab1:
            st.subheader(f"General Housing — {len(gh):,} Listings")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Listings", f"{len(gh):,}")
            c2.metric("Median Price", fmt_npr(gh["total_price"].median(), 1))
            c3.metric("Avg Beds", f"{gh['bedrooms'].mean():.1f}")
            c4.metric("Avg Floors", f"{gh['floors'].mean():.2f}")
            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                top_n = gh[~gh["neighborhood"].str.contains("Zone|Unknown", na=False)]\
                          .groupby("neighborhood")["total_price"].agg(["median","count"]).reset_index()
                top_n = top_n[top_n["count"] >= 5].sort_values("median", ascending=False).head(15)
                top_n["Label"] = top_n["median"].apply(fmt_npr)
                fig = px.bar(top_n.sort_values("median"), x="median", y="neighborhood",
                             orientation="h", color="median", color_continuous_scale="Purples",
                             text="Label", title="Top 15 Most Expensive Neighborhoods")
                fig.update_traces(textposition="outside")
                fig.update_layout(xaxis_title="", xaxis_showticklabels=False,
                                  coloraxis_showscale=False, yaxis_title="")
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight("The most expensive neighborhoods are concentrated in central Kathmandu. Location is the #1 price driver.")

            with c2:
                fig = go.Figure(data=[go.Histogram(x=gh["total_price"], nbinsx=40,
                                                   marker_color="#667eea", opacity=0.75)])
                med_val = gh["total_price"].median()
                fig.add_vline(x=med_val, line_dash="dash", line_color="#764ba2",
                              annotation_text=f"Median: {fmt_npr(med_val, 1)}")
                fig.update_xaxes(title_text="Price (NPR)")
                fig.update_yaxes(title_text="Frequency")
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight("Heavily right-skewed — most properties priced between ₹0.2–0.5 Cr, long tail to ₹4 Cr.")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                _leakage = {"price_per_aana","calculated_total_price","price_bucket",
                            "log_price","log_total_price","price_per_sqft","price_per_ana"}
                numeric_cols = gh.select_dtypes(include=[np.number]).columns.tolist()
                corr_cols = [c for c in numeric_cols if c != "total_price" and c not in _leakage]
                if corr_cols:
                    corr_series = gh[corr_cols + ["total_price"]].corr()["total_price"].drop("total_price").sort_values()
                    corr_df = pd.DataFrame({"Feature": corr_series.index, "Correlation": corr_series.values})
                    fig = px.bar(corr_df, x="Correlation", y="Feature", orientation="h",
                                 color="Correlation", color_continuous_scale="RdBu",
                                 title="Feature Correlations with Price")
                    fig.update_xaxes(range=[-1, 1])
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(clean_chart(fig, height=450), use_container_width=True)
                    insight("Land area (0.70) and built-up area (0.67) are top predictors.")

            with c2:
                fig = go.Figure()
                for district in MAIN_DISTRICTS:
                    fig.add_trace(go.Violin(y=gh[gh["district"]==district]["total_price"],
                                            name=district, box_visible=True, meanline_visible=True,
                                            points=False, marker_color=DIST_COLORS.get(district)))
                fig.update_yaxes(title_text="Price (NPR)")
                fig.update_layout(showlegend=True, height=450)
                st.plotly_chart(clean_chart(fig, height=450), use_container_width=True)
                insight("Kathmandu has widest spread & most high-end outliers. Bhaktapur is most consistently priced.")

        with tab2:
            st.subheader(f"Lalpurja Housing — {len(lh):,} Listings")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Listings", f"{len(lh):,}")
            c2.metric("Median", fmt_npr(lh["total_price"].median()))
            c3.metric("Avg Beds", f"{lh['bedrooms'].mean():.1f}")
            c4.metric("Avg Land", f"{lh['land_size_aana'].mean():.1f} Ana")
            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                top_n = lh_named.groupby("neighborhood")["total_price"].agg(["median","count"]).reset_index()
                top_n = top_n[top_n["count"] >= 3].sort_values("median", ascending=False).head(15)
                top_n["Label"] = top_n["median"].apply(fmt_npr)
                fig = px.bar(top_n.sort_values("median"), x="median", y="neighborhood",
                             orientation="h", color="median", color_continuous_scale="Purples",
                             text="Label", title="Top 15 Most Expensive Neighborhoods (Lalpurja)")
                fig.update_traces(textposition="outside")
                fig.update_layout(xaxis_title="", xaxis_showticklabels=False,
                                  coloraxis_showscale=False, yaxis_title="")
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight("Hattisar, Jawalakhel, Bansbari reach ₹19 Cr+ — ring-road proximity is the top premium driver.")

            with c2:
                amenity_dist_cols = [c for c in lh.columns if c.endswith("_m") and "price" not in c]
                if amenity_dist_cols:
                    corr_amenity = lh[amenity_dist_cols + ["total_price"]].corr()["total_price"].drop("total_price").sort_values()
                    fig = px.bar(x=corr_amenity.values, y=corr_amenity.index, orientation="h",
                                 color=corr_amenity.values, color_continuous_scale="RdBu",
                                 title="Amenity Distance Correlation with Price",
                                 labels={"x": "Correlation", "y": "Amenity"})
                    fig.update_xaxes(range=[-1, 1])
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                    insight("All amenity correlations are negative — closer = more expensive. Ring Road (−0.16) strongest.")

    elif page == "🌍 Land Market":
        st.title("🌍 Land Market Analysis")
        tab1, tab2 = st.tabs(["General Land", "Lalpurja Land"])

        with tab1:
            st.subheader(f"General Land — {len(gl):,} Plots")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Plots", f"{len(gl):,}")
            c2.metric("Median/Ana", fmt_npr(gl["price_per_aana"].median(), 1))
            c3.metric("Avg Size", f"{gl['land_size_aana'].mean():.1f} Ana")
            c4.metric("Top Area", "Jhamsikhel/Naxal")
            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                top_n = gl_named.groupby("neighborhood")["price_per_aana"].agg(["median","count"]).reset_index()
                top_n = top_n[top_n["count"] >= 10].sort_values("median", ascending=False).head(15)
                top_n["Label"] = top_n["median"].apply(lambda v: fmt_npr(v, 1))
                fig = px.bar(top_n.sort_values("median"), x="median", y="neighborhood",
                             orientation="h", color="median", color_continuous_scale="Oranges",
                             text="Label", title="Top 15 Neighborhoods by Median Price/Ana")
                fig.update_traces(textposition="outside")
                fig.update_layout(xaxis_title="", xaxis_showticklabels=False,
                                  coloraxis_showscale=False, yaxis_title="")
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight("Jhamsikhel and Naxal command near ₹1 Cr/Ana — all inside/near the ring road.")

            with c2:
                fig = go.Figure()
                for district in MAIN_DISTRICTS:
                    fig.add_trace(go.Box(y=gl[gl["district"]==district]["price_per_aana"],
                                         name=district, marker_color=DIST_COLORS.get(district)))
                fig.update_yaxes(title_text="Price per Ana (NPR)")
                fig.update_layout(showlegend=True, height=500)
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight(f"Kathmandu: {fmt_npr(gl[gl['district']=='Kathmandu']['price_per_aana'].median(),1)}/Ana · "
                        f"Lalitpur: {fmt_npr(gl[gl['district']=='Lalitpur']['price_per_aana'].median(),1)}/Ana · "
                        f"Bhaktapur: {fmt_npr(gl[gl['district']=='Bhaktapur']['price_per_aana'].median(),1)}/Ana")

        with tab2:
            st.subheader(f"Lalpurja Land — {len(ll):,} Plots")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Plots", f"{len(ll):,}")
            c2.metric("Median/Ana", fmt_npr(ll["price_per_aana"].median(), 1))
            c3.metric("Model R²", "0.744")
            c4.metric("Airport Corr.", "−0.558")
            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                top_n = ll_named.groupby("neighborhood")["price_per_aana"].agg(["median","count"]).reset_index()
                top_n = top_n[top_n["count"] >= 3].sort_values("median", ascending=False).head(15)
                top_n["Label"] = top_n["median"].apply(lambda v: fmt_npr(v, 1))
                fig = px.bar(top_n.sort_values("median"), x="median", y="neighborhood",
                             orientation="h", color="median", color_continuous_scale="Greens",
                             text="Label", title="Top 15 Neighborhoods by Median Price/Ana (Lalpurja)")
                fig.update_traces(textposition="outside")
                fig.update_layout(xaxis_title="", xaxis_showticklabels=False,
                                  coloraxis_showscale=False, yaxis_title="")
                st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                insight("Old Baneshowr, Gaushala, Tahachal top the list at ~₹0.9 Cr/Ana.")

            with c2:
                amenity_dist_ll = [c for c in ll.columns if c.endswith("_m") and "price" not in c]
                if amenity_dist_ll:
                    corr_ll = ll[amenity_dist_ll + ["price_per_aana"]].corr()["price_per_aana"].drop("price_per_aana").sort_values()
                    fig = px.bar(x=corr_ll.values, y=corr_ll.index, orientation="h",
                                 color=corr_ll.values, color_continuous_scale="RdBu",
                                 title="Amenity Distance Correlation with Price/Ana",
                                 labels={"x": "Correlation", "y": "Amenity"})
                    fig.update_xaxes(range=[-1, 1])
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(clean_chart(fig, height=500), use_container_width=True)
                    insight("Airport (−0.558) and Ring Road (−0.504) are the STRONGEST predictors for Lalpurja land.")

    elif page == "🤖 Model Performance":
        st.title("🤖 ML Model Performance")
        st.markdown("---")
        mdf = pd.DataFrame({
            "Dataset":       ["General Housing","Lalpurja Land","General Land","Lalpurja Housing"],
            "Algorithm":     ["XGBoost","CatBoost","CatBoost","CatBoost"],
            "R² Score":      [0.777, 0.744, 0.6117, 0.648],
            "Avg Error %":   [18.8,  19.1,  27.4,   23.7],
            "Training Rows": [2005,   971,  3250,   1749],
            "Features":      [24,     29,    16,      42],
        })
        st.dataframe(mdf.style.background_gradient(subset=["R² Score"], cmap="Greens"),
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
            fig = px.bar(mdf.sort_values("Avg Error %", ascending=False), x="Avg Error %", y="Dataset",
                        orientation="h", color="Avg Error %", color_continuous_scale="Reds",
                        title="Average Prediction Error %", text="Avg Error %")
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(clean_chart(fig), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# SECTION 2 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════
elif section == "🔍 Recommendations":
    st.title("💡 Personalised Property Recommendations")
    st.markdown("Get property suggestions based on your detailed preferences")
    st.markdown("---")

    rec_property_type = st.radio("🏠 Property Type", ["🏠 Housing", "🌍 Land"], horizontal=True)
    st.markdown("---")

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
                st.error("❌ No properties found. Try widening your budget or size range.")
            else:
                must_haves = []
                if want_parking:  must_haves.append("has_parking")
                if want_drainage: must_haves.append("has_drainage")
                if want_kitchen:  must_haves.append("has_modular_kitchen")
                if want_garden:   must_haves.append("has_garden")

                filtered_data["matching_score"] = filtered_data.apply(
                    lambda row: calculate_matching_score(row, {
                        "min_price": min_budget, "max_price": max_budget,
                        "bedrooms": (min_beds + max_beds) // 2, "bathrooms": 3,
                        "must_have_amenities": must_haves, "nice_to_have_amenities": [],
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

                for idx, (_, prop) in enumerate(recommendations.iterrows(), 1):
                    score = prop["matching_score"]
                    color_icon = "🟢" if score >= 80 else ("🟡" if score >= 60 else "🔴")
                    col1, col2, col3 = st.columns([0.8, 2.5, 0.7])
                    with col1:
                        st.metric(f"#{idx}", f"{score:.1f}%")
                    with col2:
                        st.write(f"**{prop.get('neighborhood','Unknown')}** | {prop.get('district','Unknown')}")
                        st.write(f"🛏️ {prop.get('bedrooms','N/A')} BHK | 🚿 {prop.get('bathrooms','N/A')} Bath | 🏞️ {prop.get(land_col,'N/A')} Ana")
                        st.write(f"💰 **{fmt_npr(prop.get('total_price',0))}**")
                    with col3:
                        st.write(color_icon)
                    st.divider()

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
            land_col_ll = "land_size_aana" if "land_size_aana" in ll.columns else "land_area_aana"
            min_land_size = st.number_input("Min Ana", min_value=0.5, max_value=100.0, value=2.0, step=0.5, key="land_min_size")
            max_land_size = st.number_input("Max Ana", min_value=0.5, max_value=100.0, value=20.0, step=0.5, key="land_max_size")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🛣️ ROAD ACCESS**")
            preferred_road  = st.selectbox("Preferred Road Type", ["Any","High Access","Low Access"], key="land_road")
        with col2:
            st.write("**🏠 PROPERTY TYPE**")
            preferred_ptype = st.selectbox("Property Use", ["Any","Residential","Commercial","Semi-commercial"], key="land_ptype")

        st.markdown("---")
        if st.button("🔍 GET LAND RECOMMENDATIONS", key="get_land", use_container_width=True):
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
                st.error("❌ No plots found. Try widening your price or size range.")
            else:
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

                for idx, (_, prop) in enumerate(recommendations.iterrows(), 1):
                    score = prop["matching_score"]
                    color_icon = "🟢" if score >= 80 else ("🟡" if score >= 60 else "🔴")
                    col1, col2, col3 = st.columns([0.8, 2.5, 0.7])
                    with col1:
                        st.metric(f"#{idx}", f"{score:.1f}%")
                    with col2:
                        land_size = prop.get(land_col_ll, "N/A")
                        ppa       = prop.get("price_per_aana", 0)
                        total     = ppa * float(land_size) if isinstance(land_size, (int, float)) else ppa
                        st.write(f"**{prop.get('neighborhood','Unknown')}** | {prop.get('district','Unknown')}")
                        st.write(f"🏞️ {land_size} Ana | 🛣️ {prop.get('road_type','N/A')} | 🏷️ {prop.get('property_type','N/A')}")
                        st.write(f"💰 **{fmt_npr(ppa)}/Ana** (Total: {fmt_npr(total)})")
                    with col3:
                        st.write(color_icon)
                    st.divider()


# ═══════════════════════════════════════════════════════════
# SECTION 3 — INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════
elif section == "🧠 Inference Engine":

    FEATURE_META = {
        "land_area_aana":        ("Land Size",             "Ana",  "📏",  1),
        "land_size_aana":        ("Land Size",             "Ana",  "📏",  1),
        "build_up_area":         ("Built-up Area",         "sqft", "🏗️",  1),
        "built_up_sqft":         ("Built-up Area",         "sqft", "🏗️",  1),
        "floors":                ("Number of Floors",      "",     "🏢",  1),
        "total_floors":          ("Number of Floors",      "",     "🏢",  1),
        "bedrooms":              ("Bedrooms",              "",     "🛏️",  1),
        "bathrooms":             ("Bathrooms",             "",     "🚿",  1),
        "kitchens":              ("Kitchens",              "",     "🍳",  1),
        "living_rooms":          ("Living Rooms",          "",     "🛋️",  1),
        "parking_spaces":        ("Parking Spaces",        "",     "🚗",  1),
        "road_width":            ("Road Width",            "ft",   "🛣️",  1),
        "house_age":             ("House Age",             "yrs",  "📅", -1),
        "hospital_m":            ("Hospital Distance",     "m",    "🏥", -1),
        "airport_m":             ("Airport Distance",      "m",    "✈️", -1),
        "pharmacy_m":            ("Pharmacy Distance",     "m",    "💊", -1),
        "bhatbhateni_m":         ("Bhatbhateni Distance",  "m",    "🛒", -1),
        "school_m":              ("School Distance",       "m",    "🎓", -1),
        "college_m":             ("College Distance",      "m",    "🎓", -1),
        "public_transport_m":    ("Bus Stop Distance",     "m",    "🚌", -1),
        "police_station_m":      ("Police Stn Distance",   "m",    "👮", -1),
        "boudhanath_m":          ("Boudhanath Distance",   "m",    "🕍", -1),
        "ring_road_m":           ("Ring Road Distance",    "m",    "🔄", -1),
        "buildup_sqft":          ("Built-up Area",         "sqft", "🏗️",  1),
        "land_aana":             ("Land Size",             "Ana",  "📏",  1),
    }
    DELTA = 0.20

    def perturb_predict(model_fn, base_row, feature, direction):
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

    def run_perturbation_analysis(model_fn, input_kwargs, feature_list, baseline, is_land_model=False):
        results = []
        land_size = input_kwargs.get("land_aana", 1.0)
        for feat in feature_list:
            if feat not in input_kwargs: continue
            val   = input_kwargs[feat]
            p_up  = perturb_predict(model_fn, input_kwargs, feat, "up")
            p_down= perturb_predict(model_fn, input_kwargs, feat, "down")
            if p_up is None or p_down is None: continue

            impact_raw = p_up - baseline
            impact_npr = impact_raw * land_size if is_land_model else impact_raw
            sensitivity = ((p_up - p_down) / (baseline + 1e-9)) * 100

            meta = FEATURE_META.get(feat, (feat, "", "📊", 1))
            label, unit, emoji, natural_dir = meta
            display_impact      = -impact_npr if natural_dir == -1 else impact_npr
            display_sensitivity = -sensitivity if natural_dir == -1 else sensitivity

            results.append({
                "feature": feat, "label": label, "unit": unit, "emoji": emoji,
                "value": val, "impact_npr": impact_npr,
                "display_impact": display_impact, "sensitivity": display_sensitivity,
                "p_up": p_up, "p_down": p_down, "natural_dir": natural_dir,
            })
        results.sort(key=lambda x: abs(x["display_impact"]), reverse=True)
        return results

    st.title("🧠 Inference Engine")
    st.markdown("Predict property price and **explain which features drive it** using local perturbation analysis.")
    st.markdown("---")

    with st.expander("📖 How does this work?", expanded=False):
        st.markdown("""
**Local Perturbation Analysis** — model-agnostic explainability.
1. Baseline prediction P₀ computed for your inputs.
2. Each feature is increased by 20% → P_up computed.
3. Impact = P_up − P₀ — shows how sensitive the price is to each feature.
4. Features ranked by |impact| — biggest movers shown first.

🟢 Green = increasing this feature raises price.
🔴 Red = increasing this feature lowers price (e.g. being further from amenities).
        """)

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        inf_prop_type = st.radio("**Property type**",
                                  ["🏠 House / Building","🌍 Land / Plot"],
                                  horizontal=True, key="inf_ptype")
    with col_sel2:
        inf_has_lalpurja = st.radio("**Advanced features? (Less Reliable)**",
                                     ["Yes","No / Not sure"],
                                     horizontal=True, key="inf_lalpurja")

    is_house_inf    = "House" in inf_prop_type
    is_lalpurja_inf = inf_has_lalpurja == "Yes"
    model_key_inf   = ("lph_house" if is_house_inf else "lph_land") if is_lalpurja_inf \
                 else ("gen_house"  if is_house_inf else "gen_land")

    model_labels = {
        "gen_house": ("General Housing — XGBoost",   "R²=0.777 · ±18.8%", "#2ecc71"),
        "gen_land":  ("General Land — CatBoost",     "R²=0.612 · ±27.4%", "#e8a45a"),
        "lph_house": ("Lalpurja Housing — CatBoost", "R²=0.648 · ±23.7%", "#c084fc"),
        "lph_land":  ("Lalpurja Land — CatBoost",    "R²=0.744 · ±19.1%", "#5ab8e8"),
    }
    mlabel, mstats, mcolor = model_labels[model_key_inf]
    st.markdown(
        f"<div style='background:rgba(30,30,50,0.7); border-left:4px solid {mcolor}; "
        f"padding:10px 16px; border-radius:6px; margin-bottom:16px;'>"
        f"<b>Model:</b> {mlabel} &nbsp;·&nbsp; {mstats}</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    input_kwargs = {}

    if model_key_inf == "gen_house":
        st.subheader("🏠 Property Details")
        c1, c2 = st.columns(2)
        with c1:
            district    = st.selectbox("District", MAIN_DISTRICTS, key="inf_gh_dist")
            neigh_opts  = sorted([n for n in MAPS["neigh_gh"] if n not in ["Unknown","nan"]])
            neighborhood = st.selectbox("Neighborhood", neigh_opts,
                index=neigh_opts.index("Budhanilkantha") if "Budhanilkantha" in neigh_opts else 0,
                key="inf_gh_neigh")
            facing   = st.selectbox("Facing", FACING_OPTIONS, key="inf_gh_face")
            bedrooms = st.slider("Bedrooms",  1, 12, 4, key="inf_gh_bed")
            bathrooms= st.slider("Bathrooms", 1, 10, 3, key="inf_gh_bath")
            floors   = st.slider("Floors", 1.0, 8.0, 2.5, 0.5, key="inf_gh_flr")
        with c2:
            land_aana  = st.slider("Land Size (Ana)",   0.5, 20.0, 4.0, 0.5, key="inf_gh_land")
            buildup    = st.slider("Built-up (sqft)",   200, 5000, 1400, 50,  key="inf_gh_bup")
            road_width = st.slider("Road Width (ft)",     5,   60,   14,  1,  key="inf_gh_road")
            house_age  = st.slider("House Age (yrs)",     0,   50,    5,      key="inf_gh_age")
            ca, cb = st.columns(2)
            has_parking  = ca.checkbox("🅿️ Parking",      True,  key="inf_gh_pk")
            has_drainage = cb.checkbox("💧 Drainage",     True,  key="inf_gh_dr")
            has_kitchen  = ca.checkbox("🍳 Mod. Kitchen", False, key="inf_gh_kit")
            has_parquet  = cb.checkbox("🪵 Parquet",      False, key="inf_gh_pq")
            has_garden   = ca.checkbox("🌳 Garden",       False, key="inf_gh_gd")
            has_solar    = cb.checkbox("☀️ Solar",        False, key="inf_gh_sol")
        input_kwargs = dict(
            district=district, neighborhood=neighborhood, bedrooms=bedrooms, bathrooms=bathrooms,
            floors=floors, land_aana=land_aana, buildup_sqft=buildup, road_width=road_width,
            house_age=house_age, facing=facing, has_parking=has_parking, has_garden=has_garden,
            has_mod_kitchen=has_kitchen, has_parquet=has_parquet, has_drainage=has_drainage,
            has_solar=has_solar
        )
        perturb_features = ["land_aana","buildup_sqft","floors","bedrooms","bathrooms","road_width","house_age"]
        model_fn    = predict_gen_house
        is_land_mdl = False
        err_pct     = 0.188

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
            land_aana  = st.slider("Land Size (Ana)", 0.5, 50.0, 5.0, 0.5, key="inf_gl_land")
            road_type  = st.selectbox("Road Type", ["High Access","Mid Access","Low Access"], key="inf_gl_rt")
            road_width = st.slider("Road Width (ft)", 5, 80, 16, 1, key="inf_gl_road")
            facing     = st.selectbox("Facing", FACING_OPTIONS, key="inf_gl_face")
        input_kwargs = dict(district=district, neighborhood=neighborhood, land_aana=land_aana,
                            road_type=road_type, road_width=road_width, facing=facing)
        perturb_features = ["land_aana","road_width"]
        model_fn    = predict_gen_land
        is_land_mdl = True
        err_pct     = 0.274

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
            bedrooms      = st.slider("Bedrooms",      1, 15, 4,   key="inf_lh_bed")
            kitchens      = st.slider("Kitchens",      1,  5, 1,   key="inf_lh_kit")
            bathrooms     = st.slider("Bathrooms",     1, 10, 3,   key="inf_lh_bath")
            living_rooms  = st.slider("Living Rooms",  1,  5, 1,   key="inf_lh_lr")
            parking_sp    = st.slider("Parking Spaces",0,  5, 1,   key="inf_lh_pk")
            total_floors  = st.slider("Total Floors", 0.5, 10.0, 2.5, 0.5, key="inf_lh_flr")
            land_aana     = st.slider("Land Size (Ana)", 1.0, 20.0, 4.0, 0.5, key="inf_lh_land")
            buildup       = st.slider("Built-up (sqft)", 200, 5000, 1200, 100, key="inf_lh_bup")
            house_age     = st.slider("House Age (yrs)",  0,  60, 5, key="inf_lh_age")
            road_width    = st.slider("Road Width (ft)",  5,  80, 14, 1, key="inf_lh_road")
        with c2:
            st.markdown("**📡 Amenity Distances (m)** — Auto-filled from neighborhood")
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
            st.warning(f"⚠️ {land_aana} Ana is above the 90th percentile — confidence is lower for large plots.")
        input_kwargs = dict(
            neighborhood=neighborhood, property_type=property_type, road_type=road_type,
            furnishing=furnishing, property_face=property_face, bedrooms=bedrooms,
            kitchens=kitchens, bathrooms=bathrooms, living_rooms=living_rooms,
            parking_spaces=parking_sp, total_floors=total_floors, house_age=house_age,
            road_width=road_width, land_aana=land_aana, buildup_sqft=buildup,
            hospital_m=hospital_m, airport_m=airport_m, pharmacy_m=pharmacy_m,
            bhatbhateni_m=bhatbhateni_m, school_m=school_m, college_m=college_m,
            public_transport_m=public_trans, police_station_m=police_m,
            boudhanath_m=boudhanath_m, ring_road_m=ring_road_m
        )
        perturb_features = ["land_aana","buildup_sqft","total_floors","bedrooms","bathrooms",
                            "kitchens","living_rooms","parking_spaces","road_width","house_age",
                            "hospital_m","airport_m","ring_road_m","boudhanath_m",
                            "pharmacy_m","bhatbhateni_m","school_m","college_m",
                            "public_transport_m","police_station_m"]
        model_fn    = predict_lph_house
        is_land_mdl = False
        err_pct     = 0.237

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
            st.markdown("**📡 Amenity Distances (m)** — Auto-filled from neighborhood")
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
            neighborhood=neighborhood, property_type=property_type, road_type=road_type,
            property_face=property_face, land_aana=land_aana, road_width=road_width,
            hospital_m=hospital_m, airport_m=airport_m, pharmacy_m=pharmacy_m,
            bhatbhateni_m=bhatbhateni_m, school_m=school_m, public_transport_m=public_trans,
            police_station_m=police_m, ring_road_m=ring_road_m
        )
        perturb_features = ["land_aana","road_width","hospital_m","airport_m","ring_road_m",
                            "bhatbhateni_m","pharmacy_m","school_m","public_transport_m","police_station_m"]
        model_fn    = predict_lph_land
        is_land_mdl = True
        err_pct     = 0.191

    st.markdown("---")
    if st.button("🧠 Run Inference Analysis", type="primary", use_container_width=True, key="inf_run_btn"):
        with st.spinner("Running prediction + perturbation analysis..."):
            try:
                baseline             = model_fn(**input_kwargs)
                land_size_for_total  = input_kwargs.get("land_aana", 1.0)
                total_price          = baseline if not is_land_mdl else baseline * land_size_for_total

                st.markdown("---")
                st.subheader("📊 Prediction Result")
                
    # ✅ REPLACE with smarter confidence intervals:

                def get_prediction_interval(price, model_r2, n_train, land_aana):
                    """
                    Approximate 90% prediction interval.
                    - Lower R² → wider interval
                    - Fewer training samples → wider interval  
                    - Very large plots → wider interval (extrapolation risk)
                    """
                    base_uncertainty = (1 - model_r2) * 0.45
                    sample_factor    = max(0, (1000 - n_train) / 10000)  # shrinks as n grows
                    size_factor      = max(0, (land_aana - 10) * 0.005)  # grows for large plots
                    total_uncertainty = base_uncertainty + sample_factor + size_factor
                    total_uncertainty = max(0.10, min(0.50, total_uncertainty))  # clamp 10%–50%
                    return price * (1 - total_uncertainty), price * (1 + total_uncertainty), total_uncertainty * 100

                land_sz = input_kwargs.get("land_aana", 4.0)
                lower, upper, unc_pct = get_prediction_interval(
                    total_price,
                    MODEL_INFO[model_key_inf]["r2"],
                    MODEL_INFO[model_key_inf]["samples"],
                    land_sz
                )

                c1, c2, c3, c4 = st.columns(4)
                if is_land_mdl:
                    c1.metric("💰 Price per Ana",   fmt_npr(baseline))
                    c2.metric("🏷️ Total Value",     fmt_npr(total_price), f"× {land_sz} Ana")
                    c3.metric("📉 90% Lower Bound", fmt_npr(lower),  f"−{unc_pct:.0f}%")
                    c4.metric("📈 90% Upper Bound", fmt_npr(upper),  f"+{unc_pct:.0f}%")
                else:
                    c1.metric("💰 Predicted Price", fmt_npr(baseline))
                    c2.metric("📉 90% Lower Bound", fmt_npr(lower),  f"−{unc_pct:.0f}%")
                    c3.metric("📈 90% Upper Bound", fmt_npr(upper),  f"+{unc_pct:.0f}%")
                    conf = get_confidence_score(land_sz,
                                                input_kwargs.get("neighborhood", ""),
                                                MODEL_INFO[model_key_inf]["r2"],
                                                MODEL_INFO[model_key_inf]["samples"],
                                                model_key_inf)
                    conf_cls = "confidence-high" if conf >= 75 else ("confidence-medium" if conf >= 60 else "confidence-low")
                    c4.markdown(
                        f"<p class='{conf_cls}' style='margin-top:28px'>Confidence: {conf:.0f}%</p>",
                        unsafe_allow_html=True
                    )

                st.markdown("---")
                st.subheader("🧠 What's Driving This Price?")
                results = run_perturbation_analysis(model_fn, input_kwargs, perturb_features,
                                                    baseline, is_land_mdl)

                if not results:
                    st.warning("Could not compute feature impacts.")
                else:
                    tab1, tab2, tab3 = st.tabs(["📊 Impact Chart", "🔢 Full Table", "💡 What-If Scenarios"])

                    with tab1:
                        top = results[:min(10, len(results))]
                        labels  = [f"{r['emoji']} {r['label']}" for r in top]
                        impacts = [r["display_impact"] for r in top]
                        colors  = ["#2ecc71" if v >= 0 else "#e74c3c" for v in impacts]
                        text    = [f"+{fmt_npr(v)}" if v >= 0 else f"-{fmt_npr(abs(v))}" for v in impacts]
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=impacts, y=labels, orientation="h",
                                             marker_color=colors, text=text, textposition="outside"))
                        fig.add_vline(x=0, line_color="#ffffff", line_width=1.5, opacity=0.4)
                        fig.update_layout(
                            title=f"Price impact of +20% change per feature (Baseline: {fmt_npr(total_price)})",
                            xaxis_title="Price Impact (NPR)", height=max(400, len(top)*55),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e0e0e0"), margin=dict(l=10, r=120, t=60, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("#### 💬 Plain-Language Summary")
                        for r in results[:3]:
                            imp = r["display_impact"]
                            direction = "increases" if imp > 0 else "decreases"
                            st.markdown(f"- **{r['emoji']} {r['label']}** �� a 20% change {direction} value by **{fmt_npr(abs(imp))}**.")

                    with tab2:
                        rows = []
                        for r in results:
                            sign = "▲" if r["display_impact"] >= 0 else "▼"
                            rows.append({
                                "Feature":        f"{r['emoji']} {r['label']}",
                                "Current Value":  f"{r['value']:.1f} {r['unit']}".strip(),
                                "Price if +20%":  fmt_npr(r["p_up"] * (land_size_for_total if is_land_mdl else 1)),
                                "Price if −20%":  fmt_npr(r["p_down"] * (land_size_for_total if is_land_mdl else 1)),
                                "Impact of +20%": f"{sign} {fmt_npr(abs(r['display_impact']))}",
                                "Sensitivity":    f"{abs(r['sensitivity']):.1f}%",
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    with tab3:
                        st.markdown("#### Top 5 Value Levers")
                        rows_wi = []
                        for r in results[:5]:
                            cur  = total_price
                            new  = r["p_up"] if not is_land_mdl else r["p_up"] * land_size_for_total
                            chg  = new - cur
                            rows_wi.append({
                                "Feature":       f"{r['emoji']} {r['label']}",
                                "Current":       f"{r['value']:.1f} {r['unit']}".strip(),
                                "If +20%":       f"{r['value']*1.2:.1f} {r['unit']}".strip(),
                                "Price Becomes": fmt_npr(new),
                                "Change":        f"+{fmt_npr(chg)}" if chg >= 0 else f"-{fmt_npr(abs(chg))}",
                            })
                        st.dataframe(pd.DataFrame(rows_wi), use_container_width=True, hide_index=True)

            except ValueError as e:
                st.error(f"⚠️ Input error: {e}")
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.info("Try adjusting your inputs or selecting a different neighborhood.")


# ═══════════════════════════════════════════════════════════
# SECTION 4 — PROPERTY ASSISTANT (RAG CHATBOT)
# ═══════════════════════════════════════════════════════════
elif section == "💬 Property Assistant":
    st.title("💬 Nepal Real Estate Assistant")
    st.markdown(
        "Ask me anything about the **Kathmandu Valley property market** — prices, neighborhoods, "
        "investment advice, model performance, or how this app works."
    )
    st.markdown("---")

    if not RAG_AVAILABLE:
        st.error(
            "❌ RAG dependencies not installed. Run:\n\n"
            "```\npip install langchain-text-splitters langchain-community langchain-huggingface "
            "langchain-openai langchain-core faiss-cpu sentence-transformers\n```"
        )
        st.stop()

    openai_key = GITHUB_API_KEY  # loaded from .env at startup

    if not openai_key:
        st.error(
            "❌ GitHub API key not found. Please add `GITHUB_TOKEN=your_token` to your `.env` file "
            "and restart the app."
        )
        st.stop()

    # ── Build knowledge base (cached) ───────────────────────────────────────
    vectorstore, _ = build_rag_knowledge_base()
    if vectorstore is None:
        st.error("❌ Failed to build knowledge base.")
        st.stop()

    # ── Build RAG chain ──────────────────────────────────────────────────────
    try:
        rag_chain = build_rag_chain(vectorstore, openai_key)
    except Exception as e:
        st.error(f"❌ Failed to initialise RAG chain: {e}")
        st.stop()

    # ── Chat interface ───────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggested questions
    st.markdown("**💡 Suggested questions:**")
    sugg_cols = st.columns(3)
    suggestions = [
        "What's the median house price in Kathmandu?",
        "Which neighborhoods have the highest land prices?",
        "How does road type affect land value?",
        "Which model has the best accuracy?",
        "What's a good budget for buying land in Lalitpur?",
    ]
    for i, sugg in enumerate(suggestions):
        if sugg_cols[i % 3].button(sugg, key=f"sugg_{i}", use_container_width=True):
            st.session_state["prefill_question"] = sugg

    st.markdown("---")

    # ── Clear chat button ────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", key="chat_clear"):
        st.session_state.chat_history = []
        st.rerun()

    # ── Display chat history ─────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # ── Prefill handling for suggested questions ─────────────────────────────
    prefill = st.session_state.pop("prefill_question", "")
    if prefill:
        st.session_state["_prefill_pending"] = prefill

    # ── Chat input ───────────────────────────────────────────────────────────
    user_question = st.chat_input(
        placeholder="e.g. What is the average price per Ana in Bhaktapur?",
        key="chat_input"
    )

    # Use prefill if no direct input
    if not user_question and st.session_state.get("_prefill_pending"):
        user_question = st.session_state.pop("_prefill_pending")

    if user_question and user_question.strip():
        with st.chat_message("user"):
            st.markdown(user_question.strip())
        st.session_state.chat_history.append({"role": "user", "content": user_question.strip()})

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            full_answer = ""
            try:
                with st.spinner("🔍 Searching knowledge base..."):
                    for chunk in rag_chain.stream(user_question.strip()):
                        full_answer += chunk
                        answer_placeholder.markdown(full_answer + "▌")
                answer_placeholder.markdown(full_answer)
                st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
            except Exception as e:
                err_msg = f"❌ Error: {str(e)}"
                if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                    err_msg = "❌ Invalid API key. Please check your GitHub PAT and try again."
                elif "rate_limit" in str(e).lower():
                    err_msg = "❌ Rate limit reached. Please wait a moment and try again."
                answer_placeholder.error(err_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": err_msg})

    # ── Architecture info ────────────────────────────────────────────────────
    with st.expander("🏗️ How does this chatbot work?", expanded=False):
        st.markdown("""
**Retrieval-Augmented Generation (RAG) Pipeline:**

1. **Knowledge Base** — Built in-memory from dataset statistics + domain knowledge (no external files needed).
   Covers: market prices, district comparisons, model performance, buying guides, neighborhood data.

2. **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` converts text chunks to vectors (runs locally on CPU).

3. **Vector Store** — FAISS stores and retrieves the top-5 most relevant chunks for each question.

4. **LLM** — GPT-4o-mini via GitHub Models API (`models.inference.ai.azure.com`) generates the final answer
   grounded in the retrieved context.

5. **Streaming** — Responses stream token-by-token for a real-time typing effect.
        """)