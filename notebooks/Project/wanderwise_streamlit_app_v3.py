
# ============================================================
# WanderWise Travel Recommender App
# ============================================================
# Run using:
# streamlit run travel_recommender_app.py
# ============================================================

import re
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WanderWise Travel Recommender",
    page_icon="✈️",
    layout="wide"
)



# ============================================================
# DESTINATION PHOTOS
# ============================================================

DESTINATION_IMAGES = {
    "Tokyo, Japan": "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf",
    "Amsterdam, Netherlands": "https://images.unsplash.com/photo-1512470876302-972faa2aa9a4",
    "Paris, France": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34",
    "Bali, Indonesia": "https://images.unsplash.com/photo-1537996194471-e657df975ab4",
    "Sydney, Australia": "https://images.unsplash.com/photo-1506973035872-a4ec16b8e8d9",
    "Seoul, South Korea": "https://images.unsplash.com/photo-1538485399081-7c8ed904c7ca",
    "New York City, United States": "https://images.unsplash.com/photo-1485871981521-5b1fd3805eee",
    "London, United Kingdom": "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad",
    "Barcelona, Spain": "https://images.unsplash.com/photo-1583422409516-2895a77efded",
    "Dubai, United Arab Emirates": "https://images.unsplash.com/photo-1512453979798-5ea266f8880c",
    "Rome, Italy": "https://images.unsplash.com/photo-1529260830199-42c24126f198",
    "Bangkok, Thailand": "https://images.unsplash.com/photo-1508009603885-50cf7c579365",
    "Phuket, Thailand": "https://images.unsplash.com/photo-1589394815804-964ed0be2eb5",
    "Rio de Janeiro, Brazil": "https://images.unsplash.com/photo-1483729558449-99ef09a8c325",
    "Cape Town, South Africa": "https://images.unsplash.com/photo-1580060839134-75a5edca2e99",
    "Santorini, Greece": "https://images.unsplash.com/photo-1570077188670-e3a8d69ac5ff",
    "Athens, Greece": "https://images.unsplash.com/photo-1603565816030-6b389eeb23cb",
    "Vancouver, Canada": "https://images.unsplash.com/photo-1559511260-66a654ae982a",
    "Cancun, Mexico": "https://images.unsplash.com/photo-1552074284-5e88ef1aef18",
    "Honolulu, Hawaii": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
}

DEFAULT_IMAGE = "https://images.unsplash.com/photo-1488646953014-85cb44e25828"

def get_destination_image(destination):
    return DESTINATION_IMAGES.get(destination, DEFAULT_IMAGE)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_column_names(dataframe):
    df = dataframe.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def clean_money(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    value = re.sub(r"[^0-9.]", "", value)

    if value == "":
        return np.nan

    return float(value)


def normalize_destination_key(value):
    """
    Converts destination text into a matching key.

    Handles:
    - case differences
    - commas and punctuation
    - extra spaces
    - common country abbreviations such as USA, UK, UAE, AUS
    """

    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()

    abbreviation_map = {
        "u.s.a.": "usa",
        "u.s.": "usa",
        "united states of america": "united states",
        "usa": "united states",
        "us": "united states",
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "uae": "united arab emirates",
        "u.a.e.": "united arab emirates",
        "aus": "australia",
        "thai": "thailand",
        "sa": "south africa",
        "korea": "south korea",
    }

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    for short, full in abbreviation_map.items():
        short_clean = re.sub(r"[^a-z0-9\s]", " ", short)
        short_clean = re.sub(r"\s+", " ", short_clean).strip()
        text = re.sub(rf"\b{re.escape(short_clean)}\b", full, text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


DESTINATION_ALIAS_MAP = {
    "sydney": "Sydney, Australia",
    "sydney australia": "Sydney, Australia",
    "australia": "Sydney, Australia",

    "paris": "Paris, France",
    "paris france": "Paris, France",
    "france": "Paris, France",

    "tokyo": "Tokyo, Japan",
    "tokyo japan": "Tokyo, Japan",
    "japan": "Tokyo, Japan",

    "bali": "Bali, Indonesia",
    "bali indonesia": "Bali, Indonesia",
    "indonesia": "Bali, Indonesia",

    "new york": "New York City, United States",
    "new york city": "New York City, United States",
    "new york united states": "New York City, United States",
    "new york city united states": "New York City, United States",
    "nyc": "New York City, United States",
    "los angeles": "Los Angeles, United States",
    "los angeles united states": "Los Angeles, United States",
    "la": "Los Angeles, United States",
    "hawaii": "Honolulu, Hawaii",
    "honolulu": "Honolulu, Hawaii",
    "honolulu hawaii": "Honolulu, Hawaii",
    "united states": "New York City, United States",

    "rome": "Rome, Italy",
    "rome italy": "Rome, Italy",
    "italy": "Rome, Italy",

    "bangkok": "Bangkok, Thailand",
    "bangkok thailand": "Bangkok, Thailand",
    "phuket": "Phuket, Thailand",
    "phuket thailand": "Phuket, Thailand",
    "thailand": "Bangkok, Thailand",

    "barcelona": "Barcelona, Spain",
    "barcelona spain": "Barcelona, Spain",
    "spain": "Barcelona, Spain",

    "london": "London, United Kingdom",
    "london united kingdom": "London, United Kingdom",
    "united kingdom": "London, United Kingdom",

    "cape town": "Cape Town, South Africa",
    "cape town south africa": "Cape Town, South Africa",
    "south africa": "Cape Town, South Africa",

    "dubai": "Dubai, United Arab Emirates",
    "dubai united arab emirates": "Dubai, United Arab Emirates",
    "united arab emirates": "Dubai, United Arab Emirates",

    "amsterdam": "Amsterdam, Netherlands",
    "amsterdam netherlands": "Amsterdam, Netherlands",
    "netherlands": "Amsterdam, Netherlands",

    "rio": "Rio de Janeiro, Brazil",
    "rio de janeiro": "Rio de Janeiro, Brazil",
    "rio de janeiro brazil": "Rio de Janeiro, Brazil",
    "brazil": "Rio de Janeiro, Brazil",

    "athens": "Athens, Greece",
    "athens greece": "Athens, Greece",
    "santorini": "Santorini, Greece",
    "santorini greece": "Santorini, Greece",
    "greece": "Athens, Greece",

    "seoul": "Seoul, South Korea",
    "seoul south korea": "Seoul, South Korea",
    "south korea": "Seoul, South Korea",

    "vancouver": "Vancouver, Canada",
    "vancouver canada": "Vancouver, Canada",
    "canada": "Vancouver, Canada",

    "cancun": "Cancun, Mexico",
    "cancun mexico": "Cancun, Mexico",
    "mexico": "Cancun, Mexico",

    "auckland": "Auckland, New Zealand",
    "auckland new zealand": "Auckland, New Zealand",
    "new zealand": "Auckland, New Zealand",
    "berlin": "Berlin, Germany",
    "berlin germany": "Berlin, Germany",
    "germany": "Berlin, Germany",
    "edinburgh": "Edinburgh, Scotland",
    "edinburgh scotland": "Edinburgh, Scotland",
    "scotland": "Edinburgh, Scotland",
    "marrakech": "Marrakech, Morocco",
    "marrakech morocco": "Marrakech, Morocco",
    "morocco": "Marrakech, Morocco",
    "phnom penh": "Phnom Penh, Cambodia",
    "phnom penh cambodia": "Phnom Penh, Cambodia",
    "cambodia": "Phnom Penh, Cambodia",
    "egypt": "Cairo, Egypt",
    "cairo": "Cairo, Egypt",
    "cairo egypt": "Cairo, Egypt",
}



def canonicalize_destination(value):
    """
    Standardizes destination names so duplicate variants are treated as one destination.
    """

    key = normalize_destination_key(value)

    if key in DESTINATION_ALIAS_MAP:
        return DESTINATION_ALIAS_MAP[key]

    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ,", ",").replace(", ", ", ")

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) >= 2:
        city = parts[0].title()
        country_key = normalize_destination_key(parts[-1])
        country_name = DESTINATION_ALIAS_MAP.get(country_key, parts[-1].title())

        if "," in country_name:
            country_name = country_name.split(",")[-1].strip()

        return f"{city}, {country_name}"

    return text.title() if text else "Unknown"


def normalize_text(value):
    if pd.isna(value):
        return "Unknown"
    return str(value).strip().title()


def mode_or_unknown(series):
    series = series.dropna()
    if len(series) == 0:
        return "Unknown"
    return series.mode().iloc[0]


def get_season_from_date(value):
    if pd.isna(value):
        return "Unknown"

    date_value = pd.to_datetime(value, errors="coerce")

    if pd.isna(date_value):
        return "Unknown"

    month = date_value.month

    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


def safe_minmax(series):
    series = series.astype(float)

    if series.max() == series.min():
        return pd.Series(np.ones(len(series)), index=series.index)

    return (series - series.min()) / (series.max() - series.min())


def similarity_score(user_value, destination_value, tolerance):
    if pd.isna(destination_value):
        return 0

    diff = abs(user_value - destination_value)
    score = 1 - (diff / tolerance)
    return max(0, min(1, score))


def preference_score(user_choice, destination_choice):
    if user_choice == "Any":
        return 1

    if pd.isna(destination_choice):
        return 0

    return 1 if str(user_choice).lower() == str(destination_choice).lower() else 0


def generate_comment(row):
    reasons = []

    if row["profile_similarity"] >= 0.70:
        reasons.append("strong profile similarity")
    elif row["profile_similarity"] >= 0.40:
        reasons.append("moderate profile similarity")

    if row["budget_fit"] >= 0.80:
        reasons.append("strong budget fit")
    elif row["budget_fit"] >= 0.50:
        reasons.append("acceptable budget fit")

    if row["duration_fit"] >= 0.80:
        reasons.append("strong duration fit")
    elif row["duration_fit"] >= 0.50:
        reasons.append("acceptable duration fit")

    if row["preference_match"] >= 0.75:
        reasons.append("matches most stated preferences")
    elif row["preference_match"] >= 0.40:
        reasons.append("matches some stated preferences")

    if row["popularity_score"] >= 0.70:
        reasons.append("high historical popularity")

    if row["age_fit"] >= 0.70:
        reasons.append("similar traveler age profile")

    if len(reasons) == 0:
        return "Recommended based on overall weighted score."

    return "Recommended because of " + ", ".join(reasons) + "."



def get_transport_icon(transport):
    """Returns a UI icon that matches the transportation type."""
    transport = str(transport).strip().lower()

    if any(word in transport for word in ["air", "plane", "airplane", "flight", "fly"]):
        return "✈️"
    if "train" in transport or "rail" in transport:
        return "🚆"
    if "bus" in transport or "coach" in transport:
        return "🚌"
    if any(word in transport for word in ["car", "taxi", "cab", "drive", "rental"]):
        return "🚗"
    if any(word in transport for word in ["boat", "ferry", "ship", "cruise"]):
        return "⛴️"
    if "walk" in transport:
        return "🚶"
    return "🧳"


def get_season_icon(season):
    """Returns a UI icon that matches the season."""
    season = str(season).strip().lower()

    if season == "summer":
        return "☀️"
    if season == "winter":
        return "❄️"
    if season == "spring":
        return "🌸"
    if season in ["fall", "autumn"]:
        return "🍂"
    return "🌍"


def get_accommodation_icon(accommodation):
    """Returns a UI icon that matches the accommodation type."""
    accommodation = str(accommodation).strip().lower()

    if "hotel" in accommodation:
        return "🏨"
    if "resort" in accommodation:
        return "🏝️"
    if "hostel" in accommodation:
        return "🛏️"
    if any(word in accommodation for word in ["airbnb", "apartment", "condo", "flat"]):
        return "🏠"
    if "villa" in accommodation:
        return "🏡"
    if "camp" in accommodation:
        return "⛺"
    return "🛌"


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

@st.cache_data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = clean_column_names(df)

    required_cols = [
        "destination",
        "duration_days",
        "traveler_age",
        "accommodation_type",
        "transportation_type",
        "accommodation_cost",
        "transportation_cost",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    df["destination"] = df["destination"].apply(canonicalize_destination)

    df["duration_days"] = pd.to_numeric(df["duration_days"], errors="coerce")
    df["traveler_age"] = pd.to_numeric(df["traveler_age"], errors="coerce")

    df["accommodation_cost_clean"] = df["accommodation_cost"].apply(clean_money)
    df["transportation_cost_clean"] = df["transportation_cost"].apply(clean_money)
    df["total_cost"] = df["accommodation_cost_clean"] + df["transportation_cost_clean"]

    df["accommodation_type"] = df["accommodation_type"].apply(normalize_text)
    df["transportation_type"] = df["transportation_type"].apply(normalize_text)

    if "start_date" in df.columns:
        df["season"] = df["start_date"].apply(get_season_from_date)
    else:
        df["season"] = "Unknown"

    df = df.dropna(
        subset=[
            "destination",
            "duration_days",
            "traveler_age",
            "total_cost"
        ]
    )

    destination_profiles = (
        df.groupby("destination")
        .agg(
            median_total_cost=("total_cost", "median"),
            median_duration_days=("duration_days", "median"),
            median_traveler_age=("traveler_age", "median"),
            common_accommodation=("accommodation_type", mode_or_unknown),
            common_transportation=("transportation_type", mode_or_unknown),
            common_season=("season", mode_or_unknown),
            trip_count=("destination", "count")
        )
        .reset_index()
    )

    destination_profiles["popularity_score"] = safe_minmax(destination_profiles["trip_count"])

    return df, destination_profiles


def recommend_destinations(
    destination_profiles,
    user_budget,
    user_duration,
    user_age,
    preferred_accommodation,
    preferred_transportation,
    preferred_season,
    top_n=10
):
    result = destination_profiles.copy()

    result["budget_fit"] = result["median_total_cost"].apply(
        lambda x: similarity_score(user_budget, x, tolerance=max(user_budget, 1))
    )

    result["duration_fit"] = result["median_duration_days"].apply(
        lambda x: similarity_score(user_duration, x, tolerance=max(user_duration, 1))
    )

    result["age_fit"] = result["median_traveler_age"].apply(
        lambda x: similarity_score(user_age, x, tolerance=30)
    )

    result["accommodation_match"] = result["common_accommodation"].apply(
        lambda x: preference_score(preferred_accommodation, x)
    )

    result["transportation_match"] = result["common_transportation"].apply(
        lambda x: preference_score(preferred_transportation, x)
    )

    result["season_match"] = result["common_season"].apply(
        lambda x: preference_score(preferred_season, x)
    )

    result["preference_match"] = (
        result["accommodation_match"]
        + result["transportation_match"]
        + result["season_match"]
    ) / 3

    result["profile_similarity"] = (
        0.40 * result["budget_fit"]
        + 0.30 * result["duration_fit"]
        + 0.30 * result["age_fit"]
    )

    result["final_score"] = (
        0.35 * result["profile_similarity"]
        + 0.25 * result["budget_fit"]
        + 0.15 * result["duration_fit"]
        + 0.15 * result["preference_match"]
        + 0.05 * result["popularity_score"]
        + 0.05 * result["age_fit"]
    )

    result["recommendation_comment"] = result.apply(generate_comment, axis=1)

    result = result.sort_values("final_score", ascending=False).head(top_n)

    display_cols = [
        "destination",
        "final_score",
        "profile_similarity",
        "budget_fit",
        "duration_fit",
        "preference_match",
        "popularity_score",
        "age_fit",
        "median_total_cost",
        "median_duration_days",
        "common_accommodation",
        "common_transportation",
        "common_season",
        "trip_count",
        "recommendation_comment"
    ]

    return result[display_cols].reset_index(drop=True)


# ============================================================
# APP UI
# ============================================================

DATA_PATH = "/Users/wpmangapot/Desktop/WorkFolder/Projects/stat280_pml/notebooks/Project/Travel details dataset.csv"

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #eef2ff 0%, transparent 34%), linear-gradient(135deg, #f8faff 0%, #f4f7ff 52%, #fffaf6 100%);
    }

    .block-container {
        padding-top: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1800px;
    }

    section[data-testid="stSidebar"] {
        width: 430px !important;
        min-width: 430px !important;
        background: rgba(255, 255, 255, 0.96);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 8px 0 30px rgba(15, 23, 42, 0.05);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.7rem;
        padding-left: 1.35rem;
        padding-right: 1.35rem;
    }

    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin-bottom: 1.7rem;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #eef2ff, #f5f3ff);
        color: #4f46e5;
        font-size: 1.6rem;
        box-shadow: 0 10px 24px rgba(79, 70, 229, 0.14);
    }

    .brand-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #312e81;
        line-height: 1.05;
        letter-spacing: -0.04em;
    }

    .brand-subtitle {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }

    .sidebar-heading {
        font-size: 1.05rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0.35rem;
    }

    .sidebar-help {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.55;
        margin-bottom: 1.05rem;
    }

    .hero-card {
        min-height: 285px;
        border-radius: 24px;
        padding: 2.55rem 2.55rem;
        background:
            linear-gradient(105deg, rgba(14, 35, 93, 0.98) 0%, rgba(24, 58, 145, 0.92) 58%, rgba(37, 99, 235, 0.78) 100%),
            url('https://images.unsplash.com/photo-1436491865332-7a61a109cc05?auto=format&fit=crop&w=1800&q=80');
        background-size: cover;
        background-position: center;
        color: white;
        box-shadow: 0 28px 70px rgba(30, 64, 175, 0.18);
        margin-bottom: 1.45rem;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-bottom: 1.8rem;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.55rem 0.95rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.82rem;
        font-weight: 700;
        backdrop-filter: blur(8px);
    }

    .hero-card h1 {
        font-size: clamp(2rem, 4vw, 3.05rem);
        line-height: 1.05;
        margin: 0 0 1.15rem 0;
        letter-spacing: -0.06em;
        font-weight: 800;
    }

    .hero-card p {
        font-size: 1.07rem;
        opacity: 0.95;
        max-width: 760px;
        line-height: 1.7;
        margin: 0;
    }

    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 3.7rem;
    }

    .metric-card {
        min-height: 112px;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
        padding: 1.3rem 1.35rem;
        display: flex;
        align-items: center;
        gap: 1.05rem;
    }

    .metric-icon {
        width: 62px;
        height: 62px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.75rem;
        flex-shrink: 0;
    }

    .metric-icon.purple { background: #ede9fe; color: #7c3aed; }
    .metric-icon.green { background: #dcfce7; color: #059669; }
    .metric-icon.orange { background: #ffedd5; color: #ea580c; }
    .metric-icon.blue { background: #dbeafe; color: #2563eb; }

    .metric-label {
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 2rem;
        color: #1f2937;
        line-height: 1;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .empty-state {
        min-height: 390px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #1f2937;
        position: relative;
        margin-top: 0.2rem;
    }

    .empty-icon {
        width: 190px;
        height: 190px;
        border-radius: 999px;
        background: radial-gradient(circle at 35% 35%, #ffffff 0%, #ede9fe 55%, #ddd6fe 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 5.4rem;
        color: #7c3aed;
        opacity: 0.84;
        margin-bottom: 1.9rem;
        box-shadow: 0 25px 70px rgba(124, 58, 237, 0.12);
    }

    .empty-state h2 {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.7rem;
    }

    .empty-state p {
        color: #64748b;
        line-height: 1.7;
        max-width: 510px;
        margin: 0;
    }

    .trail {
        position: absolute;
        left: 1rem;
        bottom: 1.9rem;
        color: #a78bfa;
        font-size: 4rem;
        opacity: 0.78;
        transform: rotate(-8deg);
    }

    .recommend-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #1f2937;
        letter-spacing: -0.05em;
        margin-bottom: 0.2rem;
    }

    .recommend-subtitle {
        color: #64748b;
        margin-bottom: 1.2rem;
    }

    .best-match {
        border-radius: 16px;
        padding: 0.95rem 1.05rem;
        background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
        border: 1px solid rgba(34, 197, 94, 0.16);
        color: #047857;
        font-weight: 700;
        margin: 0.9rem 0 1rem 0;
    }

    .destination-card {
        border-radius: 22px;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.17);
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.07);
        margin-bottom: 0.9rem;
        padding: 0.8rem;
    }

    .destination-card img {
        border-radius: 18px;
        object-fit: cover;
    }

    .rank-badge {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-weight: 800;
        font-size: 0.78rem;
        margin-bottom: 0.7rem;
    }

    .score-badge {
        display: inline-block;
        padding: 0.45rem 0.78rem;
        border-radius: 999px;
        background: #ecfeff;
        color: #155e75;
        font-weight: 800;
        font-size: 0.82rem;
        margin-bottom: 0.7rem;
    }

    .muted {
        color: #64748b;
        font-size: 0.94rem;
        line-height: 1.65;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        padding: 0.42rem 0.68rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border: 1px solid #e2e8f0;
        color: #334155;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label {
        font-weight: 700;
        color: #1f2937;
    }

    .stSlider [data-baseweb="slider"] {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    .stSlider [role="slider"] {
        width: 22px !important;
        height: 22px !important;
        box-shadow: 0 5px 14px rgba(239, 68, 68, 0.20);
    }

    .stSlider div[data-testid="stTickBar"] {
        height: 8px;
    }

    .stButton > button {
        width: 100%;
        min-height: 56px;
        border-radius: 14px;
        font-weight: 800;
        border: 0;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        box-shadow: 0 16px 32px rgba(79, 70, 229, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 20px 38px rgba(79, 70, 229, 0.32);
    }

    @media (max-width: 980px) {
        .metric-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .hero-card { padding: 2rem; }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Dataset is fixed to your local project path. No CSV upload is shown in the app.
try:
    df, destination_profiles = load_and_prepare_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        "Dataset not found. Please check DATA_PATH in the app file.\n\n"
        f"Current DATA_PATH: {DATA_PATH}"
    )
    st.stop()

if "show_recommendations" not in st.session_state:
    st.session_state.show_recommendations = False

with st.sidebar:
    st.markdown(
        """
        <div class="brand-wrap">
            <div class="brand-icon">✈️</div>
            <div>
                <div class="brand-title">WanderWise</div>
                <div class="brand-subtitle">Travel Recommender</div>
            </div>
        </div>
        <div class="sidebar-heading">✨ Build your traveler profile</div>
        <div class="sidebar-help">Customize your preferences to get destinations that fit your travel style.</div>
        """,
        unsafe_allow_html=True,
    )

    min_budget = int(max(0, destination_profiles["median_total_cost"].min()))
    max_budget = int(destination_profiles["median_total_cost"].max() * 1.5)

    user_budget = st.slider(
        "Budget ($)",
        min_value=min_budget,
        max_value=max_budget,
        value=int(destination_profiles["median_total_cost"].median()),
        step=50,
    )

    user_duration = st.slider(
        "Preferred trip duration (days)",
        min_value=1,
        max_value=int(max(30, destination_profiles["median_duration_days"].max())),
        value=int(destination_profiles["median_duration_days"].median()),
        step=1,
    )

    user_age = st.slider(
        "Traveler age",
        min_value=18,
        max_value=80,
        value=int(destination_profiles["median_traveler_age"].median()),
        step=1,
    )

    accommodation_options = ["Any"] + sorted(df["accommodation_type"].dropna().unique().tolist())
    transportation_options = ["Any"] + sorted(df["transportation_type"].dropna().unique().tolist())
    season_options = ["Any"] + sorted(df["season"].dropna().unique().tolist())

    preferred_accommodation = st.selectbox("Preferred accommodation", accommodation_options)
    preferred_transportation = st.selectbox("Preferred transportation", transportation_options)
    preferred_season = st.selectbox("Preferred season", season_options)

    top_n = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    if st.button("Recommend destinations ✈️", type="primary"):
        st.session_state.show_recommendations = True

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-pills">
            <span class="pill">Profile-based recommendation</span>
            <span class="pill">Budget + duration fit</span>
            <span class="pill">Destination photos</span>
        </div>
        <h1>WanderWise Travel Recommender</h1>
        <p>Find destinations that best match your budget, trip length, travel style, and preferred season.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-icon purple">📍</div>
            <div>
                <div class="metric-label">Unique Destinations</div>
                <div class="metric-value">{destination_profiles['destination'].nunique():,}</div>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-icon green">🧳</div>
            <div>
                <div class="metric-label">Historical Trips</div>
                <div class="metric-value">{len(df):,}</div>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-icon orange">$</div>
            <div>
                <div class="metric-label">Median Trip Cost</div>
                <div class="metric-value">{destination_profiles['median_total_cost'].median():,.0f}</div>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-icon blue">🗓️</div>
            <div>
                <div class="metric-label">Median Duration</div>
                <div class="metric-value">{destination_profiles['median_duration_days'].median():.0f} days</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.show_recommendations:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">🗺️</div>
            <h2>Ready to discover amazing places?</h2>
            <p>Customize your preferences in the sidebar and click the button to get your personalized recommendations.</p>
            <div class="trail">⌁</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

recommendations = recommend_destinations(
    destination_profiles=destination_profiles,
    user_budget=user_budget,
    user_duration=user_duration,
    user_age=user_age,
    preferred_accommodation=preferred_accommodation,
    preferred_transportation=preferred_transportation,
    preferred_season=preferred_season,
    top_n=top_n,
)

hero = recommendations.iloc[0]
st.markdown('<div class="recommend-title">🌍 Top Recommended Destinations</div>', unsafe_allow_html=True)
st.markdown('<div class="recommend-subtitle">Here are the destinations that best match your traveler profile.</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="best-match">Best match: {hero["destination"]} with a match score of {hero["final_score"] * 100:.1f}%</div>',
    unsafe_allow_html=True,
)

for idx, row in recommendations.iterrows():
    image_url = get_destination_image(row["destination"])
    score_pct = row["final_score"] * 100

    st.markdown('<div class="destination-card">', unsafe_allow_html=True)
    col_img, col_info, col_score = st.columns([0.9, 2.4, 0.7])

    with col_img:
        st.image(image_url, use_container_width=True)

    with col_info:
        st.markdown(
            f"""
            <div class="rank-badge">Rank #{idx + 1}</div>
            <h3 style="margin:0 0 0.45rem 0; color:#1f2937; font-size:1.2rem; letter-spacing:-0.04em;">{row['destination']}</h3>
            <p class="muted">{row['recommendation_comment']}</p>
            <div>
                <span class="chip">💰 {row['median_total_cost']:,.0f} typical cost</span>
                <span class="chip">🗓️ {row['median_duration_days']:.0f} days</span>
                <span class="chip">{get_accommodation_icon(row['common_accommodation'])} {row['common_accommodation']}</span>
                <span class="chip">{get_transport_icon(row['common_transportation'])} {row['common_transportation']}</span>
                <span class="chip">{get_season_icon(row['common_season'])} {row['common_season']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_score:
        st.markdown(f'<span class="score-badge">Score: {score_pct:.1f}%</span>', unsafe_allow_html=True)
        st.progress(float(row["final_score"]))
        st.caption(f"Budget fit: {row['budget_fit'] * 100:.0f}%")
        st.caption(f"Duration fit: {row['duration_fit'] * 100:.0f}%")
        st.caption(f"Preference fit: {row['preference_match'] * 100:.0f}%")

    st.markdown('</div>', unsafe_allow_html=True)
