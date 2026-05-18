"""
Travel Destination Recommender — Improved Edition
Run: streamlit run travel_recommender_app.py
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Where To Next?",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — editorial travel magazine aesthetic
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ---- global ---- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F5F0E8;
    color: #1A1A1A;
}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {
    background-color: #1A1A1A !important;
    color: #F5F0E8;
}
[data-testid="stSidebar"] * {
    color: #F5F0E8 !important;
}
[data-testid="stSidebar"] .stSlider > div > div {
    background: #C8A26B;
}
[data-testid="stSidebar"] label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #C8A26B !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Playfair Display', serif !important;
    color: #F5F0E8 !important;
}

/* ---- headings ---- */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
}

/* ---- metric cards ---- */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E0D8CC;
    border-radius: 4px;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.6rem !important;
    color: #1A1A1A !important;
}

/* ---- destination cards ---- */
.dest-card {
    background: #FFFFFF;
    border: 1px solid #E0D8CC;
    border-radius: 6px;
    padding: 0;
    margin-bottom: 20px;
    overflow: hidden;
    display: flex;
    gap: 0;
    transition: box-shadow 0.2s;
}
.dest-card:hover {
    box-shadow: 0 8px 32px rgba(0,0,0,0.10);
}
.dest-card-rank {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: #E0D8CC;
    font-style: italic;
    padding: 24px 12px 24px 24px;
    min-width: 64px;
    display: flex;
    align-items: flex-start;
    line-height: 1;
}
.dest-card-body {
    flex: 1;
    padding: 24px 24px 24px 0;
}
.dest-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    font-weight: 700;
    margin: 0 0 4px 0;
    color: #1A1A1A;
}
.dest-card-comment {
    font-size: 0.88rem;
    color: #555;
    margin: 0 0 12px 0;
    line-height: 1.5;
}
.dest-card-meta {
    font-size: 0.80rem;
    color: #888;
    margin-bottom: 14px;
}
.dest-card-meta span {
    margin-right: 16px;
}
.score-bar-label {
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin-bottom: 2px;
}
.score-bar-track {
    background: #F0EAE0;
    border-radius: 2px;
    height: 6px;
    width: 100%;
    margin-bottom: 6px;
}
.score-bar-fill {
    height: 6px;
    border-radius: 2px;
    background: #C8A26B;
}
.dest-card-score {
    min-width: 90px;
    background: #1A1A1A;
    color: #F5F0E8;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 24px 16px;
    text-align: center;
}
.dest-score-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #C8A26B;
    line-height: 1;
}
.dest-score-label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-top: 4px;
}
.dest-trips {
    font-size: 0.75rem;
    color: #aaa;
    margin-top: 8px;
}

/* ---- section divider ---- */
.section-rule {
    border: none;
    border-top: 2px solid #E0D8CC;
    margin: 32px 0 24px 0;
}

/* ---- upload prompt ---- */
.upload-hero {
    text-align: center;
    padding: 80px 40px;
    color: #888;
}
.upload-hero h2 {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    color: #1A1A1A;
    margin-bottom: 8px;
}

/* ---- tag pills ---- */
.tag {
    display: inline-block;
    background: #F5F0E8;
    border: 1px solid #E0D8CC;
    border-radius: 2px;
    font-size: 0.72rem;
    padding: 2px 8px;
    margin-right: 6px;
    color: #555;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

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
# UTILITY FUNCTIONS (unchanged logic, same as original)
# ============================================================

def clean_column_names(dataframe):
    df = dataframe.copy()
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r"[^\w]+", "_", regex=True).str.strip("_")
    )
    return df

def clean_money(value):
    if pd.isna(value):
        return np.nan
    value = re.sub(r"[^0-9.]", "", str(value).strip())
    return float(value) if value else np.nan

def normalize_destination_key(value):
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    abbreviation_map = {
        "u.s.a.": "usa", "u.s.": "usa",
        "united states of america": "united states", "usa": "united states",
        "us": "united states", "uk": "united kingdom", "u.k.": "united kingdom",
        "uae": "united arab emirates", "u.a.e.": "united arab emirates",
        "aus": "australia", "thai": "thailand",
        "sa": "south africa", "korea": "south korea",
    }
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for short, full in abbreviation_map.items():
        short_clean = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", short)).strip()
        text = re.sub(rf"\b{re.escape(short_clean)}\b", full, text)
    return re.sub(r"\s+", " ", text).strip()

DESTINATION_ALIAS_MAP = {
    "sydney": "Sydney, Australia", "sydney australia": "Sydney, Australia", "australia": "Sydney, Australia",
    "paris": "Paris, France", "paris france": "Paris, France", "france": "Paris, France",
    "tokyo": "Tokyo, Japan", "tokyo japan": "Tokyo, Japan", "japan": "Tokyo, Japan",
    "bali": "Bali, Indonesia", "bali indonesia": "Bali, Indonesia", "indonesia": "Bali, Indonesia",
    "new york": "New York City, United States", "new york city": "New York City, United States",
    "new york united states": "New York City, United States", "nyc": "New York City, United States",
    "new york city united states": "New York City, United States",
    "hawaii": "Honolulu, Hawaii", "honolulu": "Honolulu, Hawaii", "honolulu hawaii": "Honolulu, Hawaii",
    "united states": "New York City, United States",
    "rome": "Rome, Italy", "rome italy": "Rome, Italy", "italy": "Rome, Italy",
    "bangkok": "Bangkok, Thailand", "bangkok thailand": "Bangkok, Thailand",
    "phuket": "Phuket, Thailand", "phuket thailand": "Phuket, Thailand", "thailand": "Bangkok, Thailand",
    "barcelona": "Barcelona, Spain", "barcelona spain": "Barcelona, Spain", "spain": "Barcelona, Spain",
    "london": "London, United Kingdom", "london united kingdom": "London, United Kingdom", "united kingdom": "London, United Kingdom",
    "cape town": "Cape Town, South Africa", "cape town south africa": "Cape Town, South Africa", "south africa": "Cape Town, South Africa",
    "dubai": "Dubai, United Arab Emirates", "dubai united arab emirates": "Dubai, United Arab Emirates", "united arab emirates": "Dubai, United Arab Emirates",
    "amsterdam": "Amsterdam, Netherlands", "amsterdam netherlands": "Amsterdam, Netherlands", "netherlands": "Amsterdam, Netherlands",
    "rio": "Rio de Janeiro, Brazil", "rio de janeiro": "Rio de Janeiro, Brazil", "rio de janeiro brazil": "Rio de Janeiro, Brazil", "brazil": "Rio de Janeiro, Brazil",
    "athens": "Athens, Greece", "athens greece": "Athens, Greece",
    "santorini": "Santorini, Greece", "santorini greece": "Santorini, Greece", "greece": "Athens, Greece",
    "seoul": "Seoul, South Korea", "seoul south korea": "Seoul, South Korea", "south korea": "Seoul, South Korea",
    "vancouver": "Vancouver, Canada", "vancouver canada": "Vancouver, Canada", "canada": "Vancouver, Canada",
    "cancun": "Cancun, Mexico", "cancun mexico": "Cancun, Mexico", "mexico": "Cancun, Mexico",
    "auckland": "Auckland, New Zealand", "auckland new zealand": "Auckland, New Zealand", "new zealand": "Auckland, New Zealand",
    "berlin": "Berlin, Germany", "berlin germany": "Berlin, Germany", "germany": "Berlin, Germany",
    "edinburgh": "Edinburgh, Scotland", "edinburgh scotland": "Edinburgh, Scotland", "scotland": "Edinburgh, Scotland",
    "marrakech": "Marrakech, Morocco", "marrakech morocco": "Marrakech, Morocco", "morocco": "Marrakech, Morocco",
    "phnom penh": "Phnom Penh, Cambodia", "phnom penh cambodia": "Phnom Penh, Cambodia", "cambodia": "Phnom Penh, Cambodia",
    "egypt": "Cairo, Egypt", "cairo": "Cairo, Egypt", "cairo egypt": "Cairo, Egypt",
}

def canonicalize_destination(value):
    key = normalize_destination_key(value)
    if key in DESTINATION_ALIAS_MAP:
        return DESTINATION_ALIAS_MAP[key]
    if pd.isna(value):
        return "Unknown"
    text = re.sub(r"\s+", " ", str(value).strip()).replace(" ,", ",").replace(", ", ", ")
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
    return "Unknown" if pd.isna(value) else str(value).strip().title()

def mode_or_unknown(series):
    series = series.dropna()
    return "Unknown" if len(series) == 0 else series.mode().iloc[0]

def get_season_from_date(value):
    if pd.isna(value):
        return "Unknown"
    d = pd.to_datetime(value, errors="coerce")
    if pd.isna(d):
        return "Unknown"
    return {12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer"}.get(d.month, "Fall")

def safe_minmax(series):
    series = series.astype(float)
    if series.max() == series.min():
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def similarity_score(user_value, destination_value, tolerance):
    if pd.isna(destination_value):
        return 0
    return max(0, min(1, 1 - abs(user_value - destination_value) / tolerance))

def preference_score(user_choice, destination_choice):
    if user_choice == "Any":
        return 1
    if pd.isna(destination_choice):
        return 0
    return 1 if str(user_choice).lower() == str(destination_choice).lower() else 0

def generate_comment(row):
    reasons = []
    if row["profile_similarity"] >= 0.70:
        reasons.append("strong traveler profile match")
    elif row["profile_similarity"] >= 0.40:
        reasons.append("moderate profile similarity")
    if row["budget_fit"] >= 0.80:
        reasons.append("excellent budget fit")
    elif row["budget_fit"] >= 0.50:
        reasons.append("reasonable budget fit")
    if row["duration_fit"] >= 0.80:
        reasons.append("ideal trip length")
    if row["preference_match"] >= 0.75:
        reasons.append("matches your stated preferences")
    elif row["preference_match"] >= 0.40:
        reasons.append("partial preference match")
    if row["popularity_score"] >= 0.70:
        reasons.append("highly popular destination")
    if row["age_fit"] >= 0.70:
        reasons.append("popular with travelers your age")
    if not reasons:
        return "Recommended based on overall weighted score."
    return "Recommended for its " + ", ".join(reasons) + "."

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_and_prepare_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = clean_column_names(df)

    required_cols = ["destination", "duration_days", "traveler_age",
                     "accommodation_type", "transportation_type",
                     "accommodation_cost", "transportation_cost"]
    missing_cols = [c for c in required_cols if c not in df.columns]
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
    df["season"] = df["start_date"].apply(get_season_from_date) if "start_date" in df.columns else "Unknown"
    df = df.dropna(subset=["destination", "duration_days", "traveler_age", "total_cost"])

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

# ============================================================
# SCORING — fixed double-counting of profile_similarity
# ============================================================

def recommend_destinations(destination_profiles, user_budget, user_duration,
                            user_age, preferred_accommodation,
                            preferred_transportation, preferred_season, top_n=10):
    result = destination_profiles.copy()

    result["budget_fit"] = result["median_total_cost"].apply(
        lambda x: similarity_score(user_budget, x, max(user_budget, 1)))
    result["duration_fit"] = result["median_duration_days"].apply(
        lambda x: similarity_score(user_duration, x, max(user_duration, 1)))
    result["age_fit"] = result["median_traveler_age"].apply(
        lambda x: similarity_score(user_age, x, 30))

    result["accommodation_match"] = result["common_accommodation"].apply(
        lambda x: preference_score(preferred_accommodation, x))
    result["transportation_match"] = result["common_transportation"].apply(
        lambda x: preference_score(preferred_transportation, x))
    result["season_match"] = result["common_season"].apply(
        lambda x: preference_score(preferred_season, x))
    result["preference_match"] = (
        result["accommodation_match"] + result["transportation_match"] + result["season_match"]
    ) / 3

    result["profile_similarity"] = (
        0.40 * result["budget_fit"]
        + 0.30 * result["duration_fit"]
        + 0.30 * result["age_fit"]
    )

    # Fixed: weights now sum to 1.0 (removed double-counting of profile_similarity)
    result["final_score"] = (
        0.30 * result["budget_fit"]
        + 0.20 * result["duration_fit"]
        + 0.20 * result["age_fit"]
        + 0.20 * result["preference_match"]
        + 0.10 * result["popularity_score"]
    )

    result["recommendation_comment"] = result.apply(generate_comment, axis=1)
    result = result.sort_values("final_score", ascending=False).head(top_n)

    return result[[
        "destination", "final_score", "profile_similarity", "budget_fit",
        "duration_fit", "preference_match", "popularity_score", "age_fit",
        "median_total_cost", "median_duration_days", "common_accommodation",
        "common_transportation", "common_season", "trip_count", "recommendation_comment"
    ]].reset_index(drop=True)

# ============================================================
# SCORE BAR HTML HELPER
# ============================================================

def score_bar(label, value, color="#C8A26B"):
    pct = int(value * 100)
    return f"""
    <div class="score-bar-label">{label}</div>
    <div class="score-bar-track">
      <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    """

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌍 Where To Next?")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload travel dataset CSV", type=["csv"])

    if uploaded_file:
        st.markdown("### Your Profile")

# ============================================================
# MAIN — upload gate
# ============================================================

if not uploaded_file:
    st.markdown("""
    <div class="upload-hero">
      <h2>Where To Next?</h2>
      <p style="font-size:1.1rem; margin-bottom:32px;">
        Upload your travel history CSV in the sidebar to discover your perfect destination.
      </p>
      <p style="font-size:0.85rem; color:#aaa;">
        Required columns: destination · duration_days · traveler_age ·
        accommodation_type · transportation_type · accommodation_cost · transportation_cost
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df, destination_profiles = load_and_prepare_data(uploaded_file)

# ---- sidebar controls ----
with st.sidebar:
    min_budget = int(max(0, destination_profiles["median_total_cost"].min()))
    max_budget = int(destination_profiles["median_total_cost"].max() * 1.5)

    user_budget = st.slider("Budget (total trip cost)",
        min_value=min_budget, max_value=max_budget,
        value=int(destination_profiles["median_total_cost"].median()), step=100)

    user_duration = st.slider("Trip duration (days)",
        min_value=1, max_value=int(max(30, destination_profiles["median_duration_days"].max())),
        value=int(destination_profiles["median_duration_days"].median()), step=1)

    user_age = st.slider("Your age",
        min_value=18, max_value=80,
        value=int(destination_profiles["median_traveler_age"].median()), step=1)

    st.markdown("### Preferences")

    preferred_accommodation = st.selectbox("Accommodation",
        ["Any"] + sorted(df["accommodation_type"].dropna().unique().tolist()))
    preferred_transportation = st.selectbox("Transportation",
        ["Any"] + sorted(df["transportation_type"].dropna().unique().tolist()))
    preferred_season = st.selectbox("Season",
        ["Any"] + sorted(df["season"].dropna().unique().tolist()))

    st.markdown("### Results")
    top_n = st.slider("Number of recommendations", min_value=3, max_value=15, value=5, step=1)

# ============================================================
# COMPUTE RECOMMENDATIONS
# ============================================================

recommendations = recommend_destinations(
    destination_profiles=destination_profiles,
    user_budget=user_budget, user_duration=user_duration, user_age=user_age,
    preferred_accommodation=preferred_accommodation,
    preferred_transportation=preferred_transportation,
    preferred_season=preferred_season, top_n=top_n
)

# ============================================================
# HEADER + SUMMARY METRICS
# ============================================================

st.markdown(
    "<h1 style='margin-bottom:4px;'>Travel Recommendations</h1>"
    "<p style='color:#888; margin-bottom:24px; font-size:0.95rem;'>"
    "Personalized destinations ranked by budget fit, duration, preferences, and popularity.</p>",
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique Destinations", destination_profiles["destination"].nunique())
c2.metric("Historical Trips", f"{len(df):,}")
c3.metric("Median Trip Cost", f"${destination_profiles['median_total_cost'].median():,.0f}")
c4.metric("Top Pick", recommendations.iloc[0]["destination"].split(",")[0])

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

# ============================================================
# DESTINATION CARDS  — fixed rank using enumerate
# ============================================================

st.markdown("### ✈️ Top Destinations For You")

for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
    image_url = get_destination_image(row["destination"]) + "?w=400&q=80"
    tags_html = (
        f'<span class="tag">🏨 {row["common_accommodation"]}</span>'
        f'<span class="tag">🚌 {row["common_transportation"]}</span>'
        f'<span class="tag">🌤 {row["common_season"]}</span>'
    )
    bars_html = (
        score_bar("Budget Fit", row["budget_fit"]) +
        score_bar("Duration Fit", row["duration_fit"], "#7C9E87") +
        score_bar("Preferences", row["preference_match"], "#9B7EB8") +
        score_bar("Popularity", row["popularity_score"], "#D4846A")
    )

    col_img, col_body = st.columns([1, 2.8])

    with col_img:
        st.image(image_url, use_container_width=True)

    with col_body:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
              <div>
                <div style="font-size:0.75rem; letter-spacing:0.12em; color:#C8A26B;
                            text-transform:uppercase; font-weight:500;">
                  #{rank}
                </div>
                <div style="font-family:'Playfair Display',serif; font-size:1.5rem;
                            font-weight:700; margin:2px 0 6px;">
                  {row["destination"]}
                </div>
                <div style="font-size:0.85rem; color:#666; margin-bottom:10px; line-height:1.5;">
                  {row["recommendation_comment"]}
                </div>
                <div style="margin-bottom:12px;">{tags_html}</div>
                <div style="font-size:0.80rem; color:#888; margin-bottom:14px;">
                  Typical cost <strong>${row['median_total_cost']:,.0f}</strong> &nbsp;·&nbsp;
                  {row['median_duration_days']:.0f} days &nbsp;·&nbsp;
                  {int(row['trip_count'])} trips in dataset
                </div>
              </div>
              <div style="text-align:center; background:#1A1A1A; border-radius:4px;
                          padding:16px 20px; min-width:80px; margin-left:20px;">
                <div style="font-family:'Playfair Display',serif; font-size:1.9rem;
                            color:#C8A26B; line-height:1;">{row['final_score']:.2f}</div>
                <div style="font-size:0.60rem; letter-spacing:0.12em;
                            text-transform:uppercase; color:#888; margin-top:4px;">Score</div>
              </div>
            </div>
            {bars_html}
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

# ============================================================
# CHARTS
# ============================================================

st.markdown("### Score Breakdown")

score_chart_data = recommendations.melt(
    id_vars="destination",
    value_vars=["profile_similarity", "budget_fit", "duration_fit",
                "preference_match", "popularity_score", "age_fit"],
    var_name="score_component", value_name="score"
)
score_chart_data["score_component"] = (
    score_chart_data["score_component"]
    .str.replace("_", " ").str.title()
)

chart = (
    alt.Chart(score_chart_data)
    .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
    .encode(
        x=alt.X("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("destination:N", sort="-x", title=None),
        color=alt.Color("score_component:N", title="Component",
                        scale=alt.Scale(scheme="tableau10")),
        tooltip=["destination", "score_component",
                 alt.Tooltip("score:Q", format=".3f")]
    )
    .properties(height=max(300, top_n * 40))
    .configure_view(strokeWidth=0)
    .configure_axis(grid=False, labelFont="DM Sans", titleFont="DM Sans")
    .configure_legend(labelFont="DM Sans", titleFont="DM Sans")
)
st.altair_chart(chart, use_container_width=True)

st.markdown("### Cost vs. Duration")

recommended_set = set(recommendations["destination"])
scatter_data = destination_profiles.copy()
scatter_data["is_recommended"] = scatter_data["destination"].isin(recommended_set)
scatter_data["label"] = scatter_data["destination"].apply(
    lambda d: d.split(",")[0] if d in recommended_set else "")

base = alt.Chart(scatter_data).encode(
    x=alt.X("median_total_cost:Q", title="Median Total Cost ($)"),
    y=alt.Y("median_duration_days:Q", title="Median Duration (days)"),
    size=alt.Size("trip_count:Q", title="Trip Count", scale=alt.Scale(range=[40, 400])),
    tooltip=["destination",
             alt.Tooltip("median_total_cost:Q", format=",.0f", title="Cost"),
             alt.Tooltip("median_duration_days:Q", format=".0f", title="Days"),
             "trip_count"]
)

points = base.mark_circle().encode(
    color=alt.condition(
        alt.datum.is_recommended,
        alt.value("#C8A26B"),
        alt.value("#D0C8BC")
    ),
    opacity=alt.condition(
        alt.datum.is_recommended,
        alt.value(1.0),
        alt.value(0.45)
    )
)

labels = (
    alt.Chart(scatter_data[scatter_data["is_recommended"]])
    .mark_text(dy=-12, fontSize=10, font="DM Sans", color="#1A1A1A")
    .encode(
        x="median_total_cost:Q",
        y="median_duration_days:Q",
        text="label:N"
    )
)

scatter = (
    (points + labels)
    .properties(height=380)
    .configure_view(strokeWidth=0)
    .configure_axis(grid=True, gridColor="#F0EAE0", labelFont="DM Sans", titleFont="DM Sans")
)
st.altair_chart(scatter, use_container_width=True)

# ============================================================
# RAW DATA EXPANDERS
# ============================================================

with st.expander("📊 Full recommendation table"):
    st.dataframe(recommendations, use_container_width=True, hide_index=True)

with st.expander("🗺️ Destination profile data"):
    st.dataframe(destination_profiles, use_container_width=True, hide_index=True)

with st.expander("🔍 Raw data sample (first 50 rows)"):
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="margin-top:48px; padding-top:24px; border-top:1px solid #E0D8CC;
            font-size:0.75rem; color:#aaa; text-align:center;">
  Score = 30% budget fit · 20% duration · 20% age fit · 20% preferences · 10% popularity
</div>
""", unsafe_allow_html=True)
