
# ============================================================
# Travel Destination Recommender App
# ============================================================
# Run using:
# streamlit run travel_recommender_app.py
# ============================================================

import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Travel Destination Recommender",
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


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

@st.cache_data
def load_and_prepare_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
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

st.title("✈️ Travel Destination Recommender")
st.caption("A content-based and profile-based recommender using traveler history, budget fit, duration fit, preference matching, and destination popularity.")

uploaded_file = st.sidebar.file_uploader(
    "Upload travel dataset CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload your travel dataset CSV to start.")
    st.stop()

df, destination_profiles = load_and_prepare_data(uploaded_file)

st.sidebar.header("Traveler Profile")

min_budget = int(max(0, destination_profiles["median_total_cost"].min()))
max_budget = int(destination_profiles["median_total_cost"].max() * 1.5)

user_budget = st.sidebar.slider(
    "Budget",
    min_value=min_budget,
    max_value=max_budget,
    value=int(destination_profiles["median_total_cost"].median()),
    step=100
)

user_duration = st.sidebar.slider(
    "Preferred trip duration (days)",
    min_value=1,
    max_value=int(max(30, destination_profiles["median_duration_days"].max())),
    value=int(destination_profiles["median_duration_days"].median()),
    step=1
)

user_age = st.sidebar.slider(
    "Traveler age",
    min_value=18,
    max_value=80,
    value=int(destination_profiles["median_traveler_age"].median()),
    step=1
)

accommodation_options = ["Any"] + sorted(df["accommodation_type"].dropna().unique().tolist())
transportation_options = ["Any"] + sorted(df["transportation_type"].dropna().unique().tolist())
season_options = ["Any"] + sorted(df["season"].dropna().unique().tolist())

preferred_accommodation = st.sidebar.selectbox(
    "Preferred accommodation",
    accommodation_options
)

preferred_transportation = st.sidebar.selectbox(
    "Preferred transportation",
    transportation_options
)

preferred_season = st.sidebar.selectbox(
    "Preferred season",
    season_options
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=3,
    max_value=15,
    value=5,
    step=1
)

recommendations = recommend_destinations(
    destination_profiles=destination_profiles,
    user_budget=user_budget,
    user_duration=user_duration,
    user_age=user_age,
    preferred_accommodation=preferred_accommodation,
    preferred_transportation=preferred_transportation,
    preferred_season=preferred_season,
    top_n=top_n
)


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Unique Destinations", destination_profiles["destination"].nunique())

with col2:
    st.metric("Historical Trips", len(df))

with col3:
    st.metric("Median Trip Cost", f"{destination_profiles['median_total_cost'].median():,.0f}")

with col4:
    st.metric("Top Recommendation", recommendations.iloc[0]["destination"])


# ============================================================
# RECOMMENDATION OUTPUT
# ============================================================

st.subheader("Top Recommended Destinations")

for idx, row in recommendations.iterrows():
    image_url = get_destination_image(row["destination"])

    with st.container(border=True):
        col_img, col_info, col_score = st.columns([1.2, 2.2, 0.8])

        with col_img:
            st.image(image_url, use_container_width=True)

        with col_info:
            st.markdown(f"### {idx + 1}. {row['destination']}")
            st.write(row["recommendation_comment"])

            st.write(
                f"**Typical cost:** {row['median_total_cost']:,.0f} | "
                f"**Typical duration:** {row['median_duration_days']:.0f} days"
            )

            st.write(
                f"**Accommodation:** {row['common_accommodation']} | "
                f"**Transport:** {row['common_transportation']} | "
                f"**Season:** {row['common_season']}"
            )

        with col_score:
            st.metric("Final Score", f"{row['final_score']:.3f}")
            st.metric("Trip Count", int(row["trip_count"]))


st.subheader("Recommendation Table")

st.dataframe(
    recommendations,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# VISUALS
# ============================================================

st.subheader("Score Breakdown")

score_chart_data = recommendations.melt(
    id_vars="destination",
    value_vars=[
        "profile_similarity",
        "budget_fit",
        "duration_fit",
        "preference_match",
        "popularity_score",
        "age_fit"
    ],
    var_name="score_component",
    value_name="score"
)

chart = (
    alt.Chart(score_chart_data)
    .mark_bar()
    .encode(
        x=alt.X("score:Q", title="Score"),
        y=alt.Y("destination:N", sort="-x", title="Destination"),
        color=alt.Color("score_component:N", title="Component"),
        tooltip=["destination", "score_component", alt.Tooltip("score:Q", format=".3f")]
    )
    .properties(height=400)
)

st.altair_chart(chart, use_container_width=True)


st.subheader("Cost vs Duration Profile")

scatter = (
    alt.Chart(destination_profiles)
    .mark_circle(size=100)
    .encode(
        x=alt.X("median_total_cost:Q", title="Median Total Cost"),
        y=alt.Y("median_duration_days:Q", title="Median Duration Days"),
        size=alt.Size("trip_count:Q", title="Trip Count"),
        tooltip=[
            "destination",
            alt.Tooltip("median_total_cost:Q", format=",.0f"),
            alt.Tooltip("median_duration_days:Q", format=".0f"),
            "trip_count"
        ]
    )
    .properties(height=400)
)

st.altair_chart(scatter, use_container_width=True)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

with st.expander("View cleaned destination profile data"):
    st.dataframe(destination_profiles, use_container_width=True, hide_index=True)

with st.expander("View cleaned raw data sample"):
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Scoring logic: final score combines profile similarity, budget fit, duration fit, preference match, popularity, and age fit."
)
