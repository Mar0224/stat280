# ============================================================
# Travel Destination Recommender App
# Updated algorithm version
# ============================================================
# Run using:
# streamlit run travel_recommender_app_updated_algorithm.py
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
    page_title="WanderWise Travel Recommender",
    page_icon="✈️",
    layout="wide",
)


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
    "new york city usa": "New York City, United States",
    "new york usa": "New York City, United States",
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
    Examples treated as one destination:
    - New York City, USA
    - New York City, United States
    - Seoul
    - Seoul, South Korea
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


def month_name_from_number(value):
    if pd.isna(value):
        return "Unknown"
    try:
        month_num = int(value)
        if 1 <= month_num <= 12:
            return pd.Timestamp(year=2026, month=month_num, day=1).strftime("%B")
    except Exception:
        return "Unknown"
    return "Unknown"


def infer_travel_month(df):
    """
    Uses explicit month column first. If not available, derives month from a date column.
    """
    month_candidates = [
        "travel_month",
        "month",
        "start_month",
        "trip_month",
    ]
    date_candidates = [
        "start_date",
        "travel_date",
        "trip_start_date",
        "date",
    ]

    for col in month_candidates:
        if col in df.columns:
            month = pd.to_numeric(df[col], errors="coerce")
            if month.notna().sum() > 0:
                return month

    for col in date_candidates:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                return parsed.dt.month

    return pd.Series(np.nan, index=df.index)


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


def month_similarity_score(user_month, destination_month):
    """
    Cyclical month similarity. January is close to December.
    Score is 1 for same month, 0 for six months apart.
    """
    if user_month == "Any" or pd.isna(destination_month):
        return 1

    user_month = int(user_month)
    destination_month = int(destination_month)
    diff = abs(user_month - destination_month)
    cyclical_diff = min(diff, 12 - diff)
    return max(0, 1 - (cyclical_diff / 6))


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

    if row["accommodation_per_night_fit"] >= 0.80:
        reasons.append("strong accommodation-per-night fit")
    elif row["accommodation_per_night_fit"] >= 0.50:
        reasons.append("acceptable accommodation-per-night fit")

    if row["preference_match"] >= 0.75:
        reasons.append("matches most stated preferences")
    elif row["preference_match"] >= 0.40:
        reasons.append("matches some stated preferences")

    if row["month_fit"] >= 0.80:
        reasons.append("good travel-month match")

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

    df["accommodation_per_night"] = np.where(
        df["duration_days"] > 0,
        df["accommodation_cost_clean"] / df["duration_days"],
        np.nan
    )

    df["accommodation_type"] = df["accommodation_type"].apply(normalize_text)
    df["transportation_type"] = df["transportation_type"].apply(normalize_text)

    df["travel_month"] = infer_travel_month(df)
    df["travel_month_name"] = df["travel_month"].apply(month_name_from_number)

    df = df.dropna(
        subset=[
            "destination",
            "duration_days",
            "traveler_age",
            "total_cost",
            "accommodation_per_night"
        ]
    )

    destination_profiles = (
        df.groupby("destination")
        .agg(
            median_total_cost=("total_cost", "median"),
            median_duration_days=("duration_days", "median"),
            median_traveler_age=("traveler_age", "median"),
            median_accommodation_per_night=("accommodation_per_night", "median"),
            common_accommodation=("accommodation_type", mode_or_unknown),
            common_transportation=("transportation_type", mode_or_unknown),
            common_travel_month=("travel_month", mode_or_unknown),
            common_travel_month_name=("travel_month_name", mode_or_unknown),
            trip_count=("destination", "count")
        )
        .reset_index()
    )

    destination_profiles["common_travel_month"] = pd.to_numeric(
        destination_profiles["common_travel_month"], errors="coerce"
    )

    destination_profiles["popularity_score"] = safe_minmax(destination_profiles["trip_count"])

    return df, destination_profiles


def recommend_destinations(
    destination_profiles,
    user_budget,
    user_duration,
    user_age,
    user_accommodation_per_night,
    preferred_accommodation,
    preferred_transportation,
    preferred_month,
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

    result["accommodation_per_night_fit"] = result["median_accommodation_per_night"].apply(
        lambda x: similarity_score(
            user_accommodation_per_night,
            x,
            tolerance=max(user_accommodation_per_night, 1)
        )
    )

    result["accommodation_match"] = result["common_accommodation"].apply(
        lambda x: preference_score(preferred_accommodation, x)
    )

    result["transportation_match"] = result["common_transportation"].apply(
        lambda x: preference_score(preferred_transportation, x)
    )

    result["month_fit"] = result["common_travel_month"].apply(
        lambda x: month_similarity_score(preferred_month, x)
    )

    result["preference_match"] = (
        result["accommodation_match"]
        + result["transportation_match"]
        + result["month_fit"]
    ) / 3

    result["profile_similarity"] = (
        0.35 * result["budget_fit"]
        + 0.25 * result["duration_fit"]
        + 0.20 * result["age_fit"]
        + 0.20 * result["accommodation_per_night_fit"]
    )

    result["final_score"] = (
        0.30 * result["profile_similarity"]
        + 0.20 * result["budget_fit"]
        + 0.15 * result["duration_fit"]
        + 0.15 * result["preference_match"]
        + 0.10 * result["accommodation_per_night_fit"]
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
        "accommodation_per_night_fit",
        "preference_match",
        "month_fit",
        "popularity_score",
        "age_fit",
        "median_total_cost",
        "median_duration_days",
        "median_accommodation_per_night",
        "common_accommodation",
        "common_transportation",
        "common_travel_month_name",
        "trip_count",
        "recommendation_comment"
    ]

    return result[display_cols].reset_index(drop=True)


# ============================================================
# APP UI
# ============================================================


st.title("✈️ WanderWise Travel Recommender")
st.caption(
    "A content-based and profile-based recommender using traveler history, budget fit, duration fit, accommodation-per-night fit, month matching, preference matching, and destination popularity."
)

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

min_accom_per_night = int(max(0, destination_profiles["median_accommodation_per_night"].min()))
max_accom_per_night = int(max(1, destination_profiles["median_accommodation_per_night"].max() * 1.5))

user_accommodation_per_night = st.sidebar.slider(
    "Accommodation per night budget",
    min_value=min_accom_per_night,
    max_value=max_accom_per_night,
    value=int(destination_profiles["median_accommodation_per_night"].median()),
    step=10
)

accommodation_options = ["Any"] + sorted(df["accommodation_type"].dropna().unique().tolist())
transportation_options = ["Any"] + sorted(df["transportation_type"].dropna().unique().tolist())

month_options = {"Any": "Any"}
for month_num in range(1, 13):
    month_options[month_name_from_number(month_num)] = month_num

preferred_accommodation = st.sidebar.selectbox(
    "Preferred accommodation",
    accommodation_options
)

preferred_transportation = st.sidebar.selectbox(
    "Preferred transportation",
    transportation_options
)

preferred_month_label = st.sidebar.selectbox(
    "Preferred travel month",
    list(month_options.keys())
)
preferred_month = month_options[preferred_month_label]

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
    user_accommodation_per_night=user_accommodation_per_night,
    preferred_accommodation=preferred_accommodation,
    preferred_transportation=preferred_transportation,
    preferred_month=preferred_month,
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
    with st.container(border=True):
        left, right = st.columns([2, 1])

        with left:
            st.markdown(f"### {idx + 1}. {row['destination']}")
            st.write(row["recommendation_comment"])

            st.write(
                f"**Typical cost:** {row['median_total_cost']:,.0f} | "
                f"**Typical duration:** {row['median_duration_days']:.0f} days | "
                f"**Accommodation/night:** {row['median_accommodation_per_night']:,.0f} | "
                f"**Common accommodation:** {row['common_accommodation']} | "
                f"**Common transport:** {row['common_transportation']} | "
                f"**Common travel month:** {row['common_travel_month_name']}"
            )

        with right:
            st.metric("Final Score", f"{row['final_score']:.3f}")
            st.metric("Trip Count", int(row["trip_count"]))


st.subheader("Recommendation Table")

st.dataframe(
    recommendations,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    label="Download recommendations as CSV",
    data=recommendations.to_csv(index=False).encode("utf-8"),
    file_name="travel_recommendations.csv",
    mime="text/csv"
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
        "accommodation_per_night_fit",
        "preference_match",
        "month_fit",
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
            alt.Tooltip("median_accommodation_per_night:Q", format=",.0f"),
            "common_travel_month_name",
            "trip_count"
        ]
    )
    .properties(height=400)
)

st.altair_chart(scatter, use_container_width=True)


st.subheader("Accommodation Per Night vs Total Cost")

accommodation_scatter = (
    alt.Chart(destination_profiles)
    .mark_circle(size=100)
    .encode(
        x=alt.X("median_accommodation_per_night:Q", title="Median Accommodation Cost Per Night"),
        y=alt.Y("median_total_cost:Q", title="Median Total Cost"),
        size=alt.Size("trip_count:Q", title="Trip Count"),
        tooltip=[
            "destination",
            alt.Tooltip("median_accommodation_per_night:Q", format=",.0f"),
            alt.Tooltip("median_total_cost:Q", format=",.0f"),
            alt.Tooltip("median_duration_days:Q", format=".0f"),
            "common_travel_month_name",
            "trip_count"
        ]
    )
    .properties(height=400)
)

st.altair_chart(accommodation_scatter, use_container_width=True)


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
    "Scoring logic: final score combines profile similarity, budget fit, duration fit, accommodation-per-night fit, preference match, month fit, popularity, and age fit."
)