# Travel Destination Recommender System

A Streamlit-based travel destination recommender system developed for STAT280 Practical Machine Learning.  
The app recommends travel destinations based on traveler profile, budget, duration, accommodation preference, transportation preference, travel month, and historical destination patterns.

## Project Overview

This project uses a profile-based and content-based recommendation approach. The system cleans traveler history data, standardizes destination names, engineers travel-related features, and ranks destinations using a weighted scoring framework.

Key features include:

- Destination name normalization  
  - Example: `New York City, USA` and `New York City, United States` are treated as the same destination.
  - Example: `Seoul` and `Seoul, South Korea` are treated as the same destination.
- Accommodation cost per night calculation  
  - `accommodation_per_night = accommodation_cost / duration_days`
- Month-based travel preference instead of broad seasonal grouping
- Budget fit, duration fit, accommodation fit, transportation fit, month fit, popularity score, and age fit
- Interactive Streamlit dashboard
- Downloadable recommendation results

## Repository Structure

```text
STAT280_PML/
│
├── 01-recommender-systems/
│
├── 02-fraud-detection/
│   ├── fraud_detection_sna.ipynb
│   ├── fraud_detection.ipynb
│   ├── README.md
│   ├── data/
│   └── notebooks/
│       ├── Homework/
│       │   ├── 01 Unsupervised Fraud Detection.ipynb
│       │   ├── 02 Group 2 Fraud Detection Homework.ipynb
│       │   ├── 03 Recommender System.html
│       │   ├── 03 Recommender System.ipynb
│       │   └── sample_zomato_recommendations.csv
│       │
│       └── Project/
│           ├── Recommender System Project V1.ipynb
│           ├── WanderWise_Recommender_System_Clean_Final.ipynb
│           ├── wanderwise_streamlit_app.py
│           ├── Travel details dataset.csv
│           └── requirements.txt
│
└── README.md
```

## Main Project Files

| File | Description |
|---|---|
| `WanderWise_Recommender_System_Clean_Final.ipynb` | Clean notebook containing the full recommender system workflow |
| `wanderwise_streamlit_app.py` | Streamlit app for interactive recommendation |
| `Travel details dataset.csv` | Travel history dataset used by the recommender |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files and folders excluded from Git tracking |

## Methodology

The recommender system follows these steps:

1. **Data Cleaning**
   - Standardizes column names
   - Cleans monetary fields
   - Converts numeric fields
   - Handles missing values

2. **Destination Normalization**
   - Converts destination variants into canonical destination names
   - Prevents duplicated recommendations caused by inconsistent naming

3. **Feature Engineering**
   - Computes total trip cost
   - Computes accommodation cost per night
   - Extracts travel month from start date
   - Builds destination-level profiles using historical traveler data

4. **Recommendation Scoring**
   The final recommendation score combines:

   - Budget fit
   - Duration fit
   - Accommodation per night fit
   - Traveler age fit
   - Accommodation preference match
   - Transportation preference match
   - Travel month similarity
   - Historical destination popularity

5. **Streamlit Deployment**
   - Users upload a CSV file
   - Users input traveler preferences
   - The app returns ranked destination recommendations

## Scoring Framework

The final score is a weighted combination of the major recommendation components:

```text
final_score =
    profile_similarity
  + budget_fit
  + duration_fit
  + preference_match
  + month_fit
  + popularity_score
  + age_fit
```

The weights are designed to balance financial feasibility, trip compatibility, stated preferences, historical popularity, and traveler profile similarity.

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository-name>.git
cd <your-repository-name>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

For macOS/Linux:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r "02-fraud-detection/notebooks/Project/requirements.txt"
```

### 4. Run the Streamlit app

```bash
streamlit run "02-fraud-detection/notebooks/Project/wanderwise_streamlit_app.py"
```

## Expected Input Dataset

The uploaded CSV should contain the following minimum columns:

| Column | Description |
|---|---|
| `Destination` | Travel destination |
| `Duration (days)` | Number of travel days |
| `Traveler age` | Traveler age |
| `Accommodation type` | Hotel, Airbnb, Hostel, etc. |
| `Transportation type` | Airplane, Train, Bus, etc. |
| `Accommodation cost` | Total accommodation cost |
| `Transportation cost` | Total transportation cost |
| `Start date` | Travel start date, used to extract travel month |

Column names may be cleaned automatically by the app, but keeping a consistent format is recommended.

## Example Use Case

A traveler provides:

- Budget: 1,500
- Trip duration: 7 days
- Age: 28
- Preferred accommodation: Airbnb
- Preferred transportation: Airplane
- Preferred travel month: November

The app ranks destinations based on how similar each destination's historical profile is to the user's preferred travel profile.

## Notes

- The app uses historical patterns and profile similarity. It does not guarantee actual real-time flight or hotel prices.
- Recommendation quality depends on the completeness and quality of the uploaded travel dataset.
- Destination normalization is intentionally included to avoid duplicate recommendations caused by inconsistent destination naming.

## Author

Wilmar Mangapot  
STAT280 Practical Machine Learning
