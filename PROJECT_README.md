# WanderWise Travel Recommender

This folder contains the final notebook, dataset, requirements file, and Streamlit app for the travel destination recommender system.

## Files

| File | Purpose |
|---|---|
| `WanderWise_Recommender_System_Clean_Final.ipynb` | Final documented notebook |
| `wanderwise_streamlit_app.py` | Streamlit application |
| `Travel details dataset.csv` | Travel dataset |
| `requirements.txt` | Required Python packages |

## Run App

From the repository root:

```bash
pip install -r "02-fraud-detection/notebooks/Project/requirements.txt"
streamlit run "02-fraud-detection/notebooks/Project/wanderwise_streamlit_app.py"
```

## Main Updates

- Uses accommodation cost per night instead of raw accommodation cost
- Uses travel month instead of season
- Normalizes destination names to avoid duplicates
- Provides dashboard-style recommendation outputs
