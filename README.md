# SuperKart Sales Forecast Deployment

This repository contains a deployable sales forecasting solution for SuperKart.

## Recommended free deployment

Use Streamlit Community Cloud with the self-contained app in `frontend/app.py`.

- Main app file: `frontend/app.py`
- Requirements file: `frontend/requirements.txt`
- Dataset: `frontend/SuperKart.csv`

The Streamlit app trains and caches the Random Forest forecasting pipeline on startup, then provides a user interface for product-store sales prediction.

## Optional API backend

The `backend/` folder contains a Flask API implementation that can be deployed on a Python web host if a backend service is available.
