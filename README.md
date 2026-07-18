# SuperKart Sales Forecast Deployment

This repository contains a deployable sales forecasting solution for SuperKart.

## Structure

- `backend/`: Flask API for model training/loading and prediction
- `frontend/`: Streamlit interface for business users

## Backend deployment on Render

Create a Render Web Service from this GitHub repository with:

- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

After deployment, use these URLs:

- Backend API: `https://YOUR-RENDER-BACKEND.onrender.com`
- Prediction endpoint: `https://YOUR-RENDER-BACKEND.onrender.com/predict`

## Frontend deployment on Streamlit Community Cloud

Create a Streamlit app from this GitHub repository with:

- Main file path: `frontend/app.py`

Update the default backend URL in `frontend/app.py` after Render provides the backend URL.
