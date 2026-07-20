import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = "superkart_sales_forecast_pipeline.joblib"
DATA_PATH = "SuperKart.csv"
TARGET = "Product_Store_Sales_Total"


def add_features(df):
    df = df.copy()
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})
    df["Store_Age_Years"] = 2026 - df["Store_Establishment_Year"]
    df["Product_Id_Family"] = df["Product_Id"].astype(str).str[:2]
    perishables = {"Fruits and Vegetables", "Dairy", "Meat", "Seafood", "Breakfast", "Breads"}
    df["Product_Type_Category"] = np.where(df["Product_Type"].isin(perishables), "Perishables", "Non Perishables")
    df["MRP_x_Allocated_Area"] = df["Product_MRP"] * df["Product_Allocated_Area"]
    df["MRP_per_Weight"] = df["Product_MRP"] / df["Product_Weight"].replace(0, np.nan)
    df["MRP_per_Weight"] = df["MRP_per_Weight"].fillna(df["Product_MRP"])
    return df


@st.cache_resource(show_spinner="Training SuperKart forecast model...")
def load_model():
    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        return saved["pipeline"], saved.get("features", [])

    data = pd.read_csv(DATA_PATH)
    data = data.sample(n=min(len(data), 5000), random_state=1)
    data = add_features(data)
    X = data.drop(columns=[TARGET, "Product_Id", "Store_Establishment_Year"])
    y = data[TARGET]

    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=80,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features=1.0,
        max_depth=None,
        random_state=1,
        n_jobs=1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=1)
    pipeline.fit(X_train, y_train)
    joblib.dump({"pipeline": pipeline, "features": X.columns.tolist()}, MODEL_PATH)
    return pipeline, X.columns.tolist()


st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")
st.caption("Forecast product-store sales using SuperKart product, pricing, display, and outlet details.")

model, expected_features = load_model()

with st.form("prediction_form"):
    product_id = st.text_input("Product ID", "FD123")
    product_weight = st.number_input("Product weight", min_value=0.0, value=12.6)
    sugar = st.selectbox("Product sugar content", ["Low Sugar", "Regular", "No Sugar"])
    allocated_area = st.number_input("Product allocated area ratio", min_value=0.0, max_value=1.0, value=0.07, step=0.001)
    product_type = st.selectbox(
        "Product type",
        [
            "Fruits and Vegetables", "Snack Foods", "Dairy", "Frozen Foods", "Household",
            "Baking Goods", "Canned", "Health and Hygiene", "Meat", "Soft Drinks",
            "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood",
        ],
    )
    product_mrp = st.number_input("Product MRP", min_value=0.0, value=147.0)
    store_id = st.selectbox("Store ID", ["OUT001", "OUT002", "OUT003", "OUT004"])
    establishment_year = st.number_input("Store establishment year", min_value=1900, max_value=2026, value=2009)
    store_size = st.selectbox("Store size", ["Small", "Medium", "High"])
    city_type = st.selectbox("City type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
    submitted = st.form_submit_button("Predict sales")

if submitted:
    input_df = pd.DataFrame([{
        "Product_Id": product_id,
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Id": store_id,
        "Store_Establishment_Year": establishment_year,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
    }])
    input_df = add_features(input_df).drop(columns=["Product_Id", "Store_Establishment_Year"])
    input_df = input_df[expected_features]
    prediction = model.predict(input_df)[0]
    st.metric("Forecasted product-store sales", f"${prediction:,.2f}")


