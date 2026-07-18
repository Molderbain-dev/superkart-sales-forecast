import requests
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")

backend_url = st.text_input(
    "Backend prediction endpoint",
    "https://YOUR-RENDER-BACKEND.onrender.com/predict",
)

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
    payload = {
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
    }

    try:
        response = requests.post(backend_url, json=payload, timeout=30)
        response.raise_for_status()
        prediction = response.json()["predictions"][0]
        st.metric("Forecasted product-store sales", f"${prediction:,.2f}")
    except Exception as exc:
        st.error(f"Prediction request failed: {exc}")
