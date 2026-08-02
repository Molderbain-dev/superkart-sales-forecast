from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    APP_DIR / "superkart_sales_forecast_pipeline.joblib",
    APP_DIR.parent / "backend" / "superkart_sales_forecast_pipeline.joblib",
    Path.cwd() / "superkart_sales_forecast_pipeline.joblib",
    Path.cwd() / "backend" / "superkart_sales_forecast_pipeline.joblib",
]
EXPECTED_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_Family",
    "Product_Type_Category",
    "MRP_x_Allocated_Area",
    "MRP_per_Weight",
]
INPUT_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_Family",
    "Product_Type_Category",
]


def load_model():
    model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        st.error("The saved model file was not found. Confirm the deployment includes superkart_sales_forecast_pipeline.joblib.")
        st.stop()
    saved = joblib.load(model_path)
    if isinstance(saved, dict):
        return saved["pipeline"], saved.get("features", EXPECTED_FEATURES)
    return saved, EXPECTED_FEATURES


def prepare_prediction_frame(input_df, expected_features):
    input_df = input_df.copy()
    input_df["Product_Sugar_Content"] = input_df["Product_Sugar_Content"].replace({"reg": "Regular"})
    input_df["MRP_x_Allocated_Area"] = input_df["Product_MRP"] * input_df["Product_Allocated_Area"]
    input_df["MRP_per_Weight"] = input_df["Product_MRP"] / input_df["Product_Weight"].replace(0, pd.NA)
    input_df["MRP_per_Weight"] = input_df["MRP_per_Weight"].fillna(input_df["Product_MRP"])

    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    return input_df[expected_features]


def predict_sales(input_df, model, expected_features):
    prediction_df = prepare_prediction_frame(input_df, expected_features)
    return model.predict(prediction_df)


st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")
st.caption("Forecast product-store sales from product, price, display, and store profile details.")

model, expected_features = load_model()

st.subheader("Single prediction")
with st.form("prediction_form"):
    product_weight = st.number_input("Product weight", min_value=0.0, value=12.6)
    sugar = st.selectbox("Product sugar content", ["Low Sugar", "Regular", "No Sugar"])
    allocated_area = st.number_input("Product allocated area ratio", min_value=0.0, max_value=1.0, value=0.07, step=0.001)
    product_type = st.selectbox("Product type", ["Fruits and Vegetables", "Snack Foods", "Dairy", "Frozen Foods", "Household", "Baking Goods", "Canned", "Health and Hygiene", "Meat", "Soft Drinks", "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood"])
    product_mrp = st.number_input("Product MRP", min_value=0.0, value=147.0)
    store_year = st.number_input("Store establishment year", min_value=1900, max_value=2026, value=2009, step=1)
    store_size = st.selectbox("Store size", ["Small", "Medium", "High"])
    city_type = st.selectbox("City type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
    product_family = st.selectbox("Product ID family", ["Food", "Drink", "Non-Consumable"])
    product_category = st.selectbox("Product type category", ["Perishables", "Non Perishables"])
    submitted = st.form_submit_button("Predict sales")

if submitted:
    input_df = pd.DataFrame([{
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Establishment_Year": store_year,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
        "Product_Id_Family": product_family,
        "Product_Type_Category": product_category,
    }])
    prediction = predict_sales(input_df, model, expected_features)[0]
    st.metric("Forecasted product-store sales", f"${prediction:,.2f}")

st.divider()
st.subheader("Batch CSV predictions")

template_df = pd.DataFrame([{
    "Product_Weight": 12.6,
    "Product_Sugar_Content": "Regular",
    "Product_Allocated_Area": 0.07,
    "Product_Type": "Fruits and Vegetables",
    "Product_MRP": 147.0,
    "Store_Establishment_Year": 2009,
    "Store_Size": "Medium",
    "Store_Location_City_Type": "Tier 1",
    "Store_Type": "Supermarket Type1",
    "Product_Id_Family": "Food",
    "Product_Type_Category": "Perishables",
}])

st.download_button(
    "Download CSV template",
    template_df.to_csv(index=False).encode("utf-8"),
    file_name="superkart_batch_template.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader("Upload CSV for batch inference", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    missing_columns = [col for col in INPUT_COLUMNS if col not in batch_df.columns]

    if missing_columns:
        st.error("Missing required columns: " + ", ".join(missing_columns))
    else:
        result_df = batch_df.copy()
        predictions = predict_sales(batch_df[INPUT_COLUMNS], model, expected_features)
        result_df["Forecasted_Product_Store_Sales"] = predictions
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            "Download predictions",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="superkart_batch_predictions.csv",
            mime="text/csv",
        )
