import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = 'superkart_sales_forecast_pipeline.joblib'
model = joblib.load(MODEL_PATH)

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


def prepare_prediction_frame(input_df):
    input_df = input_df.copy()
    missing_columns = [col for col in INPUT_COLUMNS if col not in input_df.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    input_df = input_df[INPUT_COLUMNS]
    input_df["Product_Sugar_Content"] = input_df["Product_Sugar_Content"].replace({"reg": "Regular"})
    input_df["MRP_x_Allocated_Area"] = input_df["Product_MRP"] * input_df["Product_Allocated_Area"]
    input_df["MRP_per_Weight"] = input_df["Product_MRP"] / input_df["Product_Weight"].replace(0, pd.NA)
    input_df["MRP_per_Weight"] = input_df["MRP_per_Weight"].fillna(input_df["Product_MRP"])
    return input_df[EXPECTED_FEATURES]


app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    if isinstance(payload, dict):
        payload = [payload]

    try:
        input_df = prepare_prediction_frame(pd.DataFrame(payload))
        predictions = model.predict(input_df)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({'predictions': [float(x) for x in predictions]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
