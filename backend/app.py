import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def build_model():
    data = pd.read_csv(DATA_PATH)
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
        n_estimators=300,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features=1.0,
        max_depth=None,
        random_state=1,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
    }

    joblib.dump({"pipeline": pipeline, "metrics": metrics, "features": X.columns.tolist()}, MODEL_PATH)
    return pipeline, metrics, X.columns.tolist()


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        return saved["pipeline"], saved.get("metrics", {}), saved.get("features", [])
    return build_model()


model, model_metrics, expected_features = load_or_train_model()

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SuperKart sales forecast API",
        "health": "/health",
        "predict": "/predict",
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "metrics": model_metrics})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    if isinstance(payload, dict):
        payload = [payload]

    input_df = pd.DataFrame(payload)
    input_df = add_features(input_df)

    drop_cols = [c for c in ["Product_Id", "Store_Establishment_Year"] if c in input_df.columns]
    input_df = input_df.drop(columns=drop_cols)

    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = np.nan
    input_df = input_df[expected_features]

    predictions = model.predict(input_df)
    return jsonify({"predictions": [float(x) for x in predictions]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
