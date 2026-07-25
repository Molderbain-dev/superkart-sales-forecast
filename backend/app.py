import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = 'superkart_sales_forecast_pipeline.joblib'
model = joblib.load(MODEL_PATH)

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
    input_df = pd.DataFrame(payload)
    predictions = model.predict(input_df)
    return jsonify({'predictions': [float(x) for x in predictions]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
