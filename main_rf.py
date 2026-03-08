import joblib
import pandas as pd
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Fraud Detection API")

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests"
)

PREDICTION_ERRORS = Counter(
    "api_prediction_errors_total",
    "Total number of prediction errors"
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Time taken to process prediction request"
)

model = joblib.load("rf_fraud_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "API is running (RandomForest)"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(data: dict):
    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        df = pd.DataFrame([data])
        df = df.reindex(columns=feature_cols)

        if df.isnull().any().any():
            missing = [col for col in feature_cols if col not in data]
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

        rf_prediction = model.predict(df)[0]
        fraud_probability = model.predict_proba(df)[0][1]

        return {
            "fraud_prediction": int(rf_prediction),
            "fraud_probability": float(fraud_probability)
        }

    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)