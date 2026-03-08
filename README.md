### Fraud Detection API – MLOps Pipeline

A production-style Machine Learning Fraud Detection System deployed as a FastAPI service with monitoring and containerization.

This project demonstrates how a machine learning model moves from data science experimentation to a production-ready ML service using modern MLOps practices.

The system exposes a REST API for fraud prediction, provides interactive API documentation, and includes metrics monitoring using Prometheus and Grafana.

Project Overview

Financial fraud detection is a critical real-world machine learning problem where models must identify suspicious transactions quickly and accurately.

This project implements a complete ML inference pipeline including:

• Machine learning model experimentation
• API deployment for predictions
• Containerization using Docker
• Monitoring using Prometheus
• Visualization using Grafana

The goal is to simulate how production ML systems operate in real fintech environments.

System Architecture

The system follows a simplified ML production architecture:

User Request
     ↓
 FastAPI Service
     ↓
 Fraud Detection Model
     ↓
 Prediction Response
     ↓
 Monitoring Metrics
 (Prometheus → Grafana)
Tech Stack
Machine Learning

Python
Scikit-learn
Random Forest Classifier
Isolation Forest (initial experimentation)

API Layer

FastAPI
Uvicorn

Monitoring

Prometheus
Grafana

Deployment

Docker

Utilities

Pickle
Python logging

Project Structure
fraud-detection-api-mlops-pipeline/

├── main_rf.py
├── rf_fraud_model.pkl
├── feature_cols.pkl
├── requirements.txt
├── Dockerfile
├── prometheus.yml
├── .gitignore
├── README.md
│
└── docs/
    └── screenshots/
        ├── swagger-ui.png
        ├── prometheus-metrics.png
        └── grafana-dashboard.png
Machine Learning Model Selection

Fraud detection problems often involve identifying rare and unusual transaction patterns. Because of this, anomaly detection techniques are commonly explored before supervised models.

This project followed a two-stage experimentation process.

Initial Model: Isolation Forest

The first model implemented was Isolation Forest, an unsupervised anomaly detection algorithm.

Isolation Forest works by randomly partitioning data points in trees. Anomalies tend to be isolated quickly compared to normal observations.

Advantages of Isolation Forest:

• Does not require labeled fraud data
• Efficient for detecting rare anomalies
• Suitable for high-dimensional datasets
• Widely used for fraud detection and intrusion detection

However, during evaluation the model showed limited predictive performance on this dataset.

Observed limitations:

• Higher false positive rate
• Lower classification accuracy
• Difficulty distinguishing subtle fraud patterns from legitimate behavior

Because the dataset contained labeled fraud transactions, a supervised learning approach became more suitable.

Final Model: Random Forest Classifier

After experimentation, the project transitioned to a Random Forest Classifier.

Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions to improve accuracy and robustness.

Reasons for selecting Random Forest:

• Strong performance on structured financial datasets
• Handles feature interactions effectively
• Robust against overfitting
• Provides higher classification accuracy than Isolation Forest

The Random Forest model produced significantly better fraud prediction performance, which is why it was selected as the final production model used by the API service.

Model Deployment

The trained model is serialized using pickle and loaded during API startup.

Files used:

rf_fraud_model.pkl
feature_cols.pkl

These files contain:

• The trained Random Forest model
• The exact feature column structure used during training

This ensures the API performs consistent real-time inference.

API Service (FastAPI)

The machine learning model is deployed using FastAPI, a modern Python framework designed for high-performance APIs.

FastAPI provides:

• Automatic API documentation
• Request validation
• High performance asynchronous execution
• Production-ready architecture

Prediction Endpoint

POST endpoint:

/predict

Example request:

{
  "features": [0.1, 0.5, 1.2, 3.4]
}

Example response:

{
  "prediction": 1
}

Prediction meaning:

0 = Legitimate transaction
1 = Fraudulent transaction
API Documentation (Swagger UI)

FastAPI automatically generates interactive API documentation.

After running the server, open:

http://localhost:8001/docs

This interface allows:

• Testing API endpoints
• Viewing request schemas
• Viewing response schemas
• Interactive API exploration

Screenshot available in:

docs/screenshots/swagger-ui.png
Monitoring with Prometheus

Monitoring is essential for production ML systems.

Prometheus collects system metrics such as:

• API request count
• Latency
• Service health
• Model inference metrics

Configuration file:

prometheus.yml

Metrics endpoint:

/metrics

Prometheus periodically scrapes this endpoint to gather monitoring data.

Grafana Dashboard

Grafana connects to Prometheus to visualize collected metrics.

The dashboard can show:

• API request trends
• System performance
• Service uptime
• Model usage statistics

Screenshot included:

docs/screenshots/grafana-dashboard.png
Running the Project Locally
Install Dependencies
pip install -r requirements.txt
Start FastAPI Server
uvicorn main_rf:app --host 0.0.0.0 --port 8001
Open API Documentation

Open browser:

http://localhost:8001/docs
Docker Deployment

Build Docker image:

docker build -t fraud-api .

Run container:

docker run -p 8001:8001 fraud-api

The API will now be available at:

http://localhost:8001/docs
Monitoring Setup

Start Prometheus:

prometheus --config.file=prometheus.yml

Prometheus UI:

http://localhost:9090

Grafana UI:

http://localhost:3000
Screenshots

Fraud Detection API

docs/screenshots/Fraud-Detection-API-Page.png

Prometheus Metrics

docs/screenshots/prometheus-metrics.png

Grafana Dashboard

docs/screenshots/grafana-dashboard.png
Key Learning Outcomes

This project demonstrates several important MLOps concepts:

• Deploying ML models as APIs
• Model experimentation and selection
• Monitoring machine learning services
• Containerizing ML systems with Docker
• Designing production ML pipelines

Future Improvements

Possible extensions for this project:

• CI/CD pipeline integration
• Model versioning
• Feature store integration
• Streaming fraud detection
• Kubernetes deployment