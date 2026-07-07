## 🚦 Traffic Demand Prediction API

A Machine Learning-powered REST API that predicts traffic demand for a given location and time using a trained LightGBM regression model. The project focuses on the backend inference pipeline and provides a simple API for making predictions.

Note: This repository contains the machine learning model and backend API only. No frontend interface has been developed.

## ✨ Features
🚀 Traffic demand prediction using a trained LightGBM model
🧠 Feature engineering on:
Time of day
Geohash hierarchy
Road type
Number of lanes
Presence of large vehicles
Nearby landmarks
Weather conditions
⚡ High-performance REST API built with FastAPI
📄 Interactive Swagger API documentation
🌐 Deployed on Render
📦 JSON-based request and response

## 🛠️ Tech Stack
Python
FastAPI
LightGBM
Pandas
NumPy
Scikit-learn
Uvicorn
Render

## 📚 Live API Documentation

Test the deployed API using the interactive Swagger interface:

https://traffic-demand-api.onrender.com/docs

## 📡 API Endpoint
POST /predict

Predicts the expected traffic demand for the provided input.

Example Request
{
  "geohash": "tdr1x2",
  "timestamp": "14:30",
  "RoadType": "Highway",
  "NumberofLanes": 4,
  "LargeVehicles": "Yes",
  "Landmarks": "Mall",
  "Weather": "Clear"
}
Example Response
{
  "predicted_traffic_demand": 87.42
}

## ⚙️ Running Locally
1. Clone the repository
git clone https://github.com/KavyanshJain13/traffic-demand-api.git
cd traffic-demand-api
2. Create a virtual environment
python -m venv venv
Windows
venv\Scripts\activate
Linux/macOS
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Start the server
uvicorn app:app --reload

The API will be available at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

## 📁 Project Structure
traffic-demand-api/
│── app.py
│── model.pkl
│── encoder.pkl
│── requirements.txt
│── README.md
└── ...

## 🎯 Use Cases
Smart city traffic analytics
Intelligent transportation systems
Traffic simulation pipelines
Demand forecasting services
Integration into traffic management platforms

## 🧑🏻‍💻 Author

Kavyansh Jain